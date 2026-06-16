"""
rtcc.py: Real-time coupled object that provides data for an ODE propagator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psi4
import numpy as np
from pycc.ccwfn import HAS_TORCH
if HAS_TORCH:
    import torch
from pycc.utils import zeros_like, clone, conj, reshape, dot
import pickle as pk
from os.path import exists
import opt_einsum

from pycc._typing import Tensor

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn
    from pycc.cclambda import cclambda
    from pycc.ccdensity import ccdensity


class rtcc(object):
    """
    A Real-time CC object for ODE propagation.

    Attributes
    -----------
    ccwfn: PyCC ccwfn object
        the coupled cluster T amplitudes and supporting data structures
    cclambda : PyCC cclambda object
        the coupled cluster Lambda amplitudes and supporting data structures
    ccdensity : PyCC ccdensity object
        the coupled cluster one- and two-electron densities
    V: the time-dependent laser field
        must accept only the current time as an argument, e.g., as defined in lasers.py
    mu: list of NumPy arrays
        the dipole integrals for each Cartesian direction (taken from Hamiltonian object)
    mu_tot: NumPy arrays
        1/sqrt(3) * sum of dipole integrals (for isotropic field)
    magnetic: bool
        whether or not to compute the magnetic dipole integrals and value (default = False)
    m: list of NumPy arrays
        the magnetic dipole integrals for each Cartesian direction (only if magnetic = True) (taken from Hamiltonian object)

    Parameters
    ----------
    magnetic: bool
        optionally store magnetic dipole integrals (default = False)
    kick: bool or str
        optionally isolate 'x', 'y', or 'z' electric field kick (default = False)

    Methods
    -------
    f(): Returns a flattened NumPy array of cluster residuals
        The ODE defining function (the right-hand-side of a Runge-Kutta solver)
    collect_amps():
        Collect the cluster amplitudes and phase into a single vector
    extract_amps():
        Separate a flattened array of amplitudes (and phase) into the t1, t2, l1, and l2 components
    dipole()
        Compute the electronic or magnetic dipole moment for a given time t
    energy()
        Compute the CC correlation energy for a given time t
    lagrangian()
        Compute the CC Lagrangian energy for a given time t
    """
    def __init__(self, ccwfn: "CCwfn", cclambda: "cclambda", ccdensity: "ccdensity", V: Tensor, magnetic: bool = False, kick: Any = None) -> None:
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.ccdensity = ccdensity
        self.contract = self.ccwfn.contract
        self.V = V
         
        # Grab the requested dipole integrals from the Hamiltonian
        self.mu = self.ccwfn.H.mu
        if self.ccwfn.precision == 'SP':
            self.mu = np.complex64(self.mu)

        if kick:
            s_to_i = {"x":0, "y":1, "z":2}
            self.mu_tot = self.mu[s_to_i[kick.lower()]]
        else:
            self.mu_tot = sum(self.mu)/np.sqrt(3.0)  # isotropic field
  
        if HAS_TORCH and isinstance(self.ccwfn.t1, torch.Tensor):
            if self.ccwfn.precision == 'DP':
                self.mu = torch.tensor(self.mu, dtype=torch.complex128, device=self.ccwfn.device1)
            elif self.ccwfn.precision == 'SP':
                self.mu = torch.tensor(self.mu, dtype=torch.complex64, device=self.ccwfn.device1)
            
            if kick:
                s_to_i = {"x":0, "y":1, "z":2}
                self.mu_tot = self.mu[s_to_i[kick.lower()]]
            else:
                self.mu_tot = sum(self.mu) / (torch.sqrt(torch.tensor(3.0)).item())

        if magnetic:
            self.magnetic = True
            self.m = self.ccwfn.H.m
            if self.ccwfn.precision == 'SP':
                self.m = np.complex64(self.m)
            if HAS_TORCH and isinstance(self.ccwfn.t1, torch.Tensor):
                if self.ccwfn.precision == 'DP':
                    self.m = torch.tensor(self.m, dtype=torch.complex128, device=self.ccwfn.device1)
                elif self.ccwfn.precision == 'SP':
                    self.m = torch.tensor(self.m, dtype=torch.complex64, device=self.ccwfn.device1)
        else:
            self.magnetic = False

    def f(self, t: float, y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : float
            Current time step in the external ODE solver
        y : NumPy array
            flattened array of cluster amplitudes

        Returns
        -------
        f(t, y): NumPy array
            flattened array of cluster residuals and phase
        """
        # Extract amplitude tensors
        t1, t2, l1, l2, phase = self.extract_amps(y)

        # Add the field to the Hamiltonian
        F = clone(self.ccwfn.H.F) + self.mu_tot * self.V(t)

        # Compute the current residuals
        rt1, rt2 = self.ccwfn.residuals(F, t1, t2, real_time=True)
        rt1 = rt1 * (-1.0j)
        rt2 = rt2 * (-1.0j)
        if self.ccwfn.local is not None:
            rt1, rt2 = self.ccwfn.Local.filter_res(rt1, rt2)

        rl1, rl2 = self.cclambda.residuals(F, t1, t2, l1, l2)
        rl1 = rl1 * (+1.0j)
        rl2 = rl2 * (+1.0j)
        if self.ccwfn.local is not None:
            rl1, rl2 = self.ccwfn.Local.filter_res(rl1, rl2)

        # Phase contribution = exp(-phase(t))
        phase = self.phase(F, t1, t2)

        # Pack up the residuals
        y = self.collect_amps(rt1, rt2, rl1, rl2, phase)

        return y

    def collect_amps(self, t1, t2, l1, l2, phase):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes or residuals
        phase : scalar
            current wave function phase

        Returns
        -------
        NumPy array
            amplitudes or residuals and phase as a vector (flattened array)
        """
        if HAS_TORCH and isinstance(t1, torch.Tensor):
            t1 = torch.flatten(t1)
            t2 = torch.flatten(t2)
            l1 = torch.flatten(l1)
            l2 = torch.flatten(l2)
            if self.ccwfn.precision == 'DP':
                return torch.cat((t1, t2, l1, l2, torch.tensor(phase, dtype=torch.complex128, device=self.ccwfn.device1).unsqueeze(0))).type(torch.complex128)
            if self.ccwfn.precision == 'SP':
                return torch.cat((t1, t2, l1, l2, torch.tensor(phase, dtype=torch.complex64, device=self.ccwfn.device1).unsqueeze(0))).type(torch.complex64)
        else:
            if self.ccwfn.precision == 'DP':
                return np.concatenate((t1, t2, l1, l2, phase), axis=None).astype('complex128')
            if self.ccwfn.precision == 'SP':
                return np.concatenate((t1, t2, l1, l2, phase), axis=None).astype('complex64')

    def extract_amps(self, y):
        """
        Parameters
        ----------
        y : NumPy array
            amplitudes or residuals and phase as a vector (flattened array)

        Returns
        -------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes or residuals
        phase : scalar
            current wave function phase
        """
        no = self.ccwfn.no
        nv = self.ccwfn.nv

        # Extract the amplitudes
        len1 = no*nv
        len2 = no*no*nv*nv
        t1 = reshape(y[:len1], (no, nv))
        t2 = reshape(y[len1:(len1+len2)], (no, no, nv, nv))
        l1 = reshape(y[(len1+len2):(len1+len2+len1)], (no, nv))
        l2 = reshape(y[(len1+len2+len1):-1], (no, no, nv, nv))
        # Extract the phase; .item() returns a Python scalar from a torch tensor
        if HAS_TORCH and isinstance(y, torch.Tensor):
            phase = y[-1].item()
        else:
            phase = y[-1]

        return t1, t2, l1, l2, phase

    def dipole(self, t1: Tensor, t2: Tensor, l1: Tensor, l2: Tensor, magnetic: bool = False, real_time: bool = False):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes
        magnetic       : Bool (default = False)
            compute magnetic dipole rather than electric

        Returns
        -------
        x, y, z : scalars
            Cartesian components of the correlated dipole moment
        """
        if self.ccwfn.model == 'CC3':
            (opdm, opdm_cc3) = self.ccdensity.compute_onepdm(t1, t2, l1, l2, real_time=real_time)
        else:
            opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2)

        if magnetic:
            ints = self.m
        else:
            ints = self.mu

        if self.ccwfn.model == 'CC3':
            # Calculating T1-transformed dipole integral  
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            ints_cc3 = zeros_like(ints)
            for i in range(3):
                if HAS_TORCH and isinstance(t1, torch.Tensor):                    
                    ints_cc3 = ints_cc3.type_as(t1)
                else:
                    ints_cc3 = ints_cc3.astype(t1.dtype)
                ints_cc3[i][:no,:no] = self.ccdensity.build_Moo(no, nv, ints[i], t1)     
                ints_cc3[i][-nv:,-nv:] = self.ccdensity.build_Mvv(no, nv, ints[i], t1)      
     
            x = dot(ints[0].flatten(), opdm.flatten())
            y = dot(ints[1].flatten(), opdm.flatten())
            z = dot(ints[2].flatten(), opdm.flatten())
            # Contractions between Doo_cc3, Dvv_cc3 and T1-transformed dipole integrals
            x += dot(ints_cc3[0].flatten(), opdm_cc3.flatten())
            y += dot(ints_cc3[1].flatten(), opdm_cc3.flatten())
            z += dot(ints_cc3[2].flatten(), opdm_cc3.flatten())
            
            return x, y, z

        else:
            x = dot(ints[0].flatten(), opdm.flatten())
            y = dot(ints[1].flatten(), opdm.flatten())
            z = dot(ints[2].flatten(), opdm.flatten())

            return x, y, z    
    
    def lagrangian(self, t, t1, t2, l1, l2):
        """
        Parameters
        ----------
        t : float
            current time step in external ODE solver
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes

        Returns
        -------
        ecc : scalars
            CC Lagrangian energy (including reference contribution, but excluding nuclear repulsion)
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        ERI = self.ccwfn.H.ERI
        if self.ccwfn.model == 'CC3':
            (opdm, opdm_cc3) = self.ccdensity.compute_onepdm(t1, t2, l1, l2)
            opdm = opdm + opdm_cc3
        else:            
            opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2)
        Doooo = self.ccdensity.build_Doooo(t1, t2, l2)
        Dvvvv = self.ccdensity.build_Dvvvv(t1, t2, l2)
        Dooov = self.ccdensity.build_Dooov(t1, t2, l1, l2)
        Dvvvo = self.ccdensity.build_Dvvvo(t1, t2, l1, l2)
        Dovov = self.ccdensity.build_Dovov(t1, t2, l1, l2)
        Doovv = self.ccdensity.build_Doovv(t1, t2, l1, l2)
        contract = self.contract
        
        F = clone(self.ccwfn.H.F) + self.mu_tot * self.V(t)
        eref = self._eref(F)

        eone = dot(F.flatten(), opdm.flatten())

        oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], Doooo)
        vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], Dvvvv)
        ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], Dooov)
        vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], Dvvvo)
        ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], Dovov)
        oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], Doovv)
        etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy
        return eref + eone + etwo

    def _eref(self, F):
        """Reference (zeroth-order) energy contribution to the CC quasienergy.

        Returns ``2 Tr F_oo - sum_{ia} L_iaia`` (i.e. ``2 np.trace(F[o,o]) -
        np.trace(np.trace(L[o,o,o,o], axis1=1, axis2=3))``). Evaluated through the
        backend ``contract`` so the single expression runs unchanged on NumPy or
        torch — ``contract`` handles any device transfer of ``H.L`` in GPU mode.
        """
        o = self.ccwfn.o
        contract = self.contract
        eref = 2.0 * contract('ii->', F[o, o])
        eref -= contract('iaia->', self.ccwfn.H.L[o, o, o, o])
        return eref

    def phase(self, F, t1, t2):
        """
        Parameters
        ----------
        F : NumPy array
            current (field-dependent Fock operator
        t1, t2: NumPy arrays
            current cluster amplitudes

        Returns
        -------
        phase: scalar
            wave function quasienergy/phase-factor with contribution defined as = exp(-phase(t))
        """
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        L = self.ccwfn.H.L

        eref = self._eref(F)

        if self.ccwfn.model == 'CCD':
            ecc = contract('ijab,ijab->', t2, L[o,o,v,v])
        else:
            ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
            ecc += contract('ijab,ijab->', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])

        return (eref + ecc) * (-1.0j)

    def autocorrelation(self, y_left: Tensor, y_right: Tensor):
        """
        Parameters
        ----------
        y_left, y_right : Numpy arrays
            amplitudes or residuals and phase as a vector (flattened array) for two different time steps

        Returns
        -------
        float
            the autocorrelation function, A(t1, t2) as defined in Eq. (18) of J. Chem. Phys. 150, 144106 (2019)
        """
        contract = opt_einsum.contract

        t1_l, t2_l, l1_l, l2_l, phase_l = self.extract_amps(y_left)
        t1_r, t2_r, l1_r, l2_r, phase_r = self.extract_amps(y_right)

        A = 1
        A += contract("ia,ia->", l1_l, (t1_r - t1_l))
        A += 0.5*contract("ijab,ijab->", l2_l, (t2_r - t2_l))
        A += 0.5*contract("ijab,ia,jb->", l2_l, t1_l, t1_l)
        A += 0.5*contract("ijab,ia,jb->", l2_l, t1_r, t1_r)
        A -= contract("ijab,ia,jb->", l2_l, t1_l, t1_r)
        A *= np.exp(-phase_l) * np.exp(phase_r)

        B = 1
        B -= contract("ia,ia->", l1_r, (t1_r - t1_l))
        B -= 0.5*contract("ijab,ijab->", l2_r, (t2_r - t2_l))
        B += 0.5*contract("ijab,ia,jb->", l2_r, t1_r, t1_r)
        B += 0.5*contract("ijab,ia,jb->", l2_r, t1_l, t1_l)
        B -= contract("ijab,ia,jb->", l2_r, t1_l, t1_r)
        B *= np.exp(-phase_r) * np.exp(phase_l)
        
        return 0.5*A + 0.5*conj(B)

    def step(self,ODE,yi,t,ref=False):
        """
        A single step in the propagation

        Parameters
        ----------
        ODE : integrators object
            callable integrator with timestep attribute
        yi : NumPy array
            flattened array of initial cluster amplitudes or residuals
        t : float
            current timestep
        ref : bool
            include reference contribution to properties (optional, default = False)

        Returns
        -------
        y : NumPy array
            flatten array of cluster amplitudes or residuals at time t + ODE.h
        ret: dict
            dict of properties at time t + ODE.h
        """
        # step
        y = ODE(self.f,t,yi)

        # calculate properties
        ret = {}
        t1, t2, l1, l2, phase = self.extract_amps(y)
        ret['ecc'] = self.lagrangian(t,t1,t2,l1,l2)
        mu_x, mu_y, mu_z = self.dipole(t1,t2,l1,l2,magnetic=False)
        ret['mu_x'] = mu_x
        ret['mu_y'] = mu_y
        ret['mu_z'] = mu_z
        if self.magnetic:
            m_x, m_y, m_z = self.dipole(t1,t2,l1,l2,magnetic=True)
            ret['m_x'] = m_x
            ret['m_y'] = m_y
            ret['m_z'] = m_z
        return y,ret

    def propagate(self, ODE: Any, yi: Tensor, tf: float, ti: float = 0, ref: bool = False, chk: bool = False, tchk: bool = False,
                  ofile: str = "output.pk", tfile: str = "t_out.pk", cfile: str = "chk.pk", k: int = 2):
        """
        Propagate the function yi from time ti to time tf

        Parameters
        ----------
        ODE : integrators object
            callable integrator with timestep attribute
        yi : NumPy array
            flattened array of initial cluster amplitudes or residuals and phase
        tf : float
            final timestep
        ti : float
            initial timestep (optional, default = 0)
        ref : bool
            include reference contribution to properties (optional, default = False)
        chk : bool
            save results and final y,t to file every step, plus ref wfn
        tchk : bool or int
            return and save {t1,t2,l1,l2} to file every tchk steps (optional, default = False)
        ofile : str
            name of output file (optional, default='output.pk')
        tfile : str
            name of amplitude output file (optional, default='t_out.pk')
        cfile : str
            name of checkpoint file (optional, default='chk.pk')
        k : int
            number of decimals to include in str keys for return dict

        Returns
        -------
        ret : dict
            dict of properties for all timesteps
        ret_t : dict
            dict of {t1,t2,l1,l2} for every tchk steps (iff type(tchk)==int)
        """
        # setup
        point = 0
        key = '%.*f' % (k,ti) 

        # pull previous chkpt or properties?
        if chk:
            if exists(cfile):
                with open(cfile,'rb') as cf:
                    chkp = pk.load(cf)
            else:
                chkp = {}
                self.ccwfn.ref.to_file('ref_wfn')
        if chk and exists(ofile):
            with open(ofile,'rb') as of:
                ret = pk.load(of)
        else:
            ret = {key: {}}

        # pull previous amplitudes?
        if tchk != False:
            save_t = True
            if chk and exists(tfile):
                with open(tfile,'rb') as ampf:
                    ret_t = pk.load(ampf)
            else:
                ret_t = {key: None}
            t1,t2,l1,l2,phase = self.extract_amps(yi)
            ret_t[key] = {"t1":t1,
                    "t2":t2,
                    "l1":l1,
                    "l2":l2,
                    "phase":phase}
        else:
            save_t = False

        # initial properties
        t1, t2, l1, l2, phase = self.extract_amps(yi)
        ret[key]['ecc'] = self.lagrangian(ti,t1,t2,l1,l2)
        mu_x, mu_y, mu_z = self.dipole(t1,t2,l1,l2,magnetic=False)
        ret[key]['mu_x'] = mu_x
        ret[key]['mu_y'] = mu_y
        ret[key]['mu_z'] = mu_z
        if self.magnetic:
            m_x, m_y, m_z = self.dipole(t1,t2,l1,l2,magnetic=True)
            ret[key]['m_x'] = m_x
            ret[key]['m_y'] = m_y
            ret[key]['m_z'] = m_z

        # propagate
        t = ti
        while t < tf:
            point += 1
            y,props = self.step(ODE,yi,t,ref)
            t += ODE.h
            key = '%.*f' % (k,t) 
            ret[key] = props
            yi = y

            # update checkpoint if asked
            if chk:
                chkp['y'] = y
                chkp['time'] = t
                with open(ofile,'wb') as of:
                    pk.dump(ret,of,pk.HIGHEST_PROTOCOL)
                with open(cfile,'wb') as cf:
                    pk.dump(chkp,cf,pk.HIGHEST_PROTOCOL)

            # save amplitudes if asked and correct timestep
            if save_t and (point%tchk<0.0001):
                t1,t2,l1,l2,phase = self.extract_amps(y)
                ret_t[key] = {"t1":t1,
                        "t2":t2,
                        "l1":l1,
                        "l2":l2}
                with open(tfile,'wb') as ampf:
                    pk.dump(ret_t,ampf,pk.HIGHEST_PROTOCOL)

        if save_t:
            return ret, ret_t
        else:
            return ret
