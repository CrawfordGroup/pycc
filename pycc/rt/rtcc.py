"""
rtcc.py: Real-time coupled object that provides data for an ODE propagator
"""

import psi4
import numpy as np
import torch
import pickle as pk
from os.path import exists

# Will be removed after generalize cc_contract
import opt_einsum


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
        the dipole integrals for each Cartesian direction
    mu_tot: NumPy arrays
        1/sqrt(3) * sum of dipole integrals (for isotropic field)
    magnetic: bool
        whether or not to compute the magnetic dipole integrals and value (default = False)
    m: list of NumPy arrays
        the magnetic dipole integrals for each Cartesian direction (only if magnetic = True)

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
    def __init__(self, ccwfn, cclambda, ccdensity, V, magnetic = False, kick = None):
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.ccdensity = ccdensity
        self.contract = self.ccwfn.contract
        self.V = V
         
        # Prep the dipole integrals in MO basis
        mints = psi4.core.MintsHelper(ccwfn.ref.basisset())
        dipole_ints = mints.ao_dipole()
        C = np.asarray(self.ccwfn.C)  # May be localized MOs, so we take them from ccwfn
        self.mu = []
        for axis in range(3):
            self.mu.append(C.T @ np.asarray(dipole_ints[axis]) @ C)
        if kick:
            s_to_i = {"x":0, "y":1, "z":2}
            self.mu_tot = self.mu[s_to_i[kick.lower()]]
        else:
            self.mu_tot = sum(self.mu)/np.sqrt(3.0)  # isotropic field
  
        if isinstance(self.ccwfn.t1, torch.Tensor):
            self.mu = torch.tensor(self.mu, dtype=torch.complex128, device=self.ccwfn.device1)
            self.mu_tot = sum(self.mu) / (torch.sqrt(torch.tensor(3.0)).item())

        if magnetic:
            self.magnetic = True
            m_ints = mints.ao_angular_momentum()
            self.m = []
            for axis in range(3):
                m = (C.T @ (np.asarray(m_ints[axis])*-0.5) @ C)
                self.m.append(m*1.0j)
        else:
            self.magnetic = False

    def f(self, t, y):
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
        if isinstance(t1, torch.Tensor):
            F = self.ccwfn.H.F.clone() + self.mu_tot * self.V(t)
        else:
            F = self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)
        # Compute the current residuals
        rt1, rt2 = self.ccwfn.residuals(F, t1, t2)
        rt1 = rt1 * (-1.0j)
        rt2 = rt2 * (-1.0j)
        if self.ccwfn.local is not None:
            rt1, rt2 = self.ccwfn.Local.filter_res(rt1, rt2)

        rl1, rl2 = self.cclambda.residuals(F, t1, t2, l1, l2)
        rl1 = rl1 * (+1.0j)
        rl2 = rl2 * (+1.0j)
        if self.ccwfn.local is not None:
            rl1, rl2 = self.ccwfn.Local.filter_res(rl1, rl2)

        # Pack up the residuals
        y = self.collect_amps(rt1, rt2, rl1, rl2, phase)

        return y

    def collect_amps(self, t1, t2, l1, l2, phase):
        """
        Parameters
        ----------
        phase : scalar
            current wave function phase
        t1, t2, l2, l2 : NumPy arrays
            current cluster amplitudes or residuals

        Returns
        -------
        NumPy array
            amplitudes or residuals and phase as a vector (flattened array)
        """
        if isinstance(t1, torch.Tensor):
            t1 = torch.flatten(t1)
            t2 = torch.flatten(t2)
            l1 = torch.flatten(l1)
            l2 = torch.flatten(l2)
            return torch.cat((t1, t2, l1, l2, phase)).type(torch.complex128)
        else:
            return np.concatenate((t1, t2, l1, l2, phase), axis=None).astype('complex128')


    def extract_amps(self, y):
        """
        Parameters
        ----------
        y : NumPy array
            flattened array of wave function phase and cluster amplitudes or residuals

        Returns
        -------
        phase : scalar
            current wave function phase
        t1, t2, l2, l2 : NumPy arrays
            current cluster amplitudes or residuals
        """
        no = self.ccwfn.no
        nv = self.ccwfn.nv

        # Extract the amplitudes
        len1 = no*nv
        len2 = no*no*nv*nv
        if isinstance(y, torch.Tensor):
            t1 = torch.reshape(y[:len1], (no, nv))
            t2 = torch.reshape(y[len1:(len1+len2)], (no, no, nv, nv))
            l1 = torch.reshape(y[(len1+len2):(len1+len2+len1)], (no, nv))
            l2 = torch.reshape(y[(len1+len2+len1):(2*len1+2*len2)], (no, no, nv, nv))
        else:
            t1 = np.reshape(y[:len1], (no, nv))
            t2 = np.reshape(y[len1:(len1+len2)], (no, no, nv, nv))
            l1 = np.reshape(y[(len1+len2):(len1+len2+len1)], (no, nv))
            l2 = np.reshape(y[(len1+len2+len1):(2*len1+2*len2)], (no, no, nv, nv))

        # Extract the phase
        phase = y[y.size-1]

        return t1, t2, l1, l2, phase

    def dipole(self, t1, t2, l1, l2, withref = True, magnetic = False):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes
        withref        : Bool (default = True)
            include reference contribution to the OPDM
        magnetic       : Bool (default = False)
            compute magnetic dipole rather than electric

        Returns
        -------
        x, y, z : complex128
            Cartesian components of the dipole moment
        """
        opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2, withref=withref)
        if magnetic:
            ints = self.m
        else:
            ints = self.mu

        x = ints[0].flatten().dot(opdm.flatten())
        y = ints[1].flatten().dot(opdm.flatten())
        z = ints[2].flatten().dot(opdm.flatten())
        return x, y, z

    def energy(self, t, t1, t2, l1, l2):
        """
        Parameters
        ----------
        t : float
            current time step in external ODE solver
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes

        Returns
        -------
        ecc : complex128
            CC correlation energy
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        if isinstance(t1, torch.Tensor):
            F = self.ccwfn.H.F.clone() + self.mu_tot * self.V(t)
        else:
            F = self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)
        
        contract = self.contract 

        ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
        L = self.ccwfn.H.L
        ecc = ecc + contract('ijab,ijab->', build_tau(t1, t2), L[o,o,v,v])
        return ecc

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
        ecc : complex128
            CC Lagrangian energy (including reference contribution, but excluding nuclear repulsion)
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        ERI = self.ccwfn.H.ERI
        opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2)
        Doooo = self.ccdensity.build_Doooo(t1, t2, l2)
        Dvvvv = self.ccdensity.build_Dvvvv(t1, t2, l2)
        Dooov = self.ccdensity.build_Dooov(t1, t2, l1, l2)
        Dvvvo = self.ccdensity.build_Dvvvo(t1, t2, l1, l2)
        Dovov = self.ccdensity.build_Dovov(t1, t2, l1, l2)
        Doovv = self.ccdensity.build_Doovv(t1, t2, l1, l2)
        contract = self.contract
        
        if isinstance(t1, torch.Tensor):
            F = self.ccwfn.H.F.clone() + self.mu_tot * self.V(t)
    
            eref = 2.0 * torch.trace(F[o,o])
            # torch.trace doesn't have "axis" argument
            #eref -= torch.trace(torch.trace(tmp, axis1=1, axis2=3))
            tmp = self.ccwfn.H.L[o,o,o,o].to(self.ccwfn.device1)
            eref -= torch.trace(opt_einsum.contract('i...i', tmp.swapaxes(0,1), backend='torch'))
            del tmp

        else:
            F = self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)

            eref = 2.0 * np.trace(F[o,o])
            eref -= np.trace(np.trace(self.ccwfn.H.L[o,o,o,o], axis1=1, axis2=3))

        eone = F.flatten().dot(opdm.flatten())

        oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], Doooo)
        vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], Dvvvv)
        ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], Dooov)
        vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], Dvvvo)
        ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], Dovov)
        oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], Doovv)
        etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy
        return eref + eone + etwo

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
        mu_x, mu_y, mu_z = self.dipole(t1,t2,l1,l2,withref=ref,magnetic=False)
        ret['mu_x'] = mu_x
        ret['mu_y'] = mu_y
        ret['mu_z'] = mu_z
        if self.magnetic:
            m_x, m_y, m_z = self.dipole(t1,t2,l1,l2,withref=ref,magnetic=True)
            ret['m_x'] = m_x
            ret['m_y'] = m_y
            ret['m_z'] = m_z
        return y,ret

    def propagate(self, ODE, yi, tf, ti=0, ref=False, chk=False, tchk=False,
                  ofile="output.pk",tfile="t_out.pk",cfile="chk.pk",k=2):
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
        mu_x, mu_y, mu_z = self.dipole(t1,t2,l1,l2,withref=ref,magnetic=False)
        ret[key]['mu_x'] = mu_x
        ret[key]['mu_y'] = mu_y
        ret[key]['mu_z'] = mu_z
        if self.magnetic:
            m_x, m_y, m_z = self.dipole(t1,t2,l1,l2,withref=ref,magnetic=True)
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
