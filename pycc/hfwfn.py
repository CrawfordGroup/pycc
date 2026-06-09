"""
hfwfn.py: HF Wfn Class
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .utils import helper_diis, cc_contract
from .hamiltonian import Hamiltonian

try:
    import einsums as ein
    HAS_EINSUMS = True
except ImportError:
    HAS_EINSUMS = False

class wfn(object):
    """
    An RHF wave function and energy object.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction object
        the reference wave function built by Psi4 energy() method
    escf : float
        the energy of the HF-SCF wave function (including nuclear repulsion contribution)
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, the spin-adapted ERIs (L), and various property integrals
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace

    Methods
    -------
    """

    def __init__(self, scf_wfn, **kwargs):
        """
        Parameters
        ----------
        scf_wfn : Psi4 Wavefunction Object
            computed by Psi4 energy() method

        Returns
        -------
        None
        """

        time_init = time.time()

        self.ref = scf_wfn
        self.escf = self.ref.energy()

        self.no   = int(sum(self.ref.doccpi()))
        self.nmo  = self.ref.nmo()
        self.nv   = self.nmo - self.no

        print("NMO = %d; NO = %d; NV = %d" % (self.nmo, self.no, self.nv))

        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)
        o = self.o
        v = self.v

        self.C = self.ref.Ca_subset("AO", "ALL")

        eps_so_blocked  = self.ref.epsilon_a_subset("SO", "ALL")
        eps_active_so   = np.concatenate([np.array(eps_so_blocked.nph[h])
                                          for h in range(self.ref.nirrep())])
        sort_idx        = np.argsort(eps_active_so, kind='stable')

        irrep_labels    = self.ref.molecule().irrep_labels()
        mo_irreps       = np.array([h for h in range(self.ref.nirrep())
                                    for _ in range(self.ref.nmopi()[h])])
        mo_irreps       = mo_irreps[sort_idx]
        mo_irrep_labels = [irrep_labels[h] for h in mo_irreps]
        eps_active      = eps_active_so[sort_idx]

        # Print MO summary
        print("\nMOs by energy:")
        print(f"  {'#':>4}  {'Irrep':>6}  {'Energy':>16}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*16}")
        for i, (eps, label) in enumerate(zip(eps_active, mo_irrep_labels)):
            if i == self.no:
                print(f"  {'.'*4}  {'.'*6}  {'.'*16}")
            idx = i if i < self.no else i - self.no
            print(f"  {idx:>4}  {label:>6}  {eps:>16.10f}")

        self.H = Hamiltonian(self.ref, self.C, self.C, self.C, self.C)

        # Initiate the object for a generalized contraction function 
        # for GPU or CPU.  
        valid_device = ['CPU', 'GPU']
        device = kwargs.pop('device', 'CPU')
        if device.upper() not in valid_device:
            raise Exception("%s is not an allowed device." % (device))
        self.device = device.upper()
        if self.device == 'GPU' and HAS_TORCH == False:
            self.device = 'CPU'
            print('GPU requested, but torch not available.  Using CPU instead.')
        self.contract = cc_contract(device=self.device)

        self.einsums = kwargs.pop('einsums', False)
        if self.einsums and HAS_EINSUMS:
            self.ec = ein.einsums_contract()
            print('Contractions will use the C++ Einsums library where implemented...')
        elif self.einsums and not HAS_EINSUMS:
            self.einsums = False
            print('C++ Einsums library requested but not available.')

        print("HFWFN object initialized in %.3f seconds." % (time.time() - time_init))


    def solve_cphf(self, pert_type, pert_index):
        """
        Solve the CPHF equations for a given perturbation. Results are cached

        Parameters
        ----------
        pert_type : string
            Type of perturbation: 'nuclear', 'electric', 'magnetic' are allowed options
        pert_index : int
            Choice of perturbation: (a) if pert_type == 'electric' or 'magnetic', this is the Cartesian component of the
            field/moment; (b) if pert_type == 'nuclear', this is the combined atom number and Cartesian coordinate,
            i.e. for an N-atom system, pert_index = N*3+cart.

        Returns
        -------
        U : NumPy array
            The nmo x nmo CPHF coefficient matrix for the given perturbation
        """

        key = (pert_type, pert_index)
        if key in self._U:
            return self._U[key]

        nbf, no, nv = self.nbf, self.no, self.nv
        o, v = self.o, self.v
        C    = self.C
        U    = np.zeros((nbf, nbf), dtype=complex)

        # Nuclear displacement 
        if pert_type == 'nuclear':
            N, a = divmod(pert_index, 3)

            T_d1      = self.mints.mo_oei_deriv1('KINETIC',   N, self.C_p4, self.C_p4)
            V_d1      = self.mints.mo_oei_deriv1('POTENTIAL', N, self.C_p4, self.C_p4)
            S_d1      = self.mints.mo_oei_deriv1('OVERLAP',   N, self.C_p4, self.C_p4)
            ERI_d1    = self.mints.mo_tei_deriv1(N, self.C_p4, self.C_p4, self.C_p4, self.C_p4)
            half_S_d1 = self.mints.mo_overlap_half_deriv1('LEFT', N, self.C_p4, self.C_p4)

            T  = T_d1[a].np;  V  = V_d1[a].np
            S  = S_d1[a].np;  G  = ERI_d1[a].np.swapaxes(1, 2)
            hs = half_S_d1[a].np
            #hs_mo = C.conj().T @ hs   # (nbf, nbf) fully in MO basis
            self._half_S[pert_index] = hs

            h_d1 = T + V
            F_d1 = h_d1 + oe.contract('piqi->pq', 2*G[:,o,:,o] - G.swapaxes(2,3)[:,o,:,o])

            B = (-F_d1[v, o] + oe.contract('ai,ii->ai', S[v, o], self.F_mo[o, o]) + 0.5 * oe.contract('mn,amin->ai', S[o, o], self.A.swapaxes(1,2)[v,o,o,o]))

            U[v, o] = (self.G @ B.reshape(nv*no)).reshape(nv, no)
            U[o, v] = -(U[v, o].T + S[o, v])

            # Non-canonical dependent pairs: diagonal -0.5 S^r
            U[o, o] = -0.5 * S[o, o]
            U[v, v] = -0.5 * S[v, v]

            # Cache skeleton integrals for Hessian reuse
            self._F_R[pert_index]    = F_d1
            self._S_R[pert_index]    = S
            self._half_S[pert_index] = hs
            if pert_index == 6:
                print(f"  hf_vcd.solve_cphf nuclear U[v,o] norm: {np.linalg.norm(U[v,o]):.8f}")
        
        # Electric field 
        elif pert_type == 'electric':
            beta  = pert_index
            mu_AO = self.mints.ao_dipole()
            mu_mo = oe.contract('mp,mn,nq->pq', C.conj(), mu_AO[beta].np, C)

            B = -mu_mo[v, o]
            U[v, o] = (self.G @ B.reshape(nv*no)).reshape(nv, no)
            U[o, v] = -U[v, o].T
            # Diagonal = 0 (no overlap derivative for uniform field)

            self._h_E[beta] = mu_mo

        # Magnetic field 
        elif pert_type == 'magnetic':
            beta = pert_index
            L_AO = self.mints.ao_angular_momentum()
            mu_mag = oe.contract('mp,mn,nq->pq', C.conj(), -0.5 * L_AO[beta].np, C)

            B = mu_mag[v, o]

            U_vo    = (self.G_mag @ B.reshape(nv*no)).reshape(nv, no)
            U[v, o] =  U_vo
            U[o, v] =  U_vo.T

        else:
            raise ValueError(f"Unknown pert_type '{pert_type}'. "
                             f"Choose 'nuclear', 'electric', or 'magnetic'.")

        self._U[key] = U
        return U

    
    # AATs

    def compute_AATs(self):
        natom = self.natom
        o = self.o
        v = self.v

        U_H = [self.solve_cphf('magnetic', b) for b in range(3)]

        AAT = np.zeros((3*natom, 3))

        for la in range(3*natom):
            U_R    = self.solve_cphf('nuclear', la)
            half_S = self._half_S[la]  
            for beta in range(3):
                AAT[la, beta] += 2 * oe.contract('em,em', U_H[beta][v, o], U_R[v, o] + half_S[o, v].T)

        return AAT



    def print_tensors(self, results=None):
        """Print APT and AAT in atom/coordinate block format."""
        if results is None:
            results = self.compute_VCD()

        mol = self.H.molecule
        APT = results['APT']
        AAT = results['AAT']
        xyz = ['x', 'y', 'z']

        def _block(name, T):
            w = 72
            print(f"\n  {name}  (a.u.)")
            print("-" * w)
            print(f"  {'Label':>8}  {'x':>18}  {'y':>18}  {'z':>18}")
            print("-" * w)
            for N in range(self.natom):
                sym = mol.symbol(N)
                for a, al in enumerate(xyz):
                    la  = 3*N + a
                    tag = f"{sym}{N+1}/{al}"
                    print(f"  {tag:>8}  "
                          f"{T[la,0]:>18.10f}  "
                          f"{T[la,1]:>18.10f}  "
                          f"{T[la,2]:>18.10f}")
            print("-" * w)

        _block("Atomic Polar Tensors (APT)", APT)
        _block("Atomic Axial Tensors (AAT)", AAT)
        print()

    def __repr__(self):
        return (f"hf_vcd(natom={self.natom}, nbf={self.nbf}, "
                f"E_scf={self.E_scf.real:.10f})")
