"""
ccenergy.py: CC T-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np
from opt_einsum import contract
from .utils import helper_diis
from .cc_eqs import build_Fae, build_Fmi, build_Fme
from .cc_eqs import build_Wmnij, build_Wmbej, build_Wmbje, build_Zmbij
from .cc_eqs import r_T1, r_T2, ccsd_energy
from .hamiltonian import Hamiltonian
from .local import Local


class ccenergy(object):
    """
    An RHF-CCSD wave function and energy object.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction object
        the reference wave function built by Psi4 energy() method
    eref : float
        the energy of the reference wave function (including nuclear repulsion contribution)
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, and the spin-adapted ERIs (L)
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace
    Dia : NumPy array
        one-electron energy denominator
    Dijab : NumPy array
        two-electron energy denominator
    t1 : NumPy array
        T1 amplitudes
    t2 : NumPy array
        T2 amplitudes
    ecc | float
        the final CC correlation energy

    Methods
    -------
    solve_ccsd()
        Solves the CCSD T amplitude equations
    residuals()
        Computes the T1 and T2 residuals for a given set of amplitudes and Fock operator
    """

    def __init__(self, scf_wfn, local=False, lpno_cutoff=1e-5):
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
        self.eref = self.ref.energy()
        self.nfzc = self.ref.frzcpi()[0]                # assumes symmetry c1
        self.no = self.ref.doccpi()[0] - self.nfzc      # active occ; assumes closed-shell
        self.nmo = self.ref.nmo()                       # all MOs/AOs
        self.nv = self.nmo - self.no - self.nfzc   # active virt

        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)

        # For convenience
        o = self.o
        v = self.v

        self.H = Hamiltonian(self.ref, local=local)

        self.local = local
        if local is not False:
            self.Local = Local(self.no, self.nv, self.H, lpno_cutoff)

        # denominators
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # first-order amplitudes
        self.t1 = np.zeros((self.no, self.nv))
        if local is not False:
            self.t1, self.t2 = self.Local.filter_amps(self.t1, self.H.ERI[o,o,v,v])
        else:
            self.t1 = np.zeros((self.no, self.nv))
            self.t2 = self.H.ERI[o,o,v,v]/self.Dijab

        print("CCSD initialized in %.3f seconds." % (time.time() - time_init))

    def solve_ccsd(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):
        """
        Parameters
        ----------
        e_conv : float
            convergence condition for correlation energy (default if 1e-7)
        r_conv : float
            convergence condition for wave function rmsd (default if 1e-7)
        maxiter : int
            maximum allowed number of iterations of the CC equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        ecc : float
            CCSD correlation energy
        """
        ccsd_tstart = time.time()

        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        t1 = self.t1
        t2 = self.t2
        Dia = self.Dia
        Dijab = self.Dijab

        ecc = ccsd_energy(o, v, F, L, t1, t2)
        print("CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(t1, t2, max_diis)

        for niter in range(1, maxiter+1):

            ecc_last = ecc

            r1, r2 = self.residuals(F, self.t1, self.t2)

            if self.local is not False:
                print("Filtering amplitudes with {} PNOs".format(self.Local.dim))
                inc1, inc2 = self.Local.filter_amps(r1, r2)
                self.t1 += inc1
                self.t2 += inc2
                rms = contract('ia,ia->', inc1, inc1)
                rms += contract('ijab,ijab->', inc2, inc2)
                rms = np.sqrt(rms)
            else:
                self.t1 += r1/Dia
                self.t2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia)
                rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = np.sqrt(rms)

            ecc = ccsd_energy(o, v, F, L, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nCCSD has converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(CCSD) = %20.15f" % ecc)
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                self.ecc = ecc
                return ecc

            diis.add_error_vector(self.t1, self.t2)
            if niter >= start_diis:
                self.t1, self.t2 = diis.extrapolate(self.t1, self.t2)

    def residuals(self, F, t1, t2):
        """
        Parameters
        ----------
        F: NumPy array
            Fock matrix
        t1: NumPy array
            Current T1 amplitudes
        t2: NumPy array
            Current T2 amplitudes

        Returns
        -------
        r1, r2: NumPy arrays
            New T1 and T2 residuals: r_mu = <mu|HBAR|0>
        """

        o = self.o
        v = self.v
        ERI = self.H.ERI
        L = self.H.L

        Fae = build_Fae(o, v, F, L, t1, t2)
        Fmi = build_Fmi(o, v, F, L, t1, t2)
        Fme = build_Fme(o, v, F, L, t1)
        Wmnij = build_Wmnij(o, v, ERI, t1, t2)
        Wmbej = build_Wmbej(o, v, ERI, L, t1, t2)
        Wmbje = build_Wmbje(o, v, ERI, t1, t2)
        Zmbij = build_Zmbij(o, v, ERI, t1, t2)

        r1 = r_T1(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi)
        r2 = r_T2(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)

        return r1, r2
