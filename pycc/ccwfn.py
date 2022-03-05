"""
ccwfn.py: CC T-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import time
import numpy as np
from opt_einsum import contract
from .utils import helper_diis
from .hamiltonian import Hamiltonian
from .local import Local


class ccwfn(object):
    """
    An RHF-CC wave function and energy object.

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
    solve_cc()
        Solves the CC T amplitude equations
    residuals()
        Computes the T1 and T2 residuals for a given set of amplitudes and Fock operator
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

        valid_cc_models = ['CCD', 'CC2', 'CCSD', 'CCSD(T)', 'CC3']
        model = kwargs.pop('model','CCSD')
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model

        # models requiring singles
        self.need_singles = ['CCSD', 'CCSD(T)', 'CC2']

        # models requiring T1-transformed integrals
        self.need_t1_transform = ['CC3']

        valid_local_models = [None, 'LPNO', 'PAO']
        local = kwargs.pop('local', None)
        # TODO: case-protect this kwarg
        if local not in valid_local_models:
            raise Exception("%s is not an allowed local-CC model." % (local))
        self.local = local
        self.local_cutoff = kwargs.pop('local_cutoff', 1e-5)

        valid_local_MOs = ['PIPEK_MEZEY', 'BOYS']
        local_MOs = kwargs.pop('local_mos', 'PIPEK_MEZEY')
        if local_MOs not in valid_local_MOs:
            raise Exception("%s is not an allowed MO localization method." % (local_MOs))
        self.local_MOs = local_MOs

        self.ref = scf_wfn
        self.eref = self.ref.energy()
        self.nfzc = self.ref.frzcpi()[0]                # assumes symmetry c1
        self.no = self.ref.doccpi()[0] - self.nfzc      # active occ; assumes closed-shell
        self.nmo = self.ref.nmo()                       # all MOs/AOs
        self.nv = self.nmo - self.no - self.nfzc        # active virt
        self.nact = self.no + self.nv                   # all active MOs

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (self.nmo, self.nact, self.no, self.nv))

        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)

        # For convenience
        o = self.o
        v = self.v

        # Get MOs
        C = self.ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)  # as numpy array
        self.C = C

        # Localize occupied MOs if requested
        if (local is not None):
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC")
            LMOS = psi4.core.Localizer.build(self.local_MOs, self.ref.basisset(), C_occ)
            LMOS.localize()
            npL = np.asarray(LMOS.L)
            npC[:,:self.no] = npL
            C = psi4.core.Matrix.from_array(npC)
            self.C = C

        self.H = Hamiltonian(self.ref, self.C, self.C, self.C, self.C)

        if local is not None:
            self.Local = Local(local, self.C, self.nfzc, self.no, self.nv, self.H, self.local_cutoff)

        # denominators
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # first-order amplitudes
        self.t1 = np.zeros((self.no, self.nv))
        if local is not None:
            self.t1, self.t2 = self.Local.filter_amps(self.t1, self.H.ERI[o,o,v,v])
        else:
            self.t1 = np.zeros((self.no, self.nv))
            self.t2 = self.H.ERI[o,o,v,v]/self.Dijab

        print("CC object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_cc(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):
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
            CC correlation energy
        """
        ccsd_tstart = time.time()

        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        Dia = self.Dia
        Dijab = self.Dijab

        ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(self.t1, self.t2, max_diis)

        for niter in range(1, maxiter+1):

            ecc_last = ecc

            r1, r2 = self.residuals(F, self.t1, self.t2)

            if self.local is not None:
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

            ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nCC has converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(%s) = %20.15f" % (self.model, ecc))
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

        Fae = self.build_Fae(o, v, F, L, t1, t2)
        Fmi = self.build_Fmi(o, v, F, L, t1, t2)
        Fme = self.build_Fme(o, v, F, L, t1)
        Wmnij = self.build_Wmnij(o, v, ERI, t1, t2)
        Wmbej = self.build_Wmbej(o, v, ERI, L, t1, t2)
        Wmbje = self.build_Wmbje(o, v, ERI, t1, t2)
        Zmbij = self.build_Zmbij(o, v, ERI, t1, t2)

        r1 = self.r_T1(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi)
        r2 = self.r_T2(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)

        return r1, r2

    def build_tau(self, t1, t2, fact1=1.0, fact2=1.0):
        return fact1 * t2 + fact2 * contract('ia,jb->ijab', t1, t1)


    def build_Fae(self, o, v, F, L, t1, t2):
        if self.model == 'CCD':
            Fae = F[v,v].copy()
            Fae = Fae - contract('mnaf,mnef->ae', t2, L[o,o,v,v])
        else:
            Fae = F[v,v].copy()
            Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
            Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
            Fae = Fae - contract('mnaf,mnef->ae', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fae


    def build_Fmi(self, o, v, F, L, t1, t2):
        if self.model == 'CCD':
            Fmi = F[o,o].copy()
            Fmi = Fmi + contract('inef,mnef->mi', t2, L[o,o,v,v])
        else:
            Fmi = F[o,o].copy()
            Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
            Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
            Fmi = Fmi + contract('inef,mnef->mi', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fmi


    def build_Fme(self, o, v, F, L, t1):
        if self.model == 'CCD':
            return
        else:
            Fme = F[o,v].copy()
            Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Fme


    def build_Wmnij(self, o, v, ERI, t1, t2):
        if self.model == 'CCD':
            Wmnij = ERI[o,o,o,o].copy()
            Wmnij = Wmnij + contract('ijef,mnef->mnij', t2, ERI[o,o,v,v])
        else:
            Wmnij = ERI[o,o,o,o].copy()
            Wmnij = Wmnij + contract('je,mnie->mnij', t1, ERI[o,o,o,v])
            Wmnij = Wmnij + contract('ie,mnej->mnij', t1, ERI[o,o,v,o])
            if self.model == 'CC2':
                Wmnij = Wmnij + contract('jf, mnif->mnij', t1, contract('ie,mnef->mnif', t1, ERI[o,o,v,v]))
            else:
                Wmnij = Wmnij + contract('ijef,mnef->mnij', self.build_tau(t1, t2), ERI[o,o,v,v])
        return Wmnij


    def build_Wmbej(self, o, v, ERI, L, t1, t2):
        if self.model == 'CCD':
            Wmbej = ERI[o,v,v,o].copy()
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', 0.5*t2, ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        elif self.model == 'CC2':
            return
        else:
            Wmbej = ERI[o,v,v,o].copy()
            Wmbej = Wmbej + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
            Wmbej = Wmbej - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Wmbej


    def build_Wmbje(self, o, v, ERI, t1, t2):
        if self.model == 'CCD':
            Wmbje = -1.0 * ERI[o,v,o,v].copy()
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', 0.5*t2, ERI[o,o,v,v])
        elif self.model == 'CC2':
            return
        else:
            Wmbje = -1.0 * ERI[o,v,o,v].copy()
            Wmbje = Wmbje - contract('jf,mbfe->mbje', t1, ERI[o,v,v,v])
            Wmbje = Wmbje + contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
        return Wmbje


    def build_Zmbij(self, o, v, ERI, t1, t2):
        if self.model == 'CCD':
            return
        elif self.model == 'CC2':
            return contract('mbif,jf->mbij', contract('mbef,ie->mbif', ERI[o,v,v,v], t1), t1)
        else:
            return contract('mbef,ijef->mbij', ERI[o,v,v,v], self.build_tau(t1, t2))


    def r_T1(self, o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi):
        if self.model == 'CCD':
            r_T1 = np.zeros_like(t1)
        else:
            r_T1 = F[o,v].copy()
            r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
            r_T1 = r_T1 - contract('ma,mi->ia', t1, Fmi)
            r_T1 = r_T1 + contract('imae,me->ia', (2.0*t2 - t2.swapaxes(2,3)), Fme)
            r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
            r_T1 = r_T1 + contract('mief,maef->ia', (2.0*t2 - t2.swapaxes(2,3)), ERI[o,v,v,v])
            r_T1 = r_T1 - contract('mnae,nmei->ia', t2, L[o,o,v,o])
        return r_T1


    def r_T2(self, o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):
        if self.model == 'CCD':
            r_T2 = 0.5 * ERI[o,o,v,v].copy()
            r_T2 = r_T2 + contract('ijae,be->ijab', t2, Fae)
            r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
            r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', t2, Wmnij)
            r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', t2, ERI[v,v,v,v])
            r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
            r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
            r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
        elif self.model == 'CC2':
            r_T2 = 0.5 * ERI[o,o,v,v].copy()
            r_T2 = r_T2 + contract('ijae,be->ijab', t2, (F[v,v] - 0.5 * contract('me,ma->ae', F[o,v], t1)))
            tmp = contract('mb,me->be', t1, F[o,v])
            r_T2 = r_T2 - 0.5 * contract('ijae,be->ijab', t2, tmp)
            r_T2 = r_T2 - contract('imab,mj->ijab', t2, (F[o,o] + 0.5 * contract('ie,me->mi', t1, F[o,v])))
            tmp = contract('je,me->jm', t1, F[o,v])
            r_T2 = r_T2 - 0.5 * contract('imab,jm->ijab', t2, tmp)
            r_T2 = r_T2 + 0.5 * contract('ma,mbij->ijab', t1, contract('nb,mnij->mbij', t1, Wmnij))
            r_T2 = r_T2 + 0.5 * contract('jf,abif->ijab', t1, contract('ie,abef->abif', t1, ERI[v,v,v,v]))
            r_T2 = r_T2 - contract('ma,mbij->ijab', t1, Zmbij)
            r_T2 = r_T2 - contract('ma,mbij->ijab', t1, contract('ie,mbej->mbij', t1, ERI[o,v,v,o])) 
            r_T2 = r_T2 - contract('mb,maji->ijab', t1, contract('ie,maje->maji', t1, ERI[o,v,o,v]))
            r_T2 = r_T2 + contract('ie,abej->ijab', t1, ERI[v,v,v,o])
            r_T2 = r_T2 - contract('ma,mbij->ijab', t1, ERI[o,v,o,o])
        else:
            r_T2 = 0.5 * ERI[o,o,v,v].copy()
            r_T2 = r_T2 + contract('ijae,be->ijab', t2, Fae)
            tmp = contract('mb,me->be', t1, Fme)
            r_T2 = r_T2 - 0.5 * contract('ijae,be->ijab', t2, tmp)
            r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
            tmp = contract('je,me->jm', t1, Fme)
            r_T2 = r_T2 - 0.5 * contract('imab,jm->ijab', t2, tmp)
            r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', self.build_tau(t1, t2), Wmnij)
            r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', self.build_tau(t1, t2), ERI[v,v,v,v])
            r_T2 = r_T2 - contract('ma,mbij->ijab', t1, Zmbij)
            r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
            r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
            r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
            tmp = contract('ie,ma->imea', t1, t1)
            r_T2 = r_T2 - contract('imea,mbej->ijab', tmp, ERI[o,v,v,o])
            r_T2 = r_T2 - contract('imeb,maje->ijab', tmp, ERI[o,v,o,v])
            r_T2 = r_T2 + contract('ie,abej->ijab', t1, ERI[v,v,v,o])
            r_T2 = r_T2 - contract('ma,mbij->ijab', t1, ERI[o,v,o,o])

        r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
        return r_T2


    def cc_energy(self, o, v, F, L, t1, t2):
        if self.model == 'CCD':
            ecc = contract('ijab,ijab->', t2, L[o,o,v,v])
        else:
            ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
            ecc = ecc + contract('ijab,ijab->', self.build_tau(t1, t2), L[o,o,v,v])
        return ecc
