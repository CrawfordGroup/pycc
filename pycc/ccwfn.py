import psi4
import time
import numpy as np
from opt_einsum import contract
from .ccsd_eqs import *
from .utils import helper_diis


class ccwfn(object):

    def __init__(self, mol, scf_wfn, memory=2):

        time_init = time.time()

        ref = scf_wfn
        self.eref = ref.energy()
        self.nfzc = ref.frzcpi()[0]                # assumes symmetry c1
        self.no = ref.doccpi()[0] - self.nfzc      # active occ; assumes closed-shell
        self.nmo = ref.nmo()                       # all MOs/AOs
        self.nv = self.nmo - self.no - self.nfzc   # active virt

        # Get MOs
        C = ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)  # as numpy array

        # Get MO Fock matrix
        self.F = np.asarray(ref.Fa())
        self.F = np.einsum('uj,vi,uv', npC, npC, self.F)

        # Get MO two-electron integrals in Dirac notation
        mints = psi4.core.MintsHelper(ref.basisset())
        self.ERI = np.asarray(mints.mo_eri(C, C, C, C))     # (pr|qs)
        self.ERI = self.ERI.swapaxes(1,2)                   # <pq|rs>
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2,3)    # 2 <pq|rs> - <pq|sr>
        
        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)

        # For convenience
        o = self.o
        v = self.v

        # denominators
        eps_occ = np.diag(self.F)[o]
        eps_vir = np.diag(self.F)[v]
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # first-order amplitudes
        self.t1 = np.zeros((self.no, self.nv))
        self.t2 = self.ERI[o,o,v,v]/self.Dijab

        print("CCSD initialized in %.3f seconds." % (time.time() - time_init))

    def solve_ccsd(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):

        ccsd_tstart = time.time()

        o = self.o
        v = self.v
        F = self.F
        ERI = self.ERI
        L = self.L
        t1 = self.t1
        t2 = self.t2
        Dia = self.Dia
        Dijab = self.Dijab

        ecc = ccsd_energy(o, v, F, L, t1, t2)
        print("CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(t1, t2, max_diis)

        ediff = ecc
        rms = 0
        niter = 0

        for niter in range(maxiter+1):

            ecc_last = ecc

            t1 = self.t1
            t2 = self.t2
            Fae = build_Fae(o, v, F, L, t1, t2)
            Fmi = build_Fmi(o, v, F, L, t1, t2)
            Fme = build_Fme(o, v, F, L, t1)
            Wmnij = build_Wmnij(o, v, ERI, t1, t2)
            Wmbej = build_Wmbej(o, v, ERI, L, t1, t2)
            Wmbje = build_Wmbje(o, v, ERI, t1, t2)
            Zmbij = build_Zmbij(o, v, ERI, t1, t2)

            r1 = r_T1(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi)
            r2 = r_T2(o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)

            self.t1 += r1/Dia
            self.t2 += r2/Dijab

            rms = contract('ia,ia->', r1/Dia, r1/Dia)
            rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
            rms = np.sqrt(rms)

            ecc = ccsd_energy(o, v, F, L, t1, t2)
            ediff = ecc - ecc_last
            print('CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  rms = % .5E' % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nCCSD has converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(CCSD) = %20.15f" % ecc)
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                return ecc

            diis.add_error_vector(self.t1, self.t2)
            if niter >= start_diis:
                self.t1, self.t2 = diis.extrapolate(self.t1, self.t2)
