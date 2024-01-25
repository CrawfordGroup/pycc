"""
ccwfn.py: CC T-amplitude Solver
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import time
from time import process_time
import numpy as np
import torch
from .utils import helper_diis, cc_contract
from .hamiltonian import Hamiltonian
from .local import Local
from .cctriples import t_tjl, t3c_ijk, t3d_ijk, t3c_abc, t3d_abc
from .lccwfn import lccwfn

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
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, the spin-adapted ERIs (L), and various property integrals
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
        model = kwargs.pop('model','CCSD').upper()
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model

        # models requiring singles
        self.need_singles = ['CCSD', 'CCSD(T)', 'CC2', 'CC3']

        # models requiring T1-transformed integrals
        self.need_t1_transform = ['CC3']

        self.make_t3_density = kwargs.pop('make_t3_density', False)

        valid_local_models = [None, 'PNO', 'PAO','PNO++']
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

        valid_it2_opt = [True,False]
        it2_opt = kwargs.pop('it2_opt', True)
        # TODO: case-protect this kwarg
        if it2_opt not in valid_it2_opt:
            raise Exception("%s is not an allowed initial t2 amplitudes." % (it2_opt))
        self.it2_opt = it2_opt

        valid_filter = [True,False]
        # TODO: case-protect this kwarg
        filter = kwargs.pop('filter', False)
        if filter not in valid_filter:
            raise Exception("%s is not an allowed local filter." % (filter))
        self.filter = filter

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
            self.Local = Local(local, self.C, self.nfzc, self.no, self.nv, self.H, self.local_cutoff,self.it2_opt)
            if filter is not True:
                self.Local.trans_integrals(self.o, self.v)
                self.Local.overlaps(self.Local.QL)
                self.lccwfn = lccwfn(self.o, self.v,self.no, self.nv, self.H, self.local, self.model, self.eref, self.Local)

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

        valid_precision = ['SP', 'DP']
        precision = kwargs.pop('precision', 'DP')
        if precision.upper() not in valid_precision:
            raise Exception('%s is not an allowed precision arithmetic.' % (precision))
        self.precision = precision.upper()

        valid_device = ['CPU', 'GPU']
        device = kwargs.pop('device', 'CPU')
        if device.upper() not in valid_device:
            raise Exception("%s is not an allowed device." % (device))
        self.device = device.upper()

        if self.precision == 'SP':
            self.H.F = np.float32(self.H.F)
            self.t1 = np.float32(self.t1)
            self.t2 = np.float32(self.t2)
            self.Dia = np.float32(self.Dia)
            self.Dijab = np.float32(self.Dijab)
            self.H.ERI = np.float32(self.H.ERI)
            self.H.L = np.float32(self.H.L)

        # Initiate the object for a generalized contraction function 
        # for GPU or CPU.  
        self.contract = cc_contract(device=self.device)

        # Convert the arrays to torch.Tensors if the calculation is on GPU.
        # Send the copy of F, t1, t2 to GPU.
        # ERI will be kept on GPU
        if self.device == 'GPU':
            if self.precision == 'DP':
                self.device0 = torch.device('cpu')
                self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                # Storing on GPU
                self.H.F = torch.tensor(self.H.F, dtype=torch.complex128, device=self.device1)
                self.t1 = torch.tensor(self.t1, dtype=torch.complex128, device=self.device1)
                self.t2 = torch.tensor(self.t2, dtype=torch.complex128, device=self.device1)
                self.Dia = torch.tensor(self.Dia, dtype=torch.complex128, device=self.device1)
                self.Dijab = torch.tensor(self.Dijab, dtype=torch.complex128, device=self.device1)
                # Storing on CPU
                self.H.ERI = torch.tensor(self.H.ERI, dtype=torch.complex128, device=self.device0)
                self.H.L = torch.tensor(self.H.L, dtype=torch.complex128, device=self.device0)
            elif self.precision == 'SP':
                self.device0 = torch.device('cpu')
                self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                # Storing on GPU
                self.H.F = torch.tensor(self.H.F, dtype=torch.complex64, device=self.device1)
                self.t1 = torch.tensor(self.t1, dtype=torch.complex64, device=self.device1)
                self.t2 = torch.tensor(self.t2, dtype=torch.complex64, device=self.device1)
                self.Dia = torch.tensor(self.Dia, dtype=torch.complex64, device=self.device1)
                self.Dijab = torch.tensor(self.Dijab, dtype=torch.complex64, device=self.device1)
                # Storing on CPU
                self.H.ERI = torch.tensor(self.H.ERI, dtype=torch.complex64, device=self.device0)
                self.H.L = torch.tensor(self.H.L, dtype=torch.complex64, device=self.device0)
                
        print("CCWFN object initialized in %.3f seconds." % (time.time() - time_init))


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
       
         #initialize variables for timing each function
        self.fae_tl = 0
        self.fme_tl = 0
        self.fmi_tl = 0
        self.wmnij_tl = 0
        self.zmbij_tl = 0
        self.wmbej_tl = 0
        self.wmbje_tl = 0
        self.tau_tl = 0
        self.r1_tl = 0
        self.r2_tl = 0
        self.energy_tl = 0
        

        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        Dia = self.Dia
        Dijab = self.Dijab

        contract = self.contract

        ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        #diis = helper_diis(self.t1, self.t2, max_diis, self.precision)

        for niter in range(1, maxiter+1):

            ecc_last = ecc

            r1, r2 = self.residuals(F, self.t1, self.t2)

            if self.local is not None:
                inc1, inc2 = self.Local.filter_amps(r1, r2)
                self.t1 += inc1
                self.t2 += inc2
                rms = contract('ia,ia->', inc1, inc1)
                rms += contract('ijab,ijab->', inc2, inc2)
                if isinstance(r1, torch.Tensor):
                    rms = torch.sqrt(rms)
                else:
                    rms = np.sqrt(rms)
            else:
                self.t1 += r1/Dia
                self.t2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia)
                rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                if isinstance(r1, torch.Tensor):
                    rms = torch.sqrt(rms)
                else:
                    rms = np.sqrt(rms)

            ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if isinstance(self.t1, torch.Tensor):
                if ((torch.abs(ediff) < e_conv) and torch.abs(rms) < r_conv):
                    print("\nCCWFN converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                    print("E(REF)  = %20.15f" % self.eref)
                    print("E(%s) = %20.15f" % (self.model, ecc))
                    print("E(TOT)  = %20.15f" % (ecc + self.eref))
                    self.ecc = ecc
                    return ecc
            else:
                if ((abs(ediff) < e_conv) and abs(rms) < r_conv):
                    print("\nCCWFN converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                    print("E(REF)  = %20.15f" % self.eref)
                    if (self.model == 'CCSD(T)'):
                        print("E(CCSD) = %20.15f" % ecc)
                        if self.make_t3_density is True:
                            et = self.t3_density()
                        else:
                            et = t_tjl(self)
                        print("E(T)    = %20.15f" % et)
                        ecc = ecc + et
                    else:
                        print("E(%s) = %20.15f" % (self.model, ecc))
                    self.ecc = ecc
                    print("E(TOT)  = %20.15f" % (ecc + self.eref))
                    print('Time table for intermediates')
                    print("Fae = %6.6f" % self.fae_tl)
                    print("Fme = %6.6f" % self.fme_tl)
                    print("Fmi = %6.6f" % self.fmi_tl)
                    print("Wmnij = %6.6f" % self.wmnij_tl)
                    print("Zmbij = %6.6f" % self.zmbij_tl)
                    print("Wmbej = %6.6f" % self.wmbej_tl)
                    print("Wmbje = %6.6f" % self.wmbje_tl)
                    print("Tau_t = %6.6f" % self.tau_tl)
                    print("r1_t = %6.6f" % self.r1_tl)
                    print("r2_t = %6.6f" % self.r2_tl)
                    print("Energy_t = %6.6f" % self.energy_tl)
                    return ecc

            #diis.add_error_vector(self.t1, self.t2)
            #if niter >= start_diis:
                #self.t1, self.t2 = diis.extrapolate(self.t1, self.t2)

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

        contract = self.contract

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
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

        if isinstance(Fae, torch.Tensor):
            del Fae, Fmi, Wmnij, Wmbej, Wmbje, Zmbij

        if self.model == 'CC3':
            Wmnij_cc3 = self.build_cc3_Wmnij(o, v, ERI, t1)
            Wmbij_cc3 = self.build_cc3_Wmbij(o, v, ERI, t1, Wmnij_cc3)
            Wmnie_cc3 = self.build_cc3_Wmnie(o, v, ERI, t1)
            Wamef_cc3 = self.build_cc3_Wamef(o, v, ERI, t1)
            Wabei_cc3 = self.build_cc3_Wabei(o, v, ERI, t1)

            if isinstance(t1, torch.Tensor):
                X1 = torch.zeros_like(t1)
                X2 = torch.zeros_like(t2)
            else:
                X1 = np.zeros_like(t1)
                X2 = np.zeros_like(t2)

            for i in range(no):
                for j in range(no):
                    for k in range(no):
                        t3 = t3c_ijk(o, v, i, j, k, t2, Wabei_cc3, Wmbij_cc3, F,
contract, WithDenom=True)

                        X1[i] += contract('abc,bc->a', t3 - t3.swapaxes(0,2), L[j,k,v,v])
                        X2[i,j] += contract('abc,c->ab', t3 - t3.swapaxes(0,2), Fme[k])
                        X2[i,j] += contract('abc,dbc->ad', 2 * t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2), Wamef_cc3.swapaxes(0,1)[k])
                        X2[i] -= contract('abc,lc->lab', 2 * t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2), Wmnie_cc3[j,k])

            r1 += X1
            r2 += X2 + X2.swapaxes(0,1).swapaxes(2,3)

            if isinstance(t3, torch.Tensor):
                del Fme, Wmnij_cc3, Wmbij_cc3, Wmnie_cc3, Wamef_cc3, Wabei_cc3

        return r1, r2

    def build_tau(self, t1, t2, fact1=1.0, fact2=1.0):
        contract = self.contract
        return fact1 * t2 + fact2 * contract('ia,jb->ijab', t1, t1)


    def build_Fae(self, o, v, F, L, t1, t2):
        fae_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Fae = F[v,v].clone()
            else:
                Fae = F[v,v].copy()
            Fae = Fae - contract('mnaf,mnef->ae', t2, L[o,o,v,v])
        else:
            if isinstance(t1, torch.Tensor):
                Fae = F[v,v].clone()
            else:
                Fae = F[v,v].copy()
            Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
            Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
            Fae = Fae - contract('mnaf,mnef->ae', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        
        fae_end = process_time()
        self.fae_tl += fae_end - fae_start
        return Fae


    def build_Fmi(self, o, v, F, L, t1, t2):
        fmi_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Fmi = F[o,o].clone()
            else:
                Fmi = F[o,o].copy()
            Fmi = Fmi + contract('inef,mnef->mi', t2, L[o,o,v,v])
        else:
            if isinstance(t1, torch.Tensor):
                Fmi = F[o,o].clone()
            else:
                Fmi = F[o,o].copy()
            Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
            Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
            Fmi = Fmi + contract('inef,mnef->mi', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        fmi_end = process_time()
        self.fmi_tl += fmi_end - fmi_start
        return Fmi


    def build_Fme(self, o, v, F, L, t1):
        fme_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            return
        else:
            if isinstance(t1, torch.Tensor):
                Fme = F[o,v].clone()
            else:
                Fme = F[o,v].copy()
            Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
        fme_end = process_time()
        self.fme_tl += fme_end - fme_start
        return Fme


    def build_Wmnij(self, o, v, ERI, t1, t2):
        wmnij_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Wmnij = ERI[o,o,o,o].clone().to(self.device1)
            else:
                Wmnij = ERI[o,o,o,o].copy()
            Wmnij = Wmnij + contract('ijef,mnef->mnij', t2, ERI[o,o,v,v])
        else:
            if isinstance(t1, torch.Tensor):
                Wmnij = ERI[o,o,o,o].clone().to(self.device1)
            else:
                Wmnij = ERI[o,o,o,o].copy()
            Wmnij = Wmnij + contract('je,mnie->mnij', t1, ERI[o,o,o,v])
            Wmnij = Wmnij + contract('ie,mnej->mnij', t1, ERI[o,o,v,o])
            if self.model == 'CC2':
                Wmnij = Wmnij + contract('jf, mnif->mnij', t1, contract('ie,mnef->mnif', t1, ERI[o,o,v,v]))
            else:
                Wmnij = Wmnij + contract('ijef,mnef->mnij', self.build_tau(t1, t2), ERI[o,o,v,v])
        wmnij_end = process_time()
        self.wmnij_tl += wmnij_end - wmnij_start
        return Wmnij


    def build_Wmbej(self, o, v, ERI, L, t1, t2):
        wmbej_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Wmbej = ERI[o,v,v,o].clone().to(self.device1)
            else:
                Wmbej = ERI[o,v,v,o].copy()
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', 0.5*t2, ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        elif self.model == 'CC2':
            return
        else:
           if isinstance(t1, torch.Tensor):
                Wmbej = ERI[o,v,v,o].clone().to(self.device1)
           else:
                Wmbej = ERI[o,v,v,o].copy()
           Wmbej = Wmbej + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
           Wmbej = Wmbej - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
           Wmbej = Wmbej - contract('jnfb,mnef->mbej', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
           Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        wmbej_end = process_time()
        self.wmbej_tl += wmbej_end - wmbej_start
        return Wmbej


    def build_Wmbje(self, o, v, ERI, t1, t2):
        wmbje_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Wmbje = -1.0 * ERI[o,v,o,v].clone().to(self.device1)
            else:
                Wmbje = -1.0 * ERI[o,v,o,v].copy()
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', 0.5*t2, ERI[o,o,v,v])
        elif self.model == 'CC2':
            return
        else:
           if isinstance(t1, torch.Tensor):
                Wmbje = -1.0 * ERI[o,v,o,v].clone().to(self.device1)
           else:
                Wmbje = -1.0 * ERI[o,v,o,v].copy()
           Wmbje = Wmbje - contract('jf,mbfe->mbje', t1, ERI[o,v,v,v])
           Wmbje = Wmbje + contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
           Wmbje = Wmbje + contract('jnfb,mnfe->mbje', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
        wmbje_end = process_time()
        self.wmbje_tl += wmbje_end - wmbje_start
        return Wmbje


    def build_Zmbij(self, o, v, ERI, t1, t2):
        zmbij_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            return
        elif self.model == 'CC2':
            return contract('mbif,jf->mbij', contract('mbef,ie->mbif', ERI[o,v,v,v], t1), t1)
        else:
            zmbij_end = process_time()
            self.zmbij_tl += zmbij_end - zmbij_start
            return contract('mbef,ijef->mbij', ERI[o,v,v,v], self.build_tau(t1, t2))


    def r_T1(self, o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi):
        r1_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                r_T1 = torch.zero_like(t1)
            else:
                r_T1 = np.zeros_like(t1)
        else:
            if isinstance(t1, torch.Tensor):
                r_T1 = F[o,v].clone()
            else:
                r_T1 = F[o,v].copy()
            r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
            r_T1 = r_T1 - contract('ma,mi->ia', t1, Fmi)
            r_T1 = r_T1 + contract('imae,me->ia', (2.0*t2 - t2.swapaxes(2,3)), Fme)
            r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
            r_T1 = r_T1 + contract('mief,maef->ia', (2.0*t2 - t2.swapaxes(2,3)), ERI[o,v,v,v])
            r_T1 = r_T1 - contract('mnae,nmei->ia', t2, L[o,o,v,o])
        r1_end = process_time()
        self.r1_tl += r1_end - r1_start
        return r_T1


    def r_T2(self, o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):
        r2_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                r_T2 = 0.5 * ERI[o,o,v,v].clone().to(self.device1)
            else:
                r_T2 = 0.5 * ERI[o,o,v,v].copy()
            r_T2 = r_T2 + contract('ijae,be->ijab', t2, Fae)
            r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
            r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', t2, Wmnij)
            r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', t2, ERI[v,v,v,v])
            r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
            r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
            r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
        elif self.model == 'CC2':
            if isinstance(t1, torch.Tensor):
                r_T2 = 0.5 * ERI[o,o,v,v].clone().to(self.device1)
            else:
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
            if isinstance(tmp, torch.Tensor):
                del tmp
        else:
            if isinstance(t1, torch.Tensor):
                r_T2 = 0.5 * ERI[o,o,v,v].clone().to(self.device1)
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

            if isinstance(tmp, torch.Tensor):
                del tmp

        r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
        r2_end = process_time()
        self.r2_tl += r2_end - r2_start
        return r_T2

    # Intermedeates needed for CC3
    def build_cc3_Wmnij(self, o, v, ERI, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[o,o,o,o].clone().to(self.device1)
        else:
            W = ERI[o,o,o,o].copy()
        tmp = contract('ijma,na->ijmn', ERI[o,o,o,v], t1)
        W = W + tmp + tmp.swapaxes(0,1).swapaxes(2,3)
        tmp = contract('ia,mnaf->mnif', t1, ERI[o,o,v,v])
        W = W + contract('mnif,jf->mnij', tmp, t1)
        return W

    def build_cc3_Wmbij(self, o, v, ERI, t1, Wmnij):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[o,v,o,o].clone().to(self.device1)
        else:
            W = ERI[o,v,o,o].copy()
        W = W - contract('mnij,nb->mbij', Wmnij, t1)
        W = W + contract('mbie,je->mbij', ERI[o,v,o,v], t1)
        if isinstance(t1, torch.Tensor):
            tmp = ERI[o,v,v,o].clone().to(self.device1) + contract('mbef,jf->mbej', ERI[o,v,v,v], t1)
        else:
            tmp = ERI[o,v,v,o].copy() + contract('mbef,jf->mbej', ERI[o,v,v,v], t1)
        W = W + contract('ie,mbej->mbij', t1, tmp)
        return W

    def build_cc3_Wmnie(self, o, v, ERI, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[o,o,o,v].clone().to(self.device1)
        else:
            W = ERI[o,o,o,v].copy()
        W = W + contract('if,mnfe->mnie', t1, ERI[o,o,v,v])
        return W

    def build_cc3_Wamef(self, o, v, ERI, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            W = ERI[v,o,v,v].clone().to(self.device1)
        else:
            W = ERI[v,o,v,v].copy()
        W = W - contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return W

    def build_cc3_Wabei(self, o, v, ERI, t1):
        contract =self.contract
        # eiab
        if isinstance(t1, torch.Tensor):
            Z = ERI[v,o,v,v].clone().to(self.device1)
        else:
            Z = ERI[v,o,v,v].copy()
        tmp_ints = ERI[v,v,v,v] + ERI[v,v,v,v].swapaxes(2,3)
        Z1 = 0.5 * contract('if,abef->eiab', t1, tmp_ints)
        tmp_ints = ERI[v,v,v,v] - ERI[v,v,v,v].swapaxes(2,3)
        Z2 = 0.5 * contract('if,abef->eiab', t1, tmp_ints)
        Z_eiab = Z + Z1 + Z2

        #eiab
        if isinstance(t1, torch.Tensor):
            Zeiam = ERI[v,o,v,o].clone().to(self.device1)
        else:
            Zeiam = ERI[v,o,v,o].copy()
        Zamei = contract('amef,if->amei', ERI[v,o,v,v], t1)
        Zeiam = Zeiam + Zamei.swapaxes(0,2).swapaxes(1,3)
        Z_eiab = Z_eiab - contract('eiam,mb->eiab', Zeiam, t1)

        #eiab
        if isinstance(t1, torch.Tensor):
            Zmnei = ERI[o,o,v,o].clone().to(self.device1) + contract('mnef,if->mnei', ERI[o,o,v,v], t1)
        else:
            Zmnei = ERI[o,o,v,o].copy() + contract('mnef,if->mnei', ERI[o,o,v,v], t1)
        Zanei = contract('ma,mnei->anei', t1, Zmnei)
        Z_eiab = Z_eiab + contract('anei,nb->eiab', Zanei, t1)

        #abei
        if isinstance(t1, torch.Tensor):
            Zmbei = ERI[o,v,v,o].clone().to(self.device1)
        else:
            Zmbei = ERI[o,v,v,o].copy()
        Zmbei = Zmbei + contract('mbef,if->mbei', ERI[o,v,v,v], t1)
        Z_abei = -1 * contract('ma,mbei->abei', t1, Zmbei)

        # Wabei
        W = Z_abei + Z_eiab.swapaxes(0,2).swapaxes(1,3)
        return W

    def cc_energy(self, o, v, F, L, t1, t2):
        energy_start = process_time()
        contract = self.contract
        if self.model == 'CCD':
            ecc = contract('ijab,ijab->', t2, L[o,o,v,v])
        else:
            ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
            ecc = ecc + contract('ijab,ijab->', self.build_tau(t1, t2), L[o,o,v,v])
        energy_end = process_time()
        self.energy_tl += energy_end - energy_start
        return ecc

    def t3_density(self):
        """
        Computes (T) contributions to Lambda equations and one-/two-electron densities
        """

        contract = self.contract

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        t1 = self.t1
        t2 = self.t2
        F = self.H.F
        ERI = self.H.ERI
        L = self.H.L

        Dvv = np.zeros((nv,nv))
        Doo = np.zeros((no,no))
        Dov = np.zeros((no,nv))
        Goovv = np.zeros_like(t2)
        Gooov = np.zeros((no,no,no,nv))
        Gvvvo = np.zeros((nv,nv,nv,no))
        S1 = np.zeros_like(t1)
        S2 = np.zeros_like(t2)
        Z3 = np.zeros((nv,nv,nv))
        X1 = np.zeros_like(t1)
        X2 = np.zeros_like(t2)

        for i in range(no):
            for j in range(no):
                for k in range(no):
                    M3 = t3c_ijk(o, v, i, j, k, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract, True)
                    N3 = t3d_ijk(o, v, i, j, k, t1, t2, ERI[o,o,v,v], F, contract, True)
                    X3 = 8*M3 - 4*M3.swapaxes(0,1) - 4*M3.swapaxes(1,2) - 4*M3.swapaxes(0,2) + 2*np.moveaxis(M3, 0, 2) + 2*np.moveaxis(M3, 2, 0)
                    Y3 = 8*N3 - 4*N3.swapaxes(0,1) - 4*N3.swapaxes(1,2) - 4*N3.swapaxes(0,2) + 2*np.moveaxis(N3, 0, 2) + 2*np.moveaxis(N3, 2, 0)

                    # Doubles contribution (T) correction (Viking's formulation)
                    X2[i,j] += contract('abc,c->ab',(M3 - M3.swapaxes(0,2)), F[k,v])
                    X2[i,j] += contract('abc,dbc->ad', (2*M3 - M3.swapaxes(1,2) - M3.swapaxes(0,2)),ERI[v,k,v,v])
                    X2[i] -= contract('abc,lc->lab', (2*M3 - M3.swapaxes(1,2) - M3.swapaxes(0,2)),ERI[j,k,o,v])

                    # (T) contribution to vir-vir block of one-electron density
                    Dvv += 0.5 * contract('acd,bcd->ab', M3, (X3 + Y3))

                    # (T) contribution to occ-vir block of one-electron density
                    Dov[i] += contract('abc,bc->a', (M3 - M3.swapaxes(0,2)), (4*t2[j,k] - 2*t2[j,k].T))

                    # (T) contributions to two-electron density
                    Z3 = 2*(M3 - M3.swapaxes(1,2)) - (M3.swapaxes(0,1) - np.moveaxis(M3, 2, 0))
                    Goovv[i,j,:,:] += 4*contract('c,abc->ab', t1[k,:], Z3)
                    Gooov[j,i] -= contract('abc,lbc->la', (2*X3 + Y3), t2[:,k])
                    Gvvvo[:,:,:,j] += contract('abc,cd->abd', (2*X3 + Y3), t2[k,i,:,:])

                    # (T) contribution to Lambda_1 residual
                    S1[i] += contract('abc,bc->a', 2*(M3 - M3.swapaxes(0,1)), L[j,k,v,v])
                    # (T) contribution to Lambda_2 residual
                    S2[i] -= contract('abc,lc->lab', (2*X3 + Y3), ERI[j,k,o,v])
                    S2[i,j] += contract('abc,dcb->ad', (2*X3 + Y3), ERI[k,v,v,v])

        S2 = S2 + S2.swapaxes(0,1).swapaxes(2,3)

        # (T) contribution to occ-occ block of one-electron density
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    M3 = t3c_abc(o, v, a, b, c, t2, ERI[v,v,v,o], ERI[o,v,o,o], F, contract, True)
                    N3 = t3d_abc(o, v, a, b, c, t1, t2, ERI[o,o,v,v], F, contract, True)
                    X3 = 8*M3 - 4*M3.swapaxes(0,1) - 4*M3.swapaxes(1,2) - 4*M3.swapaxes(0,2) + 2*np.moveaxis(M3, 0, 2) + 2*np.moveaxis(M3, 2, 0)
                    Y3 = 8*N3 - 4*N3.swapaxes(0,1) - 4*N3.swapaxes(1,2) - 4*N3.swapaxes(0,2) + 2*np.moveaxis(N3, 0, 2) + 2*np.moveaxis(N3, 2, 0)
                    Doo -= 0.5 * contract('ikl,jkl->ij', M3, (X3 + Y3))

        self.Dvv = Dvv
        self.Doo = Doo
        self.Dov = Dov # Need to add this even though it doesn't contribute to the energy for RHF references

        self.Goovv = Goovv
        self.Gooov = Gooov
        self.Gvvvo = Gvvvo
        self.S1 = S1
        self.S2 = S2

        # (T) correction
        ET = contract('ia,ia->', t1, S1) # NB: Factor of two is already included in S1 definition
        ET += contract('ijab,ijab->', (4.0*t2 - 2.0*t2.swapaxes(2,3)), X2)

#        print("Dvv:")
#        it = np.nditer(self.Dvv, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("Doo:")
#        it = np.nditer(self.Doo, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("Dov:")
#        it = np.nditer(self.Dov, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("S1 Amplitudes:")
#        it = np.nditer(self.S1, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("S2 Amplitudes:")
#        it = np.nditer(self.S2, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("Goovv Density:")
#        it = np.nditer(self.Goovv, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("Gooov Density:")
#        it = np.nditer(self.Gooov, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))
#
#        print("Gvvvo Density:")
#        it = np.nditer(self.Gvvvo, flags=['multi_index'])
#        for val in it:
#            if np.abs(val) > 1e-12:
#                print("%s %20.15f" % (it.multi_index, val))

        return ET
