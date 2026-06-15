"""
ccwfn.py: CC T-amplitude Solver
"""

from __future__ import annotations

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from typing import Any

from .utils import helper_diis, zeros_like, clone, sqrt
from .wavefunction import Wavefunction
from .mpwfn import MPwfn
from .local import Local
from .cctriples import t_tjl, t3c_ijk, t3d_ijk, t3c_abc, t3d_abc, t3_pert_ijk
from .lccwfn import lccwfn
from ._typing import Tensor
from .exceptions import InvalidKeywordError

class ccwfn(Wavefunction):
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

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
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
            raise InvalidKeywordError('model', model, valid_cc_models)
        self.model = model

        # models requiring singles
        self.need_singles = ['CCSD', 'CCSD(T)', 'CC2', 'CC3']

        # models requiring T1-transformed integrals
        self.need_t1_transform = ['CC3']

        self.make_t3_density = kwargs.pop('make_t3_density', False)

        # RT-CC3 calculations requiring additional terms when an external perturbation is present 
        self.real_time = kwargs.pop('real_time', False)

        valid_local_models = [None, 'PNO', 'PAO','CPNO++','PNO++']
        local = kwargs.pop('local', None)
        if isinstance(local, str):
            local = local.upper()
        if local not in valid_local_models:
            raise InvalidKeywordError('local', local, valid_local_models)
        self.local = local
        self.local_cutoff = kwargs.pop('local_cutoff', 1e-5)

        valid_it2_opt = [True,False]
        it2_opt = kwargs.pop('it2_opt', True)
        if it2_opt not in valid_it2_opt:
            raise InvalidKeywordError('it2_opt', it2_opt, valid_it2_opt)
        self.it2_opt = it2_opt

        valid_filter = [True,False]
        filter = kwargs.pop('filter', False)
        if filter not in valid_filter:
            raise InvalidKeywordError('filter', filter, valid_filter)
        self.filter = filter

        # Reference, orbital spaces, MO coefficients (occupied localized when a
        # local-CC scheme is requested, so the base's single H build uses LMOs),
        # the seeded MO-basis integrals, and the device/precision manager all come
        # from the Wavefunction base. We pop only CC-specific kwargs above and
        # forward the rest (device/precision/local_mos) to the base, which owns
        # them -- so adding MPwfn/HFwfn/... needs no device/precision boilerplate.
        super().__init__(scf_wfn, localize_occ=(local is not None), **kwargs)
        o = self.o
        v = self.v
        mgr = self.device_manager

        if local is not None:
            self.Local = Local(local, self.C, self.nfzc, self.no, self.nv, self.H, self.local_cutoff,self.it2_opt)
            if filter is not True:
                self.Local.trans_integrals(self.o, self.v)
                self.Local.overlaps(self.Local.QL)
                self.lccwfn = lccwfn(self.o, self.v,self.no, self.nv, self.H, self.local, self.model, self.eref, self.Local)

        # The MP2 wavefunction supplies the energy denominators and the CC initial
        # guess. from_wavefunction reuses this object's already-built base (no second
        # integral transform), so the denominator/MP2-amplitude code lives only in
        # MPwfn. CC's singles denominator (Dia) is built here from the MP2 wfn's
        # orbital energies -- MP2 has no singles.
        self.mp = MPwfn.from_wavefunction(self)
        self.Dijab = self.mp.Dijab
        self.Dia = self.mp.eps_occ.reshape(-1, 1) - self.mp.eps_vir

        # CC initial-guess amplitudes. CC mutates t2 in place while iterating, so it
        # takes its own copy of the MP2 doubles (the local path filters instead).
        if local is not None:
            self.t1, self.t2 = self.Local.filter_amps(np.zeros((self.no, self.nv)),
                                                       self.H.ERI[o,o,v,v])
        else:
            self.t1 = mgr.seed_compute(np.zeros((self.no, self.nv)))
            self.t2 = clone(self.mp.t2)

        print("CCWFN object initialized in %.3f seconds." % (time.time() - time_init))


    def solve_cc(self, e_conv: float = 1e-7, r_conv: float = 1e-7, maxiter: int = 100, max_diis: int = 8, start_diis: int = 1) -> float:
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

        contract = self.contract

        ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(self.t1, self.t2, max_diis, self.precision)

        for niter in range(1, maxiter+1):

            ecc_last = ecc

            r1, r2 = self.residuals(F, self.t1, self.t2)

            if self.local is not None:
                inc1, inc2 = self.Local.filter_amps(r1, r2)
                self.t1 += inc1
                self.t2 += inc2
                rms = contract('ia,ia->', inc1, inc1) + contract('ijab,ijab->', inc2, inc2)
                rms = sqrt(rms)
            else:
                self.t1 += r1/Dia
                self.t2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia) + contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = sqrt(rms)

            ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence. abs() dispatches to __abs__ on both NumPy scalars
            # and 0-d torch tensors, so this single block (and the (T) correction below)
            # runs on either backend -- previously the torch arm was a separate copy that
            # silently skipped the (T) step, so CCSD(T) on a torch tensor returned the
            # bare CCSD energy.
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
                return ecc

            diis.add_error_vector(self.t1, self.t2)
            if niter >= start_diis:
                self.t1, self.t2 = diis.extrapolate(self.t1, self.t2)

    def residuals(self, F, t1, t2, real_time=False):
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
        r2 = self.r_T2(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)

        if HAS_TORCH and isinstance(Fae, torch.Tensor):
            del Fae, Fmi, Wmnij, Wmbej, Wmbje, Zmbij

        if self.model == 'CC3':
            X1, X2 = self._cc3_t_residual(o, v, F, ERI, L, t1, t2, Fme, real_time=real_time)
            r1 += X1
            r2 += X2 + X2.swapaxes(0,1).swapaxes(2,3)

            if HAS_TORCH and isinstance(r1, torch.Tensor):
                del Fme

        return r1, r2

    def _cc3_t_residual(self, o, v, F, ERI, L, t1, t2, Fme, real_time=False):
        """Compute the CC3 connected-triples contributions (X1, X2) to the T residuals.

        Builds the T1-dressed CC3 W-intermediates, then accumulates the connected
        triples' contribution to the T1 and T2 residuals. Single-instance helper
        that parallels cclambda._cc3_lambda_triples (the t3 side of the l3 work).

        Parameters
        ----------
        o, v : slice
            occupied/virtual orbital slices
        F, ERI, L : ndarray or torch.Tensor
            Fock matrix, two-electron integrals, and L = 2*ERI - ERI.swapaxes
        t1, t2 : ndarray or torch.Tensor
            current T1/T2 amplitudes
        Fme : ndarray or torch.Tensor
            the one-body H_me intermediate (already built by residuals)
        real_time : bool
            if True, subtract the explicit time-dependent perturbation from the
            connected triples (real-time CC3 path)

        Returns
        -------
        X1, X2 : ndarray or torch.Tensor
            the CC3 triples contributions to the T1 and T2 residuals
        """
        no = self.no
        contract = self.contract

        Wmnij_cc3 = self.build_cc3_Wmnij(o, v, ERI, t1)
        Wmbij_cc3 = self.build_cc3_Wmbij(o, v, ERI, t1, Wmnij_cc3)
        Wmnie_cc3 = self.build_cc3_Wmnie(o, v, ERI, t1)
        Wamef_cc3 = self.build_cc3_Wamef(o, v, ERI, t1)
        Wabei_cc3 = self.build_cc3_Wabei(o, v, ERI, t1)

        X1 = zeros_like(t1)
        X2 = zeros_like(t2)

        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk(o, v, i, j, k, t2, Wabei_cc3, Wmbij_cc3, F, contract, WithDenom=True)

                    if real_time is True:
                        V = F - clone(self.H.F)
                        t3 -= t3_pert_ijk(o, v, i, j, k, t2, V, F, contract)

                    X1[i] += contract('abc,bc->a', t3 - t3.swapaxes(0,2), L[j,k,v,v])

                    X2[i,j] += contract('abc,dbc->ad', 2 * t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2), Wamef_cc3.swapaxes(0,1)[k])
                    X2[i] -= contract('lc,abc->lab', Wmnie_cc3[j,k], (2 * t3 - t3.swapaxes(1,2) - t3.swapaxes(0,2)))
                    X2[i,j] += contract('abc,c->ab', t3 - t3.swapaxes(0,2), Fme[k])

        if HAS_TORCH and isinstance(t3, torch.Tensor):
            del Wmnij_cc3, Wmbij_cc3, Wmnie_cc3, Wamef_cc3, Wabei_cc3

        return X1, X2

    def build_tau(self, t1, t2, fact1=1.0, fact2=1.0):
        """Build the effective doubles amplitude tau = fact1*t2 + fact2*(t1 t1).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, nv, nv)
            tau indexed [i, j, a, b]; t1 is [i, a], t2 is [i, j, a, b].

        Notes
        -----
        ::

            tau_ijab = fact1 * t2_ijab + fact2 * t_ia t_jb

        (defaults fact1 = fact2 = 1).
        """
        contract = self.contract
        return fact1 * t2 + fact2 * contract('ia,jb->ijab', t1, t1)


    def build_Fae(self, o, v, F, L, t1, t2):
        """Build the F_ae similarity-transformed one-body intermediate.

        Parameters
        ----------
        o, v : slice
            Occupied and virtual orbital subspaces.
        F, L : ndarray or torch.Tensor
            Fock matrix and spin-adapted ERIs (L = 2<pq|rs> - <pq|sr>).

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv)
            Virtual-virtual intermediate indexed [a, e].

        Notes
        -----
        CCSD form (repeated indices summed; CCD keeps only the f_ae and final
        t2 terms)::

            F_ae = f_ae - 1/2 f_me t_ma + t_mf L_mafe
                        - (t2_mnaf + 1/2 t_ma t_nf) L_mnef
        """
        contract = self.contract

        if self.model == 'CCD':
            Fae = clone(F[v,v])
            Fae = Fae - contract('mnaf,mnef->ae', t2, L[o,o,v,v])
        else:
            Fae = clone(F[v,v])
            Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
            Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
            Fae = Fae - contract('mnaf,mnef->ae', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fae


    def build_Fmi(self, o, v, F, L, t1, t2):
        """Build the F_mi similarity-transformed one-body intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no)
            Occupied-occupied intermediate indexed [m, i].

        Notes
        -----
        CCSD form (repeated indices summed; CCD keeps only the f_mi and final
        t2 terms)::

            F_mi = f_mi + 1/2 t_ie f_me + t_ne L_mnie
                        + (t2_inef + 1/2 t_ie t_nf) L_mnef
        """
        contract = self.contract

        if self.model == 'CCD':
            Fmi = clone(F[o,o])
            Fmi = Fmi + contract('inef,mnef->mi', t2, L[o,o,v,v])
        else:
            Fmi = clone(F[o,o])
            Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
            Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
            Fmi = Fmi + contract('inef,mnef->mi', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fmi


    def build_Fme(self, o, v, F, L, t1):
        """Build the F_me similarity-transformed one-body intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv)
            Occupied-virtual intermediate indexed [m, e]. Returns None for the
            CCD model, which has no singles.

        Notes
        -----
        Repeated indices summed::

            F_me = f_me + t_nf L_mnef
        """
        contract = self.contract

        if self.model == 'CCD':
            return
        else:
            Fme = clone(F[o,v])
            Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Fme


    def build_Wmnij(self, o, v, ERI, t1, t2):
        """Build the W_mnij similarity-transformed two-body intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, no, no)
            Intermediate indexed [m, n, i, j]. ERI is in Dirac order <pq|rs>.

        Notes
        -----
        CCSD form (repeated indices summed)::

            W_mnij = <mn|ij> + t_je <mn|ie> + t_ie <mn|ej>
                            + (t2_ijef + t_ie t_jf) <mn|ef>
        """
        contract = self.contract

        if self.model == 'CCD':
            Wmnij = clone(ERI[o,o,o,o], device=self.device1)
            Wmnij = Wmnij + contract('ijef,mnef->mnij', t2, ERI[o,o,v,v])
        else:
            Wmnij = clone(ERI[o,o,o,o], device=self.device1)
            Wmnij = Wmnij + contract('je,mnie->mnij', t1, ERI[o,o,o,v])
            Wmnij = Wmnij + contract('ie,mnej->mnij', t1, ERI[o,o,v,o])
            if self.model == 'CC2':
                tmp = contract('mnef,ei->mnif', ERI[o,o,v,v], t1.T)
                Wmnij = Wmnij + contract('mnif,fj->mnij', tmp, t1.T)
            else:
                Wmnij = Wmnij + contract('ijef,mnef->mnij', self.build_tau(t1, t2), ERI[o,o,v,v])
        return Wmnij


    def build_Wmbej(self, o, v, ERI, L, t1, t2):
        """Build the W_mbej similarity-transformed two-body intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, nv, no)
            Intermediate indexed [m, b, e, j]. Returns None for the CC2 model,
            which omits this intermediate.

        Notes
        -----
        CCSD form (repeated indices summed)::

            W_mbej = <mb|ej> + t_jf <mb|ef> - t_nb <mn|ej>
                            - (1/2 t2_jnfb + t_jf t_nb) <mn|ef>
                            + 1/2 t2_njfb L_mnef
        """
        contract = self.contract

        if self.model == 'CCD':
            Wmbej = clone(ERI[o,v,v,o], device=self.device1)
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', 0.5*t2, ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        elif self.model == 'CC2':
            return
        else:
            Wmbej = clone(ERI[o,v,v,o], device=self.device1)
            Wmbej = Wmbej + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
            Wmbej = Wmbej - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Wmbej


    def build_Wmbje(self, o, v, ERI, t1, t2):
        """Build the W_mbje similarity-transformed two-body intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, no, nv)
            Intermediate indexed [m, b, j, e]. Returns None for the CC2 model,
            which omits this intermediate.

        Notes
        -----
        CCSD form (repeated indices summed)::

            W_mbje = -<mb|je> - t_jf <mb|fe> + t_nb <mn|je>
                             + (1/2 t2_jnfb + t_jf t_nb) <mn|fe>
        """
        contract = self.contract

        if self.model == 'CCD':
            Wmbje = -1.0 * clone(ERI[o,v,o,v], device=self.device1)
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', 0.5*t2, ERI[o,o,v,v])
        elif self.model == 'CC2':
            return
        else:
            Wmbje = -1.0 * clone(ERI[o,v,o,v], device=self.device1)
            Wmbje = Wmbje - contract('jf,mbfe->mbje', t1, ERI[o,v,v,v])
            Wmbje = Wmbje + contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
        return Wmbje


    def build_Zmbij(self, o, v, ERI, t1, t2):
        """Build the Z_mbij similarity-transformed intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, no, no)
            Intermediate indexed [m, b, i, j]. Returns None for the CCD model.

        Notes
        -----
        CCSD form (repeated indices summed)::

            Z_mbij = <mb|ef> (t2_ijef + t_ie t_jf)
        """
        contract = self.contract

        if self.model == 'CCD':
            return
        elif self.model == 'CC2':
            tmp = contract('mbef,ie->mbif', ERI[o,v,v,v], t1)
            return contract('mbif,fj->mbij', tmp, t1.T)
        else:
            return contract('mbef,ijef->mbij', ERI[o,v,v,v], self.build_tau(t1, t2))


    def r_T1(self, o, v, F, ERI, L, t1, t2, Fae, Fme, Fmi):
        """Compute the T1 (singles) amplitude residual.

        solve_cc drives this residual to zero. Fae/Fme/Fmi are the precomputed
        one-body intermediates. Returns zeros for CCD (no singles).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv)
            T1 residual indexed [i, a].

        Notes
        -----
        CCSD form (repeated indices summed)::

            r_t1_ia = f_ia + t_ie F_ae - F_mi t_ma
                    + (2 t2_imae - t2_imea) F_me
                    + t_nf L_nafi
                    + (2 t2_mief - t2_mife) <ma|ef>
                    - t2_mnae L_nmei
        """
        contract = self.contract

        if self.model == 'CCD':
            r_T1 = zeros_like(t1)
        else:
            r_T1 = clone(F[o,v])
            r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
            r_T1 = r_T1 - contract('mi,ma->ia', Fmi, t1)
            r_T1 = r_T1 + contract('imae,me->ia', (2.0*t2 - t2.swapaxes(2,3)), Fme)
            r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
            r_T1 = r_T1 + contract('mief,maef->ia', (2.0*t2 - t2.swapaxes(2,3)), ERI[o,v,v,v])
            r_T1 = r_T1 - contract('mnae,nmei->ia', t2, L[o,o,v,o])
        return r_T1


    def r_T2(self, o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):
        """Compute the T2 (doubles) amplitude residual.

        solve_cc drives this residual to zero. This method dispatches on the CC
        model to the per-model builder (CCD / CC2 / CCSD; CC3 and CCSD(T) use the
        CCSD form), then symmetrizes the spin-adapted "half" residual as
        r_t2_ijab += r_t2_jiba. Fae/Fmi/Fme and Wmnij/Wmbej/Wmbje/Zmbij are the
        precomputed intermediates.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, nv, nv)
            T2 residual indexed [i, j, a, b].
        """
        if self.model == 'CCD':
            r_T2 = self._r_T2_ccd(o, v, ERI, t2, Fae, Fmi, Wmnij, Wmbej, Wmbje)
        elif self.model == 'CC2':
            r_T2 = self._r_T2_cc2(o, v, F, ERI, t1, t2, Wmnij, Zmbij)
        else:
            r_T2 = self._r_T2_ccsd(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)

        r_T2 = r_T2 + r_T2.swapaxes(0,1).swapaxes(2,3)
        return r_T2

    def _r_T2_ccd(self, o, v, ERI, t2, Fae, Fmi, Wmnij, Wmbej, Wmbje):
        """CCD doubles residual (no singles), before the r_T2 symmetrization.

        Notes
        -----
        The CCSD form with every singles term dropped (so tau -> t2); F_be = Fae
        and F_mj = Fmi are the CCD intermediates. Before the i<->j / a<->b
        symmetrization (repeated indices summed; W_mbje[e<->j] is Wmbje with its
        last two axes swapped)::

            r_t2_ijab = 1/2 <ij|ab>
                      + t2_ijae F_be - t2_imab F_mj
                      + 1/2 t2_mnab W_mnij + 1/2 t2_ijef <ab|ef>
                      + (t2_imae - t2_imea) W_mbej
                      + t2_imae (W_mbej + W_mbje[e<->j])
                      + t2_mjae W_mbie
        """
        contract = self.contract

        r_T2 = 0.5 * clone(ERI[o,o,v,v], device=self.device1)
        r_T2 = r_T2 + contract('ijae,eb->ijab', t2, Fae.T)
        r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
        r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', t2, ERI[v,v,v,v])
        r_T2 = r_T2 + 0.5 * contract('mnij,mnab->ijab', Wmnij, t2)
        r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
        r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
        r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
        return r_T2

    def _r_T2_cc2(self, o, v, F, ERI, t1, t2, Wmnij, Zmbij):
        """CC2 doubles residual (bare-Fock Fae/Fmi forms), before the r_T2 symmetrization.

        Notes
        -----
        Like CCSD, but the Fae/Fmi dressings are truncated to their bare-Fock + t1
        forms (f is the Fock matrix; the W_mnij and <ab|ef> terms are evaluated on
        t1.t1 rather than tau). Before the i<->j / a<->b symmetrization (repeated
        indices summed)::

            r_t2_ijab = 1/2 <ij|ab>
                      + t2_ijae (f_be - 1/2 f_me t_mb) - 1/2 t2_ijae f_me t_mb
                      - t2_imab (f_mj + 1/2 f_me t_je) - 1/2 t2_imab f_me t_je
                      + 1/2 t_ma t_nb W_mnij + 1/2 t_ie t_jf <ab|ef>
                      - t_ma Z_mbij
                      - t_ie t_ma <mb|ej> - t_ie t_mb <ma|je>
                      + t_ie <ab|ej> - t_ma <mb|ij>
        """
        contract = self.contract

        r_T2 = 0.5 * clone(ERI[o,o,v,v], device=self.device1)

        tmp = F[v,v] - 0.5 * contract('me,ma->ae', F[o,v], t1)
        r_T2 = r_T2 + contract('ijae,eb->ijab', t2, tmp.T)
        tmp = contract('mb,me->be', t1, F[o,v])
        r_T2 = r_T2 - 0.5 * contract('ijae,eb->ijab', t2, tmp.T)
        tmp = F[o,o] + 0.5 * contract('ie,me->mi', t1, F[o,v])
        r_T2 = r_T2 - contract('imab,mj->ijab', t2, tmp)
        tmp = contract('je,me->jm', t1, F[o,v])
        r_T2 = r_T2 - 0.5 * contract('imab,jm->ijab', t2, tmp)
        r_T2 = r_T2 + 0.5 * contract('ma,mbij->ijab', t1, contract('nb,mnij->mbij', t1, Wmnij))
        r_T2 = r_T2 + 0.5 * contract('jf,abif->ijab', t1, contract('ie,abef->abif', t1, ERI[v,v,v,v]))
        r_T2 = r_T2 - contract('ma,mbij->ijab', t1, Zmbij)
        r_T2 = r_T2 - contract('ma,mbij->ijab', t1, contract('ie,mbej->mbij', t1, ERI[o,v,v,o]))
        r_T2 = r_T2 - contract('mb,maji->ijab', t1, contract('ie,maje->maji', t1, ERI[o,v,o,v]))
        r_T2 = r_T2 + contract('ie,abej->ijab', t1, ERI[v,v,v,o])
        r_T2 = r_T2 - contract('ma,mbij->ijab', t1, ERI[o,v,o,o])
        contract = self.contract
        if HAS_TORCH and isinstance(tmp, torch.Tensor):
            del tmp
        return r_T2

    def _r_T2_ccsd(self, o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):
        """CCSD doubles residual (also used by CC3 / CCSD(T)), before the r_T2 symmetrization.

        Notes
        -----
        CCSD form, before the i<->j / a<->b symmetrization (repeated indices
        summed; W_mbje[e<->j] is Wmbje with its last two axes swapped)::

            r_t2_ijab = 1/2 <ij|ab>
                      + t2_ijae F_be  - 1/2 t2_ijae t_mb F_me
                      - t2_imab F_mj  - 1/2 t2_imab t_je F_me
                      + 1/2 tau_mnab W_mnij + 1/2 tau_ijef <ab|ef>
                      - t_ma Z_mbij
                      + (t2_imae - t2_imea) W_mbej
                      + t2_imae (W_mbej + W_mbje[e<->j])
                      + t2_mjae W_mbie
                      - t_ie t_ma <mb|ej> - t_ie t_mb <ma|je>
                      + t_ie <ab|ej> - t_ma <mb|ij>
        """
        contract = self.contract

        r_T2 = 0.5 * clone(ERI[o,o,v,v], device=self.device1)
        r_T2 = r_T2 + contract('ijae,eb->ijab', t2, Fae.T)
        tmp = contract('bm,me->be', t1.T, Fme)
        r_T2 = r_T2 - 0.5 * contract('ijae,eb->ijab', t2, tmp.T)
        r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
        tmp = contract('je,em->jm', t1, Fme.T)
        r_T2 = r_T2 - 0.5 * contract('imab,mj->ijab', t2, tmp.T)
        tau = self.build_tau(t1, t2)
        r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', tau, Wmnij)
        r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', tau, ERI[v,v,v,v])
        r_T2 = r_T2 - contract('ma,mbij->ijab', t1, Zmbij)
        r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
        r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
        r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
        tmp = contract('ei,am->aemi', t1.T, t1.T)
        r_T2 = r_T2 - contract('imea,mbej->ijab', tmp.swapaxes(0,3).swapaxes(1,2), ERI[o,v,v,o])
        r_T2 = r_T2 - contract('imeb,maje->ijab', tmp.swapaxes(0,3).swapaxes(1,2), ERI[o,v,o,v])
        r_T2 = r_T2 + contract('ie,abej->ijab', t1, ERI[v,v,v,o])
        r_T2 = r_T2 - contract('ma,mbij->ijab', t1, ERI[o,v,o,o])

        if HAS_TORCH and isinstance(tmp, torch.Tensor):
            del tmp
        return r_T2

    # Intermedeates needed for CC3
    def build_cc3_Wmnij(self, o, v, ERI, t1):
        """Build the CC3 W_mnij intermediate (T1-dressed integrals).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, no, no)
            Indexed [m, n, i, j].

        Notes
        -----
        Repeated indices summed::

            W_mnij = <mn|ij> + t_ja <mn|ia> + t_ia <nm|ja>
                            + t_ie t_jf <mn|ef>
        """
        contract = self.contract
        W = clone(ERI[o,o,o,o], device=self.device1)
        tmp = contract('ijma,na->ijmn', ERI[o,o,o,v], t1)
        W = W + tmp + tmp.swapaxes(0,1).swapaxes(2,3)
        tmp = contract('ia,mnaf->mnif', t1, ERI[o,o,v,v])
        W = W + contract('mnif,jf->mnij', tmp, t1)
        return W

    def build_cc3_Wmbij(self, o, v, ERI, t1, Wmnij):
        """Build the CC3 W_mbij intermediate (T1-dressed integrals).

        Reuses the CC3 W_mnij intermediate.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, no, no)
            Indexed [m, b, i, j].

        Notes
        -----
        Repeated indices summed::

            W_mbij = <mb|ij> - W_mnij t_nb + t_je <mb|ie>
                            + t_ie ( <mb|ej> + t_jf <mb|ef> )
        """
        contract = self.contract
        W = clone(ERI[o,v,o,o], device=self.device1)
        W = W - contract('mnij,nb->mbij', Wmnij, t1)
        W = W + contract('mbie,je->mbij', ERI[o,v,o,v], t1)
        tmp = clone(ERI[o,v,v,o], device=self.device1) + contract('mbef,jf->mbej', ERI[o,v,v,v], t1)
        W = W + contract('ie,mbej->mbij', t1, tmp)
        return W

    def build_cc3_Wmnie(self, o, v, ERI, t1):
        """Build the CC3 W_mnie intermediate (T1-dressed integrals).

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, no, nv)
            Indexed [m, n, i, e].

        Notes
        -----
        Repeated indices summed::

            W_mnie = <mn|ie> + t_if <mn|fe>
        """
        contract = self.contract
        W = clone(ERI[o,o,o,v], device=self.device1)
        W = W + contract('if,mnfe->mnie', t1, ERI[o,o,v,v])
        return W

    def build_cc3_Wamef(self, o, v, ERI, t1):
        """Build the CC3 W_amef intermediate (T1-dressed integrals).

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, no, nv, nv)
            Indexed [a, m, e, f].

        Notes
        -----
        Repeated indices summed::

            W_amef = <am|ef> - t_na <nm|ef>
        """
        contract = self.contract
        W = clone(ERI[v,o,v,v], device=self.device1)
        W = W - contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return W

    def build_cc3_Wabei(self, o, v, ERI, t1):
        """Build the CC3 W_abei intermediate (T1-dressed integrals).

        The most involved CC3 intermediate. It is assembled in two parts that
        are combined as W_abei = Z_abei + Z_eiab[swap (a,e) and (b,i)]:

        - Z_eiab: <ei|ab> dressed by t1 through the vir-vir-vir-vir integrals
          (symmetric + antisymmetric pieces) and ladder/ring corrections;
        - Z_abei: the -t_ma (<mb|ei> + t_if <mb|ef>) particle-attachment term.

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv, nv, no)
            Indexed [a, b, e, i]. See the inline construction for the exact
            term-by-term T1 dressing.
        """
        contract = self.contract
        # eiab
        Z = clone(ERI[v,o,v,v], device=self.device1)
        tmp_ints = ERI[v,v,v,v] + ERI[v,v,v,v].swapaxes(2,3)
        Z1 = 0.5 * contract('if,abef->eiab', t1, tmp_ints)
        tmp_ints = ERI[v,v,v,v] - ERI[v,v,v,v].swapaxes(2,3)
        Z2 = 0.5 * contract('if,abef->eiab', t1, tmp_ints)
        Z_eiab = Z + Z1 + Z2

        #eiab
        Zeiam = clone(ERI[v,o,v,o], device=self.device1)
        Zamei = contract('amef,if->amei', ERI[v,o,v,v], t1)
        Zeiam = Zeiam + Zamei.swapaxes(0,2).swapaxes(1,3)
        Z_eiab = Z_eiab - contract('eiam,mb->eiab', Zeiam, t1)

        #eiab
        Zmnei = clone(ERI[o,o,v,o], device=self.device1) + contract('mnef,if->mnei', ERI[o,o,v,v], t1)
        Zanei = contract('ma,mnei->anei', t1, Zmnei)
        Z_eiab = Z_eiab + contract('anei,nb->eiab', Zanei, t1)

        #abei
        Zmbei = clone(ERI[o,v,v,o], device=self.device1)
        Zmbei = Zmbei + contract('mbef,if->mbei', ERI[o,v,v,v], t1)
        Z_abei = -1 * contract('ma,mbei->abei', t1, Zmbei)

        # Wabei
        W = Z_abei + Z_eiab.swapaxes(0,2).swapaxes(1,3)
        return W

    def cc_energy(self, o, v, F, L, t1, t2):
        """Compute the CC correlation energy from the current amplitudes.

        Returns
        -------
        float
            The CC correlation energy (repeated indices summed)::

                CCD:   E = t2_ijab L_ijab
                else:  E = 2 f_ia t_ia + tau_ijab L_ijab   (tau = t2 + t1 t1)
        """
        contract = self.contract
        if self.model == 'CCD':
            ecc = contract('ijab,ijab->', t2, L[o,o,v,v])
        else:
            ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
            ecc = ecc + contract('ijab,ijab->', self.build_tau(t1, t2), L[o,o,v,v])
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

        return ET
