"""
ciwfn.py: Configuration-interaction (CISD) wavefunction.
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import numpy as np

from .wavefunction import Wavefunction
from .mpwfn import MPwfn
from .utils import helper_diis, clone, sqrt, zeros_like
from .exceptions import InvalidKeywordError

if TYPE_CHECKING:
    from ._typing import Tensor


class CIwfn(Wavefunction):
    """A configuration-interaction (CISD) wavefunction on the shared
    :class:`Wavefunction` base.

    Attributes
    ----------
    model : str
        'CISD' (singles + doubles) or 'CID' (doubles only)
    mp : MPwfn
        composed MP2 wavefunction supplying ``Dijab`` and the MP1 guess
    Dia, Dijab : Tensor
        one- and two-electron energy denominators (from ``diag(F)``)
    c1, c2 : Tensor
        current singles / doubles CI coefficients
    eci : float
        the CI correlation energy (set by :meth:`solve_ci`)
    """

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
        time_init = time.time()

        valid_ci_models = ['CISD', 'CID']
        model = kwargs.pop('model', 'CISD').upper()
        if model not in valid_ci_models:
            raise InvalidKeywordError('model', model, valid_ci_models)
        self.model = model
        self.need_singles = model in ['CISD']

        super().__init__(scf_wfn, **kwargs)
        mgr = self.device_manager

        # The MP2 wavefunction supplies the energy denominators and the CISD initial
        # guess (MP1 doubles), reusing this object's base - the same pattern ccwfn
        # uses. CI's singles denominator (Dia) is built from the MP2 orbital energies.
        self.mp = MPwfn.from_wavefunction(self)
        self.Dijab = self.mp.Dijab
        self.Dia = self.mp.eps_occ.reshape(-1, 1) - self.mp.eps_vir

        # Initial guess: c1 = 0, c2 = MP1 doubles (CI mutates c2 in place, so copy).
        self.c1 = mgr.seed_compute(np.zeros((self.no, self.nv)))
        self.c2 = clone(self.mp.t2)

        print("CIWFN object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_ci(self, e_conv: float = 1e-7, r_conv: float = 1e-7, maxiter: int = 100, max_diis: int = 8, start_diis: int = 1) -> "Tensor":
        """Iterate the projected CISD equations to convergence and return E_c.
        """
        ci_tstart = time.time()

        o, v = self.o, self.v
        F, ERI = self.H.F, self.H.ERI
        L = self.H.L if self.orbital_basis == 'spatial' else None  # no L in spin orbitals
        Dia, Dijab = self.Dia, self.Dijab
        contract = self.contract

        eci = self.ci_energy(o, v, F, L, self.c1, self.c2)
        print("CI Iter %3d: CI Ecorr = %.15f  dE = % .5E  MP2" % (0, eci, -eci))

        diis = helper_diis(self.c1, self.c2, max_diis, self.precision)

        for niter in range(1, maxiter + 1):
            eci_last = eci

            s1 = self.sigma1(o, v, F, ERI, L, self.c1, self.c2)
            s2 = self.sigma2(o, v, F, ERI, L, self.c1, self.c2)

            r1 = s1 - eci * self.c1
            r2 = s2 - eci * self.c2
            inc1 = r1 / (Dia + eci)
            inc2 = r2 / (Dijab + eci)
            self.c1 = self.c1 + inc1
            self.c2 = self.c2 + inc2
            rms = contract('ia,ia->', inc1, inc1) + contract('ijab,ijab->', inc2, inc2)
            rms = sqrt(rms)

            eci = self.ci_energy(o, v, F, L, self.c1, self.c2)
            ediff = eci - eci_last
            print("CI Iter %3d: CI Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, eci, ediff, rms))

            if (abs(ediff) < e_conv) and abs(rms) < r_conv:
                print("\nCIWFN converged in %.3f seconds.\n" % (time.time() - ci_tstart))
                print("E(REF)   = %20.15f" % self.eref)
                print("E(%s)  = %20.15f" % (self.model, eci))
                print("E(TOT)   = %20.15f" % (eci + self.eref))
                self.eci = eci
                return eci

            diis.add_error_vector(self.c1, self.c2)
            if niter >= start_diis:
                self.c1, self.c2 = diis.extrapolate(self.c1, self.c2)

    def ci_energy(self, o, v, F, L, c1, c2) -> "Tensor":
        """CISD correlation energy, E_c = <Phi_0|H_N|Psi> = 2 f_ia c_ia + c_ijab L_ijab.

        The singles term is kept general (non-zero only for non-Brillouin references);
        the doubles term is the linear (CI) form of the CC energy (no t1.t1).

        Spin-orbital path: E_c = f_ia c_ia + 1/4 <ij||ab> c_ijab.
        """
        if self.orbital_basis == 'spinorbital':
            return self._so_ci_energy(o, v, F, self.H.ERI, c1, c2)
        contract = self.contract
        e = 2.0 * contract('ia,ia->', F[o, v], c1)
        e = e + contract('ijab,ijab->', c2, L[o, o, v, v])
        return e

    def _so_ci_energy(self, o, v, F, ERI, c1, c2) -> "Tensor":
        """Spin-orbital CISD correlation energy, E_c = f_ia c_ia + 1/4 <ij||ab> c_ijab."""
        contract = self.contract
        e = contract('ia,ia->', F[o, v], c1)
        e = e + 0.25 * contract('ijab,ijab->', c2, ERI[o, o, v, v])
        return e

    def sigma1(self, o, v, F, ERI, L, c1, c2) -> "Tensor":
        """Singles sigma vector, sigma1_ia = <Phi_i^a|H_N|Psi>.

        The linear part of the CCSD T1 residual (t -> c) with the dressed one-body
        intermediates replaced by the bare Fock blocks (no diagonal-Fock assumption)::

            sigma1_ia = f_ia + c1_ie f_ae - f_mi c1_ma + (2 c2_imae - c2_imea) f_me
                      + c1_nf L_nafi + L_amef c_imef - c2_mnae L_mnie
        """
        if not self.need_singles:
            return zeros_like(c1)

        if self.orbital_basis == 'spinorbital':
            return self._so_sigma1(o, v, F, self.H.ERI, c1, c2)

        contract = self.contract
        s1 = clone(F[o, v], device=self.device1)
        s1 = s1 + contract('ie,ae->ia', c1, F[v, v])
        s1 = s1 - contract('mi,ma->ia', F[o, o], c1)
        s1 = s1 + contract('imae,me->ia', (2.0 * c2 - c2.swapaxes(2, 3)), F[o, v])
        s1 = s1 + contract('nf,nafi->ia', c1, L[o, v, v, o])
        s1 = s1 + contract('amef,imef->ia', L[v, o, v, v], c2)
        s1 = s1 - contract('mnae,mnie->ia', c2, L[o, o, o, v])
        return s1

    def _so_sigma1(self, o, v, F, ERI, c1, c2) -> "Tensor":
        """Spin-orbital singles sigma vector: the linear part of the spin-orbital CCSD
        T1 residual (t -> c) with the dressed one-body intermediates replaced by the
        bare Fock blocks (Fae -> f_vv, Fmi -> f_oo, Fme -> f_ov)."""
        contract = self.contract
        s1 = clone(F[o, v], device=self.device1)
        s1 = s1 + contract('ie,ae->ia', c1, F[v, v])
        s1 = s1 - contract('ma,mi->ia', c1, F[o, o])
        s1 = s1 + contract('imae,me->ia', c2, F[o, v])
        s1 = s1 - contract('nf,naif->ia', c1, ERI[o, v, o, v])
        s1 = s1 - 0.5 * contract('imef,maef->ia', c2, ERI[o, v, v, v])
        s1 = s1 - 0.5 * contract('mnae,nmei->ia', c2, ERI[o, o, v, o])
        return s1

    def sigma2(self, o, v, F, ERI, L, c1, c2) -> "Tensor":
        """Doubles sigma vector, sigma2_ijab = <Phi_ij^ab|H_N|Psi>.

        The linear part of the CCSD T2 residual (t -> c, tau -> c2) with the dressed
        two-body intermediates replaced by their bare integrals. Built as a "half"
        residual and then symmetrized (i<->j, a<->b), as in ``r_T2``::

            sigma2(half) = 1/2 <ij|ab> + c2_ijae f_be - c2_imab f_mj
                         + 1/2 c2_mnab <mn|ij> + 1/2 c2_ijef <ab|ef>
                         + c2_miea L_mbej - c2_imea <mb|ej> - c2_imeb <ma|je>
                         + c1_ie <ab|ej> - c1_ma <mb|ij>
            sigma2 = sigma2(half) + sigma2(half)[i<->j, a<->b]
        """
        if self.orbital_basis == 'spinorbital':
            return self._so_sigma2(o, v, F, self.H.ERI, c1, c2)

        contract = self.contract

        s2 = 0.5 * clone(ERI[o, o, v, v], device=self.device1)
        s2 = s2 + contract('ijae,eb->ijab', c2, F[v, v].T)
        s2 = s2 - contract('imab,mj->ijab', c2, F[o, o])
        s2 = s2 + 0.5 * contract('mnab,mnij->ijab', c2, ERI[o, o, o, o])
        s2 = s2 + 0.5 * contract('ijef,abef->ijab', c2, ERI[v, v, v, v])
        s2 = s2 + contract('miea,mbej->ijab', c2, L[o, v, v, o])
        s2 = s2 - contract('imea,mbej->ijab', c2, ERI[o, v, v, o])
        s2 = s2 - contract('imeb,maje->ijab', c2, ERI[o, v, o, v])
        if self.need_singles:
            s2 = s2 + contract('ie,abej->ijab', c1, ERI[v, v, v, o])
            s2 = s2 - contract('ma,mbij->ijab', c1, ERI[o, v, o, o])

        s2 = s2 + s2.swapaxes(0, 1).swapaxes(2, 3)
        return s2

    def _so_sigma2(self, o, v, F, ERI, c1, c2) -> "Tensor":
        """Spin-orbital doubles sigma vector: the linear part of the spin-orbital CCSD
        T2 residual (t -> c) with the dressed two-body intermediates replaced by their
        bare antisymmetrized integrals. Built as the full (already i<->j, a<->b
        antisymmetric) residual, as in the spin-orbital ``_so_r_T2`` -- no separate
        symmetrization step."""
        contract = self.contract

        s2 = clone(ERI[o, o, v, v], device=self.device1)
        s2 = s2 + (contract('ijae,be->ijab', c2, F[v, v])
                   - contract('ijbe,ae->ijab', c2, F[v, v]))
        s2 = s2 - (contract('imab,mj->ijab', c2, F[o, o])
                   - contract('jmab,mi->ijab', c2, F[o, o]))
        s2 = s2 + 0.5 * contract('mnab,mnij->ijab', c2, ERI[o, o, o, o])
        s2 = s2 + 0.5 * contract('ijef,abef->ijab', c2, ERI[v, v, v, v])
        s2 = s2 + contract('imae,mbej->ijab', c2, ERI[o, v, v, o])
        s2 = s2 - contract('imbe,maej->ijab', c2, ERI[o, v, v, o])
        s2 = s2 - contract('jmae,mbei->ijab', c2, ERI[o, v, v, o])
        s2 = s2 + contract('jmbe,maei->ijab', c2, ERI[o, v, v, o])
        if self.need_singles:
            s2 = s2 + (contract('ie,abej->ijab', c1, ERI[v, v, v, o])
                       - contract('je,abei->ijab', c1, ERI[v, v, v, o]))
            s2 = s2 - (contract('ma,mbij->ijab', c1, ERI[o, v, o, o])
                       - contract('mb,maij->ijab', c1, ERI[o, v, o, o]))
            tmp = contract('ia,jb->ijab', c1, F[o, v])
            s2 = s2 + (tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3)
                       + tmp.swapaxes(0, 1).swapaxes(2, 3))
        return s2

    def _normalized_amplitudes(self):
        if getattr(self, '_ci_namp', None) is None:
            c = self.contract
            t1, t2 = self.c1, self.c2      # CIwfn's intermediate-normalized amplitudes
            norm2 = (2.0 * c('ia,ia->', t1.conj(), t1)
                     + c('ijab,ijab->', t2.conj(), 2.0 * t2 - t2.swapaxes(2, 3)))
            N = 1.0 / np.sqrt(1.0 + norm2)
            n0 = N.real if np.isrealobj(N) else N
            n1 = N * t1
            n2 = N * t2
            tau_n = 2.0 * n2 - n2.swapaxes(2, 3)
            self._ci_namp = (n0, n1, n2, tau_n)
        return self._ci_namp

    def _cisd_densities(self):
        """Cache (D_pq, D_pq_corr, D_pqrs): full 1-PDM, correlation-only
        1-PDM, and 2-PDM."""
        if getattr(self, '_ci_dens', None) is None:
            c = self.contract
            o, v, nmo, no = self.o, self.v, self.nmo, self.no
            n0, n1, n2, tau_n = self._normalized_amplitudes()

            D = np.zeros((nmo, nmo), dtype=n1.dtype)
            for i in range(no):
                D[i, i] += 2.0
            D[o, o] -= 2.0 * c('ja,ia->ij', n1.conj(), n1)
            D[o, o] -= 2.0 * c('jkab,ikab->ij', tau_n.conj(), n2)
            D[v, v] += 2.0 * c('ia,ib->ab', n1.conj(), n1)
            D[v, v] += 2.0 * c('ijac,ijbc->ab', tau_n.conj(), n2)
            D[o, v] += (2.0 * n0 * n1 + 2.0 * c('jb,ijab->ia', n1.conj(), 2.0 * n2 - n2.swapaxes(2, 3)))
            D[v, o] += (2.0 * n0 * n1.conj().T + 2.0 * c('ijab,jb->ai', (2.0 * n2 - n2.swapaxes(2, 3)).conj(), n1))
            D_corr = D.copy()
            for i in range(no):
                D_corr[i, i] -= 2.0

            G = np.zeros((nmo, nmo, nmo, nmo), dtype=n1.dtype)
            G[o, o, o, o] += c('klab,ijab->ijkl', n2, tau_n)
            G[v, v, v, v] += c('ijab,ijcd->abcd', n2, tau_n)
            G[o, v, v, o] += 4.0 * c('ja,ib->iabj', n1, n1)
            G[o, v, o, v] -= 2.0 * c('ja,ib->iajb', n1, n1)
            G[v, o, o, v] += 2.0 * c('jkac,ikbc->aijb', tau_n, tau_n)
            G[v, o, v, o] -= 4.0 * c('jkac,ikbc->aibj', n2, n2)
            G[v, o, v, o] += 2.0 * c('jkac,ikcb->aibj', n2, n2)
            G[v, o, v, o] += 2.0 * c('jkca,ikbc->aibj', n2, n2)
            G[v, o, v, o] -= 4.0 * c('jkca,ikcb->aibj', n2, n2)
            G[o, o, v, v] += n0 * tau_n
            tau_swp = (2.0 * n2.swapaxes(0, 2).swapaxes(1, 3) - n2.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3))
            G[v, v, o, o] += np.conjugate(tau_swp) * n0
            G[v, o, v, v] += 2.0 * c('ja,ijcb->aibc', n1, tau_n)
            G[o, v, o, o] -= 2.0 * c('kjab,ib->iajk', tau_n, n1)
            G[v, v, v, o] += 2.0 * c('jiab,jc->abci', tau_n, n1)
            G[o, o, o, v] -= 2.0 * c('kb,ijba->ijka', n1, tau_n)

            self._ci_dens = (D, D_corr, G)
        return self._ci_dens

    def _cisd_dn0(self, dn1, dn2):
        """First derivative of the normalization factor n0 along a perturbation. """
        c = self.contract
        t1, t2 = self.c1, self.c2
        tau = 2.0 * t2 - t2.swapaxes(2, 3)
        n0 = self._normalized_amplitudes()[0]
        return -n0**3 * (2.0 * c('ia,ia->', t1.conj(), dn1) + c('ijab,ijab->', tau.conj(), dn2))

    # coupled-perturbed CI 

    def _psi4_mints(self):
        """psi4 MintsHelper + C as psi4.core.Matrix"""
        if getattr(self, '_mints_cache', None) is None:
            import psi4
            mints = psi4.core.MintsHelper(self.H.basisset)
            C_p4 = psi4.core.Matrix.from_array(np.asarray(self.C))
            self._mints_cache = (mints, C_p4)
        return self._mints_cache

    def _build_magnetic_ints(self, beta):
        ct = self.contract
        nbf = self.nmo
        o, v = self.o, self.v
        t = slice(0, nbf)
        no, nv = self.no, self.nv
        ERI = np.asarray(self.H.ERI)
        F = np.asarray(self.H.F)
        C = np.asarray(self.C)
        eps = np.diag(F)
        mints, _ = self._psi4_mints()

        # A-matrix (antisymmetric perturbation)
        A_mag = -(2 * ERI - ERI.swapaxes(2, 3)) + (2 * ERI - ERI.swapaxes(2, 3)).swapaxes(1, 3)
        A_mag = A_mag.swapaxes(1, 2)
        G_mag = (ct('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no)) + A_mag[v, o, v, o])
        G_mag = np.linalg.inv(G_mag.reshape(nv * no, nv * no))

        # Magnetic dipole integrals
        L_AO = mints.ao_angular_momentum()
        h_mag = ct('mp,mn,nq->pq', C.conj(), -0.5 * L_AO[beta].np, C)

        # U_H  (full 4-block solve - exactly as in working code)
        U_H = np.zeros((nbf, nbf), dtype=complex)
        B_vo = h_mag[v, o]
        U_H[v, o] += (G_mag @ B_vo.reshape(nv * no)).reshape(nv, no)
        U_H[o, v] += U_H[v, o].T

        D_oo = (eps[o] - eps[o].reshape(-1, 1)) + np.eye(no)
        B_oo = (-h_mag[o, o].copy() + ct('em,iejm->ij', U_H[v, o], A_mag.swapaxes(1, 2)[o, v, o, o]))
        U_H[o, o] += B_oo / D_oo

        D_vv = (eps[v] - eps[v].reshape(-1, 1)) + np.eye(nv)
        B_vv = (-h_mag[v, v].copy() + ct('em,aebm->ab', U_H[v, o], A_mag.swapaxes(1, 2)[v, v, v, o]))
        U_H[v, v] += B_vv / D_vv

        for j in range(no):
            U_H[j, j] = 0
        for cc in range(no, nbf):
            U_H[cc, cc] = 0

        # dF/dH
        dF = np.zeros((nbf, nbf), dtype=complex)
        dF[o, o] -= h_mag[o, o].copy()
        dF[o, o] += (U_H[o, o] * eps[o].reshape(-1, 1) - U_H[o, o].swapaxes(0, 1) * eps[o])
        dF[o, o] += ct('em,iejm->ij', U_H[v, o], A_mag.swapaxes(1, 2)[o, v, o, o])

        dF[v, v] -= h_mag[v, v].copy()
        dF[v, v] += (U_H[v, v] * eps[v].reshape(-1, 1) - U_H[v, v].swapaxes(0, 1) * eps[v])
        dF[v, v] += ct('em,aebm->ab', U_H[v, o], A_mag.swapaxes(1, 2)[v, v, v, o])

        # dERI/dH
        dERI = np.zeros(ERI.shape, dtype=complex)
        dERI += ct('tr,pqts->pqrs', U_H[:, t], ERI[t, t, :, t])
        dERI += ct('ts,pqrt->pqrs', U_H[:, t], ERI[t, t, t, :])
        dERI -= ct('tp,tqrs->pqrs', U_H[:, t], ERI[:, t, t, t])
        dERI -= ct('tq,ptrs->pqrs', U_H[:, t], ERI[t, :, t, t])

        return dF, dERI, U_H

    def _build_vecpot_ints(self, gamma):
        ct = self.contract
        nbf = self.nmo
        o, v = self.o, self.v
        t = slice(0, nbf)
        no, nv = self.no, self.nv
        ERI = np.asarray(self.H.ERI)
        F = np.asarray(self.H.F)
        C = np.asarray(self.C)
        eps = np.diag(F)
        mints, _ = self._psi4_mints()

        # A-matrix for antisymmetric perturbation (same as magnetic)
        A_mag = -(2 * ERI - ERI.swapaxes(2, 3)) + (2 * ERI - ERI.swapaxes(2, 3)).swapaxes(1, 3)
        A_mag = A_mag.swapaxes(1, 2)
        G_mag = (ct('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no)) + A_mag[v, o, v, o])
        G_mag = np.linalg.inv(G_mag.reshape(nv * no, nv * no))

        # Linear momentum integrals: p_gamma = -i * nabla_gamma
        nabla_AO = mints.ao_nabla()
        h_A = ct('mp,mn,nq->pq', C.conj(), -nabla_AO[gamma].np, C)

        # Solve CPHF for U^A (identical structure to U^H)
        U_A = np.zeros((nbf, nbf), dtype=complex)
        B_vo = h_A[v, o]
        U_A[v, o] += (G_mag @ B_vo.reshape(nv * no)).reshape(nv, no)
        U_A[o, v] += U_A[v, o].T

        D_oo = (eps[o] - eps[o].reshape(-1, 1)) + np.eye(no)
        B_oo = (-h_A[o, o].copy() + ct('em,iejm->ij', U_A[v, o], A_mag.swapaxes(1, 2)[o, v, o, o]))
        U_A[o, o] += B_oo / D_oo

        D_vv = (eps[v] - eps[v].reshape(-1, 1)) + np.eye(nv)
        B_vv = (-h_A[v, v].copy() + ct('em,aebm->ab', U_A[v, o], A_mag.swapaxes(1, 2)[v, v, v, o]))
        U_A[v, v] += B_vv / D_vv

        for j in range(no):
            U_A[j, j] = 0
        for cc in range(no, nbf):
            U_A[cc, cc] = 0

        # dF/dA_gamma
        dF = np.zeros((nbf, nbf), dtype=complex)
        dF[o, o] -= h_A[o, o].copy()
        dF[o, o] += (U_A[o, o] * eps[o].reshape(-1, 1) - U_A[o, o].swapaxes(0, 1) * eps[o])
        dF[o, o] += ct('em,iejm->ij', U_A[v, o], A_mag.swapaxes(1, 2)[o, v, o, o])

        dF[v, v] -= h_A[v, v].copy()
        dF[v, v] += (U_A[v, v] * eps[v].reshape(-1, 1) - U_A[v, v].swapaxes(0, 1) * eps[v])
        dF[v, v] += ct('em,aebm->ab', U_A[v, o], A_mag.swapaxes(1, 2)[v, v, v, o])

        # dERI/dA_gamma (orbital response only, same sign pattern as magnetic)
        dERI = np.zeros(ERI.shape, dtype=complex)
        dERI += ct('tr,pqts->pqrs', U_A[:, t], ERI[t, t, :, t])
        dERI += ct('ts,pqrt->pqrs', U_A[:, t], ERI[t, t, t, :])
        dERI -= ct('tp,tqrs->pqrs', U_A[:, t], ERI[:, t, t, t])
        dERI -= ct('tq,ptrs->pqrs', U_A[:, t], ERI[t, :, t, t])

        return dF, dERI, U_A

    def _cpci_ints(self, pert):
        """(dF, dERI, U) for a cphf.Perturbation. Nuclear goes through pycc's
        CPHF (validated exact by LG-APT/VG-APT)"""
        if getattr(self, '_cpci_ints_cache', None) is None:
            self._cpci_ints_cache = {}
        if pert in self._cpci_ints_cache:
            return self._cpci_ints_cache[pert]
        cphf = self.cphf
        if pert.kind == 'nuclear':
            dF = np.asarray(cphf.perturbed_fock(pert))
            dERI = np.asarray(cphf.perturbed_eri(pert))
            U = np.asarray(cphf._full_U(pert))
            result = (dF, dERI, U)
        elif pert.kind == 'magnetic':
            result = self._build_magnetic_ints(pert.comp)
        elif pert.kind == 'vecpot':
            result = self._build_vecpot_ints(pert.comp)
        else:
            raise ValueError(f"unknown perturbation kind {pert.kind!r}")
        self._cpci_ints_cache[pert] = result
        return result

    def _solve_cpci(self, pert, maxiter=100, diis_start=2, diis_max=8,
                     e_convergence=1e-11, d_convergence=1e-11):
        """Coupled-perturbed CI
        Returns (dc1, dc2, dc0v)
        """
        if getattr(self, '_cpci_cache', None) is None:
            self._cpci_cache = {}
        if pert in self._cpci_cache:
            return self._cpci_cache[pert]

        from .utils import helper_diis
        c = self.contract
        o, v = self.o, self.v
        t1, t2 = self.c1, self.c2
        F, ERI = np.asarray(self.H.F), np.asarray(self.H.ERI)
        Dia, Dijab = self.Dia, self.Dijab
        E_cisd = self.eci
        n0 = self._normalized_amplitudes()[0]

        dF, dERI, U = self._cpci_ints(pert)
        dF, dERI = np.asarray(dF), np.asarray(dERI)

        D_pq, D_pq_corr, D_pqrs = self._cisd_densities()
        dE = c('pq,pq->', dF, D_pq) + c('pqrs,pqrs->', dERI, D_pqrs)

        dt1 = -(dE * t1).astype(complex)
        dt1 = dt1 - c('ji,ja->ia', dF[o, o], t1)
        dt1 = dt1 + c('ab,ib->ia', dF[v, v], t1)
        dt1 = dt1 + c('jabi,jb->ia', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t1)
        dt1 = dt1 + c('jb,ijab->ia', dF[o, v], 2.0 * t2 - t2.swapaxes(2, 3))
        dt1 = dt1 + c('ajbc,ijbc->ia', 2.0 * dERI[v, o, v, v] - dERI.swapaxes(2, 3)[v, o, v, v], t2)
        dt1 = dt1 - c('kjib,kjab->ia', 2.0 * dERI[o, o, o, v] - dERI.swapaxes(2, 3)[o, o, o, v], t2)
        dt1 = dt1 / Dia

        dt2 = -(dE * t2).astype(complex)
        dt2 = dt2 + c('abcj,ic->ijab', dERI[v, v, v, o], t1)
        dt2 = dt2 + c('abic,jc->ijab', dERI[v, v, o, v], t1)
        dt2 = dt2 - c('kbij,ka->ijab', dERI[o, v, o, o], t1)
        dt2 = dt2 - c('akij,kb->ijab', dERI[v, o, o, o], t1)
        dt2 = dt2 + c('ac,ijcb->ijab', dF[v, v], t2)
        dt2 = dt2 + c('bc,ijac->ijab', dF[v, v], t2)
        dt2 = dt2 - c('ki,kjab->ijab', dF[o, o], t2)
        dt2 = dt2 - c('kj,ikab->ijab', dF[o, o], t2)
        dt2 = dt2 + c('klij,klab->ijab', dERI[o, o, o, o], t2)
        dt2 = dt2 + c('abcd,ijcd->ijab', dERI[v, v, v, v], t2)
        dt2 = dt2 - c('kbcj,ikca->ijab', dERI[o, v, v, o], t2)
        dt2 = dt2 + c('kaci,kjcb->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
        dt2 = dt2 - c('kbic,kjac->ijab', dERI[o, v, o, v], t2)
        dt2 = dt2 - c('kaci,kjbc->ijab', dERI[o, v, v, o], t2)
        dt2 = dt2 + c('kbcj,ikac->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
        dt2 = dt2 - c('kajc,ikcb->ijab', dERI[o, v, o, v], t2)
        dt2 = dt2 / Dijab

        dE_proj = (2.0 * c('ia,ia->', t1, dF[o, v])
                   + c('ijab,ijab->', t2, 2.0 * dERI[o, o, v, v] - dERI.swapaxes(2, 3)[o, o, v, v])
                   + 2.0 * c('ia,ia->', dt1, F[o, v])
                   + c('ijab,ijab->', dt2, 2.0 * ERI[o, o, v, v] - ERI.swapaxes(2, 3)[o, o, v, v]))

        diis = helper_diis(dt1, dt2, diis_max, getattr(self, 'precision', 1e-12))

        for iteration in range(1, maxiter + 1):
            dE_proj_old = dE_proj
            dt1_old, dt2_old = dt1.copy(), dt2.copy()

            # singles residual - driving terms (dF/dERI acting on t1/t2)
            dRt1 = dF.copy().swapaxes(0, 1)[o, v].astype(complex)
            dRt1 = dRt1 - dE_proj * t1
            dRt1 = dRt1 - c('ji,ja->ia', dF[o, o], t1)
            dRt1 = dRt1 + c('ab,ib->ia', dF[v, v], t1)
            dRt1 = dRt1 + c('jabi,jb->ia', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t1)
            dRt1 = dRt1 + c('jb,ijab->ia', dF[o, v], 2.0 * t2 - t2.swapaxes(2, 3))
            dRt1 = dRt1 + c('ajbc,ijbc->ia', 2.0 * dERI[v, o, v, v] - dERI.swapaxes(2, 3)[v, o, v, v], t2)
            dRt1 = dRt1 - c('kjib,kjab->ia', 2.0 * dERI[o, o, o, v] - dERI.swapaxes(2, 3)[o, o, o, v], t2)
            # singles residual - response terms (F/ERI acting on dt1/dt2)
            dRt1 = dRt1 - E_cisd * dt1
            dRt1 = dRt1 - c('ji,ja->ia', F[o, o], dt1)
            dRt1 = dRt1 + c('ab,ib->ia', F[v, v], dt1)
            dRt1 = dRt1 + c('jabi,jb->ia', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dt1)
            dRt1 = dRt1 + c('jb,ijab->ia', F[o, v], 2.0 * dt2 - dt2.swapaxes(2, 3))
            dRt1 = dRt1 + c('ajbc,ijbc->ia', 2.0 * ERI[v, o, v, v] - ERI.swapaxes(2, 3)[v, o, v, v], dt2)
            dRt1 = dRt1 - c('kjib,kjab->ia', 2.0 * ERI[o, o, o, v] - ERI.swapaxes(2, 3)[o, o, o, v], dt2)

            # doubles residual - driving terms
            dRt2 = dERI.copy().swapaxes(0, 2).swapaxes(1, 3)[o, o, v, v].astype(complex)
            dRt2 = dRt2 - dE_proj * t2
            dRt2 = dRt2 + c('abcj,ic->ijab', dERI[v, v, v, o], t1)
            dRt2 = dRt2 + c('abic,jc->ijab', dERI[v, v, o, v], t1)
            dRt2 = dRt2 - c('kbij,ka->ijab', dERI[o, v, o, o], t1)
            dRt2 = dRt2 - c('akij,kb->ijab', dERI[v, o, o, o], t1)
            dRt2 = dRt2 + c('ac,ijcb->ijab', dF[v, v], t2)
            dRt2 = dRt2 + c('bc,ijac->ijab', dF[v, v], t2)
            dRt2 = dRt2 - c('ki,kjab->ijab', dF[o, o], t2)
            dRt2 = dRt2 - c('kj,ikab->ijab', dF[o, o], t2)
            dRt2 = dRt2 + c('klij,klab->ijab', dERI[o, o, o, o], t2)
            dRt2 = dRt2 + c('abcd,ijcd->ijab', dERI[v, v, v, v], t2)
            dRt2 = dRt2 - c('kbcj,ikca->ijab', dERI[o, v, v, o], t2)
            dRt2 = dRt2 + c('kaci,kjcb->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
            dRt2 = dRt2 - c('kbic,kjac->ijab', dERI[o, v, o, v], t2)
            dRt2 = dRt2 - c('kaci,kjbc->ijab', dERI[o, v, v, o], t2)
            dRt2 = dRt2 + c('kbcj,ikac->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
            dRt2 = dRt2 - c('kajc,ikcb->ijab', dERI[o, v, o, v], t2)
            # doubles residual - response terms
            dRt2 = dRt2 - E_cisd * dt2
            dRt2 = dRt2 + c('abcj,ic->ijab', ERI[v, v, v, o], dt1)
            dRt2 = dRt2 + c('abic,jc->ijab', ERI[v, v, o, v], dt1)
            dRt2 = dRt2 - c('kbij,ka->ijab', ERI[o, v, o, o], dt1)
            dRt2 = dRt2 - c('akij,kb->ijab', ERI[v, o, o, o], dt1)
            dRt2 = dRt2 + c('ac,ijcb->ijab', F[v, v], dt2)
            dRt2 = dRt2 + c('bc,ijac->ijab', F[v, v], dt2)
            dRt2 = dRt2 - c('ki,kjab->ijab', F[o, o], dt2)
            dRt2 = dRt2 - c('kj,ikab->ijab', F[o, o], dt2)
            dRt2 = dRt2 + c('klij,klab->ijab', ERI[o, o, o, o], dt2)
            dRt2 = dRt2 + c('abcd,ijcd->ijab', ERI[v, v, v, v], dt2)
            dRt2 = dRt2 - c('kbcj,ikca->ijab', ERI[o, v, v, o], dt2)
            dRt2 = dRt2 + c('kaci,kjcb->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dt2)
            dRt2 = dRt2 - c('kbic,kjac->ijab', ERI[o, v, o, v], dt2)
            dRt2 = dRt2 - c('kaci,kjbc->ijab', ERI[o, v, v, o], dt2)
            dRt2 = dRt2 + c('kbcj,ikac->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dt2)
            dRt2 = dRt2 - c('kajc,ikcb->ijab', ERI[o, v, o, v], dt2)

            dt1 = dt1 + dRt1 / Dia
            dt2 = dt2 + dRt2 / Dijab

            diis.add_error_vector(dt1, dt2)
            if iteration >= diis_start:
                dt1, dt2 = diis.extrapolate(dt1, dt2)

            dE_proj = (2.0 * c('ia,ia->', t1, dF[o, v])
                       + c('ijab,ijab->', t2, 2.0 * dERI[o, o, v, v] - dERI.swapaxes(2, 3)[o, o, v, v])
                       + 2.0 * c('ia,ia->', dt1, F[o, v])
                       + c('ijab,ijab->', dt2, 2.0 * ERI[o, o, v, v] - ERI.swapaxes(2, 3)[o, o, v, v]))

            delta_dE = abs(dE_proj - dE_proj_old)
            rms_dt1 = np.sqrt(np.sum((dt1 - dt1_old) ** 2))
            rms_dt2 = np.sqrt(np.sum((dt2 - dt2_old) ** 2))
            if iteration > 1 and (delta_dE < e_convergence and rms_dt1 < d_convergence
                                   and rms_dt2 < d_convergence):
                break

        if getattr(self, '_cpci_raw_cache', None) is None:
            self._cpci_raw_cache = {}
        self._cpci_raw_cache[pert] = (dt1, dt2)

        dc0 = self._cisd_dn0(dt1, dt2)
        if pert.kind in ('magnetic', 'vecpot'):
            dc0v = 0.0
            dc1 = n0 * dt1
            dc2 = n0 * dt2
        else:
            dc0v = dc0
            dc1 = dc0 * t1 + n0 * dt1
            dc2 = dc0 * t2 + n0 * dt2

        result = (dc1, dc2, dc0v)
        self._cpci_cache[pert] = result
        return result

    def _cpci_raw(self, pert):
        if getattr(self, '_cpci_raw_cache', None) is None or pert not in self._cpci_raw_cache:
            self._solve_cpci(pert)
        return self._cpci_raw_cache[pert]

    # length-gauge APT  (cisd_apt_lg.py: _CISDAPTEngine)

    def _zvector(self):
        """CISD unperturbed Z-vector: relaxed density Q_relaxed, Lagrangian
        X, h_mo, and the CPHF orbital-response z-amplitudes."""
        if getattr(self, '_cizvec', None) is None:
            c = self.contract
            o, v, no, nmo = self.o, self.v, self.no, self.nmo
            D_pq, D_pq_corr, D_pqrs = self._cisd_densities()

            ERI = np.asarray(self.H.ERI)
            F = np.asarray(self.H.F)
            h_mo = F - c('piqi->pq', 2.0 * ERI[:, o, :, o] - ERI.swapaxes(2, 3)[:, o, :, o])

            Q = D_pq_corr.copy()
            for i in range(no):
                Q[i, i] += 2.0

            G_sym = 0.25 * (D_pqrs + D_pqrs.transpose(1, 0, 3, 2)
                             + D_pqrs.transpose(2, 3, 0, 1) + D_pqrs.transpose(3, 2, 1, 0))
            G = G_sym.copy()
            for i in range(no):
                for j in range(no):
                    G[i, j, i, j] += 2.0
                    G[i, j, j, i] -= 1.0
            G_cross = np.zeros_like(G)
            for i in range(no):
                G_cross[i, :, i, :] += 2.0 * D_pq_corr
                G_cross[:, i, :, i] += 2.0 * D_pq_corr
                G_cross[i, :, :, i] -= D_pq_corr.T
                G_cross[:, i, i, :] -= D_pq_corr
            G = G + 0.5 * G_cross

            X = c('jm,im->ij', Q, h_mo) + 2.0 * c('jmkl,imkl->ij', G, ERI)

            rhs = -(X[v, o] - X[o, v].T)
            z_ov = self.cphf.solve(rhs.T)
            z_ai = z_ov.T

            Q_relaxed = Q.copy()
            Q_relaxed[v, o] += z_ai
            Q_relaxed[o, v] += z_ai.T

            self._cizvec = dict(Q=Q, G=G, Q_relaxed=Q_relaxed, X=X, h_mo=h_mo, z_ai=z_ai)
        return self._cizvec

    def _corr_QGX(self):
        """Correlation-only (Q_corr, G_corr, X_corr): the full Z-vector
        quantities with the pure-HF density blocks removed (Q minus 2*delta_oo;
        G minus the closed-shell HF 2-RDM block 2*d_ik*d_jl - d_il*d_jk; X
        rebuilt linearly from the corr densities). Cached. Used by the
        correlation-only property assemblies; the FULL quantities in
        _zvector() stay untouched (the perturbed-Lagrangian z response is a
        full-density equation and remains as validated)."""
        if getattr(self, '_ci_corr_qgx', None) is None:
            ct = self.contract
            no = self.no
            zv = self._zvector()
            ERI = np.asarray(self.H.ERI)
            Q_corr = zv['Q'].copy()
            for i in range(no):
                Q_corr[i, i] -= 2.0
            G_corr = zv['G'].copy()
            for i in range(no):
                for j in range(no):
                    G_corr[i, j, i, j] -= 2.0
                    G_corr[i, j, j, i] += 1.0
            X_corr = (ct('jm,im->ij', Q_corr, zv['h_mo'])
                      + 2.0 * ct('jmkl,imkl->ij', G_corr, ERI))
            self._ci_corr_qgx = (Q_corr, G_corr, X_corr)
        return self._ci_corr_qgx

    def _reference_hf(self):
        """The all-electron :class:`HFwfn` for the SCF reference (cached),
        supplying the reference (electronic) contribution to the total CISD
        properties via the pycc property facade - same pattern as
        MPwfn._reference_hf."""
        if getattr(self, '_ref_hf', None) is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.ref, orbital_basis=self.orbital_basis)
        return self._ref_hf

    def dipole_derivatives(self, route: str = 'explicit') -> np.ndarray:
        """CISD CORRELATION-ONLY LG-APT, shape (natom, 3, 3), [A, beta, alpha].
        Port of _CISDAPTEngine.assemble() with the HF density block removed
        from T1/T2 and the nuclear charge term dropped - both supplied by the
        pycc.apt facade (HFwfn reference + _nuclear_apt), matching MPwfn's
        contract."""
        from .cphf import Perturbation
        c = self.contract
        o, v = self.o, self.v
        zv = self._zvector()
        # Correlation-only relaxed density: Q_relaxed minus the HF 2*delta_oo.
        # T3/T3z are already pure correlation (amplitude/Z-vector response).
        Qt = zv['Q_relaxed'].copy()
        for i in range(self.no):
            Qt[i, i] -= 2.0
        natom = self.derivatives.natom
        h_E = [np.asarray(self.H.mu[f]) for f in range(3)]

        APT = np.zeros((natom, 3, 3))
        for A in range(natom):
            dip = self.derivatives.dipole(A)
            for beta in range(3):
                pert = Perturbation('nuclear', (A, beta))
                U_R = np.asarray(self.cphf._full_U(pert))
                dc1, dc2, dc0v = self._solve_cpci(pert)
                dc1, dc2 = dc1.real, dc2.real
                dc0v = dc0v.real if hasattr(dc0v, 'real') else dc0v
                dz, dD_corr = self._solve_dz_dR(pert, dc1, dc2, dc0v)

                for alpha in range(3):
                    h_RE = np.asarray(dip[alpha * 3 + beta])
                    T1 = c('ij,ij->', Qt, h_RE)
                    T2 = (c('rp,pq,rq->', U_R, Qt, h_E[alpha])
                          + c('pq,ps,sq->', Qt, h_E[alpha], U_R))
                    T3 = c('pq,pq->', dD_corr, h_E[alpha])
                    T3z = 2.0 * c('ai,ai->', dz, h_E[alpha][v, o])
                    APT[A, beta, alpha] = (T1 + T2 + T3 + T3z).real
        return APT

    def _perturbed_cisd_corr_opdm(self, dc1, dc2):
        c = self.contract
        o, v, nmo = self.o, self.v, self.nmo
        n0, n1, n2, tau_n = self._normalized_amplitudes()
        dtau_n = 2.0 * dc2 - dc2.swapaxes(2, 3)
        sigma = n2 - n2.swapaxes(2, 3)
        dsigma = dc2 - dc2.swapaxes(2, 3)

        dD = np.zeros((nmo, nmo), dtype=dc1.dtype)
        dD[o, o] -= 2.0 * c('ja,ia->ij', dc1, n1) + 2.0 * c('ja,ia->ij', n1, dc1)
        dD[o, o] -= 2.0 * c('jkab,ikab->ij', dtau_n, n2) + 2.0 * c('jkab,ikab->ij', tau_n, dc2)
        dD[v, v] += 2.0 * c('ia,ib->ab', dc1, n1) + 2.0 * c('ia,ib->ab', n1, dc1)
        dD[v, v] += 2.0 * c('ijac,ijbc->ab', dtau_n, n2) + 2.0 * c('ijac,ijbc->ab', tau_n, dc2)
        dD[o, v] += 2.0 * dc1
        dD[o, v] += 2.0 * c('jb,ijab->ia', dc1, sigma) + 2.0 * c('jb,ijab->ia', n1, dsigma)
        dD[v, o] = dD[o, v].T
        return dD

    def _perturbed_cisd_tpdm(self, dc1, dc2, dc0v):
        c = self.contract
        o, v, nmo = self.o, self.v, self.nmo
        n0, n1, n2, tau_n = self._normalized_amplitudes()
        dtau_n = 2.0 * dc2 - dc2.swapaxes(2, 3)

        dG = np.zeros((nmo, nmo, nmo, nmo), dtype=dc1.dtype)
        dG[o, o, o, o] = c('klab,ijab->ijkl', dc2, tau_n) + c('klab,ijab->ijkl', n2, dtau_n)
        dG[v, v, v, v] = c('ijab,ijcd->abcd', dc2, tau_n) + c('ijab,ijcd->abcd', n2, dtau_n)
        dG[o, v, v, o] = 4.0 * (c('ja,ib->iabj', dc1, n1) + c('ja,ib->iabj', n1, dc1))
        dG[o, v, o, v] = -2.0 * (c('ja,ib->iajb', dc1, n1) + c('ja,ib->iajb', n1, dc1))
        dG[v, o, o, v] = 2.0 * (c('jkac,ikbc->aijb', dtau_n, tau_n) + c('jkac,ikbc->aijb', tau_n, dtau_n))
        dG[v, o, v, o] = (
            -4.0 * (c('jkac,ikbc->aibj', dc2, n2) + c('jkac,ikbc->aibj', n2, dc2))
            + 2.0 * (c('jkac,ikcb->aibj', dc2, n2) + c('jkac,ikcb->aibj', n2, dc2))
            + 2.0 * (c('jkca,ikbc->aibj', dc2, n2) + c('jkca,ikbc->aibj', n2, dc2))
            - 4.0 * (c('jkca,ikcb->aibj', dc2, n2) + c('jkca,ikcb->aibj', n2, dc2)))
        dG[o, o, v, v] = dc0v * tau_n + n0 * dtau_n
        tau_swp = 2.0 * n2.swapaxes(0, 2).swapaxes(1, 3) - n2.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3)
        dtau_swp = 2.0 * dc2.swapaxes(0, 2).swapaxes(1, 3) - dc2.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3)
        dG[v, v, o, o] = dc0v * tau_swp + n0 * dtau_swp
        dG[v, o, v, v] = 2.0 * (c('ja,ijcb->aibc', dc1, tau_n) + c('ja,ijcb->aibc', n1, dtau_n))
        dG[o, v, o, o] = -2.0 * (c('kjab,ib->iajk', dtau_n, n1) + c('kjab,ib->iajk', tau_n, dc1))
        dG[v, v, v, o] = 2.0 * (c('jiab,jc->abci', dtau_n, n1) + c('jiab,jc->abci', tau_n, dc1))
        dG[o, o, o, v] = -2.0 * (c('kb,ijba->ijka', dc1, tau_n) + c('kb,ijba->ijka', n1, dtau_n))
        return dG

    def _solve_dz_dR(self, pert, dc1, dc2, dc0v):
        c = self.contract
        o, v, no, nv = self.o, self.v, self.no, self.nv
        zv = self._zvector()
        z = zv['z_ai']

        dF, dERI, _ = self._cpci_ints(pert)
        dD_corr = self._perturbed_cisd_corr_opdm(dc1, dc2)
        dD_pqrs = self._perturbed_cisd_tpdm(dc1, dc2, dc0v)

        o_ = self.o
        dh_mo = dF - c('piqi->pq', 2.0 * dERI[:, o_, :, o_] - dERI.swapaxes(2, 3)[:, o_, :, o_])
        dG_sym = 0.25 * (dD_pqrs + dD_pqrs.transpose(1, 0, 3, 2)
                          + dD_pqrs.transpose(2, 3, 0, 1) + dD_pqrs.transpose(3, 2, 1, 0))
        dG = dG_sym.copy()
        dG_cross = np.zeros_like(dG)
        for i in range(no):
            dG_cross[i, :, i, :] += 2.0 * dD_corr
            dG_cross[:, i, :, i] += 2.0 * dD_corr
            dG_cross[i, :, :, i] -= dD_corr.T
            dG_cross[:, i, i, :] -= dD_corr
        dG = dG + 0.5 * dG_cross

        ERI = np.asarray(self.H.ERI)
        dX = c('jm,im->ij', dD_corr, zv['h_mo']) + c('jm,im->ij', zv['Q'], dh_mo)
        dX = dX + 2.0 * c('jmkl,imkl->ij', dG, ERI) + 2.0 * c('jmkl,imkl->ij', zv['G'], dERI)

        db = -(dX[v, o] - dX[o, v].T)

        dL = 2.0 * dERI - dERI.swapaxes(2, 3)
        dG_mat_vovo = (dL[v, o, o, v].transpose(0, 2, 3, 1)
                       + dL[v, v, o, o].transpose(0, 2, 1, 3))
        dG_mat_vovo = dG_mat_vovo + c('ab,ij->aibj', dF[v, v], np.eye(no))
        dG_mat_vovo = dG_mat_vovo - c('ab,ij->aibj', np.eye(nv), dF[o, o])
        dGz = c('aibj,bj->ai', dG_mat_vovo, z)

        rhs = db - dGz
        dz_ov = self.cphf.solve(rhs.T)
        dz = dz_ov.T
        return dz, dD_corr

    # AAT 

    def _build_Dtilde(self, dc1, dc2, dc0):
        c = self.contract
        n0, n1, n2, tau_n = self._normalized_amplitudes()
        o, v, nmo = self.o, self.v, self.nmo

        R = np.zeros((nmo, nmo), dtype=complex)
        R[o, o] -= 2.0 * c('ja,ia->ij', n1, dc1)
        R[o, o] -= 2.0 * c('jkab,ikab->ij', tau_n, dc2)
        R[v, v] += 2.0 * c('ia,ib->ab', n1, dc1)
        R[v, v] += 2.0 * c('ijac,ijbc->ab', tau_n, dc2)
        R[o, v] += 2.0 * n0 * dc1 + 2.0 * dc0 * n1
        R[o, v] += 2.0 * c('jb,ijab->ia', n1, 2.0 * dc2 - dc2.swapaxes(2, 3))
        R[v, o] += 2.0 * c('ijab,jb->ai', tau_n, dc1)
        return R

    def _aat_dc_normalized(self, pert):
        dc1, dc2, dc0v = self._solve_cpci(pert)
        return dc0v, dc1, dc2

    def compute_Icc_AATs(self):
        """Term 1 of the AAT: direct state-vector overlap <dPsi_R|dPsi_H>."""
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        I_cc = np.zeros((3 * natom, 3))
        magH = {b: self._aat_dc_normalized(Perturbation('magnetic', b)) for b in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, dn1_R, dn2_R = self._aat_dc_normalized(pR)
            for beta in range(3):
                _, dn1_H, dn2_H = magH[beta]
                term = 2.0 * c('ia,ia->', dn1_R.conj(), dn1_H)
                term = term + c('ijab,ijab->', (2.0 * dn2_R - dn2_R.swapaxes(2, 3)).conj(), dn2_H)
                I_cc[la, beta] = term.real
        return I_cc

    def compute_Iphic_AATs(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        AAT_phic = np.zeros((3 * natom, 3), dtype=complex)
        magH = {b: self._aat_dc_normalized(Perturbation('magnetic', b)) for b in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, _, U_R = self._cpci_ints(pR)
            half_S = np.asarray(self.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                dn0_H, dn1_H, dn2_H = magH[beta]
                R_pq = self._build_Dtilde(dn1_H, dn2_H, dn0_H)
                AAT_phic[la, beta] = c('pq,qp->', R_pq, Ur_eff)
        return AAT_phic.real

    def compute_Iphiphi_AATs(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        _, D_pq, _ = self._cisd_densities()   # correlation-only 1-PDM
        I_pp = np.zeros((3 * natom, 3))
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, _, U_R = self._cpci_ints(pR)
            half_S = np.asarray(self.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                _, _, U_H = self._cpci_ints(Perturbation('magnetic', beta))
                I_pp[la, beta] = c('pq,pq->', D_pq, U_H.T @ Ur_eff).real
        return I_pp

    def compute_Icphi_AATs(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        AAT_cphi = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            dn0_R, dn1_R, dn2_R = self._aat_dc_normalized(pR)
            R_pq = self._build_Dtilde(dn1_R, dn2_R, dn0_R)
            for beta in range(3):
                _, _, U_H = self._cpci_ints(Perturbation('magnetic', beta))
                AAT_cphi[la, beta] = c('pq,pq->', R_pq, U_H)
        return AAT_cphi.real

    def atomic_axial_tensors(self, gauge: str = 'canonical') -> np.ndarray:
        """CISD CORRELATION-ONLY AAT, shape (natom, 3, 3): the four electronic
        overlap blocks with the correlation 1-PDM in Iphiphi (the 2*delta_oo
        HF block of Iphiphi is the SCF reference AAT; Icc/Icphi/Iphic are pure
        correlation, vanishing with the amplitudes). The SCF reference and the
        nuclear term (Z_A/4) eps_abc R_c are supplied by the pycc.aat facade
        (HFwfn.atomic_axial_tensors + _nuclear_aat), matching MPwfn's
        contract."""
        natom = self.ref.molecule().natom()
        total = (self.compute_Icc_AATs() + self.compute_Iphic_AATs()
                 + self.compute_Iphiphi_AATs() + self.compute_Icphi_AATs())
        return total.reshape(natom, 3, 3)

    # velocity-gauge APT  
    def compute_Icc_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        I_cc = np.zeros((3 * natom, 3), dtype=complex)
        vecA = {g: self._aat_dc_normalized(Perturbation('vecpot', g)) for g in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            _, dn1_R, dn2_R = self._aat_dc_normalized(Perturbation('nuclear', (A, beta_)))
            for gamma in range(3):
                _, dn1_A, dn2_A = vecA[gamma]
                I_cc[la, gamma] = (2.0 * c('ia,ia->', dn1_R.conj(), dn1_A)
                                   + c('ijab,ijab->', (2.0 * dn2_R - dn2_R.swapaxes(2, 3)).conj(), dn2_A))
        return I_cc

    def compute_Icphi_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        Icphi = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            dn0_R, dn1_R, dn2_R = self._aat_dc_normalized(Perturbation('nuclear', (A, beta_)))
            D_tilde_R = self._build_Dtilde(dn1_R, dn2_R, dn0_R)
            for gamma in range(3):
                _, _, U_A = self._cpci_ints(Perturbation('vecpot', gamma))
                Icphi[la, gamma] = c('pq,pq->', D_tilde_R, U_A)
        return Icphi

    def compute_Iphic_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        Iphic = np.zeros((3 * natom, 3), dtype=complex)
        vecA = {g: self._aat_dc_normalized(Perturbation('vecpot', g)) for g in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, _, U_R = self._cpci_ints(pR)
            half_S = np.asarray(self.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                dn0_A, dn1_A, dn2_A = vecA[gamma]
                D_tilde_A = self._build_Dtilde(dn1_A, dn2_A, dn0_A)
                Iphic[la, gamma] = c('pq,qp->', D_tilde_A.conj(), Ur_eff)
        return Iphic

    def compute_Iphiphi_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.derivatives.natom
        _, D_pq, _ = self._cisd_densities()   # correlation-only 1-PDM
        I_pp = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            _, _, U_R = self._cpci_ints(Perturbation('nuclear', (A, beta_)))
            half_S = np.asarray(self.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                _, _, U_A = self._cpci_ints(Perturbation('vecpot', gamma))
                I_pp[la, gamma] = c('pq,pq->', D_pq, U_A.T @ Ur_eff)
        return I_pp

    def velocity_dipole_derivatives(self, gauge: str = 'canonical') -> np.ndarray:
        """CISD CORRELATION-ONLY VG-APT, shape (natom, 3, 3): -2 times the four
        overlap blocks with the correlation 1-PDM in Iphiphi (the HF block of
        Iphiphi is the SCF reference; the other three blocks are pure
        correlation). The SCF reference and the Z_A delta nuclear term are
        supplied by the pycc.apt(gauge='velocity') facade, matching MPwfn's
        contract. """
        natom = self.ref.molecule().natom()
        overlap_total = (self.compute_Icc_VG_APT() + self.compute_Icphi_VG_APT()
                          + self.compute_Iphic_VG_APT() + self.compute_Iphiphi_VG_APT())
        return (-2.0 * overlap_total).real.reshape(natom, 3, 3)

    # Hessian  (cisd_analytic_hessian.py)

    def _hess_build_Y(self):
        """Y_ijkl double-UU tensor, correlation densities only (linear in Q/G,
        so the HF part belongs to the SCF reference Hessian)."""
        ct = self.contract
        zv = self._zvector()
        Q, G, _ = self._corr_QGX()
        h_mo = zv['h_mo']
        ERI = np.asarray(self.H.ERI)
        Y = ct('jl,ik->ijkl', Q, h_mo)
        Y = Y + 2.0 * ct('jlmn,ikmn->ijkl', G, ERI)
        Y = Y + 2.0 * ct('jmln,imkn->ijkl', G, ERI)
        Y = Y + 2.0 * ct('jmnl,imnk->ijkl', G, ERI)
        return Y

    def _hess_X_tilde(self):
        """Correlation-only X_tilde: X_corr (linear in the corr densities) with
        the Z-vector-corrected vo block. z itself is solved from the FULL
        Lagrangian in _zvector() (unchanged from the validated code); it is a
        pure correlation quantity since the HF part of X satisfies Brillouin
        (X_HF[v,o] - X_HF[o,v].T = 0) and contributes nothing to the rhs."""
        ct = self.contract
        o, v, no, nv = self.o, self.v, self.no, self.nv
        zv = self._zvector()
        _, _, X_corr = self._corr_QGX()
        F, ERI = np.asarray(self.H.F), np.asarray(self.H.ERI)
        A = (2.0 * ERI - ERI.swapaxes(2, 3)) + (2.0 * ERI - ERI.swapaxes(2, 3)).swapaxes(1, 3)
        A = A.swapaxes(1, 2)
        G_mat = (ct('ab,ij->aibj', np.eye(nv), np.eye(no))
                  * (F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no))
                  + A[v, o, v, o])
        X_tilde = X_corr.copy()
        X_tilde[v, o] += ct('aibj,bj->ai', G_mat, zv['z_ai'])
        return X_tilde

    def build_xi_ab(self, U_a, U_b, S_a, S_b, S_ab):
        c = self.contract
        xi = S_ab.copy()
        xi = xi + c('im,jm->ij', U_a, U_b) + c('im,jm->ij', U_b, U_a)
        xi = xi - c('im,jm->ij', S_a, S_b) - c('im,jm->ij', S_b, S_a)
        return xi

    def build_X_deriv(self, h_a_skel, g_a_skel):
        """Skeleton-derivative Lagrangian, correlation densities only."""
        ct = self.contract
        Q, G, _ = self._corr_QGX()
        Xa = ct('jm,im->ij', Q, h_a_skel)
        Xa = Xa + 2.0 * ct('jmkl,imkl->ij', G, g_a_skel)
        return Xa

    def build_cpci_term(self, dc1_a, dc2_a, dc1_b, dc2_b):
        """The pure linear CI-Jacobian action on dc1_b/dc2_b."""
        c = self.contract
        o, v = self.o, self.v
        F, ERI = np.asarray(self.H.F), np.asarray(self.H.ERI)
        E_tot = self.eci

        sig1 = -E_tot * dc1_b
        sig1 = sig1 - c('ji,ja->ia', F[o, o], dc1_b)
        sig1 = sig1 + c('ab,ib->ia', F[v, v], dc1_b)
        sig1 = sig1 + c('jabi,jb->ia', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dc1_b)
        sig1 = sig1 + c('jb,ijab->ia', F[o, v], 2.0 * dc2_b - dc2_b.swapaxes(2, 3))
        sig1 = sig1 + c('ajbc,ijbc->ia', 2.0 * ERI[v, o, v, v] - ERI.swapaxes(2, 3)[v, o, v, v], dc2_b)
        sig1 = sig1 - c('kjib,kjab->ia', 2.0 * ERI[o, o, o, v] - ERI.swapaxes(2, 3)[o, o, o, v], dc2_b)

        sig2 = -E_tot * dc2_b
        sig2 = sig2 + c('abcj,ic->ijab', ERI[v, v, v, o], dc1_b)
        sig2 = sig2 + c('abic,jc->ijab', ERI[v, v, o, v], dc1_b)
        sig2 = sig2 - c('kbij,ka->ijab', ERI[o, v, o, o], dc1_b)
        sig2 = sig2 - c('akij,kb->ijab', ERI[v, o, o, o], dc1_b)
        sig2 = sig2 + c('ac,ijcb->ijab', F[v, v], dc2_b)
        sig2 = sig2 + c('bc,ijac->ijab', F[v, v], dc2_b)
        sig2 = sig2 - c('ki,kjab->ijab', F[o, o], dc2_b)
        sig2 = sig2 - c('kj,ikab->ijab', F[o, o], dc2_b)
        sig2 = sig2 + c('klij,klab->ijab', ERI[o, o, o, o], dc2_b)
        sig2 = sig2 + c('abcd,ijcd->ijab', ERI[v, v, v, v], dc2_b)
        sig2 = sig2 - c('kbcj,ikca->ijab', ERI[o, v, v, o], dc2_b)
        sig2 = sig2 + c('kaci,kjcb->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dc2_b)
        sig2 = sig2 - c('kbic,kjac->ijab', ERI[o, v, o, v], dc2_b)
        sig2 = sig2 - c('kaci,kjbc->ijab', ERI[o, v, v, o], dc2_b)
        sig2 = sig2 + c('kbcj,ikac->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dc2_b)
        sig2 = sig2 - c('kajc,ikcb->ijab', ERI[o, v, o, v], dc2_b)

        return -2.0 * (2.0 * c('ia,ia->', dc1_a, sig1) + c('ijab,ijab->', 2.0 * dc2_a - dc2_a.swapaxes(2, 3), sig2)).real

    def hessian(self, route: str = 'explicit') -> np.ndarray:
        """CISD CORRELATION-ONLY nuclear Hessian, shape (3*natom, 3*natom):
        the T0-T5 assembly with the correlation densities in the density with linear
        terms (T0, xi-X_tilde, T3, T4, the HF blocks of those terms are the
        SCF reference Hessian) and the unchanged pure-correlation Z-vector (Bab.z)
        and CPCI (T5) terms; nuclear-repulsion block dropped. Reference and
        nuclear are supplied by the pycc.hessian facade, matching MPwfn's
        contract."""
        from .cphf import Perturbation
        ct = self.contract
        o, v, no = self.o, self.v, self.no
        mints, C_p4 = self._psi4_mints()
        mol = self.ref.molecule()
        natom = mol.natom()
        ndof = 3 * natom
        F = np.asarray(self.H.F)
        eps = np.diag(F)

        zv = self._zvector()
        Q, G, _ = self._corr_QGX()   # correlation densities (T0/T3/T4/xi-X)
        X_t = self._hess_X_tilde()
        Y = self._hess_build_Y()
        ERI = np.asarray(self.H.ERI)
        z = zv['z_ai']
        n0 = self._normalized_amplitudes()[0]

        ERI_M = ERI.transpose(0, 2, 1, 3)
        W_M = 2 * ERI_M - ERI_M.transpose(0, 2, 1, 3)
        A_M = 4 * ERI_M - ERI_M.transpose(0, 2, 1, 3) - ERI_M.transpose(0, 3, 2, 1)

        def to_mulliken(g_dirac):
            return g_dirac.transpose(0, 2, 1, 3)

        h_skel_1, g_skel_1, S_skel_1 = {}, {}, {}
        U_all, dT1_all, dT2_all = {}, {}, {}

        for N1 in range(natom):
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            E_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            for a in range(3):
                la = 3 * N1 + a
                h_skel_1[la] = T_core[a].np + V_core[a].np
                S_skel_1[la] = S_core[a].np
                g_skel_1[la] = E_core[a].np.swapaxes(1, 2)
                pert = Perturbation('nuclear', (N1, a))
                U_all[la] = np.asarray(self.cphf._full_U(pert))
                dT1_all[la], dT2_all[la] = self._cpci_raw(pert)

        X_deriv = {la: self.build_X_deriv(h_skel_1[la], g_skel_1[la])
                   for la in range(ndof)}

        Fa_M, Aa_M = {}, {}
        for la in range(ndof):
            g_a_M = to_mulliken(g_skel_1[la])
            Fa = h_skel_1[la].copy()
            Fa += (ct('ijkk->ij', 2 * g_a_M[:, :, o, o]) - ct('ikjk->ij', g_a_M[:, o, :, o]))
            Fa_M[la] = Fa
            Aa_M[la] = (4 * g_a_M - g_a_M.transpose(0, 2, 1, 3) - g_a_M.transpose(0, 3, 2, 1))

        Hessian = np.zeros((ndof, ndof))

        def _assemble_ab(a, b, h_ab_sk, S_ab_sk, g_ab_sk, N1, N2):
            la, lb = 3 * N1 + a, 3 * N2 + b
            U_a, U_b = U_all[la], U_all[lb]

            # T1: skeleton density contraction
            t0_val = ct('ij,ij->', Q, h_ab_sk)
            t0_val += ct('pqrs,pqrs->', G, g_ab_sk)

            # T2: Z-vector replacement
            xi = self.build_xi_ab(U_a, U_b, S_skel_1[la], S_skel_1[lb], S_ab_sk)
            t2_xiXt = -ct('ij,ji->', xi, X_t)
            g_ab_sk_M = to_mulliken(g_ab_sk)
            F_ab_M = (h_ab_sk + ct('ijkk->ij', 2 * g_ab_sk_M[:, :, o, o]) - ct('ikjk->ij', g_ab_sk_M[:, o, :, o]))
            Bab = F_ab_M.copy()
            Bab -= ct('ij,j->ij', xi, eps)
            Bab -= ct('kl,ijkl->ij', xi[o, o], W_M[:, :, o, o])
            Bab += ct('ki,kj->ij', U_a, Fa_M[lb])
            Bab += ct('ki,kj->ij', U_b, Fa_M[la])
            Bab += ct('kj,ik->ij', U_a, Fa_M[lb])
            Bab += ct('kj,ik->ij', U_b, Fa_M[la])
            Bab += ct('ki,kj,k->ij', U_a, U_b, eps)
            Bab += ct('ki,kj,k->ij', U_b, U_a, eps)
            UaUbT = U_a[:, o] @ U_b[:, o].T
            Bab += ct('kl,ijkl->ij', UaUbT, A_M)
            temp5_b = ct('lm,kjlm->kj', U_b[:, o], A_M[:, :, :, o])
            temp5_a = ct('lm,kjlm->kj', U_a[:, o], A_M[:, :, :, o])
            Bab += ct('ki,kj->ij', U_a, temp5_b)
            Bab += ct('ki,kj->ij', U_b, temp5_a)
            temp6_b = ct('lm,iklm->ik', U_b[:, o], A_M[:, :, :, o])
            temp6_a = ct('lm,iklm->ik', U_a[:, o], A_M[:, :, :, o])
            Bab += ct('kj,ik->ij', U_a, temp6_b)
            Bab += ct('kj,ik->ij', U_b, temp6_a)
            Bab += ct('kl,ijkl->ij', U_a[:, o], Aa_M[lb][:, :, :, o])
            Bab += ct('kl,ijkl->ij', U_b[:, o], Aa_M[la][:, :, :, o])
            t2_zvec = 2.0 * ct('ai,ai->', Bab[v, o], z)
            t2_val = t2_xiXt + t2_zvec

            # T3+T4: orbital-response Lagrangian + Y tensor
            t3_val = 2.0 * (ct('ij,ij->', U_b, X_deriv[la])+ ct('ij,ij->', U_a, X_deriv[lb]))
            t4_val = 2.0 * ct('ij,kl,ijkl->', U_a, U_b, Y)

            # T5: CPCI amplitude response
            t5_val = n0**2 * self.build_cpci_term(
                dT1_all[la], dT2_all[la], dT1_all[lb], dT2_all[lb]).real

            return la, lb, (t0_val + t2_val + t3_val + t4_val + t5_val).real

        for N1 in range(natom):
            for N2 in range(natom):
                T_ab = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_ab = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_ab = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)
                E_ab = mints.mo_tei_deriv2(N1, N2, C_p4, C_p4, C_p4, C_p4)

                h_ab = {(a, b): T_ab[3 * a + b].np + V_ab[3 * a + b].np for a in range(3) for b in range(3)}
                S_ab_np = {(a, b): S_ab[3 * a + b].np for a in range(3) for b in range(3)}
                g_ab = {(a, b): E_ab[3 * a + b].np.swapaxes(1, 2) for a in range(3) for b in range(3)}

                for a in range(3):
                    for b in range(3):
                        la, lb, val = _assemble_ab(
                            a, b, h_ab[(a, b)], S_ab_np[(a, b)], g_ab[(a, b)], N1, N2)
                        Hessian[la, lb] = val

                del T_ab, V_ab, S_ab, E_ab, h_ab, S_ab_np, g_ab

        Hessian = 0.5 * (Hessian + Hessian.T)
        return Hessian
