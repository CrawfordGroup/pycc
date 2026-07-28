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
            s2 = s2 + (contract('ie,abej->ijab', c1, ERI[v, v, v, o]) - contract('je,abei->ijab', c1, ERI[v, v, v, o]))
            s2 = s2 - (contract('ma,mbij->ijab', c1, ERI[o, v, o, o]) - contract('mb,maij->ijab', c1, ERI[o, v, o, o]))
            tmp = contract('ia,jb->ijab', c1, F[o, v])
            s2 = s2 + (tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3) + tmp.swapaxes(0, 1).swapaxes(2, 3))
        return s2

    def _normalized_amplitudes(self):
        if getattr(self, '_ci_namp', None) is None:
            c = self.contract
            t1, t2 = self.c1, self.c2    
            norm2 = (2.0 * c('ia,ia->', t1.conj(), t1) + c('ijab,ijab->', t2.conj(), 2.0 * t2 - t2.swapaxes(2, 3)))
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
