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

    The CI coefficients use intermediate normalization (the reference coefficient is
    fixed at 1), and the amplitude equations are the projected form -- directly
    analogous to the CCSD amplitude equations, but with only the *linear* terms (so no
    similarity-transformed intermediates like F_ae or W_mbej arise) and with the
    correlation energy appearing on the right-hand side::

        <Phi_i^a  | H_N | Psi> = E_c c_i^a
        <Phi_ij^ab| H_N | Psi> = E_c c_ij^ab
        E_c = <Phi_0 | H_N | Psi>

    Like ``ccwfn``, the energy denominators and the initial guess come from a composed
    :class:`MPwfn` (the MP1 doubles), built over this object's already-constructed base
    (no second integral transform). The Fock matrix is treated as a general (possibly
    non-diagonal) matrix throughout -- only the denominators use its diagonal -- so
    non-canonical / non-HF reference orbitals are supported, as in the CC code.

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

        # Reference, orbital spaces, seeded integrals, and the device manager all come
        # from the base; CIwfn pops only its own kwargs (model) and forwards the rest.
        super().__init__(scf_wfn, **kwargs)
        mgr = self.device_manager

        # The MP2 wavefunction supplies the energy denominators and the CISD initial
        # guess (MP1 doubles), reusing this object's base -- the same pattern ccwfn
        # uses. CI's singles denominator (Dia) is built from the MP2 orbital energies.
        self.mp = MPwfn.from_wavefunction(self)
        self.Dijab = self.mp.Dijab
        self.Dia = self.mp.eps_occ.reshape(-1, 1) - self.mp.eps_vir

        # Initial guess: c1 = 0, c2 = MP1 doubles (CI mutates c2 in place, so copy).
        self.c1 = mgr.seed_compute(np.zeros((self.no, self.nv)))
        self.c2 = clone(self.mp.t2)

        print("CIWFN object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_ci(self, e_conv: float = 1e-7, r_conv: float = 1e-7, maxiter: int = 100,
                 max_diis: int = 8, start_diis: int = 1) -> "Tensor":
        """Iterate the projected CISD equations to convergence and return E_c.

        The coefficients satisfy the eigenvalue equations ``sigma = E_c c`` (``sigma``
        is ``<excited|H_N|Psi>``); the residual is ``r = sigma - E_c c`` and the update
        uses the CI diagonal preconditioner -- the orbital-energy denominator shifted by
        the correlation energy, ``c += r / (D + E_c)`` -- so the energy enters both the
        residual and the denominator. DIIS accelerates convergence as in ``solve_cc``.
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
            print("CI Iter %3d: CI Ecorr = %.15f  dE = % .5E  rms = % .5E"
                  % (niter, eci, ediff, rms))

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
            return self._ci_energy_spinorbital(o, v, F, self.H.ERI, c1, c2)
        contract = self.contract
        e = 2.0 * contract('ia,ia->', F[o, v], c1)
        e = e + contract('ijab,ijab->', c2, L[o, o, v, v])
        return e

    def _ci_energy_spinorbital(self, o, v, F, ERI, c1, c2) -> "Tensor":
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
            return self._sigma1_spinorbital(o, v, F, self.H.ERI, c1, c2)

        contract = self.contract
        s1 = clone(F[o, v], device=self.device1)
        s1 = s1 + contract('ie,ae->ia', c1, F[v, v])
        s1 = s1 - contract('mi,ma->ia', F[o, o], c1)
        s1 = s1 + contract('imae,me->ia', (2.0 * c2 - c2.swapaxes(2, 3)), F[o, v])
        s1 = s1 + contract('nf,nafi->ia', c1, L[o, v, v, o])
        s1 = s1 + contract('amef,imef->ia', L[v, o, v, v], c2)
        s1 = s1 - contract('mnae,mnie->ia', c2, L[o, o, o, v])
        return s1

    def _sigma1_spinorbital(self, o, v, F, ERI, c1, c2) -> "Tensor":
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
            return self._sigma2_spinorbital(o, v, F, self.H.ERI, c1, c2)

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

    def _sigma2_spinorbital(self, o, v, F, ERI, c1, c2) -> "Tensor":
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
            # Bare-Fock singles->doubles coupling <Phi_ij^ab|F_N|Phi_k^c> =
            # P(ij)P(ab) f_jb c1_ia. Zero for a canonical reference (f_ov = 0); in CCSD
            # it is absorbed by the T1 similarity transformation, but linear CISD needs
            # it explicitly for a non-canonical (ROHF) reference.
            tmp = contract('ia,jb->ijab', c1, F[o, v])
            s2 = s2 + (tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3)
                       + tmp.swapaxes(0, 1).swapaxes(2, 3))
        return s2
