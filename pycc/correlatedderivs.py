"""Shared base for correlated analytic-derivative property drivers (MP2, CC; CI to follow).

`CorrelatedDerivs` owns the method-agnostic orbital-response and assembly machinery -- the pieces
that depend only on the reduced densities and the SCF reference, not on how the correlated
wavefunction was obtained.  Method-specific subclasses (`MPderiv`, `CCderiv`) supply the reduced
densities and their first-order responses.  See docs/DERIVATIVES_PLAN_2026-06.md section 9 for the
base/leaf split and the phased plan; more machinery moves here in later phases.
"""

from __future__ import annotations

import numpy as np


class CorrelatedDerivs:
    """Base class for correlated derivative-property drivers.

    Holds the correlated wavefunction and the method-agnostic derivative machinery.  A subclass
    (`MPderiv`, `CCderiv`) is constructed from a converged correlated wavefunction and supplies the
    method-specific reduced densities and their perturbed responses.
    """

    def __init__(self, wfn) -> None:
        self.wfn = wfn
        self.contract = wfn.contract
        self._ref_hf = None

    def _reference_hf(self):
        """All-electron :class:`~pycc.hfwfn.HFwfn` for the SCF reference (cached) -- supplies the
        reference contribution the :mod:`pycc.properties` facade pairs with the correlation part,
        and the CPHF orbital Hessian used by the Z-vector solve."""
        if self._ref_hf is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.wfn.ref, orbital_basis=self.wfn.orbital_basis)
        return self._ref_hf

    # ---- generalized-Fock orbital Lagrangian I'(D, Gam) ----
    # The Gauss/Stanton/Bartlett orbital-gradient Lagrangian: a method-agnostic function of a
    # full-MO 1-PDM ``D`` and cumulant 2-PDM ``Gam`` (both supplied by the leaf) plus the SCF
    # reference integrals/Fock.  Its occupied-virtual antisymmetric part ``X_ai = I'_ia - I'_ai``
    # drives the Z-vector; evaluated at the relaxed density it is the energy-weighted density ``W``.
    # Basis-dispatched: antisymmetrized ``<pq||rs>`` (spin-orbital) vs the spin-adapted ``L``
    # (closed-shell).  ``Gam`` must carry the proper 2-PDM permutational symmetry (the caller's
    # densities -- MP2 seeds / :meth:`ccdensity.gradient_densities` -- ensure this).

    def _lagrangian(self, D, Gam) -> np.ndarray:
        """Generalized-Fock orbital Lagrangian ``I'(D, Gam)`` (``nmo x nmo``)

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (w_rpsq + w_rqsp)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        with the two-electron kernel ``w`` = the antisymmetrized ``<pq||rs>`` (spin-orbital,
        :meth:`_so_lagrangian`) or the spin-adapted ``L`` (closed-shell, :meth:`_spatial_lagrangian`),
        dispatched on the orbital basis."""
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_lagrangian(D, Gam)
        return self._spatial_lagrangian(D, Gam)

    def _so_lagrangian(self, D, Gam) -> np.ndarray:
        """Spin-orbital generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) from a full-MO
        1-PDM ``D`` and cumulant 2-PDM ``Gam``

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (<rp||sq> + <rq||sp>)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        The 1-PDM term's column index runs over the full occupied space ``ofull`` (= core +
        active), so the frozen-core rows/columns are built (for ``nfzc=0`` this is the whole
        occupied space)."""
        nmo = self.wfn.nmo
        ofull = slice(0, self.wfn.o.stop)
        ERI = np.asarray(self.wfn.H.ERI)
        eps = np.diag(np.asarray(self.wfn.H.F))
        termA = eps[:, None] * (D + D.T)                       # f_pp (D_pq + D_qp)
        termB = np.zeros((nmo, nmo))
        termB[:, ofull] = (self.contract('rs,rpsq->pq', D, ERI[:, :, :, ofull])
                           + self.contract('rs,rqsp->pq', D, ERI[:, ofull, :, :]))
        termC = 4.0 * self.contract('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    def _spatial_lagrangian(self, D, Gam) -> np.ndarray:
        """Spin-adapted (closed-shell) generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) -- the
        closed-shell analogue of :meth:`_so_lagrangian`

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (L_rpsq + L_rqsp)
                           + 4 sum_rst <pr|st> Gamma_qrst ]

        with the spin-adapted ``L_pqrs = 2 <pq|rs> - <pq|sr>`` (= ``H.L``) carrying the closed-shell
        spin sum in the two-electron 1-PDM term, and the bare ``<pr|st>`` (= ``H.ERI``) with
        ``Gamma`` in the 2-PDM term.  ``ofull`` is the full occupied space (core + active)."""
        nmo = self.wfn.nmo
        ofull = slice(0, self.wfn.nfzc + self.wfn.no)
        ERI = np.asarray(self.wfn.H.ERI)
        L = np.asarray(self.wfn.H.L)
        eps = np.diag(np.asarray(self.wfn.H.F))
        termA = eps[:, None] * (D + D.T)
        termB = np.zeros((nmo, nmo))
        termB[:, ofull] = (self.contract('rs,rpsq->pq', D, L[:, :, :, ofull])
                           + self.contract('rs,rqsp->pq', D, L[:, ofull, :, :]))
        termC = 4.0 * self.contract('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    @staticmethod
    def _dependent_pairs(Iblock, eps_block, thresh=1e-8):
        """Canonical dependent-pair rotation ``P_mn = (I'_mn - I'_nm)/(eps_m - eps_n)`` for a square
        occ-occ or virt-virt Lagrangian block ``Iblock`` and its orbital energies ``eps_block``.
        Numerator-gated (``|Delta I'| < thresh`` -> 0), skipping the diagonal (``m=n``) and
        near-degenerate pairs.  ``P`` is symmetric (numerator and denominator both antisymmetric).

        This is the frozen-core core<->active-occupied divide generalized to an arbitrary square
        block; methods that break occ-occ/virt-virt rotation invariance (CCSD(T), CI) need it over
        the full oo and vv blocks."""
        num = np.asarray(Iblock) - np.asarray(Iblock).T
        den = eps_block[:, None] - eps_block[None, :]
        P = np.zeros_like(num)
        m = np.abs(num) > thresh
        P[m] = num[m] / den[m]
        return P

    @staticmethod
    def _perturbed_dependent_pairs(dIblock, Pblock0, eps_block, dfdiag_block, thresh=1e-8):
        """Field derivative ``dP`` of :meth:`_dependent_pairs` (quotient rule):
        ``dP_mn = (dI'_mn - dI'_nm)/(eps_m - eps_n) - P0_mn (df_mm - df_nn)/(eps_m - eps_n)`` -- the
        second term the canonical-``df``-diagonal denominator derivative, using the unperturbed
        ``Pblock0``.  Gated on ``|eps_m - eps_n| > thresh`` (diagonal + near-degenerate -> 0)."""
        dnum = np.asarray(dIblock) - np.asarray(dIblock).T
        gap = eps_block[:, None] - eps_block[None, :]
        dgap = dfdiag_block[:, None] - dfdiag_block[None, :]
        dP = np.zeros_like(dnum)
        m = np.abs(gap) > thresh
        dP[m] = (dnum[m] - np.asarray(Pblock0)[m] * dgap[m]) / gap[m]
        return dP
