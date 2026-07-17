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
