"""Orbital-unrelaxed (response) CCSD dipole polarizability via the density reformulation
-- CCderiv.response_polarizability(omega=0) -- cross-checked against the independent
symmetric linear-response code, ccresponse.polarizability(0).

The two routes share NO machinery, which makes the agreement a strong check:
  * response_polarizability contracts the orbital-unrelaxed perturbed 1-PDM (the field
    derivative of the CC density, solved with the BARE MO dipole and no CPHF folding)
    with the bare dipole,  alpha = -Tr(dD . mu)  (the asymmetric / density route);
  * ccresponse builds it from the symmetric response function <<mu;mu>> assembled from
    the right-hand perturbed amplitudes X only.

Static (omega = 0) and dynamic (omega != 0) CCSD; spatial (closed-shell RHF) and spin-
orbital.  The frequency enters the shared perturbed-amplitude/multiplier solvers as the
-/+ omega residual shift (-omega on the right dt, +omega on the left dl) with (D + omega)
denominators.  This is the RESPONSE (orbital-unrelaxed) polarizability, distinct from the
relaxed derivative CCderiv.polarizability (test_084) even at omega = 0 -- the two differ by
the HF orbital relaxation.  See docs/ccresponse_reformulation_plan.md."""

import psi4
import pycc
import numpy as np


# Equilibrium H2O (C2v, molecular plane yz); frame locked (matches test_084).
WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
symmetry c1
units bohr
no_com
no_reorient
"""

# Planar HOF (Cs, molecular plane xy), tilted off the axes so the in-plane block is full.
HOF = """
O  0.00000  0.00000  0.00000
F  2.56070  0.93200  0.00000
H -0.83450  1.62360  0.00000
symmetry c1
units bohr
no_com
no_reorient
"""


def _ccresponse_polar(wfn, route, omega=0.0):
    """CCSD polarizability at frequency omega from the independent symmetric linear-response
    code -- the oracle for the reformulation."""
    cc = pycc.ccwfn(wfn, orbital_basis=route)
    cc.solve_cc(1e-12, 1e-12, 100)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(1e-12, 1e-12, 100)
    dens = pycc.ccdensity(cc, lam, onlyone=(route == 'spinorbital'))
    return np.asarray(pycc.ccresponse(dens).polarizability(omega))


def test_response_polarizability_spatial_vs_ccresponse(rhf_wfn):
    """Spatial (closed-shell RHF) static unrelaxed CCSD polarizability (all 9 elements) matches
    the symmetric linear-response oracle; the tensor is naturally symmetric (not imposed); and it
    is genuinely DIFFERENT from the relaxed derivative polarizability (the orbital-relaxation piece
    the response omits)."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    alpha = np.asarray(pycc.CCderiv(cc).response_polarizability(0.0))
    ref = _ccresponse_polar(wfn, 'spatial')
    assert np.max(np.abs(alpha - ref)) < 1e-10, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-10            # naturally symmetric

    relaxed = np.asarray(pycc.CCderiv(cc).polarizability())   # the derivative (orbital-relaxed) tensor
    assert np.max(np.abs(alpha - relaxed)) > 1e-3             # response != derivative (distinct quantities)
    psi4.core.clean()


def test_response_polarizability_hof_offdiagonal(rhf_wfn):
    """Off-diagonal case: planar HOF/cc-pVDZ (Cs).  The unrelaxed response polarizability matches the
    symmetric-response oracle across the full tensor -- the large in-plane off-diagonal alpha_xy and
    the vanishing out-of-plane block (z perpendicular to the molecular plane)."""
    wfn = rhf_wfn(HOF, "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    alpha = np.asarray(pycc.CCderiv(cc).response_polarizability(0.0))
    ref = _ccresponse_polar(wfn, 'spatial')
    assert np.max(np.abs(alpha - ref)) < 1e-10, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-10
    assert abs(alpha[0, 1]) > 0.5                             # genuine in-plane off-diagonal
    assert abs(alpha[0, 2]) < 1e-9 and abs(alpha[1, 2]) < 1e-9  # out-of-plane block vanishes (Cs)
    psi4.core.clean()


def test_dynamic_response_polarizability_spatial_vs_ccresponse(rhf_wfn):
    """Dynamic (omega != 0) spatial unrelaxed polarizability vs the symmetric-response oracle at a
    positive and a negative frequency (the -/+ omega residual shift and (D + omega) denominators in
    the perturbed-amplitude/multiplier solvers).  The static tensor is recovered as omega -> 0 and
    the dynamic value disperses away from it (normal dispersion, |alpha(omega)| > |alpha(0)|)."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    d = pycc.CCderiv(cc)
    for omega in (0.07, -0.07):
        alpha = np.asarray(d.response_polarizability(omega))
        ref = _ccresponse_polar(wfn, 'spatial', omega)
        assert np.max(np.abs(alpha - ref)) < 1e-10, (omega, alpha)
        assert np.max(np.abs(alpha - alpha.T)) < 1e-10
    static = np.asarray(d.response_polarizability(0.0))
    dyn = np.asarray(d.response_polarizability(0.07))
    assert np.all(np.diag(dyn) > np.diag(static))             # normal dispersion (real, sub-resonance)
    psi4.core.clean()


def test_fc_response_polarizability_vs_ccresponse(rhf_wfn):
    """Frozen-core (O 1s frozen) spatial unrelaxed response polarizability matches the symmetric-
    response oracle -- the perturbed amplitudes/multipliers run on the active o/v slices while the
    bare dipole is the full-MO operator."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="true")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    assert cc.nfzc > 0
    alpha = np.asarray(pycc.CCderiv(cc).response_polarizability(0.0))
    ref = _ccresponse_polar(wfn, 'spatial')
    assert np.max(np.abs(alpha - ref)) < 1e-10, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-10
    psi4.core.clean()


def test_so_response_polarizability_vs_ccresponse(rhf_wfn):
    """SO == spatial keystone: the spin-orbital unrelaxed response polarizability of an RHF reference
    forced into the spin-orbital basis matches the spin-orbital symmetric-response oracle, carrying
    the check onto the SO machinery (SO Jacobian, SO HBAR, SO r_L) -- static and dynamic."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital')
    cc.solve_cc(1e-12, 1e-12, 100)
    d = pycc.CCderiv(cc)
    for omega in (0.0, 0.07):
        alpha = np.asarray(d.response_polarizability(omega))
        ref = _ccresponse_polar(wfn, 'spinorbital', omega)
        assert np.max(np.abs(alpha - ref)) < 1e-10, (omega, alpha)
        assert np.max(np.abs(alpha - alpha.T)) < 1e-10
    psi4.core.clean()
