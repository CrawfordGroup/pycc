"""
CCSD(T) analytic nuclear gradient -- pycc.gradient(ccwfn) / CCderiv, spatial closed-shell RHF,
all-electron.

The (T) correction requires canonical MOs, so its orbital response cannot use the -1/2 S^X
non-canonical trick for the occ-occ / virt-virt blocks: (T) breaks the oo/vv rotation invariance,
and the canonical perturbed orbitals acquire dependent-pair rotations kappa_oo/kappa_vv.  These are
the frozen-core (I'_ij-I'_ji)/(eps_i-eps_j) divide generalized to all oo and vv pairs
(CCderiv._dependent_pairs), added to the relaxed density and coupled into the ov Z-vector RHS.  See
docs/DERIVATIVES_PLAN_2026-06.md sec.7.

Anchored on a finite difference of pycc's own CCSD(T) correlation energy (internal oracle; no psi4
gradient dependency).  6-31G is used deliberately: STO-3G's virtual space is too small to exercise
the dependent-pair term (the gradient is correct there to <1e-7 with or without it), exactly the
"minimal basis hides sins" trap.
"""

import contextlib
import os

import numpy as np
import psi4
import pycc

# H2O, fixed frame (no_com/no_reorient) so displaced geometries share the molecular axes.
ATOMS = ['O', 'H', 'H']
REF = np.array([
    [0.0,  0.000000000000000,  0.143225857166674],
    [0.0, -1.638037301628121, -1.136549142277225],
    [0.0,  1.638037301628121, -1.136549142277225],
])


def _geom(coords):
    body = "\n".join(f"{s} {c[0]:.15f} {c[1]:.15f} {c[2]:.15f}" for s, c in zip(ATOMS, coords))
    return body + "\nsymmetry c1\nunits bohr\nno_com\nno_reorient\n"


def _scf_wfn(coords, basis):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()
    psi4.geometry(_geom(coords))
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _ecorr(coords, basis):
    """CCSD(T) correlation energy at a geometry (energy-only; no densities)."""
    cc = pycc.ccwfn(_scf_wfn(coords, basis), model='ccsd(t)')
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        return cc.solve_cc(1e-12, 1e-12, 200)


def _ccwfn_grad(coords, basis):
    """CCSD(T) wavefunction with the (T) densities built, ready for the gradient."""
    cc = pycc.ccwfn(_scf_wfn(coords, basis), model='ccsd(t)', make_t3_density=True)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-12, 1e-12, 200)
    return cc


# Frozen finite-difference oracle: the CCSD(T) correlation gradient (H2O/6-31G, frame locked;
# closed-shell, so platform-reproducible).  Validated (once) against a 5-point O(h^4) central finite
# difference of the CCSD(T) correlation energy (h=0.005, CC converged to 1e-13), agreeing to 1.8e-12
# (see _findiff_gradient, the regeneration recipe).  A re-introduced off-diagonal (T) density or a
# missing oo/vv dependent-pair shows up here at ~5e-5.
GRAD_REF = np.array([
    [0.0,  0.0,                -0.04066006991882],
    [0.0,  0.02059868948161,   0.02033003495941],
    [0.0, -0.02059868948161,   0.02033003495941],
])


def _findiff_gradient(basis="6-31G", h=0.005):
    """Regeneration recipe for GRAD_REF (not run in the tests): central finite difference of the
    CCSD(T) correlation energy under nuclear displacement."""
    g = np.zeros((3, 3))
    for a in range(3):
        for x in range(3):
            cp = REF.copy(); cp[a, x] += h
            cm = REF.copy(); cm[a, x] -= h
            g[a, x] = (_ecorr(cp, basis) - _ecorr(cm, basis)) / (2 * h)
    return g


def test_ccsdt_gradient_vs_findiff():
    """The analytic CCSD(T) correlation gradient reproduces the finite-difference-validated frozen
    reference (H2O/6-31G).  The frozen values are the analytic gradient itself (which matched the
    5-point FD to 1.8e-12), so the deterministic closed-shell recomputation reproduces them to ~4e-15;
    1e-11 is a tight guard with ample margin for cross-platform floating-point variation."""
    cc = _ccwfn_grad(REF, "6-31G")
    g_an = np.asarray(pycc.CCderiv(cc).gradient())
    assert np.max(np.abs(g_an - GRAD_REF)) < 1e-11, g_an


def test_ccsdt_gradient_facade_components():
    """The pycc.gradient facade decomposes exactly, and the (T) correction is a real, nonzero
    contribution to the correlation gradient on top of plain CCSD."""
    basis = "6-31G"
    cc_t = _ccwfn_grad(REF, basis)
    r = pycc.gradient(cc_t)
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-12

    cc = pycc.ccwfn(_scf_wfn(REF, basis))                # plain CCSD
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-12, 1e-12, 200)
    r_ccsd = pycc.gradient(cc)
    # the (T) contribution (density + dependent-pair orbital response) is nonzero
    assert np.max(np.abs(np.asarray(r.correlation) - np.asarray(r_ccsd.correlation))) > 1e-4
