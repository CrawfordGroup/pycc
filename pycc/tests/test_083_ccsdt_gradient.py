"""
CCSD(T) analytic nuclear gradient -- pycc.gradient(ccwfn) / CCderiv.  Spatial closed-shell RHF
(all-electron and frozen-core) and the spin-orbital path (closed-shell keystones plus the open-shell
UHF case), both routed through the same (T) density and oo/vv dependent-pair orbital response.

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


def _scf_wfn(coords, basis, frozen_core=False):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()
    psi4.geometry(_geom(coords))
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'freeze_core': 'true' if frozen_core else 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _ecorr(coords, basis, frozen_core=False):
    """CCSD(T) correlation energy at a geometry (energy-only; no densities)."""
    cc = pycc.ccwfn(_scf_wfn(coords, basis, frozen_core), model='ccsd(t)')
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        return cc.solve_cc(1e-12, 1e-12, 200)


def _ccwfn_grad(coords, basis, frozen_core=False, orbital_basis='spatial'):
    """CCSD(T) wavefunction with the (T) densities built, ready for the gradient."""
    cc = pycc.ccwfn(_scf_wfn(coords, basis, frozen_core), model='ccsd(t)',
                    make_t3_density=True, orbital_basis=orbital_basis)
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


# Frozen-core reference (H2O/6-31G, O 1s frozen): the analytic FC CCSD(T) correlation gradient,
# validated once against a 5-point O(h^4) FD of the FC CCSD(T) energy to 1.9e-12 (same frame/geometry
# as the all-electron GRAD_REF).  This confirms the core<->active-occupied (T) response is carried by
# P_co (the (T)-inclusive Lagrangian) -- no extra core term is needed beyond the active oo/vv pairs.
GRAD_REF_FC = np.array([
    [0.0,  0.0,                -4.069702231895e-02],
    [0.0,  2.061319366917e-02,  2.034851115948e-02],
    [0.0, -2.061319366917e-02,  2.034851115948e-02],
])


def _findiff_gradient(basis="6-31G", frozen_core=False, h=0.005):
    """Regeneration recipe for GRAD_REF / GRAD_REF_FC (not run in the tests): central finite
    difference of the CCSD(T) correlation energy under nuclear displacement."""
    g = np.zeros((3, 3))
    for a in range(3):
        for x in range(3):
            cp = REF.copy(); cp[a, x] += h
            cm = REF.copy(); cm[a, x] -= h
            g[a, x] = (_ecorr(cp, basis, frozen_core) - _ecorr(cm, basis, frozen_core)) / (2 * h)
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
    r = pycc.gradient(pycc.CCderiv(cc_t))
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-12

    cc = pycc.ccwfn(_scf_wfn(REF, basis))                # plain CCSD
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-12, 1e-12, 200)
    r_ccsd = pycc.gradient(pycc.CCderiv(cc))
    # the (T) contribution (density + dependent-pair orbital response) is nonzero
    assert np.max(np.abs(np.asarray(r.correlation) - np.asarray(r_ccsd.correlation))) > 1e-4


def test_ccsdt_gradient_frozen_core():
    """Frozen-core CCSD(T) correlation gradient (H2O/6-31G, O 1s frozen) vs the FD-validated frozen
    reference.  Exercises the core<->active-occupied (T) orbital response (carried by P_co) on top of
    the active oo/vv dependent pairs, and confirms the (T) density reconstructs the FC correlation
    energy.  psi4 has no frozen-core CC(T) gradient, so the FD of the FC CCSD(T) energy is the oracle."""
    cc = _ccwfn_grad(REF, "6-31G", frozen_core=True)
    assert cc.nfzc > 0
    g_an = np.asarray(pycc.CCderiv(cc).gradient())
    assert np.max(np.abs(g_an - GRAD_REF_FC)) < 1e-11, g_an
    # the (T) 1-/2-PDM (active space) reconstructs the frozen-core CCSD(T) correlation energy
    lam = pycc.cclambda(cc, pycc.cchbar(cc))
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        lam.solve_lambda(1e-12, 1e-12)
    assert abs(pycc.ccdensity(cc, lam).compute_energy() - cc.ecc) < 1e-11


# ---------------------------------------------------------------------------------------------------
# Spin-orbital (UHF) CCSD(T) gradient -- CCderiv._so_gradient.  The (T) enters through the spin-orbital
# density and the oo/vv dependent-pair kappa_oo/kappa_vv (the spatial branch with H.L -> <pq||rs>).
# ---------------------------------------------------------------------------------------------------

# NH2 (2-B1, bent) open-shell UHF reference.  Run in C2v with the 2-B1 ground-state occupation pinned
# (NH2_OCC) so the SCF cannot fall into the 2-A1 solution ~0.074 Eh higher that a poor guess otherwise
# reaches -- pinned, the reference is guess- and platform-independent (E_SCF/6-31G = -55.5322382101),
# hence its gradient is freezeable.  6-31G, not STO-3G: STO-3G leaves the (T) vv dependent-pair
# identically zero (nv too small), the "minimal basis hides sins" trap, so it would not guard that term.
NH2 = "0 2\nN\nH 1 1.02\nH 1 1.02 2 103.0\n"
NH2_OCC = {'docc': [3, 0, 0, 1], 'socc': [0, 0, 1, 0]}   # 2-B1 (C2v irreps A1,A2,B1,B2)


def _nh2_so_ccsdt():
    """NH2 (2-B1, C2v pinned) / 6-31G spin-orbital CCSD(T) wavefunction, (T) densities built."""
    psi4.core.clean(); psi4.core.clean_options(); psi4.core.be_quiet()
    psi4.geometry(NH2)
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'reference': 'uhf', **NH2_OCC,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12, 'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True, orbital_basis='spinorbital')
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-12, 1e-12, 300)
    return cc


# Frozen open-shell reference: the analytic SO CCSD(T) correlation gradient (NH2 2-B1, C2v pinned occ /
# 6-31G, in psi4's C2v-oriented frame).  Validated once against a 5-point O(h^4) central FD of pycc's
# own SO CCSD(T) correlation energy (h=0.005, CC/SCF converged to 1e-13), agreeing to 3.7e-12 -- and
# every FD point's SCF stayed on the 2-B1 state (E in [-55.5324, -55.5321], never the 2-A1 ~-55.46), so
# no displaced C1 geometry slipped states.  The x-components vanish by the planar (yz) symmetry.
GRAD_REF_SO_NH2 = np.array([
    [0.0,  0.0,                 2.81754964293362e-02],
    [0.0,  1.53549271288188e-02, -1.40877482146681e-02],
    [0.0, -1.53549271288188e-02, -1.40877482146681e-02],
])


def test_so_ccsdt_gradient_keystone_equals_spatial():
    """SO == spatial: a closed-shell RHF driven through the spin-orbital path reproduces the spatial
    CCSD(T) correlation gradient (H2O/6-31G).  6-31G, not STO-3G, so the oo/vv dependent-pair is
    actually exercised.  The strongest structural check -- and it validates Dov (the ov 1-PDM), which
    the (T)-density energy reconstruction is blind to."""
    g_sp = np.asarray(pycc.CCderiv(_ccwfn_grad(REF, "6-31G")).gradient())
    g_so = np.asarray(pycc.CCderiv(_ccwfn_grad(REF, "6-31G", orbital_basis='spinorbital')).gradient())
    assert np.max(np.abs(g_so - g_sp)) < 1e-10, (g_so, g_sp)


def test_so_ccsdt_gradient_keystone_frozen_core():
    """SO == spatial CCSD(T) gradient with a frozen core (the core spans 2*nfzc spin-orbitals): the
    core<->active-occupied (T) response is carried by P_co, the active oo/vv pairs by the dependent
    pair, in both bases."""
    cc_so = _ccwfn_grad(REF, "6-31G", frozen_core=True, orbital_basis='spinorbital')
    assert cc_so.nfzc > 0
    g_so = np.asarray(pycc.CCderiv(cc_so).gradient())
    g_sp = np.asarray(pycc.CCderiv(_ccwfn_grad(REF, "6-31G", frozen_core=True)).gradient())
    assert np.max(np.abs(g_so - g_sp)) < 1e-10, (g_so, g_sp)


def test_so_ccsdt_gradient_openshell_nh2():
    """Open-shell UHF SO CCSD(T) correlation gradient (NH2 2-B1, C2v pinned occ / 6-31G) vs the frozen
    5-point-FD-validated reference.  The genuinely new open-shell (T) physics.  The 1e-8 guard (looser
    than the closed-shell keystones) reflects the near-singular open-shell spin-orbital orbital Hessian
    -- the Z-vector solve carries a redundant near-zero mode whose cross-platform BLAS noise leaks into
    the gradient at that level (cf. test_078); the FD oracle itself is tight to 3.7e-12."""
    g = np.asarray(pycc.CCderiv(_nh2_so_ccsdt()).gradient())
    assert np.max(np.abs(g - GRAD_REF_SO_NH2)) < 1e-8, g
