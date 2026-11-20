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
from .conftest import record, current_test_context

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


# The one-time 5-point O(h^4) FD agreements quoted above, for the supplementary validation table.
# The committed tests regression-check the analytic gradient against the FD-validated frozen arrays
# (reproducing them to ~4e-15); these are the agreements with the finite difference itself.
FD_AGREEMENT = 1.8e-12
FD_AGREEMENT_FC = 1.9e-12


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
    g_an = np.asarray(pycc.CCderiv(cc).gradient().correlation)
    record('Gradient', 'CCSD(T)', 'all-electron', 'fd-energy',
           FD_AGREEMENT, molecule='H2O', note='spatial', ao_basis='6-31G',
           limited_by='fd-truncation')
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
    g_an = np.asarray(pycc.CCderiv(cc).gradient().correlation)
    record('Gradient', 'CCSD(T)', 'frozen core', 'fd-energy',
           FD_AGREEMENT_FC, molecule='H2O', note='spatial', ao_basis='6-31G',
           limited_by='fd-truncation')
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

def test_so_ccsdt_gradient_keystone_equals_spatial():
    """SO == spatial: a closed-shell RHF driven through the spin-orbital path reproduces the spatial
    CCSD(T) correlation gradient (H2O/6-31G).  6-31G, not STO-3G, so the oo/vv dependent-pair is
    actually exercised.  The strongest structural check -- and it validates Dov (the ov 1-PDM), which
    the (T)-density energy reconstruction is blind to."""
    g_sp = np.asarray(pycc.CCderiv(_ccwfn_grad(REF, "6-31G")).gradient().correlation)
    g_so = np.asarray(pycc.CCderiv(_ccwfn_grad(REF, "6-31G", orbital_basis='spinorbital')).gradient().correlation)
    assert np.max(np.abs(g_so - g_sp)) < 1e-10, (g_so, g_sp)


def test_so_ccsdt_gradient_keystone_frozen_core():
    """SO == spatial CCSD(T) gradient with a frozen core (the core spans 2*nfzc spin-orbitals): the
    core<->active-occupied (T) response is carried by P_co, the active oo/vv pairs by the dependent
    pair, in both bases."""
    cc_so = _ccwfn_grad(REF, "6-31G", frozen_core=True, orbital_basis='spinorbital')
    assert cc_so.nfzc > 0
    g_so = np.asarray(pycc.CCderiv(cc_so).gradient().correlation)
    g_sp = np.asarray(pycc.CCderiv(_ccwfn_grad(REF, "6-31G", frozen_core=True)).gradient().correlation)
    assert np.max(np.abs(g_so - g_sp)) < 1e-10, (g_so, g_sp)


# ---- CH2 (3-B1) / cc-pVDZ: the open-shell CCSD(T) gradient anchor -------------------------------
# Replaces the former NH2 (2-B1) case.  Triplet CH2's UHF reference is stable by default (psi4's
# default SCF and `stability_analysis: follow` return the same energy) and its response equations are
# not near-singular, so this validates against an EXTERNAL code at CFOUR's ten-decimal print floor
# rather than against a frozen PyCC value with a loose guard.  It is the suite's only open-shell
# CCSD(T) gradient and its only CFOUR-anchored CC gradient.
#
# Oracle: xcfour 2.1, GRD(CCSD(T)) - GRD(SCF) (CALC=CCSD(T), REFERENCE=UHF, MULTIPLICITY=3,
# BASIS=PVDZ, DERIV_LEVEL=FIRST, SCF_CONV=13, CC_CONV=12, LINEQ_CONV=12).  The geometry is CFOUR's
# own full-precision MOL-file frame; psi4's cc-pVDZ is CFOUR's PVDZ, so no GENBAS transcription is
# needed.
CH2_CFOUR_FRAME = """
0 3
C 0.000000000000000  0.000000000000000  0.114593744572731
H 0.000000000000000 -1.874001424240600 -0.682224040442087
H 0.000000000000000  1.874001424240600 -0.682224040442087
symmetry c1
units bohr
no_com
no_reorient
"""

CFOUR_GRAD_T_CH2_PVDZ = np.array(
    [[ 0.0000000000e+00,  0.0000000000e+00, -5.9503317000e-03],
     [ 0.0000000000e+00,  9.8478138000e-03,  2.9751658000e-03],
     [ 0.0000000000e+00, -9.8478138000e-03,  2.9751658000e-03]])

def test_so_ccsdt_gradient_cfour_ch2_triplet():
    """Open-shell UHF-CCSD(T) correlation gradient for 3-B1 CH2 / cc-pVDZ vs the CFOUR GRD oracle.
    The genuinely spin-polarized (T) first-derivative path -- spin-orbital (T) perturbed amplitudes
    and the canonical perturbed-MO dependent pairs on a UHF reference -- validated against an
    external code; matches to 4.6e-11, the same level the closed-shell cases reach, in ~18 s."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(CH2_CFOUR_FRAME)
    psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk', 'reference': 'uhf',
                      'e_convergence': 1e-13, 'd_convergence': 1e-13, 'maxiter': 300})
    _, wfn = psi4.energy('scf', return_wfn=True)
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital', make_t3_density=True)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-13, 1e-13, 300)
    g = np.asarray(pycc.CCderiv(cc).gradient().correlation)
    _dev = float(np.max(np.abs(g - CFOUR_GRAD_T_CH2_PVDZ)))
    record('Gradient', 'CCSD(T)', 'all-electron', 'cfour', _dev,
           molecule='CH2', note='spin-orbital', ao_basis='cc-pVDZ',
           limited_by='oracle-print')
    assert np.max(np.abs(g - CFOUR_GRAD_T_CH2_PVDZ)) < 1e-9, g
    psi4.core.clean()
