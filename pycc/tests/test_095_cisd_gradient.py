"""
CISD analytic nuclear gradient - pycc.gradient(pycc.CIderiv(ci)), the relaxed-density gradient
inherited from CorrelatedDerivs (relaxed 1-PDM + cumulant 2-PDM + energy-weighted density), driven
by CIderiv's density hooks.  Mirrors test_061_mp2_relaxed_density: all-electron and frozen-core in
one module, a local wavefunction factory, and a finite difference of PyCC's OWN total energy as the
ground-truth oracle.

Why an energy FD rather than a tabulated reference: the gradient is a first-order property, so the
FD needs only the CISD energy - no derivative machinery at all.  That makes it a genuinely
independent oracle for the entire relaxed-density path (the Z-vector, and under a frozen core the
core<->active divide P_co and the full-occupied-space orbital response), rather than a
self-comparison.  

This module fills the one first-order gap in the CISD derivative suite: `CIderiv.gradient()` was
otherwise reached only from `_gradfd_col` in test_082, which is a regeneration recipe that the
tests do not run.  (`relaxed_dipole()` is already exercised, all-electron and frozen-core, by the
finite-field oracle in test_091.)

Spatial orbitals (the CIderiv path is spatial-only), H2O/6-31G (C1).
"""

import numpy as np
import psi4

import pycc


# Water in a fixed Cartesian frame (bohr), C1 - the same frame test_061 uses for the MP2
# frozen-core gradient FD.
WATER_XYZ = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.814), ("H", 1.755, 0.0, -0.46)]

H = 2.0e-3          # FD step


def _geom(disp=None):
    g = [list(a) for a in WATER_XYZ]
    if disp:
        i, c, delta = disp
        g[i][1 + c] += delta
    return ("units bohr\nnocom\nnoreorient\n"
            + "\n".join(f"{a[0]} {a[1]} {a[2]} {a[3]}" for a in g) + "\nsymmetry c1\n")


def _ciwfn(disp=None, freeze_core='false', basis='6-31G'):
    """Converged spatial CISD wavefunction at a given geometry, returned with the
    SCF total energy.  Mirrors test_061's factory; no ``orbital_basis`` knob because the CIderiv
    path is spatial-only."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(_geom(disp))
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    escf, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD', orbital_basis='spatial')
    ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=250)
    return ci, escf


def _cisd_total_energy(disp, freeze_core, basis='6-31G'):
    """PyCC total CISD energy (E_SCF + E_corr) at a displaced geometry."""
    ci, escf = _ciwfn(disp, freeze_core, basis)
    return escf + ci.eci


def _gradient_fd(freeze_core, basis='6-31G'):
    """Full (natom, 3) gradient via a 5-point O(h^4) central FD of the total CISD energy."""
    g = np.zeros((3, 3))
    for i in range(3):
        for c in range(3):
            e = [_cisd_total_energy((i, c, k * H), freeze_core, basis) for k in (-2, -1, 1, 2)]
            g[i, c] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * H)
    return g


def test_cisd_gradient_vs_energy_fd_631g():
    """All-electron CISD nuclear gradient (total = nuclear + SCF reference + correlation) vs a
    5-point finite difference of PyCC's own total CISD energy, every component, H2O/6-31G."""
    ci, _ = _ciwfn()
    gA = np.asarray(pycc.gradient(pycc.CIderiv(ci)).total)
    gF = _gradient_fd('false')
    assert np.max(np.abs(gA - gF)) < 1e-8, np.max(np.abs(gA - gF))


def test_cisd_gradient_vs_energy_fd_631g_frozen_core():
    """Frozen-core CISD nuclear gradient vs the same energy FD, H2O/6-31G.

    This is the ground-truth check on the frozen-core relaxed density: the core<->active-occupied
    divide P_co coupled into the Z-vector RHS, and the Z-vector itself solved over the FULL
    occupied space.  Nothing here touches the second-derivative machinery, so a failure localizes
    to the relaxed density rather than to the 2n+1 code."""
    ci, _ = _ciwfn(freeze_core='true')
    assert ci.nfzc > 0, ci.nfzc                     # frozen core is actually active
    gA = np.asarray(pycc.gradient(pycc.CIderiv(ci)).total)
    gF = _gradient_fd('true')
    assert np.max(np.abs(gA - gF)) < 1e-8, np.max(np.abs(gA - gF))


def test_cisd_gradient_translational_sum_rule():
    """Translational invariance: the total gradient of an isolated molecule sums to zero over
    atoms, and the correlation block does so on its own.  All-electron and frozen-core; FD-free."""
    for fc in ('false', 'true'):
        comp = pycc.gradient(pycc.CIderiv(_ciwfn(freeze_core=fc)[0]))
        assert np.max(np.abs(np.sum(np.asarray(comp.total), axis=0))) < 1e-9, fc
        assert np.max(np.abs(np.sum(np.asarray(comp.correlation), axis=0))) < 1e-9, fc
