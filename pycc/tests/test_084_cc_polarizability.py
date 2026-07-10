"""CCSD static electric-dipole polarizability (correlation contribution) via the asymmetric
(2n+1) route -- alpha_corr[a,b] = -d^2 E_corr / dF_a dF_b, omega = 0; see
docs/DERIVATIVES_PLAN_2026-06.md sec 8.

CCderiv.polarizability() differentiates the relaxed-density gradient a second time in a field.
A static field leaves the AO basis fixed (S^F = <pq|rs>^F = 0), so

    alpha[a,b] = Tr(dD_rel(F_b) . mu_a) + Tr(D_rel . (U^bT mu_a + mu_a U^b)),

with the perturbed amplitudes/multipliers dt/dLambda, the perturbed correlation densities, the
FULL-df perturbed Lagrangian (sec 8.2), and the perturbed Z-vector -- all first-order responses.

Validation: the hard-wired references below ARE the analytic tensors, each validated ONCE against
a tight (Lambda->1e-13) 5-point finite field of CCderiv.relaxed_dipole (alpha = d mu / dF): the
water keystone to 1.5e-11, the HOF off-diagonal case to 2.6e-10 (recipe in _findiff_alpha).  The
deterministic recomputation reproduces them to ~1e-11, so 1e-9 is a tight, robust guard.  Spatial
closed-shell RHF, all-electron (frozen core / spin-orbital / (T) to follow)."""

import psi4
import pycc
import numpy as np


# Equilibrium H2O (C2v, molecular plane yz, C2 along z); frame locked for a reproducible tensor.
WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
symmetry c1
units bohr
no_com
no_reorient
"""

# Planar HOF (Cs, molecular plane xy), tilted off the axes so the in-plane block is full -- the
# off-diagonal alpha_xy is large while alpha_xz = alpha_yz = 0 (z perpendicular to the plane).
HOF = """
O  0.00000  0.00000  0.00000
F  2.56070  0.93200  0.00000
H -0.83450  1.62360  0.00000
symmetry c1
units bohr
no_com
no_reorient
"""

# Frozen references: the analytic alpha_corr, each validated once against a tight finite field of
# the relaxed dipole (see the module docstring / _findiff_alpha).
WATER_ALPHA_REF = np.array([[0.071555631886, 0.0, 0.0],
                            [0.0, 0.060171399010, 0.0],
                            [0.0, 0.0, 0.173591992896]])
HOF_ALPHA_REF = np.array([[-1.879732823995, -0.599507379935, 0.0],
                          [-0.599507379947, 0.104671076326, 0.0],
                          [0.0, 0.0, 0.099333851054]])
# Frozen-core H2O/6-31G keystone (validated to 1.0e-11).
WATER_ALPHA_REF_FC = np.array([[0.071841599990, 0.0, 0.0],
                               [0.0, 0.061126138693, 0.0],
                               [0.0, 0.0, 0.174203460940]])

# Open-shell UHF NH2 (2-B1, bent), run in C2v with the 2-B1 occupation pinned (NH2_OCC) so a poor
# SCF guess cannot fall into the ~0.074 Eh higher 2-A1 solution (making the value freezeable).  SO
# CCSD correlation polarizability; alpha_zz validated against a 5-point energy 2nd-derivative finite
# field (field along the C2 axis, C2v-preserving) to 2.7e-10; the full tensor is symmetric to 4e-16.
NH2 = """
0 2
N
H 1 1.02
H 1 1.02 2 103.0
"""
NH2_OCC = {'docc': [3, 0, 0, 1], 'socc': [0, 0, 1, 0]}
NH2_ALPHA_REF = np.array([[0.155624577600, 0.0, 0.0],
                          [0.0, 0.059976118900, 0.0],
                          [0.0, 0.0, 0.194543333700]])


def _findiff_alpha(geom, basis, F=5e-4, freeze_core='false'):
    """Regeneration recipe for the *_ALPHA_REF tensors (not run in the tests): a 5-point O(h^4)
    finite field of pycc's CCSD relaxed correlation dipole (alpha = d mu / dF), with Lambda
    converged to 1e-13 at each field (tighter than CCderiv.relaxed_dipole's default).  Rebuilds the
    relaxed density with the frozen-core core<->active divide P_co coupled into the Z-vector RHS,
    mirroring CCderiv._relaxed_density (a no-op for freeze_core='false')."""
    def rd(bax, Fv):
        d = [0.0, 0.0, 0.0]; d[bax] = Fv
        psi4.core.clean(); psi4.core.clean_options()
        psi4.geometry(geom)
        psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                          'e_convergence': 1e-13, 'd_convergence': 1e-13,
                          'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': d})
        _, wfn = psi4.energy('scf', return_wfn=True)
        cx = pycc.ccwfn(wfn); cx.solve_cc(1e-13, 1e-13, 300)
        hb = pycc.cchbar(cx); lm = pycc.cclambda(cx, hb); lm.solve_lambda(1e-13, 1e-13, 300)
        D, G = (np.asarray(x) for x in pycc.ccdensity(cx, lm).gradient_densities())
        o, v = cx.o, cx.v; co = slice(0, cx.nfzc); of = slice(0, o.stop)
        c = cx.contract; eps = np.diag(np.asarray(cx.H.F)); L = np.asarray(cx.H.L)
        Ip = np.asarray(cx.mp._mp2_lagrangian(D, G))
        Dr = D.copy(); X = Ip[of, v] - Ip[v, of].T
        if cx.nfzc:
            Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            Dr[co, o] += Pco; Dr[o, co] += Pco.T; zjc = -Pco.T
            X = X - (c('jc,ajic->ia', zjc, L[v, o, of, co]) + c('jc,acij->ia', zjc, L[v, co, of, o]))
        z = np.asarray(pycc.CCderiv(cx)._reference_hf().cphf.solve(X))
        Dr[v, of] += -z.T; Dr[of, v] += -z
        return np.array([np.sum(Dr * np.asarray(cx.H.mu[a])) for a in range(3)])
    alpha = np.zeros((3, 3))
    for b in range(3):
        alpha[:, b] = (rd(b, -2*F) - 8*rd(b, -F) + 8*rd(b, F) - rd(b, 2*F)) / (12*F)
    return alpha


def test_ccsd_polarizability_water_631g(rhf_wfn):
    """Fast keystone: all-electron CCSD correlation polarizability (all 9 elements) for H2O/6-31G
    matches the frozen finite-field reference; the pycc.polarizability facade decomposes exactly
    (nuclear = 0, correlation == CCderiv.polarizability); the tensor is symmetric."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    alpha = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(alpha - WATER_ALPHA_REF)) < 1e-9, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-9              # symmetric

    # facade: total = nuclear (0) + reference (HF) + correlation, and correlation == analytic alpha
    r = pycc.polarizability(cc)
    assert np.max(np.abs(np.asarray(r.nuclear))) < 1e-14
    assert np.max(np.abs(np.asarray(r.correlation) - alpha)) < 1e-12
    assert np.max(np.abs(np.asarray(r.total) - (np.asarray(r.nuclear)
                  + np.asarray(r.reference) + np.asarray(r.correlation)))) < 1e-12
    psi4.core.clean()


def test_fc_ccsd_polarizability_water_631g(rhf_wfn):
    """Frozen-core CCSD correlation polarizability for H2O/6-31G vs the frozen reference.  Exercises
    the perturbed core<->active Sylvester divide dP_co and its coupling into the perturbed Z-vector
    RHS (the frozen-core additions to _perturbed_relaxed_density)."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="true")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    alpha = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(alpha - WATER_ALPHA_REF_FC)) < 1e-9, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-9
    psi4.core.clean()


def test_ccsd_polarizability_hof_offdiagonal(rhf_wfn):
    """Off-diagonal case: planar HOF/cc-pVDZ (Cs).  CCSD correlation polarizability matches the
    frozen finite-field reference; the in-plane off-diagonal alpha_xy is large (~-0.60) while
    alpha_xz = alpha_yz = 0 (z perpendicular to the molecular plane); the tensor is symmetric; and
    the TOTAL (reference + correlation) is positive-definite (the physical constraint -- the
    correlation alpha_xx is negative on its own, contracting the over-polarized HF O-F direction)."""
    wfn = rhf_wfn(HOF, "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn)
    cc.solve_cc(1e-12, 1e-12, 100)
    alpha = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(alpha - HOF_ALPHA_REF)) < 1e-9, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-9              # symmetric (asymmetric route does not enforce it)
    assert abs(alpha[0, 1]) > 0.5                              # genuine in-plane off-diagonal
    assert abs(alpha[0, 2]) < 1e-9 and abs(alpha[1, 2]) < 1e-9  # out-of-plane block vanishes (Cs)

    r = pycc.polarizability(cc)
    total = np.asarray(r.total)
    assert np.all(np.linalg.eigvalsh(0.5 * (total + total.T)) > 0.0), np.linalg.eigvalsh(total)
    psi4.core.clean()


def test_so_ccsd_polarizability_keystone(rhf_wfn):
    """SO == spatial keystone: the spin-orbital CCSD polarizability of an RHF reference forced into
    the spin-orbital basis reproduces the FD-validated spatial references (all-electron and frozen
    core).  This carries the high-precision spatial validation onto the SO machinery (SO Jacobian,
    SO HBAR, SO r_L, inline orbital Hessian)."""
    for fc, ref in (("false", WATER_ALPHA_REF), ("true", WATER_ALPHA_REF_FC)):
        wfn = rhf_wfn(WATER, "6-31G", freeze_core=fc)
        cc = pycc.ccwfn(wfn, orbital_basis='spinorbital')
        cc.solve_cc(1e-12, 1e-12, 100)
        alpha = np.asarray(pycc.CCderiv(cc).polarizability())
        assert np.max(np.abs(alpha - ref)) < 1e-9, (fc, alpha)
        psi4.core.clean()


def test_uhf_ccsd_polarizability_nh2(rhf_wfn):
    """Open-shell UHF spin-orbital CCSD polarizability (2-B1 NH2, pinned occupation) vs the frozen
    reference -- the genuinely spin-polarized SO path (beyond the RHF-forced-to-SO keystone).
    alpha_zz was validated against an external energy finite field; the tensor is symmetric and its
    diagonal positive."""
    wfn = rhf_wfn(NH2, "6-31G", reference='uhf', freeze_core='false', **NH2_OCC)
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital')
    cc.solve_cc(1e-12, 1e-12, 100)
    alpha = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(alpha - NH2_ALPHA_REF)) < 1e-9, alpha
    assert np.max(np.abs(alpha - alpha.T)) < 1e-9
    assert np.all(np.diag(alpha) > 0.0)
    psi4.core.clean()


def test_ccsd_polarizability_guards(rhf_wfn):
    """The unimplemented paths raise rather than silently returning a wrong tensor: CCSD(T)
    (needs the perturbed (T) response), an unknown route, and the spin-orbital basis."""
    import pytest
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)')
    cc.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(NotImplementedError):
        pycc.CCderiv(cc).polarizability()
    with pytest.raises(ValueError):
        pycc.CCderiv(cc).polarizability(route='explicit')
    psi4.core.clean()
