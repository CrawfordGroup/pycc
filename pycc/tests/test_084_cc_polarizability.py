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
import pytest


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
        Ip = np.asarray(pycc.CCderiv(cx)._lagrangian(D, G))
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
    r = pycc.polarizability(pycc.CCderiv(cc))
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

    r = pycc.polarizability(pycc.CCderiv(cc))
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


# ---- CCSD(T): spin-orbital, energy-second-derivative-FD anchored ----
# The spin-orbital CCSD(T) correlation alpha (H2O/6-31G, WATER), frozen rather than re-run.  Unlike
# CCSD above, the relaxed-dipole FD is NOT a reliable oracle for (T): the Lee-Rendell dependent-pair
# numerator gate makes the relaxed dipole non-smooth under an out-of-plane (sigma_h-breaking) field,
# which poisons the finite difference (the energy is untouched -- it never uses the dependent pairs).
# So each (T) value was validated once against the ENERGY second-derivative finite field
# (alpha_aa = -d^2 E_corr/dF_a^2, density-independent) -- diagonals to 2.4e-8 (see _energy_fd_alpha).
# The deterministic analytic recomputation reproduces the frozen tensor to ~machine precision, so
# 1e-9 is a tight regression guard.  Spatial CCSD(T) is not yet available.
WATER_ALPHA_T_DIAG    = np.array([0.0749401988258554, 0.0992910106442475, 0.20111257930800613])   # SO, AE
WATER_ALPHA_T_DIAG_FC = np.array([0.0752394685506570, 0.1001122215200233, 0.20165562805738180])   # SO, FC
# frozen external anchor: alpha diagonal from the 5-point energy 2nd-derivative FD (validated once).
ALPHA_T_ENERGY_FD    = np.array([0.0749401992206078, 0.0992910218359603, 0.20111260312954962])    # AE
ALPHA_T_ENERGY_FD_FC = np.array([0.0752394689422971, 0.1001122328817920, 0.20165565204760540])    # FC


def _energy_fd_alpha(freeze_core='false', F=5e-3):
    """Regeneration recipe for ALPHA_T_ENERGY_FD[_FC] (not run in the tests): the SO CCSD(T)
    correlation alpha diagonal from a 5-point O(h^4) finite field of pycc's own correlation energy,
    alpha_aa = -d^2(E_CCSD(T) - E_SCF)/dF_a^2 -- density-independent, the reliable (T) oracle."""
    def e(Fvec):
        psi4.core.clean(); psi4.core.clean_options(); psi4.geometry(WATER)
        opt = {'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': freeze_core,
               'e_convergence': 1e-14, 'd_convergence': 1e-14}
        if Fvec is not None:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': Fvec})
        psi4.set_options(opt)
        _, wfn = psi4.energy('scf', return_wfn=True)
        return pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital').solve_cc(1e-13, 1e-13, 300)
    E0 = e(None); diag = np.zeros(3)
    for a in range(3):
        d = lambda s: e([s * F if i == a else 0.0 for i in range(3)])
        diag[a] = -(-d(2) + 16 * d(1) - 30 * E0 + 16 * d(-1) - d(-2)) / (12 * F * F)
    return diag


def test_so_ccsdt_polarizability_water_631g(rhf_wfn):
    """All-electron spin-orbital CCSD(T) correlation alpha (H2O/6-31G) vs the frozen reference and the
    frozen energy-second-derivative finite field; the tensor is diagonal by symmetry."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital', make_t3_density=True)
    cc.solve_cc(1e-13, 1e-13, 100)
    a = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(np.diag(a) - WATER_ALPHA_T_DIAG)) < 1e-9, np.diag(a)
    assert np.max(np.abs(a - np.diag(np.diag(a)))) < 1e-10, a          # diagonal by symmetry
    assert np.max(np.abs(np.diag(a) - ALPHA_T_ENERGY_FD)) < 1e-7, np.diag(a)
    psi4.core.clean()


def test_fc_so_ccsdt_polarizability_water_631g(rhf_wfn):
    """Frozen-core spin-orbital CCSD(T) correlation alpha (H2O/6-31G, O 1s frozen) vs the frozen
    reference and the frozen energy-FD diagonal -- exercises the (T) core<->active and active oo/vv
    dependent-pair response inside the perturbed relaxed density."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="true")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital', make_t3_density=True)
    cc.solve_cc(1e-13, 1e-13, 100)
    assert cc.nfzc > 0
    a = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(np.diag(a) - WATER_ALPHA_T_DIAG_FC)) < 1e-9, np.diag(a)
    assert np.max(np.abs(np.diag(a) - ALPHA_T_ENERGY_FD_FC)) < 1e-7, np.diag(a)
    psi4.core.clean()


# --- CFOUR-oracle CCSD(T) polarizability tests ----------------------------------------------------
# Oracles from CFOUR (xcfour, CALC=CCSD(T), PROP=SECOND_ORDER, SCF_CONV=13, CC_CONV=12): the
# correlation polarizability is POLAR (total) - POLARSCF (HF).  To match CFOUR to all printed digits
# the AO basis must be identical -- Psi4's "6-31G" differs from CFOUR's by ~4e-8 -- so CFOUR's exact
# GENBAS "6-31G" is transcribed here; pycc then reproduces CFOUR to ~5e-11 on both routes.
CFOUR_631G = """cartesian
****
H     0
S   3   1.00
     18.7311370             0.0334946
      2.8253944             0.2347270
      0.6401217             0.8137573
S   1   1.00
      0.1612778             1.0000000
****
O     0
S   6   1.00
   5484.6716600             0.0018311
    825.2349460             0.0139502
    188.0469580             0.0684451
     52.9645000             0.2327143
     16.8975704             0.4701929
      5.7996353             0.3585209
S   3   1.00
     15.5396162            -0.1107775
      3.5999336            -0.1480263
      1.0137618             1.1307670
S   1   1.00
      0.2700058             1.0000000
P   3   1.00
     15.5396162             0.0708743
      3.5999336             0.3397528
      1.0137618             0.7271586
P   1   1.00
      0.2700058             1.0000000
****
F     0
S   6   1.00
   7001.7130900             0.0018196
   1051.3660900             0.0139161
    239.2856900             0.0684053
     67.3974453             0.2331858
     21.5199573             0.4712674
      7.4031013             0.3566185
S   3   1.00
     20.8479528            -0.1085070
      4.8083083            -0.1464517
      1.3440699             1.1286886
S   1   1.00
      0.3581514             1.0000000
P   3   1.00
     20.8479528             0.0716287
      4.8083083             0.3459121
      1.3440699             0.7224700
P   1   1.00
      0.3581514             1.0000000
****
"""

# CFOUR CCSD(T) correlation polarizability (a.u.), POLAR - POLARSCF, keyed by (molecule, freeze_core).
CFOUR_ALPHA_T = {
    ('water', 'false'): np.diag([0.0749401813, 0.0992909720, 0.2011125406]),
    ('water', 'true'):  np.diag([0.0752394512, 0.1001121832, 0.2016555896]),
    ('hof', 'false'): np.array([[-2.2114079425, -0.7220009461, 0.0],
                                [-0.7220009461,  0.1659235582, 0.0],
                                [ 0.0,           0.0,           0.0651632870]]),
    ('hof', 'true'):  np.array([[-2.2106648649, -0.7220673958, 0.0],
                                [-0.7220673958,  0.1661531583, 0.0],
                                [ 0.0,           0.0,           0.0652603834]]),
}
_CFOUR_GEOM = {'water': WATER, 'hof': HOF}


def _cfour_wfn(geom, freeze_core):
    """RHF reference in CFOUR's exact GENBAS 6-31G (via basis_helper), for the CFOUR-oracle tests."""
    psi4.core.clean(); psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    psi4.basis_helper(CFOUR_631G, name='cfour631g')
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


# route x molecule x freeze_core; the spin-orbital HOF cases are the expensive pair (~200 s), so they
# are marked slow (skipped by default) -- the spatial HOF cases and the SO==spatial keystone still
# exercise the off-diagonal and SO paths in the default run.
_CFOUR_CASES = [
    ('spatial', 'water', 'false'), ('spatial', 'water', 'true'),
    ('spatial', 'hof', 'false'), ('spatial', 'hof', 'true'),
    ('spinorbital', 'water', 'false'), ('spinorbital', 'water', 'true'),
    pytest.param('spinorbital', 'hof', 'false', marks=pytest.mark.slow),
    pytest.param('spinorbital', 'hof', 'true', marks=pytest.mark.slow),
]


@pytest.mark.parametrize('route,mol,fc', _CFOUR_CASES)
def test_ccsdt_polarizability_cfour(route, mol, fc):
    """CCSD(T) correlation polarizability vs the CFOUR oracle (POLAR - POLARSCF), water and HOF,
    all-electron and frozen core, on both the spatial (closed-shell RHF) and spin-orbital routes, in
    CFOUR's exact GENBAS 6-31G.  HOF exercises the off-diagonal (Cs) in-plane block; matches to 1e-9."""
    wfn = _cfour_wfn(_CFOUR_GEOM[mol], fc)
    cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis=route, make_t3_density=True)
    cc.solve_cc(1e-13, 1e-13, 100)
    a = np.asarray(pycc.CCderiv(cc).polarizability())
    assert np.max(np.abs(a - CFOUR_ALPHA_T[(mol, fc)])) < 1e-9, (mol, fc, route, a)
    assert np.max(np.abs(a - a.T)) < 1e-9                      # symmetric
    psi4.core.clean()


def test_ccsd_polarizability_guards(rhf_wfn):
    """Constructing a CCderiv sets the (T) density itself -- so a CCSD(T) property no longer needs
    the user to pass make_t3_density (the former footgun).  Both the spatial and spin-orbital
    CCSD(T) paths build without the flag."""
    wfn = rhf_wfn(WATER, "6-31G", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='ccsd(t)')                              # spatial CCSD(T), no make_t3_density
    cc.solve_cc(1e-12, 1e-12, 100)
    d = pycc.CCderiv(cc)                                              # sets make_t3_density, builds the (T) density
    assert cc.make_t3_density is True
    d.polarizability()                                               # works without the user setting the flag
    cc_so = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital')   # no make_t3_density
    cc_so.solve_cc(1e-12, 1e-12, 100)
    pycc.CCderiv(cc_so).polarizability()                             # SO path likewise works without the flag
    psi4.core.clean()
