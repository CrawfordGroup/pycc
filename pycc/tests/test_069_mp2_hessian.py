"""
MP2 molecular (nuclear) Hessian (correlation contribution) via the explicit nuclear-nuclear
second derivative -- H_corr[Aa,Bb] = d^2 E_corr / dX_{Aa} dX_{Bb};
DERIVATIVES_PLAN_2026-06.md, derivints.pdf Eqs. 15/17/18/20.

The nuclear-nuclear analog of the polarizability/APT, completing the second-order machinery:
the nuclear density response d_X gamma / d_X Gamma, the nuclear first derivatives, and the
nuclear-nuclear second derivatives (CPHF.perturbed_fock2 / perturbed_eri2). The moving-basis
terms absent from the field/APT cases now activate -- the xi^{XY} overlap terms
S^{XY} - S^X S^Y (Eq. 18) and the mixed skeletons h^{XY} (core2) and <pq||rs>^{XY} (eri2).
The reference (SCF) Hessian is separate (HFwfn.hessian, carrying the nuclear-repulsion term);
the total is their sum.

Validation:
  * primary: correlation Hessian columns vs a 7-point O(h^6) finite difference of the analytic
    correlation gradient (H = d(gradient)/dX, a 1/h stencil ~1e-13);
  * symmetry H = H.T and the translational sum rule sum_B H_{Aa,Bb} = 0 (E_corr is
    translationally invariant) -- FD-free checks;
  * keystone: spin-orbital == spin-adapted (closed-shell RHF) correlation Hessian.

Geometry is Cartesian in bohr with no_com/no_reorient so displacing an atom keeps the frame
fixed and matches the analytic (bohr) integral derivatives. Oracle is pycc's own analytic
gradient (Psi4's frozen-core MP2 gradient bug rules out its analytic derivatives). 6-31G only
(the shared second-order machinery is already cc-pVDZ-tested via the polarizability/APT).
"""

import psi4
import pycc
import numpy as np


BASE = np.array([[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.814137],
                 [0.0, 1.756000, -0.454300]])
SYM = ['O', 'H', 'H']


def _geom(coords):
    s = "units bohr\nsymmetry c1\nno_com\nno_reorient\n"
    for sym, xyz in zip(SYM, coords):
        s += f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n"
    return s


def _mpwfn(coords, orbital_basis='spatial', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(_geom(coords))
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp


def _gradfd_col(orbital_basis, atom, cart, freeze_core='false', h=0.002):
    """One Hessian column (natom, 3) via a 7-point O(h^6) FD of the analytic correlation
    gradient over the nuclear coordinate (atom, cart)."""
    def g(delta):
        c = BASE.copy(); c[atom, cart] += delta
        return np.asarray(_mpwfn(c, orbital_basis, freeze_core).gradient())
    return (-g(-3*h) + 9*g(-2*h) - 45*g(-h)
            + 45*g(h) - 9*g(2*h) + g(3*h)) / (60*h)


def test_mp2_corr_hessian_symmetry_sum_rule_631g():
    """All-electron correlation Hessian: symmetric, and translationally invariant
    (sum over the second atom vanishes), H2O/6-31G -- FD-free checks."""
    H = _mpwfn(BASE, 'spatial').hessian()
    assert H.shape == (9, 9)
    assert np.max(np.abs(H - H.T)) < 1e-10
    assert np.max(np.abs(H.reshape(3, 3, 3, 3).sum(axis=2))) < 1e-10


def test_mp2_corr_hessian_gradfd_631g():
    """All-electron correlation Hessian columns vs a 7-point FD of the analytic gradient,
    H2O/6-31G (~1e-13)."""
    H = _mpwfn(BASE, 'spatial').hessian()
    for (atom, cart) in [(0, 2), (2, 1)]:
        j = atom * 3 + cart
        assert np.max(np.abs(H[:, j] - _gradfd_col('spatial', atom, cart).reshape(-1))) < 1e-9


def test_so_mp2_corr_hessian_vs_spatial_631g():
    """Keystone (6-31G): closed-shell spin-orbital == spin-adapted correlation Hessian."""
    H_so = _mpwfn(BASE, 'spinorbital').hessian()
    H_sa = _mpwfn(BASE, 'spatial').hessian()
    assert np.max(np.abs(H_so - H_sa)) < 1e-11


# ---- frozen core (the nuclear-nuclear core<->active response) ----

def test_fc_mp2_corr_hessian_gradfd_631g():
    """Frozen-core correlation Hessian columns vs the 7-point gradient FD, H2O/6-31G."""
    H = _mpwfn(BASE, 'spatial', freeze_core='true').hessian()
    for (atom, cart) in [(0, 2), (2, 1)]:
        j = atom * 3 + cart
        assert np.max(np.abs(H[:, j]
                      - _gradfd_col('spatial', atom, cart, freeze_core='true').reshape(-1))) < 1e-9


def test_fc_mp2_corr_hessian_symmetry_sum_rule_631g():
    """Frozen-core correlation Hessian: symmetric and translationally invariant."""
    H = _mpwfn(BASE, 'spatial', freeze_core='true').hessian()
    assert np.max(np.abs(H - H.T)) < 1e-10
    assert np.max(np.abs(H.reshape(3, 3, 3, 3).sum(axis=2))) < 1e-10


def test_fc_so_mp2_corr_hessian_vs_spatial_631g():
    """Keystone (6-31G, frozen core): spin-orbital == spin-adapted correlation Hessian."""
    H_so = _mpwfn(BASE, 'spinorbital', freeze_core='true').hessian()
    H_sa = _mpwfn(BASE, 'spatial', freeze_core='true').hessian()
    assert np.max(np.abs(H_so - H_sa)) < 1e-11


# ---- 2n+1 route (Phase D): O(N) first-order solves, reproduces the explicit Hessian ----

def test_mp2_hessian_2n1_vs_explicit_631g():
    """2n+1 == explicit correlation Hessian, all four spin/frozen-core combinations (H2O/6-31G).

    The 2n+1 route differentiates the relaxed nuclear gradient w.r.t. a second nucleus using
    only 3N first-order solves (perturbed relaxed density / energy-weighted density and U^Y) --
    no U^{XY}. The nuclear-nuclear analog of the '2n+1-field' APT, with the full second integral
    skeletons f^{XY}/<pq||rs>^{XY}/S^{XY} and the U^Y rotation of the X skeletons (hoisted onto
    the densities). Result is symmetric and reproduces the explicit route to ~machine
    precision."""
    import pytest
    for ob in ('spinorbital', 'spatial'):
        for fc in ('false', 'true'):
            mp = _mpwfn(BASE, ob, freeze_core=fc)
            He = np.asarray(mp.hessian())
            Hn = np.asarray(mp.hessian(route='2n+1'))
            assert np.max(np.abs(Hn - He)) < 1e-11
            assert np.max(np.abs(Hn - Hn.T)) < 1e-11
    with pytest.raises(ValueError):
        _mpwfn(BASE, 'spatial').hessian(route='bogus')
