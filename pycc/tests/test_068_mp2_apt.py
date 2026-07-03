"""
MP2 atomic polar tensors / nuclear dipole derivatives (correlation contribution) via the
explicit mixed field-nuclear second derivative -- P_corr_{A,beta,alpha} = d(mu_alpha)/dX_{A,beta}
= -d^2 E_corr / dF_alpha dX_{A,beta} (the IR-intensity tensor); DERIVATIVES_PLAN_2026-06.md.

Reuses the polarizability infrastructure: the field first derivatives d_F f / d_F <pq||rs>,
the *nuclear* density response d_X gamma / d_X Gamma (the generic MPwfn._perturbed_densities
driven by the nuclear perturbed integrals), and the mixed second derivatives d_{F X} f /
d_{F X} <pq||rs> (CPHF.perturbed_fock2 / perturbed_eri2, with the generalized skeletons: the
first-order nuclear two-electron skeleton and the -mu^X mixed one-electron skeleton). The
reference (SCF) APTs are separate (HFwfn.dipole_derivatives, carrying the Z_A term); the total
is their sum.

Validation:
  * nuclear T2 density response: gauge-invariant scalars Tr(gamma^2), ||Gamma||^2 vs a finite
    nuclear displacement (a nuclear displacement genuinely rotates pycc's -1/2 S^X gauge vs the
    canonical MOs, so these orthogonal invariants -- not the raw amplitudes -- are compared);
  * primary: correlation APT vs a 7-point O(h^6) finite difference of the analytic dipole over
    nuclear displacement (P = d mu / dX, a 1/h stencil ~1e-12);
  * keystone: spin-orbital == spin-adapted (closed-shell RHF) correlation APT;
  * translational (acoustic) sum rule: sum over atoms of the total (and correlation) APT is zero
    for a neutral molecule -- an FD-free physics check.

Geometry is Cartesian in bohr with no_com/no_reorient so displacing an atom keeps the frame
fixed and matches the analytic (bohr) integral derivatives. Oracles are pycc's own dipole
(Psi4's frozen-core MP2 gradient bug rules out its analytic derivatives).
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


def _mpwfn(coords, basis, orbital_basis='spatial', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(_geom(coords))
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp


def _dens_invariants(mp):
    """Gauge-invariant Tr(gamma^2), ||Gamma||^2 (orthogonal orbital rotations preserve both)."""
    o, v, nmo = mp.o, mp.v, mp.nmo
    if mp.orbital_basis == 'spinorbital':
        Doo, Dvv = mp._so_mp2_corr_opdm(); Gam = np.asarray(mp._so_mp2_tpdm())
    else:
        Doo, Dvv = mp._mp2_corr_opdm(); Gam = np.asarray(mp._mp2_tpdm())
    g = np.zeros((nmo, nmo)); g[o, o] = np.asarray(Doo); g[v, v] = np.asarray(Dvv)
    return np.sum(g * g), np.sum(Gam * Gam)


def _corr_dipfd(basis, orbital_basis, alpha, atom, cart, freeze_core='false', h=0.002):
    """P_corr[A,cart,alpha] via a 7-point O(h^6) FD of the analytic correlation dipole
    component `alpha` over the nuclear coordinate (atom, cart)."""
    def mu(delta):
        c = BASE.copy(); c[atom, cart] += delta
        return _mpwfn(c, basis, orbital_basis, freeze_core)._corr_dipole_explicit()[alpha]
    return (-mu(-3*h) + 9*mu(-2*h) - 45*mu(-h)
            + 45*mu(h) - 9*mu(2*h) + mu(3*h)) / (60*h)


def test_mp2_nuclear_t2_response_631g():
    """Nuclear T2 density response d_X gamma / d_X Gamma vs a finite nuclear displacement,
    via the gauge-invariant scalars d Tr(gamma^2), d ||Gamma||^2 (spatial, H2O/6-31G)."""
    from pycc.cphf import Perturbation
    mp0 = _mpwfn(BASE, '6-31G', 'spatial')
    o, v, nmo = mp0.o, mp0.v, mp0.nmo
    Doo, Dvv = mp0._mp2_corr_opdm(); Gam = np.asarray(mp0._mp2_tpdm())
    g = np.zeros((nmo, nmo)); g[o, o] = np.asarray(Doo); g[v, v] = np.asarray(Dvv)
    for (atom, cart) in [(0, 2), (2, 1)]:
        dgX, dGX = mp0._perturbed_densities(Perturbation('nuclear', (atom, cart)))
        dTrg2 = 2.0 * np.sum(g * np.asarray(dgX))
        dGam2 = 2.0 * np.sum(Gam * np.asarray(dGX))
        h = 0.005

        def inv(delta):
            c = BASE.copy(); c[atom, cart] += delta
            return _dens_invariants(_mpwfn(c, '6-31G', 'spatial'))
        v2, v1, vm1, vm2 = inv(2*h), inv(h), inv(-h), inv(-2*h)
        fd_g = (-v2[0] + 8*v1[0] - 8*vm1[0] + vm2[0]) / (12*h)
        fd_G = (-v2[1] + 8*v1[1] - 8*vm1[1] + vm2[1]) / (12*h)
        assert abs(dTrg2 - fd_g) < 1e-8
        assert abs(dGam2 - fd_G) < 1e-8


def test_mp2_corr_apt_dipfd_631g():
    """Spatial correlation APT vs a 7-point FD of the analytic dipole over nuclear
    displacement, representative components, H2O/6-31G (~1e-12)."""
    P = _mpwfn(BASE, '6-31G', 'spatial').dipole_derivatives()
    for (alpha, atom, cart) in [(2, 0, 2), (1, 2, 1), (0, 1, 2), (2, 1, 1)]:
        assert abs(P[atom, cart, alpha]
                   - _corr_dipfd('6-31G', 'spatial', alpha, atom, cart)) < 1e-10


def test_so_mp2_corr_apt_vs_spatial_631g():
    """Keystone (6-31G): closed-shell spin-orbital == spin-adapted correlation APT."""
    P_so = _mpwfn(BASE, '6-31G', 'spinorbital').dipole_derivatives()
    P_sa = _mpwfn(BASE, '6-31G', 'spatial').dipole_derivatives()
    assert np.max(np.abs(P_so - P_sa)) < 1e-11


def test_so_mp2_corr_apt_vs_spatial_ccpvdz():
    """Keystone (cc-pVDZ): spin-orbital == spin-adapted correlation APT."""
    P_so = _mpwfn(BASE, 'cc-pVDZ', 'spinorbital').dipole_derivatives()
    P_sa = _mpwfn(BASE, 'cc-pVDZ', 'spatial').dipole_derivatives()
    assert np.max(np.abs(P_so - P_sa)) < 1e-11


def test_mp2_apt_translational_sum_rule_631g():
    """Acoustic (translational) sum rule: for a neutral molecule the sum over atoms of the
    total APT vanishes (the correlation APT sums to zero on its own -- Tr(gamma_corr) = 0)."""
    mp0 = _mpwfn(BASE, '6-31G', 'spatial')
    P_corr = mp0.dipole_derivatives()
    P_tot = np.asarray(pycc.apt(mp0).total)
    assert np.max(np.abs(np.sum(P_corr, axis=0))) < 1e-10
    assert np.max(np.abs(np.sum(P_tot, axis=0))) < 1e-10


# ---- frozen core (the mixed field/nuclear core<->active response) ----

def test_fc_mp2_corr_apt_dipfd_631g():
    """Frozen-core spatial correlation APT vs the 7-point dipole-over-nuclear FD, H2O/6-31G.

    Exercises the mixed field/nuclear core<->active response: the frozen-core U^{FX} (with the
    core<->active canonical divide and the ov xi-seed) and the nuclear density response over
    the active space. Works with no APT-specific code change -- the ncore machinery carries
    over from the gradient/polarizability."""
    P = _mpwfn(BASE, '6-31G', 'spatial', freeze_core='true').dipole_derivatives()
    for (alpha, atom, cart) in [(2, 0, 2), (1, 2, 1), (2, 1, 1)]:
        assert abs(P[atom, cart, alpha]
                   - _corr_dipfd('6-31G', 'spatial', alpha, atom, cart,
                                 freeze_core='true')) < 1e-10


def test_fc_so_mp2_corr_apt_vs_spatial_631g():
    """Keystone (6-31G, frozen core): spin-orbital == spin-adapted correlation APT."""
    P_so = _mpwfn(BASE, '6-31G', 'spinorbital', freeze_core='true').dipole_derivatives()
    P_sa = _mpwfn(BASE, '6-31G', 'spatial', freeze_core='true').dipole_derivatives()
    assert np.max(np.abs(P_so - P_sa)) < 1e-11


def test_fc_mp2_apt_translational_sum_rule_631g():
    """Frozen-core acoustic sum rule: sum over atoms of the total (and correlation) APT
    vanishes for neutral water."""
    mp0 = _mpwfn(BASE, '6-31G', 'spatial', freeze_core='true')
    assert np.max(np.abs(np.sum(mp0.dipole_derivatives(), axis=0))) < 1e-10
    assert np.max(np.abs(np.sum(np.asarray(pycc.apt(mp0).total), axis=0))) < 1e-10


# ---- 2n+1 routes (Phase C): both reproduce the explicit APT to ~machine precision ----

def test_mp2_apt_2n1_routes_vs_explicit_631g():
    """Both 2n+1 APT routes == the explicit route, all four spin/frozen-core combinations
    (H2O/6-31G). '2n+1-nuclear' differentiates the relaxed dipole w.r.t. the nuclei (3N
    responses, reuses _perturbed_relaxed_opdm); '2n+1-field' differentiates the relaxed nuclear
    gradient w.r.t. the field (3 field responses -- d_F D_rel, d_F Gamma, and the perturbed
    energy-weighted density d_F W -- contracted with the nuclear skeletons + their field
    rotations). The two also agree with each other."""
    import pytest
    for ob in ('spinorbital', 'spatial'):
        for fc in ('false', 'true'):
            mp = _mpwfn(BASE, '6-31G', ob, freeze_core=fc)
            Pe = np.asarray(mp.dipole_derivatives())
            Pn = np.asarray(mp.dipole_derivatives(route='2n+1-nuclear'))
            Pf = np.asarray(mp.dipole_derivatives(route='2n+1-field'))
            assert np.max(np.abs(Pn - Pe)) < 1e-11
            assert np.max(np.abs(Pf - Pe)) < 1e-11
    with pytest.raises(ValueError):
        _mpwfn(BASE, '6-31G', 'spatial').dipole_derivatives(route='bogus')
