"""
MP2 atomic polar tensors / nuclear dipole derivatives (correlation contribution) via the 2n+1
route -- P_corr_{A,beta,alpha} = d(mu_alpha)/dX_{A,beta} = -d^2 E_corr / dF_alpha dX_{A,beta}
(the IR-intensity tensor); DERIVATIVES_PLAN_2026-06.md.

Two 2n+1 routes, using only first-order perturbed responses: '2n+1-field' (the default)
differentiates the relaxed nuclear gradient w.r.t. the field, '2n+1-nuclear' differentiates the
relaxed dipole w.r.t. the nuclei. The reference (SCF) APTs are separate
(HFwfn.dipole_derivatives, carrying the Z_A term); the total is their sum.

Validation (frozen analytic references; the finite-difference recipes that validated them once
are kept but not re-run each time):
  * nuclear T2 density response: gauge-invariant scalars Tr(gamma^2), ||Gamma||^2 vs frozen
    references (validated against a finite nuclear displacement -- a displacement genuinely rotates
    pycc's -1/2 S^X gauge vs the canonical MOs, so these orthogonal invariants, not the raw
    amplitudes, are compared);
  * primary: correlation APT components vs frozen references, each validated once against a
    7-point O(h^6) finite difference of the analytic dipole over nuclear displacement (P = d mu / dX,
    a 1/h stencil ~1e-12);
  * keystone: spin-orbital == spin-adapted (closed-shell RHF) correlation APT;
  * the two 2n+1 routes agree with each other;
  * translational (acoustic) sum rule: sum over atoms of the total (and correlation) APT is zero
    for a neutral molecule -- an FD-free physics check.

Geometry is Cartesian in bohr with no_com/no_reorient so displacing an atom keeps the frame
fixed and matches the analytic (bohr) integral derivatives. The FD recipes use pycc's own dipole
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
    """Regeneration recipe for APT_631G[_FC] / NUCLEAR_T2 (not run in the tests): P_corr[A,cart,
    alpha] via a 7-point O(h^6) FD of the analytic correlation dipole (the first-order relaxed
    dipole, independent of the 2n+1 second-derivative code the frozen values validate) over the
    nuclear coordinate (atom, cart)."""
    def mu(delta):
        c = BASE.copy(); c[atom, cart] += delta
        return _mpwfn(c, basis, orbital_basis, freeze_core).relaxed_dipole()[alpha]
    return (-mu(-3*h) + 9*mu(-2*h) - 45*mu(-h)
            + 45*mu(h) - 9*mu(2*h) + mu(3*h)) / (60*h)


# ---- frozen analytic references (each validated once against the FD recipes above; not re-run) ----
# The nuclear T2 density response d Tr(gamma^2) / d ||Gamma||^2 (validated vs a 5-point FD of the
# density invariants to ~6e-12) and the correlation APT components (validated vs the 7-point
# dipole-over-nuclear FD _corr_dipfd to ~1e-12).  Analytic values, frozen rather than re-run.
NUCLEAR_T2 = {   # (atom, cart) -> (d Tr(gamma^2), d ||Gamma||^2), H2O/6-31G spatial all-electron
    (0, 2): (-0.0023008988415012237, -0.040233168083713064),
    (2, 1): (0.0026350116216617243, 0.04796952154189671),
}
APT_631G = {     # (alpha, atom, cart) -> P_corr[atom, cart, alpha], H2O/6-31G spatial all-electron
    (2, 0, 2): 0.07434263445343849,
    (1, 2, 1): -0.058852795259439046,
    (0, 1, 2): -6.87525447497706e-17,
    (2, 1, 1): -0.005605013725530517,
}
APT_631G_FC = {  # frozen core
    (2, 0, 2): 0.07444257211108295,
    (1, 2, 1): -0.05890846005628474,
    (2, 1, 1): -0.005563997828511514,
}


def test_mp2_nuclear_t2_response_631g():
    """Nuclear T2 density response d_X gamma / d_X Gamma (the analytic MPwfn._perturbed_densities)
    vs its frozen reference, via the gauge-invariant scalars d Tr(gamma^2), d ||Gamma||^2 (spatial,
    H2O/6-31G).  The reference was validated against a 5-point FD of the invariants to ~6e-12."""
    from pycc.cphf import Perturbation
    mp0 = _mpwfn(BASE, '6-31G', 'spatial')
    o, v, nmo = mp0.o, mp0.v, mp0.nmo
    Doo, Dvv = mp0._mp2_corr_opdm(); Gam = np.asarray(mp0._mp2_tpdm())
    g = np.zeros((nmo, nmo)); g[o, o] = np.asarray(Doo); g[v, v] = np.asarray(Dvv)
    for (atom, cart) in [(0, 2), (2, 1)]:
        dgX, dGX = mp0._perturbed_densities(Perturbation('nuclear', (atom, cart)))
        dTrg2 = 2.0 * np.sum(g * np.asarray(dgX))
        dGam2 = 2.0 * np.sum(Gam * np.asarray(dGX))
        ref_g, ref_G = NUCLEAR_T2[(atom, cart)]
        assert abs(dTrg2 - ref_g) < 1e-9
        assert abs(dGam2 - ref_G) < 1e-9


def test_mp2_corr_apt_dipfd_631g():
    """Spatial correlation APT vs its frozen reference (validated once against a 7-point dipole-
    over-nuclear FD to ~1e-12), representative components, H2O/6-31G."""
    P = _mpwfn(BASE, '6-31G', 'spatial').dipole_derivatives()
    for (alpha, atom, cart), ref in APT_631G.items():
        assert abs(P[atom, cart, alpha] - ref) < 1e-9


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
    """Frozen-core spatial correlation APT vs its frozen reference (validated once against the
    7-point dipole-over-nuclear FD to ~1e-12), H2O/6-31G.

    Exercises the mixed field/nuclear core<->active response: the frozen-core nuclear density
    response over the active space, coupled to the core<->active orbital relaxation. Works with no
    APT-specific code change -- the ncore machinery carries over from the gradient/polarizability."""
    P = _mpwfn(BASE, '6-31G', 'spatial', freeze_core='true').dipole_derivatives()
    for (alpha, atom, cart), ref in APT_631G_FC.items():
        assert abs(P[atom, cart, alpha] - ref) < 1e-9


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


# ---- the two 2n+1 routes agree with each other ----

def test_mp2_apt_2n1_routes_agree_631g():
    """The two 2n+1 APT routes agree, all four spin/frozen-core combinations (H2O/6-31G).
    '2n+1-nuclear' differentiates the relaxed dipole w.r.t. the nuclei (3N responses, reuses
    _perturbed_relaxed_opdm); '2n+1-field' (the default) differentiates the relaxed nuclear
    gradient w.r.t. the field (3 field responses -- d_F D_rel, d_F Gamma, and the perturbed
    energy-weighted density d_F W -- contracted with the nuclear skeletons + their field
    rotations)."""
    import pytest
    for ob in ('spinorbital', 'spatial'):
        for fc in ('false', 'true'):
            mp = _mpwfn(BASE, '6-31G', ob, freeze_core=fc)
            Pn = np.asarray(mp.dipole_derivatives(route='2n+1-nuclear'))
            Pf = np.asarray(mp.dipole_derivatives(route='2n+1-field'))
            assert np.max(np.abs(Pn - Pf)) < 1e-11
    with pytest.raises(ValueError):
        _mpwfn(BASE, '6-31G', 'spatial').dipole_derivatives(route='bogus')
