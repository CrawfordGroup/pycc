"""
MP2 static dipole polarizability (correlation contribution) via the explicit
second-derivative route -- alpha_corr_ab = -d^2 E_corr / dF_a dF_b, omega = 0;
docs/DERIVATIVES_PLAN_2026-06.md, derivints.pdf Eq. 15.

The correlation second derivative folds the first- and second-order CPHF field response
(U^a, U^{ab}) into the full derivatives of f and <pq||rs> (CPHF.perturbed_fock/eri and
perturbed_fock2/eri2) and contracts them with the unrelaxed MP2 densities and their
first-order responses (MPwfn._perturbed_densities):

    d_ab E_corr = sum_pq [ d_a gamma d_b f + gamma d_ab f ]
                + sum_pqrs [ d_a Gamma d_b <pq||rs> + Gamma d_ab <pq||rs> ]

The reference (SCF) part is kept separate (HFwfn.polarizability); the total is their sum.

Validation:
  * primary: alpha diagonal vs a 7-point O(h^6) finite difference of the analytic MP2
    correlation *dipole* (alpha = d mu / dF). Differencing a first derivative divides by h
    (not h^2 as an energy second derivative would), so the round-off floor is ~3 orders
    lower -- agreement is ~1e-12. It is also a cross-check: the dipole uses only the
    first-order machinery (U, unrelaxed densities), the polarizability the second-order
    machinery (U^{ab}, perturbed densities), so their agreement confirms the two analytic
    derivations are mutually consistent.
  * independence guard: one component vs a finite field of the MP2 *energy* -- a fully
    external oracle (the dipole FD is pycc-vs-pycc). The energy FD is of E_MP2, not any
    analytic gradient, so it is unaffected by Psi4's frozen-core MP2 gradient bug.
  * keystone: spin-orbital == spin-adapted (closed-shell RHF) correlation alpha (both bases,
    all-electron and frozen-core), which carries the high-precision spatial checks to the
    spin-orbital path.
"""

import psi4
import pycc
import numpy as np


WATER = """
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
"""

# Open-shell UHF reference: NH2 (2-B1, non-degenerate).
NH2 = """
0 2
N
H 1 1.02
H 1 1.02 2 103.0
symmetry c1
"""


def _pycc_alpha(basis, orbital_basis='spatial', freeze_core='false', geom=WATER, reference='rhf'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'freeze_core': freeze_core, 'e_convergence': 1e-14, 'd_convergence': 1e-14})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return np.asarray(mp.polarizability())


def _dipfd_alpha_diag(basis, axis, orbital_basis='spatial', freeze_core='false', h=0.002):
    """|alpha_(axis,axis)| via a 7-point O(h^6) central first-derivative stencil of the
    analytic MP2 correlation dipole component ``axis`` under a field along ``axis``
    (alpha = d mu / dF). Returns the magnitude (the diagonal polarizability is positive;
    the mu/field sign convention is immaterial to |d mu / dF|)."""
    def mu(Fval):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(WATER)
        d = [0.0, 0.0, 0.0]
        d[axis] = Fval
        opt = {'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
               'e_convergence': 1e-14, 'd_convergence': 1e-14}
        if Fval:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': d})
        psi4.set_options(opt)
        _, wfn = psi4.energy('scf', return_wfn=True)
        mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
        mp.compute_energy()
        return mp._corr_dipole_explicit()[axis]
    d1 = (-mu(-3 * h) + 9 * mu(-2 * h) - 45 * mu(-h)
          + 45 * mu(h) - 9 * mu(2 * h) + mu(3 * h)) / (60 * h)
    return abs(d1)


def _energy_fd_alpha_diag(basis, axis, freeze_core='false', F=0.002, geom=WATER, reference='rhf'):
    """alpha_corr_(axis,axis) = -d^2(E_MP2 - E_SCF)/dF^2 by a 5-point finite field of the
    energy -- a fully external oracle (independence guard for the dipole cross-check)."""
    def e(model, Fval):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(geom)
        d = [0.0, 0.0, 0.0]
        d[axis] = Fval
        opt = {'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv', 'reference': reference,
               'freeze_core': freeze_core, 'e_convergence': 1e-13, 'd_convergence': 1e-13}
        if Fval:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': d})
        psi4.set_options(opt)
        return psi4.energy(model)

    def d2(model):
        return (-e(model, 2 * F) + 16 * e(model, F) - 30 * e(model, 0.0)
                + 16 * e(model, -F) - e(model, -2 * F)) / (12 * F * F)
    return -(d2('mp2') - d2('scf'))


# ---- primary: high-precision dipole finite difference (~1e-12) ----

def test_mp2_corr_polarizability_dipfd_631g():
    """All-electron spatial MP2 correlation alpha, all three diagonal components vs a
    7-point finite difference of the analytic correlation dipole, H2O/6-31G (~1e-12)."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial')
    for axis in range(3):
        assert abs(a[axis, axis] - _dipfd_alpha_diag('6-31G', axis)) < 1e-10


def test_fc_mp2_corr_polarizability_dipfd_631g():
    """Frozen-core spatial MP2 correlation alpha diagonal vs the 7-point dipole finite
    difference, H2O/6-31G (~1e-12).

    Exercises the frozen-core second-order response: the core<->active U^{ab} from the
    canonical d_ab f_ij = 0 divide, and the ov second-order CPHF solve seeded with the
    nonzero ov orthonormality term xi_ia (which vanishes only in the all-electron field
    case)."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial', freeze_core='true')
    for axis in range(3):
        assert abs(a[axis, axis]
                   - _dipfd_alpha_diag('6-31G', axis, freeze_core='true')) < 1e-10


# ---- independence guard: external energy finite field ----

def test_mp2_corr_polarizability_energy_fd_631g():
    """All-electron alpha_zz vs a 5-point finite field of the MP2 energy -- a fully
    external oracle (the dipole FD above is pycc-vs-pycc). Coarser (energy second
    derivatives divide by h^2), so a looser tolerance."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial')
    assert abs(a[2, 2] - _energy_fd_alpha_diag('6-31G', 2)) < 1e-7


def test_ump2_corr_polarizability_energy_fd_nh2_631g():
    """Open-shell UHF-MP2 correlation alpha_zz vs a 5-point finite field of the MP2 energy --
    the external open-shell oracle for the spin-orbital second-order response, NH2 (2-B1) / 6-31G.
    (Energy second derivatives divide by h^2, so a looser tolerance than the dipole FD.)"""
    a = _pycc_alpha('6-31G', orbital_basis='spinorbital', geom=NH2, reference='uhf')
    assert abs(a[2, 2] - _energy_fd_alpha_diag('6-31G', 2, geom=NH2, reference='uhf')) < 1e-6


# ---- keystone: spin-orbital == spin-adapted (carries the checks to the SO path) ----

def test_so_mp2_corr_polarizability_vs_spatial_631g():
    """Keystone (6-31G): closed-shell spin-orbital == spin-adapted correlation alpha."""
    a_so = _pycc_alpha('6-31G', orbital_basis='spinorbital')
    a_sa = _pycc_alpha('6-31G', orbital_basis='spatial')
    assert np.max(np.abs(a_so - a_sa)) < 1e-11


def test_so_mp2_corr_polarizability_vs_spatial_ccpvdz():
    """Keystone (cc-pVDZ, polarization functions): spin-orbital == spin-adapted alpha."""
    a_so = _pycc_alpha('cc-pVDZ', orbital_basis='spinorbital')
    a_sa = _pycc_alpha('cc-pVDZ', orbital_basis='spatial')
    assert np.max(np.abs(a_so - a_sa)) < 1e-11


def test_fc_so_mp2_corr_polarizability_vs_spatial_631g():
    """Keystone (6-31G, frozen core): spin-orbital == spin-adapted correlation alpha."""
    a_so = _pycc_alpha('6-31G', orbital_basis='spinorbital', freeze_core='true')
    a_sa = _pycc_alpha('6-31G', orbital_basis='spatial', freeze_core='true')
    assert np.max(np.abs(a_so - a_sa)) < 1e-11


def test_fc_so_mp2_corr_polarizability_vs_spatial_ccpvdz():
    """Keystone (cc-pVDZ, frozen core): spin-orbital == spin-adapted correlation alpha."""
    a_so = _pycc_alpha('cc-pVDZ', orbital_basis='spinorbital', freeze_core='true')
    a_sa = _pycc_alpha('cc-pVDZ', orbital_basis='spatial', freeze_core='true')
    assert np.max(np.abs(a_so - a_sa)) < 1e-11
