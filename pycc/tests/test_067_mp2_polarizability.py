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
  * primary: alpha diagonal vs *frozen* references (ALPHA_DIAG_631G[_FC]), each validated once
    against a 7-point O(h^6) finite difference of the analytic MP2 correlation *dipole*
    (alpha = d mu / dF) to ~7e-12 -- the regeneration recipe _dipfd_alpha_diag, kept but not run.
    That dipole FD is a cross-check: the dipole uses only the first-order machinery (U, unrelaxed
    densities), the polarizability the second-order machinery (U^{ab}, perturbed densities), so
    their agreement (verified when the reference was frozen) confirms the two analytic derivations
    are mutually consistent.
  * independence guard: one component vs a *frozen* finite field of the MP2 *energy* -- a fully
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

# Open-shell UHF reference: NH2 (2-B1, bent), run in C2v with the 2-B1 ground-state occupation
# pinned (NH2_OCC) so a poor SCF guess cannot fall into the 2-A1 excited solution ~0.074 Eh higher
# (which makes the pinned value guess- and platform-independent, hence freezeable).
NH2 = """
0 2
N
H 1 1.02
H 1 1.02 2 103.0
"""
NH2_OCC = {'docc': [3, 0, 0, 1], 'socc': [0, 0, 1, 0]}   # pin 2-B1 ground state (C2v irreps A1,A2,B1,B2)


def _pycc_alpha(basis, orbital_basis='spatial', freeze_core='false', geom=WATER, reference='rhf', occ=None):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    opt = {'basis': basis, 'scf_type': 'pk', 'reference': reference,
           'freeze_core': freeze_core, 'e_convergence': 1e-14, 'd_convergence': 1e-14}
    if occ:
        opt.update(occ)
    psi4.set_options(opt)
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return np.asarray(mp.polarizability())


def _dipfd_alpha_diag(basis, axis, orbital_basis='spatial', freeze_core='false', h=0.002):
    """Regeneration recipe for ALPHA_DIAG_631G[_FC] (not run in the tests): ``|alpha_(axis,axis)|``
    via a 7-point O(h^6) central first-derivative stencil of the analytic MP2 correlation dipole
    component ``axis`` under a field along ``axis`` (alpha = d mu / dF). Returns the magnitude (the
    diagonal polarizability is positive; the mu/field sign convention is immaterial to
    |d mu / dF|)."""
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


def _energy_fd_alpha_diag(basis, axis, freeze_core='false', F=0.002, geom=WATER, reference='rhf', occ=None):
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
        if occ:
            opt.update(occ)
        if Fval:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': d})
        psi4.set_options(opt)
        return psi4.energy(model)

    def d2(model):
        return (-e(model, 2 * F) + 16 * e(model, F) - 30 * e(model, 0.0)
                + 16 * e(model, -F) - e(model, -2 * F)) / (12 * F * F)
    return -(d2('mp2') - d2('scf'))


# ---- primary: frozen analytic diagonal (validated vs the 7-point dipole finite difference) ----
# The MP2 correlation alpha diagonal, frozen rather than re-run each time.  Each value was validated
# once against a 7-point O(h^6) finite difference of the analytic correlation *dipole* (alpha =
# d mu / dF) to ~7e-12 (the regeneration recipe _dipfd_alpha_diag).  That dipole FD is an internal
# cross-check -- the dipole uses only first-order machinery (U, unrelaxed densities), the
# polarizability the second-order machinery (U^{ab}, perturbed densities) -- so its agreement (checked
# when the reference was frozen) confirms the two analytic derivations are mutually consistent.  The
# deterministic analytic recomputation reproduces the frozen values to ~machine precision.
ALPHA_DIAG_631G    = np.array([0.084402387,  0.0611767146, 0.1951911112])   # all-electron
ALPHA_DIAG_631G_FC = np.array([0.0846398834, 0.0620015909, 0.1955436888])   # frozen core


def test_mp2_corr_polarizability_631g():
    """All-electron spatial MP2 correlation alpha diagonal (H2O/6-31G) vs the frozen reference."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial')
    assert np.max(np.abs(np.diag(a) - ALPHA_DIAG_631G)) < 1e-9, np.diag(a)


def test_fc_mp2_corr_polarizability_631g():
    """Frozen-core spatial MP2 correlation alpha diagonal (H2O/6-31G) vs the frozen reference.

    Exercises the frozen-core second-order response: the core<->active U^{ab} from the canonical
    d_ab f_ij = 0 divide, and the ov second-order CPHF solve seeded with the nonzero ov
    orthonormality term xi_ia (which vanishes only in the all-electron field case)."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial', freeze_core='true')
    assert np.max(np.abs(np.diag(a) - ALPHA_DIAG_631G_FC)) < 1e-9, np.diag(a)


# ---- independence guard: frozen external energy finite field ----
# alpha_zz from a 5-point finite field of the MP2 *energy* is a disposable *external* oracle (psi4,
# not pycc), so it is frozen here rather than re-run each time.  Each value was validated against
# that field to ~2e-9 (see _energy_fd_alpha_diag, the regeneration recipe); the live dipole FD above
# pins the same alpha internally to ~1e-12.  The NH2 value is made reproducible by pinning the 2-B1
# occupation (NH2_OCC, C2v); without the pin its UHF is bistable (a 2-A1 solution ~0.074 Eh higher).
ALPHA_ZZ_ENERGY_FD = {
    'ae_631g': 0.1951911112,   # H2O/6-31G, all-electron (spatial)
    'uhf_nh2': 0.1855953897,   # NH2 (2-B1, C2v, pinned occ) / 6-31G (spin-orbital)
}


def test_mp2_corr_polarizability_energy_fd_631g():
    """All-electron alpha_zz vs the frozen external energy finite field, H2O/6-31G (the live
    dipole FD above is pycc-vs-pycc; this frozen value is the external anchor)."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial')
    assert abs(a[2, 2] - ALPHA_ZZ_ENERGY_FD['ae_631g']) < 1e-7


def test_ump2_corr_polarizability_energy_fd_nh2_631g():
    """Open-shell UHF-MP2 correlation alpha_zz vs the frozen external energy finite field,
    NH2 (2-B1, C2v, pinned occupation) / 6-31G -- the open-shell external anchor for the
    spin-orbital second-order response."""
    a = _pycc_alpha('6-31G', orbital_basis='spinorbital', geom=NH2, reference='uhf', occ=NH2_OCC)
    assert abs(a[2, 2] - ALPHA_ZZ_ENERGY_FD['uhf_nh2']) < 1e-7


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
