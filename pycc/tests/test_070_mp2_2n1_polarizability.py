"""
MP2 dipole polarizability via the 2n+1 route -- cross-check against the explicit route;
DERIVATIVES_PLAN_2026-06.md ("2n+1 second-derivative route"), mp2_2n1_perturbed.tex.

The 2n+1 route differentiates the relaxed-density gradient and uses only the first-order
perturbed *relaxed* density (MPwfn._so_perturbed_relaxed_opdm, which carries the perturbed
Z-vector z^x -- the same orbital Hessian as the gradient's Z-vector, perturbed RHS). No
second-order CPHF U^{ab}. For a field,

    alpha_ab = sum_pq d_a D_rel_pq (mu_b)_pq + sum_pq D_rel_pq [ (U^a).T mu_b + mu_b U^a ]_pq,

the second term being the MO dipole rotating under the field. This must reproduce the
explicit-route polarizability (test_067) to ~machine precision -- an independent cross-check
of both routes (Phase A/B of the 2n+1 plan). Spin-orbital / all-electron so far.
"""

import psi4
import pycc
import numpy as np
import pytest


WATER = """
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
"""


def _mpwfn(basis, orbital_basis='spinorbital', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(WATER)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp


def test_2n1_polarizability_vs_explicit_631g():
    """2n+1 == explicit correlation polarizability, all-electron, 6-31G, both spin paths."""
    for ob in ('spinorbital', 'spatial'):
        mp = _mpwfn('6-31G', orbital_basis=ob)
        a_2n1 = np.asarray(mp.polarizability(route='2n+1'))
        a_exp = np.asarray(mp.polarizability(route='explicit'))
        assert np.max(np.abs(a_2n1 - a_exp)) < 1e-11
        assert np.max(np.abs(a_2n1 - a_2n1.T)) < 1e-11        # symmetric


def test_2n1_polarizability_vs_explicit_fc_631g():
    """2n+1 == explicit correlation polarizability, frozen core, 6-31G, both spin paths.

    The frozen-core 2n+1 carries the perturbed core-active divide ``d_x P_co`` (a Sylvester
    derivative whose off-diagonal active-active ``d_x f`` coupling is essential -- a field
    leaves the active occupied space non-canonical) and the perturbed Z-vector over the full
    occupied space. The explicit route is validated against a finite-difference of the dipole
    (test_067), so this pins the 2n+1 frozen-core path to the same ground truth."""
    for ob in ('spinorbital', 'spatial'):
        mp = _mpwfn('6-31G', orbital_basis=ob, freeze_core='true')
        a_2n1 = np.asarray(mp.polarizability(route='2n+1'))
        a_exp = np.asarray(mp.polarizability(route='explicit'))
        assert np.max(np.abs(a_2n1 - a_exp)) < 1e-11
        assert np.max(np.abs(a_2n1 - a_2n1.T)) < 1e-11        # symmetric


def test_2n1_polarizability_vs_explicit_ccpvdz():
    """2n+1 == explicit correlation polarizability (cc-pVDZ), both spin paths."""
    for ob in ('spinorbital', 'spatial'):
        mp = _mpwfn('cc-pVDZ', orbital_basis=ob)
        a_2n1 = np.asarray(mp.polarizability(route='2n+1'))
        a_exp = np.asarray(mp.polarizability(route='explicit'))
        assert np.max(np.abs(a_2n1 - a_exp)) < 1e-11


def test_2n1_polarizability_so_vs_spatial_631g():
    """Keystone: 2n+1 spin-orbital == spin-adapted correlation polarizability (6-31G),
    all-electron and frozen core."""
    for fc in ('false', 'true'):
        a_so = np.asarray(_mpwfn('6-31G', 'spinorbital', fc).polarizability(route='2n+1'))
        a_sa = np.asarray(_mpwfn('6-31G', 'spatial', fc).polarizability(route='2n+1'))
        assert np.max(np.abs(a_so - a_sa)) < 1e-11


def test_2n1_polarizability_guards():
    """An unknown route raises ValueError."""
    with pytest.raises(ValueError):
        _mpwfn('6-31G').polarizability(route='bogus')
