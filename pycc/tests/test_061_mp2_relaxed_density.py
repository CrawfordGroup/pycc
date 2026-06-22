"""
MP2 relaxed (orbital-response) one-particle density and analytic nuclear gradient, both
the spin-orbital and the spin-adapted (closed-shell RHF) paths; docs/DERIVATIVES_PLAN_2026-06.md.

The relaxed density adds the orbital-relaxation (Z-vector) contribution to the
unrelaxed MP2 correlation density; the gradient assembles it with the cumulant 2-PDM
and energy-weighted density against the skeleton derivative integrals. Both follow the
CC gradient formulation (Gauss, Stanton & Bartlett, JCP 95, 2623 (1991)).

Validation:
  * relaxed MP2 dipole -Tr(D_relaxed mu) vs a 5-point finite field of (E_MP2 - E_SCF);
  * analytic MP2 gradient vs psi4.gradient('mp2');
  * keystone: the spin-adapted and spin-orbital gradients agree on a closed shell.
"""

import psi4
import pycc
import numpy as np

WATER = """
O
H 1 0.96
H 1 0.96 2 104.5
"""


def _ff_corr_dipole(geom, basis, F=0.0005):
    """Relaxed correlation mu_z by a 5-point finite field of (E_MP2 - E_SCF)."""
    def e(model, Fz):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(geom)
        opt = {'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv',
               'freeze_core': 'false', 'e_convergence': 1e-12, 'd_convergence': 1e-12}
        if Fz:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole',
                        'perturb_dipole': [0.0, 0.0, Fz]})
        psi4.set_options(opt)
        return psi4.energy(model)

    def mu(model):
        return -(-e(model, 2 * F) + 8 * e(model, F)
                 - 8 * e(model, -F) + e(model, -2 * F)) / (12 * F)
    return mu('mp2') - mu('scf')


def _pycc_corr_dipole(geom, basis, orbital_basis='spinorbital'):
    """PyCC relaxed-MP2 electronic correlation mu_z (spin-orbital or spin-adapted)."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    D = mp.mp2_relaxed_opdm()
    return -np.einsum('pq,pq->', D, np.asarray(mp.H.mu[2]))


def test_mp2_relaxed_dipole_631g():
    """Relaxed spin-orbital MP2 dipole (mu_z) vs finite field, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G')
               - _ff_corr_dipole(geom, '6-31G')) < 1e-8


def test_sa_mp2_relaxed_dipole_631g():
    """Relaxed spin-adapted (closed-shell RHF) MP2 dipole vs finite field, H2O/6-31G."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G', orbital_basis='spatial')
               - _ff_corr_dipole(geom, '6-31G')) < 1e-8


def test_mp2_relaxed_dipole_ccpvdz():
    """Relaxed MP2 dipole vs finite field, H2O/cc-pVDZ (C2v: polarization functions
    and A2-irrep MOs). Exercises symmetry-adapted MOs in the relaxed density."""
    assert abs(_pycc_corr_dipole(WATER, 'cc-pVDZ')
               - _ff_corr_dipole(WATER, 'cc-pVDZ')) < 1e-8


def _pycc_gradient(geom, basis, orbital_basis='spinorbital'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp.gradient()


def _psi4_mp2_gradient(geom, basis):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv',
                      'freeze_core': 'false', 'e_convergence': 1e-12,
                      'd_convergence': 1e-12})
    return np.asarray(psi4.gradient('mp2'))


def test_mp2_gradient_631g():
    """Spin-orbital MP2 analytic nuclear gradient vs Psi4, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G')
                         - _psi4_mp2_gradient(geom, '6-31G'))) < 1e-8


def test_sa_mp2_gradient_631g():
    """Spin-adapted (closed-shell RHF) MP2 gradient vs Psi4, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G', orbital_basis='spatial')
                         - _psi4_mp2_gradient(geom, '6-31G'))) < 1e-8


def test_mp2_gradient_spatial_vs_so_631g():
    """Keystone: the spin-adapted and spin-orbital MP2 gradients agree on a closed shell."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G', orbital_basis='spatial')
                         - _pycc_gradient(geom, '6-31G', orbital_basis='spinorbital'))) < 1e-10


def test_mp2_gradient_ccpvdz():
    """Spin-orbital MP2 gradient vs Psi4, H2O/cc-pVDZ (C2v: polarization + A2-irrep MOs)."""
    assert np.max(np.abs(_pycc_gradient(WATER, 'cc-pVDZ')
                         - _psi4_mp2_gradient(WATER, 'cc-pVDZ'))) < 1e-8


def test_sa_mp2_gradient_ccpvdz():
    """Spin-adapted MP2 gradient vs Psi4, H2O/cc-pVDZ (C2v: polarization + A2-irrep MOs)."""
    assert np.max(np.abs(_pycc_gradient(WATER, 'cc-pVDZ', orbital_basis='spatial')
                         - _psi4_mp2_gradient(WATER, 'cc-pVDZ'))) < 1e-8
