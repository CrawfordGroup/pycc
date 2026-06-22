"""
Spin-orbital Hartree-Fock analytic nuclear gradient (HFwfn, orbital_basis='spinorbital')
-- the open-shell-capable form of the CPHF-free RHF gradient;
docs/DERIVATIVES_PLAN_2026-06.md.

    dE/dX = sum_i h^x_ii + 1/2 sum_ij <ij||ij>^x - sum_i eps_i S^x_ii + dV_NN/dX   (i,j occ)

Validation:
  * keystone -- a closed-shell RHF reference forced to spin orbitals gives the same
    gradient as the spatial RHF path (and Psi4); and
  * open-shell UHF and ROHF gradients vs Psi4.
"""

import psi4
import pycc
import numpy as np


WATER = """
O
H 1 0.96
H 1 0.96 2 104.5
"""
OH = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.0
symmetry c1
"""


def _scf_wfn(geom, basis, reference):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _psi4_scf_gradient(geom, basis, reference):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    return np.asarray(psi4.gradient('scf'))


def test_so_rhf_gradient_vs_spatial_631g():
    """Keystone (C1): closed-shell RHF forced to spin orbitals == spatial RHF gradient."""
    geom = WATER + "symmetry c1\n"
    wfn = _scf_wfn(geom, '6-31G', 'rhf')
    g_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').gradient()
    g_spatial = pycc.HFwfn(wfn).gradient()
    assert np.max(np.abs(g_so - g_spatial)) < 1e-11
    assert np.max(np.abs(g_so - _psi4_scf_gradient(geom, '6-31G', 'rhf'))) < 1e-10


def test_so_rhf_gradient_vs_spatial_ccpvdz():
    """Keystone (C2v: polarization functions + A2-irrep MOs): SO-RHF == spatial RHF."""
    wfn = _scf_wfn(WATER, 'cc-pVDZ', 'rhf')
    g_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').gradient()
    g_spatial = pycc.HFwfn(wfn).gradient()
    assert np.max(np.abs(g_so - g_spatial)) < 1e-11
    assert np.max(np.abs(g_so - _psi4_scf_gradient(WATER, 'cc-pVDZ', 'rhf'))) < 1e-10


def test_uhf_gradient_vs_psi4():
    """Open-shell UHF gradient (OH doublet) vs Psi4."""
    wfn = _scf_wfn(OH, '6-31G', 'uhf')
    g_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').gradient()
    assert np.max(np.abs(g_so - _psi4_scf_gradient(OH, '6-31G', 'uhf'))) < 1e-10


def test_rohf_gradient_vs_psi4():
    """Open-shell ROHF gradient (OH doublet, semicanonical orbitals) vs Psi4."""
    wfn = _scf_wfn(OH, '6-31G', 'rohf')
    g_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').gradient()
    assert np.max(np.abs(g_so - _psi4_scf_gradient(OH, '6-31G', 'rohf'))) < 1e-10
