"""
Spin-orbital MP2 relaxed (orbital-response) one-particle density;
docs/DERIVATIVES_PLAN_2026-06.md.

The relaxed density adds the orbital-relaxation (Z-vector) contribution to the
unrelaxed MP2 correlation density. The orbital-gradient Lagrangian and Z-vector
follow the spin-orbital CC gradient formulation (Gauss, Stanton & Bartlett, JCP 95,
2623 (1991)). It is validated by the relaxed MP2 dipole: the electronic correlation
dipole -Tr(D_relaxed mu) must equal a 5-point finite-field reference of
(E_MP2 - E_SCF) w.r.t. a z-dipole field (full orbital relaxation in the field).
"""

import psi4
import pycc
import numpy as np
import pytest

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


def _pycc_corr_dipole(geom, basis):
    """PyCC spin-orbital relaxed-MP2 electronic correlation mu_z."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis='spinorbital')
    mp.compute_energy()
    D = mp.mp2_relaxed_opdm()
    return -np.einsum('pq,pq->', D, np.asarray(mp.H.mu[2]))


def test_mp2_relaxed_dipole_631g():
    """Relaxed MP2 dipole (mu_z) vs finite field, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G')
               - _ff_corr_dipole(geom, '6-31G')) < 1e-8


@pytest.mark.slow
def test_mp2_relaxed_dipole_ccpvdz():
    """Relaxed MP2 dipole vs finite field, H2O/cc-pVDZ (C2v: polarization functions
    and A2-irrep MOs). Exercises symmetry-adapted MOs in the relaxed density."""
    assert abs(_pycc_corr_dipole(WATER, 'cc-pVDZ')
               - _ff_corr_dipole(WATER, 'cc-pVDZ')) < 1e-8
