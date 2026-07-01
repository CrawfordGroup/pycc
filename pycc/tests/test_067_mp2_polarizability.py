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

Validation (all-electron):
  * correlation alpha diagonal vs a 5-point finite field of (E_MP2 - E_SCF);
  * keystone: spin-orbital == spin-adapted (closed-shell RHF) correlation alpha.

Oracle note: the finite field is of the MP2 *energy* (not any analytic gradient), so it is
unaffected by Psi4's frozen-core MP2 gradient bug. Frozen-core second derivatives are not
yet correct (a residual ~1e-5 remains in the second-order machinery) -- see the xfail below.
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


def _ff_corr_alpha_diag(basis, axis, F=0.002, freeze_core='false'):
    """alpha_corr_(axis,axis) = -d^2(E_MP2 - E_SCF)/dF^2 by a 5-point finite field."""
    def e(model, Fval):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(WATER)
        d = [0.0, 0.0, 0.0]
        d[axis] = Fval
        opt = {'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv',
               'freeze_core': freeze_core, 'e_convergence': 1e-13, 'd_convergence': 1e-13}
        if Fval:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': d})
        psi4.set_options(opt)
        return psi4.energy(model)

    def d2(model):
        return (-e(model, 2 * F) + 16 * e(model, F) - 30 * e(model, 0.0)
                + 16 * e(model, -F) - e(model, -2 * F)) / (12 * F * F)
    return -(d2('mp2') - d2('scf'))


def _pycc_alpha(basis, orbital_basis='spinorbital', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(WATER)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return np.asarray(mp.polarizability())


def test_mp2_corr_polarizability_diag_631g():
    """Spatial (closed-shell RHF) MP2 correlation alpha, all three diagonal components
    vs a 5-point finite field of (E_MP2 - E_SCF), H2O/6-31G."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial')
    for axis in range(3):
        assert abs(a[axis, axis] - _ff_corr_alpha_diag('6-31G', axis)) < 1e-7


def test_so_mp2_corr_polarizability_diag_631g():
    """Spin-orbital MP2 correlation alpha diagonal vs finite field, H2O/6-31G."""
    a = _pycc_alpha('6-31G', orbital_basis='spinorbital')
    for axis in range(3):
        assert abs(a[axis, axis] - _ff_corr_alpha_diag('6-31G', axis)) < 1e-7


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


@pytest.mark.xfail(reason="frozen-core MP2 second derivative has a ~1e-5 residual in the "
                          "second-order machinery (see bug hunt); all-electron is exact",
                   strict=True)
def test_fc_mp2_corr_polarizability_diag_631g():
    """Frozen-core spatial MP2 correlation alpha_zz vs finite field -- KNOWN FAILING
    (documents the open frozen-core second-derivative bug)."""
    a = _pycc_alpha('6-31G', orbital_basis='spatial', freeze_core='true')
    assert abs(a[2, 2] - _ff_corr_alpha_diag('6-31G', 2, freeze_core='true')) < 1e-7
