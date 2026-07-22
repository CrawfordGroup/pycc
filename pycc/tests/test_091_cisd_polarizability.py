"""
test_091_cisd_polarizability.py

CISD static electric-dipole polarizability (correlation contribution),
alpha_corr[a,b] = -d^2 E_corr / dF_a dF_b, omega = 0 - cross-checks the two
implementation routes against each other (no finite-difference oracle here.

Routes (CIwfn.polarizability(route=...)):
  * '2n+1'     -- base-driven (CorrelatedDerivs.polarizability), from
                  CIderiv's _unrelaxed_densities/_perturbed_unrelaxed_densities
                  hooks. Confirmed via the energy-closing identity and
                  consistency with dipole_derivatives/hessian.
  * 'explicit' -- independent cross-check reusing the SAME T0-T5 / Z-vector
                  machinery as hessian() (Yamaguchi Ch. 18) with field-field
                  skeleton integrals. 
                  This test is exactly the check needed to establish whether
                  it agrees with the trusted '2n+1' route.

Both routes eliminate the second-order CPHF response U^{fg} via the
Z-vector and never compute it, so if they are both implemented correctly,
agreement is a genuine algebraic identity, not a numerical coincidence.
"""

import psi4
import pycc
import numpy as np


WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
units bohr
no_com
no_reorient
"""


def _ciwfn(basis='6-31G'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(WATER)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': False, 'e_convergence': 1e-14, 'd_convergence': 1e-14})
    _, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD')
    ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=150)
    return ci


def test_cisd_polarizability_explicit_vs_2n1_631g():
    """The two polarizability routes agree with each other, H2O/6-31G,
    every element. '2n+1' is the trusted route (energy-closing identity +
    consistency with dipole_derivatives/hessian); this test checks whether
    'explicit' reproduces it."""
    ci = _ciwfn('6-31G')
    a_2n1 = np.asarray(ci.polarizability(route='2n+1'))
    a_exp = np.asarray(ci.polarizability(route='explicit'))
    diff = np.max(np.abs(a_2n1 - a_exp))
    assert diff < 1e-9, (diff, a_2n1, a_exp)


def test_cisd_polarizability_symmetric_631g():
    """Both routes give a symmetric tensor (alpha_ab = alpha_ba) - an
    FD-free structural check, run for both routes independently."""
    ci = _ciwfn('6-31G')
    for route in ('2n+1', 'explicit'):
        alpha = np.asarray(ci.polarizability(route=route))
        assert np.max(np.abs(alpha - alpha.T)) < 1e-8, (route, alpha)


def test_cisd_polarizability_off_diagonal_zero_by_symmetry_631g():
    """H2O/6-31G is C2v (z along C2, molecular plane yz): the correlation
    polarizability must be diagonal. Checked for both routes."""
    ci = _ciwfn('6-31G')
    for route in ('2n+1', 'explicit'):
        alpha = np.asarray(ci.polarizability(route=route))
        assert abs(alpha[0, 1]) < 1e-7, (route, 'xy', alpha[0, 1])
        assert abs(alpha[0, 2]) < 1e-7, (route, 'xz', alpha[0, 2])
        assert abs(alpha[1, 2]) < 1e-7, (route, 'yz', alpha[1, 2])


def test_cisd_polarizability_guards():
    """An unknown route raises ValueError rather than silently doing
    something wrong."""
    import pytest
    ci = _ciwfn('6-31G')
    with pytest.raises(ValueError):
        ci.polarizability(route='bogus')
