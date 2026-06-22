"""
Spin-orbital Hartree-Fock molecular (nuclear) Hessian, HFwfn.hessian with
orbital_basis='spinorbital'; docs/DERIVATIVES_PLAN_2026-06.md.

The spin-orbital form of the CPHF Hessian: the second-derivative skeleton integrals plus
the spin-orbital nuclear CPHF response (shared cache), with the closed-shell prefactors
halved for singly occupied spin orbitals.

Note on the cross-spin Coulomb second derivative: Psi4's mo_tei_deriv2(A,B) does not
satisfy the integral's electron-exchange symmetry (pq|rs) = (rs|pq) term by term. For the
energy trace the same-spin terms absorb the resulting bra<->ket relabel (so RHF/SO-RHF are
already symmetric), but the UHF cross-spin Coulomb term sum_{i in a, j in b}(ii|jj) does
not -- it carries a spurious antisymmetric part. Derivatives.so_eri2 fixes this at the
integral level (symmetrizing the chemist deriv2 over the bra<->ket swap), so the assembled
Hessian is symmetric with no global symmetrization.

Validation:
  * keystone -- closed-shell RHF forced to spin orbitals == spatial RHF Hessian;
  * open-shell UHF Hessian vs psi4.hessian('scf').

ROHF is guarded (the nuclear response goes through CPHF.solve): NotImplementedError.
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
OH = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.8
units bohr
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


def _psi4_hessian(geom, basis, reference):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    return np.asarray(psi4.hessian('scf'))


def test_so_rhf_hessian_vs_spatial_631g():
    """Keystone (C1): closed-shell RHF forced to spin orbitals == spatial RHF Hessian."""
    wfn = _scf_wfn(WATER + "symmetry c1\n", '6-31G', 'rhf')
    h_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').hessian()
    h_spatial = pycc.HFwfn(wfn).hessian()
    assert np.max(np.abs(h_so - h_spatial)) < 1e-10


@pytest.mark.slow
def test_so_rhf_hessian_vs_spatial_ccpvdz():
    """Keystone (C2v: polarization functions + A2-irrep MOs): SO-RHF == spatial RHF."""
    wfn = _scf_wfn(WATER, 'cc-pVDZ', 'rhf')
    h_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').hessian()
    h_spatial = pycc.HFwfn(wfn).hessian()
    assert np.max(np.abs(h_so - h_spatial)) < 1e-10


def test_uhf_hessian_vs_psi4():
    """Open-shell UHF molecular Hessian (OH doublet) vs psi4.hessian('scf')."""
    wfn = _scf_wfn(OH, '6-31G', 'uhf')
    h_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').hessian()
    assert np.max(np.abs(h_so - h_so.T)) < 1e-10       # naturally symmetric (so_eri2 fix)
    assert np.max(np.abs(h_so - _psi4_hessian(OH, '6-31G', 'uhf'))) < 1e-8


def test_rohf_hessian_not_implemented():
    """ROHF Hessian goes through the (unsupported) ROHF CPHF response: NotImplementedError."""
    wfn = _scf_wfn(OH, '6-31G', 'rohf')
    with pytest.raises(NotImplementedError):
        pycc.HFwfn(wfn, orbital_basis='spinorbital').hessian()
