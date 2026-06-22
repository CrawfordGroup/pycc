"""
Spin-orbital Hartree-Fock static dipole polarizability (HFwfn, orbital_basis=
'spinorbital') -- the open-shell-capable electric-field CPHF response;
docs/DERIVATIVES_PLAN_2026-06.md.

    alpha_ab = -2 sum_ia mu^a_ia U^b_ia,   G U^b = -mu^b   (spin orbitals)

A pure field response: the perturbation does not move the basis functions, so there is
no overlap/derivative-integral (Pulay) contribution -- the simplest second-derivative
property.

Validation:
  * keystone -- closed-shell RHF forced to spin orbitals == spatial RHF == Psi4 CPHF;
  * open-shell UHF vs Psi4 CPHF (Psi4's iterative UHF-CPHF sets the ~1e-6 agreement).

ROHF is intentionally not supported: the semicanonical spin-orbital response is UHF-like
and does not reproduce the restricted ROHF response (the ROHF Brillouin/orbital-rotation
conventions are not uniquely defined). The CPHF response raises NotImplementedError.
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
H 0.0 0.0 1.0
symmetry c1
"""


def _scf_wfn(geom, basis, reference):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _psi4_polarizability(geom, basis, reference):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    psi4.properties('scf', properties=['DIPOLE_POLARIZABILITIES'])
    ax = 'XYZ'
    return np.array([[psi4.variable(f'DIPOLE POLARIZABILITY {ax[i]}{ax[j]}')
                      for j in range(3)] for i in range(3)])


def test_so_rhf_polarizability_vs_spatial():
    """Keystone (cc-pVDZ, C2v): closed-shell RHF forced to spin orbitals == spatial
    RHF polarizability (== Psi4 CPHF)."""
    wfn = _scf_wfn(WATER, 'cc-pVDZ', 'rhf')
    a_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').polarizability()
    a_spatial = pycc.HFwfn(wfn).polarizability()
    assert np.max(np.abs(a_so - a_spatial)) < 1e-11
    assert np.max(np.abs(a_so - _psi4_polarizability(WATER, 'cc-pVDZ', 'rhf'))) < 1e-9


def test_uhf_polarizability_vs_psi4():
    """Open-shell UHF polarizability (OH doublet) vs Psi4 CPHF (Psi4's iterative
    UHF-CPHF convergence sets the agreement; the direct SO solve is exact)."""
    wfn = _scf_wfn(OH, '6-31G', 'uhf')
    a_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').polarizability()
    assert np.max(np.abs(a_so - _psi4_polarizability(OH, '6-31G', 'uhf'))) < 1e-6


def test_rohf_polarizability_not_implemented():
    """ROHF CPHF response is intentionally unsupported (restricted-orbital response /
    non-unique Brillouin conventions): the response solve raises NotImplementedError."""
    wfn = _scf_wfn(OH, '6-31G', 'rohf')
    with pytest.raises(NotImplementedError):
        pycc.HFwfn(wfn, orbital_basis='spinorbital').polarizability()
