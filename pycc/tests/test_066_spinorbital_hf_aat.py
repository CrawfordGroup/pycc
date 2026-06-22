"""
Spin-orbital Hartree-Fock atomic axial tensors (AATs), HFwfn.atomic_axial_tensors with
orbital_basis='spinorbital'; docs/DERIVATIVES_PLAN_2026-06.md.

The electronic magnetic-dipole vibrational transition moment (for VCD): the overlap of the
nuclear- and magnetic-field wavefunction derivatives, reusing the spin-orbital nuclear CPHF
response, the magnetic-field response (antisymmetric Hessian), and the nuclear half-
derivative overlaps. Singly occupied spin orbitals drop the closed-shell prefactor 2 -> 1.

Validation:
  * keystone -- closed-shell RHF forced to spin orbitals == spatial RHF AAT (which is
    validated against DALTON's analytic SCF AATs in test_050);
  * open-shell UHF -- there is NO prior open-shell UHF AAT implementation to compare to, so
    this only checks that the spin-orbital path runs and returns a sane (finite, real,
    nonzero) tensor; it shares the code path the closed-shell keystone validates.

ROHF is guarded (the responses go through CPHF.solve): NotImplementedError.
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
H2O2 = """
O   0.0000000000   1.3192641900  -0.0952542913
O  -0.0000000000  -1.3192641900  -0.0952542913
H   1.6464858700   1.6841036400   0.7620343300
H  -1.6464858700  -1.6841036400   0.7620343300
symmetry c1
units bohr
noreorient
no_com
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


def test_so_rhf_aat_vs_spatial_631g():
    """Keystone (C1): closed-shell RHF forced to spin orbitals == spatial RHF AAT."""
    wfn = _scf_wfn(WATER, '6-31G', 'rhf')
    a_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').atomic_axial_tensors()
    a_spatial = pycc.HFwfn(wfn).atomic_axial_tensors()
    assert np.max(np.abs(a_so - a_spatial)) < 1e-10


def test_so_rhf_aat_vs_spatial_h2o2():
    """Keystone on the canonical VCD molecule (H2O2, C1): SO-RHF == spatial RHF AAT (the
    spatial AAT is validated against DALTON in test_050)."""
    wfn = _scf_wfn(H2O2, 'sto-3g', 'rhf')
    a_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').atomic_axial_tensors()
    a_spatial = pycc.HFwfn(wfn).atomic_axial_tensors()
    assert np.max(np.abs(a_so - a_spatial)) < 1e-10


def test_uhf_aat_runs():
    """Open-shell UHF AAT (OH doublet): no prior implementation to validate against, so
    only check the spin-orbital path runs and returns a finite, real, nonzero tensor."""
    wfn = _scf_wfn(OH, '6-31G', 'uhf')
    aat = pycc.HFwfn(wfn, orbital_basis='spinorbital').atomic_axial_tensors()
    assert aat.shape == (2, 3, 3)
    assert np.all(np.isfinite(aat))
    assert np.isrealobj(aat)
    assert np.max(np.abs(aat)) > 1e-6


def test_rohf_aat_not_implemented():
    """ROHF AAT goes through the (unsupported) ROHF CPHF response: NotImplementedError."""
    wfn = _scf_wfn(OH, '6-31G', 'rohf')
    with pytest.raises(NotImplementedError):
        pycc.HFwfn(wfn, orbital_basis='spinorbital').atomic_axial_tensors()
