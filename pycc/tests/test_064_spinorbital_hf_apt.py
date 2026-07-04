"""
Spin-orbital Hartree-Fock atomic polar tensors (APTs / nuclear dipole derivatives),
HFwfn.dipole_derivatives with orbital_basis='spinorbital'; docs/DERIVATIVES_PLAN_2026-06.md.

A mixed field-nuclear second derivative: it uses the spin-orbital *nuclear* CPHF response
(the deferred nuclear-RHS layer, now built) plus the spin-orbital dipole/overlap
derivative integrals. Singly occupied spin orbitals halve the closed-shell prefactors::

    d mu_a / d X_Ab = Z_A delta_ab + sum_i (d mu_a / d X_Ab)_ii
                    - sum_ik S^X_ki (mu_a)_ik + 2 sum_ia U^X_ia (mu_a)_ia

Validation:
  * keystone -- closed-shell RHF forced to spin orbitals == spatial RHF APT;
  * open-shell UHF APT (C2v) vs a finite-difference-validated frozen reference.

ROHF is guarded (the nuclear response goes through CPHF.solve): raises NotImplementedError.
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
# OH doublet in explicit bohr coordinates (for clean finite differencing).
OH_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])


def _oh_geom(coords):
    # OH along z in its computational point group (C2v -- no 'symmetry c1'), so the 2-Pi reference is
    # symmetry-adapted and reproducible; a C1 reference would break the pi_x/pi_y degeneracy.
    s = "0 2\n" + "\n".join(f"{el} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}"
                            for el, c in zip(['O', 'H'], coords))
    return s + "\nunits bohr\n"


# OH UHF APT (6-31G, C2v), frozen reference [3*atom + beta, alpha].  Validated (once) against a
# 7-point O(h^6) central finite difference of the SCF dipole under nuclear displacement, agreeing to
# ~2e-12:  apt[A, beta, alpha] = d mu_alpha / d X_Ab.
# Run in C2v so the reference is symmetry-adapted and reproducible (~1e-13); the perpendicular
# components (x = y here) would be arbitrary under a C1 symmetry-broken reference.
UHF_APT_REF = np.array([[-0.4682227123,  0.0,          0.0],
                        [ 0.0,          -0.4682227123, 0.0],
                        [ 0.0,           0.0,         -0.0546138293],
                        [ 0.4682227123,  0.0,          0.0],
                        [ 0.0,           0.4682227123, 0.0],
                        [ 0.0,           0.0,          0.0546138293]])


def _scf_wfn(geom, basis, reference):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'reference': reference,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def test_so_rhf_apt_vs_spatial_631g():
    """Keystone (C1): closed-shell RHF forced to spin orbitals == spatial RHF APT."""
    wfn = _scf_wfn(WATER + "symmetry c1\n", '6-31G', 'rhf')
    apt_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').dipole_derivatives()
    apt_spatial = pycc.HFwfn(wfn).dipole_derivatives()
    assert np.max(np.abs(apt_so - apt_spatial)) < 1e-10


def test_so_rhf_apt_vs_spatial_ccpvdz():
    """Keystone (C2v: polarization functions + A2-irrep MOs): SO-RHF == spatial RHF APT."""
    wfn = _scf_wfn(WATER, 'cc-pVDZ', 'rhf')
    apt_so = pycc.HFwfn(wfn, orbital_basis='spinorbital').dipole_derivatives()
    apt_spatial = pycc.HFwfn(wfn).dipole_derivatives()
    assert np.max(np.abs(apt_so - apt_spatial)) < 1e-10


def test_uhf_apt():
    """Open-shell UHF APT (OH doublet / 6-31G, C2v) vs the finite-difference-validated frozen
    reference (see UHF_APT_REF above)."""
    wfn = _scf_wfn(_oh_geom(OH_COORDS), '6-31G', 'uhf')
    apt = np.asarray(pycc.HFwfn(wfn, orbital_basis='spinorbital').dipole_derivatives()).reshape(-1, 3)
    assert np.max(np.abs(apt - UHF_APT_REF)) < 1e-8


def test_rohf_apt_not_implemented():
    """ROHF APT goes through the (unsupported) ROHF CPHF response: NotImplementedError."""
    wfn = _scf_wfn(_oh_geom(OH_COORDS), '6-31G', 'rohf')
    with pytest.raises(NotImplementedError):
        pycc.HFwfn(wfn, orbital_basis='spinorbital').dipole_derivatives()
