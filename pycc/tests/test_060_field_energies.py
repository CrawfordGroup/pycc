"""
Spin-orbital finite-field CC energies (static external electric-dipole field);
docs/archive/ENHANCEMENT_PLAN_2026-06.md.

A static field is added to the Fock matrix post-SCF (V = -F*mu_z, fixed orbitals),
making it non-canonical. CCSD handles this through its Fock intermediates; CC3
additionally needs the iterative [V,T3] coupling, hence store_triples=True. A
5-point finite difference of the correlation energy w.r.t. the field strength then
gives the (correlation) dipole and polarizability.

Validation (matches socc tests 010/011):
  * CCSD mu_z and alpha_zz (H2O/STO-3G) vs CFOUR.
  * CC3 alpha_zz vs an independent Dalton reference.
"""

import psi4
import pycc
import numpy as np
import pytest

# socc moldict["H2O"] geometry (Cartesian, bohr).
H2O = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""

FIELD = 0.0001  # finite-difference field strength (a.u.)


def _scf_wfn():
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'mp2_type': 'conv',
                      'freeze_core': 'false', 'reference': 'rhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _findiff(wfn, model, store_triples):
    """5-point finite-difference mu_z and alpha_zz from the field-on CC correlation
    energy. The field-free SCF (wfn) is shared; the field is applied in CCwfn."""
    def E(strength):
        kw = dict(frozen_core=False, model=model, orbital_basis='spinorbital',
                  store_triples=store_triples)
        if strength != 0.0:
            kw.update(field=True, field_strength=strength, field_axis='Z')
        return pycc.CCwfn(wfn, **kw).solve_cc(e_conv=1e-12, r_conv=1e-12)

    F = FIELD
    e0, ep, e2p, em, e2m = E(0.0), E(F), E(2 * F), E(-F), E(-2 * F)
    mu_z = -(-e2p + 8 * ep - 8 * em + e2m) / (12 * F)
    alpha_zz = -(-e2p + 16 * ep - 30 * e0 + 16 * em - e2m) / (12 * F * F)
    return mu_z, alpha_zz


def test_ccsd_field_findiff():
    """CCSD finite-field dipole + polarizability (H2O/STO-3G) vs CFOUR (socc
    test_010). Exercises the non-canonical-Fock CCSD path under an applied field."""
    mu_z, alpha_zz = _findiff(_scf_wfn(), 'CCSD', store_triples=False)
    assert abs(mu_z - 0.0724134575) < 1e-7
    assert abs(alpha_zz - 2.9745913) < 1e-6


def test_cc3_field_findiff():
    """CC3 finite-field polarizability (H2O/STO-3G) vs Dalton (socc test_011).
    Requires store_triples=True -- the field couples the triples ([V,T3])."""
    _, alpha_zz = _findiff(_scf_wfn(), 'CC3', store_triples=True)
    assert abs(alpha_zz - 2.9989468) < 1e-6


def test_cc3_field_requires_store_triples():
    """Finite-field CC3 without stored triples is rejected: the non-canonical Fock
    couples the triples, so the full T3 must be retained."""
    with pytest.raises(Exception):
        pycc.CCwfn(_scf_wfn(), frozen_core=False, model='CC3',
                   orbital_basis='spinorbital', field=True, field_strength=0.001,
                   field_axis='Z')
