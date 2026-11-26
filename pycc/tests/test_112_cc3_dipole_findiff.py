"""
CC3 unrelaxed one-particle density vs finite-field CC3 with the iterative full T3.

A static dipole field added to the Fock matrix (fixed orbitals) makes it non-canonical, which
couples the triples ([V,T3]); the finite-field CC3 energy then needs the iterative, stored-T3
spin-orbital path (field=True, store_triples=True).  Its 5-point derivative w.r.t. the field is
the unrelaxed CC3 correlation dipole, which must equal the analytic contraction of the spatial
CC3 Lambda density in rtcc.dipole (opdm with bare integrals, opdm_cc3 with T1-transformed ones).

Also covers H2O/STO-3G (nv < no), where build_cc3_Dov and build_cc3_Doo previously allocated
intermediates with the wrong shape, and the absence of a CC3 Lagrangian in RT-CC3 (no CC3 two-particle density).
"""

import numpy as np
import psi4
import pycc
import pytest
from pycc.rt.integrators import rk4
from pycc.rt.lasers import sine_square_laser

# H2O, frame locked (as in test_034).
GEOM = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
no_com
no_reorient
"""


def _rtcc3(wfn):
    cc = pycc.CCwfn(wfn, model='CC3')
    cc.solve_cc(1e-12, 1e-12, 200)
    lam = pycc.cclambda(cc, pycc.cchbar(cc))
    lam.solve_lambda(1e-12, 1e-12, 200)
    dens = pycc.ccdensity(cc, lam)
    rt = pycc.rtcc(cc, lam, dens, sine_square_laser(1.0, 2.87, 5.0))   # V(0) = 0
    y0 = rt.collect_amps(cc.t1, cc.t2, lam.l1, lam.l2, 0).astype('complex128')
    return cc, rt, y0


def _findiff_dipole_z(wfn, h=1e-4):
    """5-point finite-field CC3 correlation dipole mu_z (fixed orbitals, iterative full T3)."""
    def E(s):
        return pycc.CCwfn(wfn, model='CC3', orbital_basis='spinorbital', store_triples=True,
                          field=True, field_strength=s, field_axis='Z').solve_cc(1e-12, 1e-12)
    return -(-E(2 * h) + 8 * E(h) - 8 * E(-h) + E(-2 * h)) / (12 * h)


def _check_dipole(wfn):
    cc, rt, y0 = _rtcc3(wfn)
    mu_an = rt.dipole(*rt.extract_amps(y0)[:4])
    mu_ff = _findiff_dipole_z(wfn)
    assert abs(mu_an[2] - mu_ff) < 1e-9, (mu_an[2], mu_ff)
    psi4.core.clean()


def test_cc3_dipole_findiff_sto3g(rhf_wfn):
    """H2O/STO-3G (nv < no): analytic CC3 dipole equals the finite-field value."""
    _check_dipole(rhf_wfn(GEOM, "STO-3G", freeze_core="false"))


def test_cc3_dipole_findiff_631g(rhf_wfn):
    """H2O/6-31G (nv > no): analytic CC3 dipole equals the finite-field value."""
    _check_dipole(rhf_wfn(GEOM, "6-31G", freeze_core="false"))


def test_rtcc3_no_lagrangian(rhf_wfn):
    """There is no CC3 two-particle density: rtcc.lagrangian raises for CC3, and step()
    returns the dipole without an 'ecc' entry."""
    cc, rt, y0 = _rtcc3(rhf_wfn(GEOM, "STO-3G", freeze_core="false"))
    with pytest.raises(NotImplementedError):
        rt.lagrangian(0.0, *rt.extract_amps(y0)[:4])
    _, ret = rt.step(rk4(0.01), y0, 0.0)
    assert 'ecc' not in ret and {'mu_x', 'mu_y', 'mu_z'} <= set(ret)
    psi4.core.clean()
