"""
First-order perturbative diagonal Born-Oppenheimer correction
(MPderiv.dboc_mp1): DBOC(MP1) of Tajti, Szalay, and Gauss, J. Chem.
Phys. 127, 014102 (2007), Eq. (30) - the SCF DBOC plus the first-order
correlation correction, linear in the MP2 doubles (no second-order
amplitudes) - cc-pVDZ, at the exact HEAT geometries of Gauss et al.,
J. Chem. Phys. 125, 144111 (2006).

References are CFOUR v2.1 values (the DBOC(MP1) totals printed by
CALC=SCF, DBOC=ON runs), agreement ~5e-6 cm^-1.
"""

import psi4
import pycc

HARTREE_CM = 219474.6313632

GEOMS = {
    'h2': """
0 1
H
H 1 0.74186
symmetry c1
""",
    'hf': """
0 1
F
H 1 0.91516
symmetry c1
""",
    'h2o': """
0 1
O
H 1 0.95623
H 1 0.95623 2 104.25
symmetry c1
""",
}


def _mp1_wfn(geom, fc=False):
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk', 'freeze_core': 'true' if fc else 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    e, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _mp1_dboc_cm(geom, fc=False):
    return pycc.MPderiv(pycc.MPwfn(_mp1_wfn(geom, fc))).dboc_mp1() * HARTREE_CM


def test_mp1_dboc_h2():
    assert abs(_mp1_dboc_cm(GEOMS['h2']) - 105.955494) < 0.001


def test_mp1_dboc_hf():
    assert abs(_mp1_dboc_cm(GEOMS['hf']) - 612.580094) < 0.001


def test_mp1_dboc_h2o():
    assert abs(_mp1_dboc_cm(GEOMS['h2o']) - 610.318160) < 0.001


def test_mp1_dboc_hf_frozen_core():
    assert abs(_mp1_dboc_cm(GEOMS['hf'], fc=True) - 611.956979) < 0.001


def test_mp1_dboc_h2o_frozen_core():
    assert abs(_mp1_dboc_cm(GEOMS['h2o'], fc=True) - 609.630785) < 0.001


def test_mp1_dboc_perturbed_mo_gauge_invariance():
    """DBOC(MP1) is invariant to the perturbed-MO gauge, all-electron and frozen core."""
    for geom in ('hf', 'h2o'):
        for fc in (False, True):
            wfn = _mp1_wfn(GEOMS[geom], fc)
            nc = pycc.MPderiv(pycc.MPwfn(wfn)).dboc_mp1()
            cd = pycc.MPderiv(pycc.MPwfn(wfn))
            cd._gauge_override = 'canonical'
            assert abs(nc - cd.dboc_mp1()) < 1e-9 * max(1.0, abs(nc)), (geom, fc, nc, cd.dboc_mp1())
