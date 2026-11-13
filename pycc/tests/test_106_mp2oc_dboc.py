"""Order-consistent (Def-B / Gauss) MP2 diagonal Born-Oppenheimer correction
(MPderiv.dboc_mp2_oc): DBOC(MP2) of Tajti, Szalay, and Gauss, J. Chem. Phys. 127,
014102 (2007), Eq. (31) - the definition CFOUR's CALC=MP2, DBOC=ON evaluates -
cc-pVDZ, at the exact HEAT geometries of Gauss et al., J. Chem. Phys. 125, 144111
(2006).

References are CFOUR v2.1 values (the DBOC totals printed by CALC=MP2, DBOC=ON
runs); agreement ~1e-5 cm^-1.  Companion to test_103 (DBOC(MP1)).
"""

import psi4
import pycc

HARTREE_CM = 219474.6313632

GEOMS = {
    'h2': "0 1\nH\nH 1 0.74186\nsymmetry c1\n",
    'hf': "0 1\nF\nH 1 0.91516\nsymmetry c1\n",
    'h2o': "0 1\nO\nH 1 0.95623\nH 1 0.95623 2 104.25\nsymmetry c1\n",
}
CFOUR_MP2 = {'h2': 109.297919, 'hf': 616.175575, 'h2o': 615.278819}


CFOUR_MP2_FC = {'hf': 615.605680, 'h2o': 614.666109}


def _mp2oc_dboc_cm(geom, fc=False):
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk', 'freeze_core': 'true' if fc else 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    e, wfn = psi4.energy('scf', return_wfn=True)
    return pycc.MPderiv(pycc.MPwfn(wfn)).dboc_mp2_oc() * HARTREE_CM


def test_mp2oc_dboc_h2():
    assert abs(_mp2oc_dboc_cm(GEOMS['h2']) - CFOUR_MP2['h2']) < 1e-5


def test_mp2oc_dboc_hf():
    assert abs(_mp2oc_dboc_cm(GEOMS['hf']) - CFOUR_MP2['hf']) < 1e-5


def test_mp2oc_dboc_h2o():
    assert abs(_mp2oc_dboc_cm(GEOMS['h2o']) - CFOUR_MP2['h2o']) < 1e-5


def test_mp2oc_dboc_hf_frozen_core():
    assert abs(_mp2oc_dboc_cm(GEOMS['hf'], fc=True) - CFOUR_MP2_FC['hf']) < 1e-4


def test_mp2oc_dboc_h2o_frozen_core():
    assert abs(_mp2oc_dboc_cm(GEOMS['h2o'], fc=True) - CFOUR_MP2_FC['h2o']) < 1e-4


def test_mp2oc_dboc_perturbed_mo_gauge_invariance():
    """Order-consistent MP2 DBOC is invariant to the perturbed-MO gauge, all-electron and frozen core."""
    for fc in (False, True):
        psi4.core.clean()
        psi4.set_memory('2 GB')
        psi4.core.set_output_file('output.dat', False)
        psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk', 'freeze_core': 'true' if fc else 'false',
                          'e_convergence': 1e-12, 'd_convergence': 1e-12})
        psi4.geometry(GEOMS['h2o'])
        e, wfn = psi4.energy('scf', return_wfn=True)
        nc = pycc.MPderiv(pycc.MPwfn(wfn)).dboc_mp2_oc()
        cd = pycc.MPderiv(pycc.MPwfn(wfn))
        cd._gauge_override = 'canonical'
        assert abs(nc - cd.dboc_mp2_oc()) < 1e-9 * max(1.0, abs(nc)), (fc, nc, cd.dboc_mp2_oc())
