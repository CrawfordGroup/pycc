"""CCSD diagonal Born-Oppenheimer correction (CCderiv.dboc): Gauss, Tajti, Kallay,
Stanton, and Szalay, J. Chem. Phys. 125, 144111 (2006), cc-pVDZ, all electrons
correlated, at the HEAT geometries of Gauss et al.  References are CFOUR v2.1
values; agreement ~1e-5 cm^-1.  Companion to test_106 (order-consistent MP2 DBOC).
"""

import psi4
import pycc

HARTREE_CM = 219474.6313632

GEOMS = {
    'h2': "0 1\nH\nH 1 0.74186\nsymmetry c1\n",
    'hf': "0 1\nF\nH 1 0.91516\nsymmetry c1\n",
    'h2o': "0 1\nO\nH 1 0.95623\nH 1 0.95623 2 104.25\nsymmetry c1\n",
}
CFOUR_CCSD = {'h2': 111.906639, 'hf': 617.527474, 'h2o': 617.885361}


def _ccsd_dboc_cm(geom):
    psi4.core.clean()
    psi4.set_memory('4 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    e, wfn = psi4.energy('scf', return_wfn=True)
    return pycc.CCderiv(pycc.ccwfn(wfn), e_conv=1e-11, r_conv=1e-11).dboc() * HARTREE_CM


def test_ccsd_dboc_h2():
    assert abs(_ccsd_dboc_cm(GEOMS['h2']) - CFOUR_CCSD['h2']) < 1e-5


def test_ccsd_dboc_hf():
    assert abs(_ccsd_dboc_cm(GEOMS['hf']) - CFOUR_CCSD['hf']) < 1e-5


def test_ccsd_dboc_h2o():
    assert abs(_ccsd_dboc_cm(GEOMS['h2o']) - CFOUR_CCSD['h2o']) < 1e-5
