import psi4
import pycc
import numpy as np


# (P)-hydrogen peroxide, validation geometry (Angstrom); atom order H,H,O,O.
GEOM = """
H   -1.025917944    0.8626019238    0.216196164
H   1.025917944     -0.8626019238   0.2161961616
O   -0.7221777851   -0.1253224661   0.2161987519
O   0.7221777851    0.1253224661    0.2161987448
no_com
no_reorient
symmetry c1
"""


def _ciwfn():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(GEOM)
    psi4.set_options({'basis': 'sto-3g', 'scf_type': 'pk', 'freeze_core': False,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    _, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD', orbital_basis='spatial')
    ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=150)
    return ci


LG_APT_REF = np.array([
 [ 0.1326162556, 0.0627497934,-0.0000000461],
 [ 0.0788409115,-0.2034439684, 0.0000012770],
 [-0.0000002480, 0.0000011653, 0.2541017267],
 [ 0.1326162556, 0.0627497934, 0.0000000467],
 [ 0.0788409115,-0.2034439684,-0.0000012743],
 [ 0.0000002485,-0.0000011624, 0.2541017267],
 [-0.1326162556,-0.0627497934,-0.0000000387],
 [-0.0788409115, 0.2034439684,-0.0000017648],
 [-0.0000000825,-0.0000014301,-0.2541017267],
 [-0.1326162556,-0.0627497934, 0.0000000381],
 [-0.0788409115, 0.2034439684, 0.0000017621],
 [ 0.0000000820, 0.0000014273,-0.2541017267],
])


def test_cisd_lg_apt_vs_reference():
    P = np.asarray(_ciwfn().dipole_derivatives()).reshape(-1, 3)
    assert np.max(np.abs(P - LG_APT_REF)) < 1e-6, np.max(np.abs(P - LG_APT_REF))


def test_cisd_lg_apt_translational_sum_rule():
    P = np.asarray(_ciwfn().dipole_derivatives())
    assert np.max(np.abs(np.sum(P, axis=0))) < 1e-6
