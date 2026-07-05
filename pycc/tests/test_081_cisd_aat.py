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


AAT_REF = np.array([
 [-0.0245586668, 0.0977047211,-0.3532345859],
 [-0.0158908913, 0.0181292780,-0.0917113603],
 [ 0.1522781165, 0.3409913695,-0.0000007363],
 [-0.0245586651, 0.0977047208, 0.3532345859],
 [-0.0158908916, 0.0181292762, 0.0917113604],
 [-0.1522781168,-0.3409913697,-0.0000007362],
 [-0.0063821715, 0.6947952006, 0.2131056766],
 [-0.7231023900, 0.0127783589,-2.1375592252],
 [-0.2035256543, 2.2788946007,-0.0000010351],
 [-0.0063821716, 0.6947951795,-0.2131056766],
 [-0.7231023702, 0.0127783579, 2.1375592251],
 [ 0.2035256536,-2.2788945997,-0.0000010340],
])


def test_cisd_aat_vs_reference():
    P = np.asarray(_ciwfn().atomic_axial_tensors()).reshape(-1, 3)
    assert np.max(np.abs(P - AAT_REF)) < 1e-6, np.max(np.abs(P - AAT_REF))
