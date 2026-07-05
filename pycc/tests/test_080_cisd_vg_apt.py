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


VG_APT_REF = np.array([
 [ 0.9565985610, 0.2404439930,-0.0000006788],
 [ 0.1774893329, 0.1555861277, 0.0000019628],
 [-0.0000004434, 0.0000018991, 0.8551121460],
 [ 0.9565985610, 0.2404439930, 0.0000006761],
 [ 0.1774893329, 0.1555861277,-0.0000019592],
 [ 0.0000004413,-0.0000018960, 0.8551121460],
 [ 6.8024442952, 0.0624854667,-0.0000009738],
 [ 0.1250980895, 7.0795865038, 0.0000030620],
 [-0.0000011688, 0.0000029529, 8.0926181418],
 [ 6.8024442952, 0.0624854667, 0.0000009844],
 [ 0.1250980895, 7.0795865038,-0.0000030555],
 [ 0.0000011790,-0.0000029459, 8.0926181418],
])


def test_cisd_vg_apt_vs_reference():
    P = np.asarray(_ciwfn().velocity_dipole_derivatives()).reshape(-1, 3)
    assert np.max(np.abs(P - VG_APT_REF)) < 1e-6, np.max(np.abs(P - VG_APT_REF))
