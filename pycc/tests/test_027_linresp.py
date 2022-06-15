"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_polar_h2o_cc_pvdz():
    """H2O cc-pVDZ"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-14,
                      'd_convergence': 1e-14,
                      'r_convergence': 1e-14,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-13
    r_conv = 1e-13

    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)

    resp = pycc.ccresponse(density)
    omega = 0.01
    check = resp.pertcheck(omega) # checks perturbed wave functions for all available operators

    # From Psi4: Electric- and magnetic-dipole pseudo-responses at +/-omega
    ref = {
"MU_X_0.010000": 0.059711553704,
"MU_X_-0.010000": 0.056273457658,
"MU_Y_0.010000": 7.341419446523,
"MU_Y_-0.010000": 7.129244769943,
"MU_Z_0.010000": 3.071438076138,
"MU_Z_-0.010000": 2.989674229480,
"M_X_0.010000": 0.607770924164,
"M_Y_0.010000": 0.710225214533,
"M_Z_0.010000": 0.775111802368,
"M*_X_-0.010000": 0.586575382108,
"M*_Y_-0.010000": 0.667622954134,
"M*_Z_-0.010000": 0.736881617713, 
"P_X_-0.010000": 0.097163221394,
"P_Y_-0.010000": 2.169072875250,
"P_Z_-0.010000": 1.497365713340,
"P*_X_0.010000": 0.103276788499,
"P*_Y_0.010000": 2.228622130154,
"P*_Z_0.010000": 1.536627133369,
"Q_XX_0.010000": 5.942498696750,
"Q_XY_0.010000": 0.202389983457,
"Q_XZ_0.010000": 0.186067317836,
"Q_YX_0.010000": 0.202389983457,
"Q_YY_0.010000": 7.147772196224,
"Q_YZ_0.010000": 19.240803761856,
"Q_ZX_0.010000": 0.186067317836,
"Q_ZY_0.010000": 19.240803761856,
"Q_ZZ_0.010000": 0.250165812115,
"Q_XX_-0.010000": 5.811357442660,
"Q_XY_-0.010000": 0.192591582644,
"Q_XZ_-0.010000": 0.175163473590,
"Q_YX_-0.010000": 0.192591582644,
"Q_YY_-0.010000": 6.971750667839,
"Q_YZ_-0.010000": 18.721795464544,
"Q_ZX_-0.010000": 0.175163473590,
"Q_ZY_-0.010000": 18.721795464544,
"Q_ZZ_-0.010000": 0.241096711760}

    for key in ref:
        print(f"key = {key}; ref = {ref[key]}; check = {check[key]}")
        assert(abs(check[key] - ref[key]) < 1e-11)
