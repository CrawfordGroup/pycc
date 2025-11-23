"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pytest
import pycc
from ..data.molecules import *

def test_sym_linresp():

    psi4.core.clean_options()
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'aug-cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-12
    r_conv = 1e-12

    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)

    resp = pycc.ccresponse(density)

    omega1 = 0.0656

    # Creating dictionaries
    X_A = {}
    X_B = {}

    for axis in range(0, 3):
        string = "MU_" + resp.cart[axis]
        A = resp.pertbar[string]
        X_A[string] = resp.solve_right(A, omega1)
        X_B[string] = resp.solve_right(A, -omega1)

    # Grabbing X, Y and declaring the matrix space for LR
    polar_AB = np.zeros((3,3))

    for a in range(0, 3):
        string_a = "MU_" + resp.cart[a]
        X1_A, X2_A, _ = X_A[string_a]
        for b in range(0, 3):
            string_b = "MU_" + resp.cart[b]
            X_1B, X_2B, _ = X_B[string_b]
            polar_AB[a,b] = resp.sym_linresp(string_a, string_b, X1_A, X2_A, X_1B, X_2B)

    print(f"Dynamic Polarizability Tensor @ w = {omega1} a.u.:")
    print(polar_AB)
    print("Average Dynamic Polarizability:")
    polar_AB_avg = np.average([polar_AB[0,0], polar_AB[1,1], polar_AB[2,2]])
    print(polar_AB_avg)

    # Validating from psi4
    polar_xx = 9.932240347101651
    polar_yy = 13.446487681337629
    polar_zz = 11.344346098120035
    polar_avg = 11.574358042186

    assert(abs(polar_AB[0,0] - polar_xx) < 1e-7)
    assert(abs(polar_AB[1,1] - polar_yy) < 1e-7)
    assert(abs(polar_AB[2,2] - polar_zz) < 1e-7)
    assert(abs(polar_AB_avg - polar_avg) < 1e-7)
