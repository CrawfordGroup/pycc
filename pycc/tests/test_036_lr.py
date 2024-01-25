"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pytest
import pycc
from ..data.molecules import *

def test_linresp():
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12
    })
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
    # X_1 = X(-omega); X_2 = X(omega)
    # Y_1 = Y(-omega); Y_2 = Y(omega)
    X_1 = {}
    X_2 = {}
    Y_1 = {}
    Y_2 = {}

    for axis in range(0, 3):
        string = "MU_" + resp.cart[axis]

        A = resp.pertbar[string]

        X_2[string] = resp.solve_right(A, omega1)
        Y_2[string] = resp.solve_left(A, omega1)
        X_1[string] = resp.solve_right(A, -omega1)
        Y_1[string] = resp.solve_left(A, -omega1)

    # Grabbing X, Y and declaring the matrix space for LR
    polar_AB = np.zeros((3,3))

    for a in range(0, 3):
        string_a = "MU_" + resp.cart[a]
        for b in range(0, 3):
            string_b = "MU_" + resp.cart[b]
            Y1_B, Y2_B, _ = Y_2[string_b]
            X1_B, X2_B, _ = X_2[string_b]
            polar_AB[a,b] = resp.linresp_asym(string_a, X1_B, X2_B, Y1_B, Y2_B)

    polar_AB_avg = np.average([polar_AB[0,0], polar_AB[1,1], polar_AB[2,2]])

    #validating from psi4
    polar_XX = 9.92992070420665
    polar_YY = 13.443740151331559
    polar_ZZ = 11.342765745046526
    polar_avg = 11.572142200333

    assert(abs(polar_AB[0,0] - polar_XX) < 1e-8)
    assert(abs(polar_AB[1,1] - polar_YY) < 1e-8)
    assert(abs(polar_AB[2,2] - polar_ZZ) < 1e-8)
    assert(abs(polar_AB_avg - polar_avg) < 1e-8)
