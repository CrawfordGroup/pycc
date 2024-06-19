"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import sys
sys.path.append("/Users/jattakumi/pycc/pycc/")
import pytest
import pycc
from data.molecules import *

def test_linresp():
    h2o2 = """
    O -0.182400 -0.692195 -0.031109
    O 0.182400 0.692195 -0.031109
    H 0.533952 -1.077444 0.493728
    H -0.533952 1.077444 0.493728
    symmetry c1
    """

    # H2_4 chain
    h2_4 = """
    H 0.000000 0.000000 0.000000
    H 0.750000 0.000000 0.000000
    H 0.000000 1.500000 0.000000
    H 0.375000 1.500000 -0.649520
    H 0.000000 3.000000 0.000000
    H -0.375000 3.000000 -0.649520
    H 0.000000 4.500000 -0.000000
    H -0.750000 4.500000 -0.000000
    symmetry c1
    """

    # H2_7 chain
    h2_7 = """
    H 0.000000 0.000000 0.000000
    H 0.750000 0.000000 0.000000
    H 0.000000 1.500000 0.000000
    H 0.375000 1.500000 -0.649520
    H 0.000000 3.000000 0.000000
    H -0.375000 3.000000 -0.649520
    H 0.000000 4.500000 -0.000000
    H -0.750000 4.500000 -0.000000
    H 0.000000 6.000000 -0.000000
    H -0.375000 6.000000 0.649520
    H 0.000000 7.500000 -0.000000
    H 0.375000 7.500000 -0.649520
    H 0.000000 9.000000 -0.000000
    H 0.750000 9.000000 0.000000
    symmetry c1
    """

    threshold = [1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11, 1e-12, 0]
    lmo = 'PNO++'
    geom = "h2_4"
    psi4.core.clean_options()
    psi4.set_memory('2 GiB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
    })

    mol = psi4.geometry(h2_4)
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    for t in threshold:
        e_conv = 1e-12
        r_conv = 1e-12

        cc = pycc.ccwfn(rhf_wfn, local_mos = 'BOYS', local= lmo, local_cutoff = t, filter=True)
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
            for b in range(0,3):
                string_b = "MU_" + resp.cart[b]
                Y1_B, Y2_B, _ = Y_2[string_b]
                X1_B, X2_B, _ = X_2[string_b]
                polar_AB[a,b] = resp.linresp_asym(string_a, X1_B, X2_B, Y1_B, Y2_B)

        print("Dynamic Polarizability Tensor @ w=0.0656 a.u.:")
        print(polar_AB)
        print("Average Dynamic Polarizability:")
        polar_AB_avg = np.average([polar_AB[0,0], polar_AB[1,1], polar_AB[2,2]])
        print(polar_AB_avg)

        B_avg = str(geom) + "_Dynpolar_" + str(lmo) + ".txt"

        with open(B_avg, 'a') as f1:
            f1.write(str(polar_AB_avg) + ' ' + str(t) + '\n')

        del cc, hbar, cclambda, density, resp

    # #validating from psi4
    # polar_XX = 9.92992070420665
    # polar_YY = 13.443740151331559
    # polar_ZZ = 11.342765745046526
    # polar_avg = 11.572142200333
    #
    # assert(abs(polar_AB[0,0] - polar_XX) < 1e-8)
    # assert(abs(polar_AB[1,1] - polar_YY) < 1e-8)
    # assert(abs(polar_AB[2,2] - polar_ZZ) < 1e-8)
    # assert(abs(polar_AB_avg - polar_avg) < 1e-8)
