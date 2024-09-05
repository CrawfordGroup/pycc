"""
Test CCSD linear response functions.
"""
import sys
sys.path.append("/Users/jattakumi/pycc/")
import numpy as np
# Import package, test suite, and other packages as needed
import psi4
from opt_einsum import contract

import pycc
#from ccwfn import ccwfn
#from cchbar import cchbar
#from cclambda import cclambda
#from ccdensity import ccdensity
#from ccresponse import ccresponse

geom = """
O
H 1 1.8084679
H 1 1.8084679 2 104.5
units bohr
symmetry c1 
no_reorient
"""

hf = """
F  0.000000000000   0.000000000000  0.000000000000
H  0.000000000000   0.000000000000  -1.732800000000
units bohr
no_reorient
symmetry c1
"""

hof = """
          O          -0.947809457408    -0.132934425181     0.000000000000
          H          -1.513924046286     1.610489987673     0.000000000000
          F           0.878279174340     0.026485523618     0.000000000000
unit bohr
no_reorient
symmetry c1
"""
psi4.core.clean_options()
psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'true',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12})
mol = psi4.geometry(hof)
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

omega1 = 0.0
# omega1 = 0.0656


#A = resp.pertbar.Aoo

#resp.linresp(omega1)
# Creating dictionaries
# X_1 = X(-omega); X_2 = X(omega)
# Y_1 = Y(-omega); Y_2 = Y(omega)
# X_neg = X1(-omega) , X2(-omega)
# X_pos, Y_neg, Y_pos
X_1 = {}
X_2 = {}
Y_1 = {}
Y_2 = {}
X_A = {}
X_B = {}

for axis in range(0, 3):
    string = "MU_" + resp.cart[axis]

    A = resp.pertbar[string]
    # B = resp.pertbar[string]
    #print("Aoo",A)

    # A -> -omega
    # B -> +omega
    X_A[string] = resp.solve_right(A, omega1)
    X_B[string] = resp.solve_right(A, -omega1)

    X_2[string] = resp.solve_right(A, omega1)
    Y_2[string] = resp.solve_left(A, omega1)
    X_1[string] = resp.solve_right(A, -omega1)
    Y_1[string] = resp.solve_left(A, -omega1)

#resp.polar(omega1)
# Grabbing X, Y and declaring the matrix space for LR
polar_AB_pos = np.zeros((3,3))
polar_AB_neg = np.zeros((3,3))
# polar_AB_aveg = np.zeros((3,3))


for a in range(0, 3):
    string_a = "MU_" + resp.cart[a]
    X1_A, X2_A, _ = X_A[string_a]
    for b in range(0, 3):
        # string_a = "MU_" + resp.cart[a]
        # X1_A, X2_A, _ = X_A[string_a]
        string_b = "MU_" + resp.cart[b]
        Y1_B, Y2_B, _ = Y_2[string_b]
        X1_B, X2_B, _ = X_2[string_b]
        X_1B, X_2B, _ = X_B[string_b]
        polar_AB_pos[a,b] = resp.linresp_asym(string_a, X1_B, X2_B, Y1_B, Y2_B)
        polar_AB_neg[a, b] = resp.linresp_sym(string_a, string_b, X1_A, X2_A, X_1B, X_2B)
        # polar_AB_neg[a, b] = resp.linresp_sym(string_a, string_b, X1_A, X2_A, X_1B, X_2B)

print(polar_AB_pos[0,0])
print(polar_AB_pos[0,1])
print(polar_AB_pos[0,2])
print(polar_AB_pos[1,0])
print(polar_AB_pos[1,1])
print(polar_AB_pos[1,2])
print(polar_AB_pos[2,0])
print(polar_AB_pos[2,1])
print(polar_AB_pos[2,2])
print("Comparing asymmetric versus symmetric")
print(polar_AB_neg[0,0])
print(polar_AB_neg[0,1])
print(polar_AB_neg[0,2])
print(polar_AB_neg[1,0])
print(polar_AB_neg[1,1])
print(polar_AB_neg[1,2])
print(polar_AB_neg[2,0])
print(polar_AB_neg[2,1])
print(polar_AB_neg[2,2])
