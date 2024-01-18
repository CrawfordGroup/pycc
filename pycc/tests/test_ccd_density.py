"""
Test CCSD density equations using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import sys
sys.path.append ("/Users/jattakumi/pycc/pycc")
from data.molecules import *



"""H2O"""
# Psi4 Setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'true',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12,
                  'diis': 1})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 75
e_conv = 1e-8
r_conv = 1e-8

#simulation code
ccsd = pycc.ccwfn(rhf_wfn, model = 'CCSD', local = 'PNO', local_cutoff = 1e-7, filter = True, it2_opt = True)
eccsd = ccsd.solve_cc(e_conv, r_conv)
hbar = pycc.cchbar(ccsd)
cclambda = pycc.cclambda(ccsd, hbar)
lccsd = cclambda.solve_lambda(e_conv, r_conv)
epsi4 = -0.070616830152761
lpsi4 = -0.068826452648939
ccdensity = pycc.ccdensity(ccsd, cclambda, True)
#ecc_density = ccdensity.compute_energy()
#assert (abs(epsi4 - eccsd) < 1e-11)
#assert (abs(lpsi4 - lccsd) < 1e-11)
#assert (abs(epsi4 - ecc_density) < 1e-11)
#print(ecc_density)

#pno ccd density code
lccd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-7,it2_opt=True)
elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
lhbar = pycc.cchbar(lccd)
lcclambda = pycc.cclambda(lccd, lhbar)
l_lccd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
lccdensity = pycc.ccdensity(lccd, lcclambda)                                                 
