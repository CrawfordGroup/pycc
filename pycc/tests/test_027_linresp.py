"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
#from ..data.molecules import *
import sys
sys.path.insert(0, '/Users/crawdad/src/pycc/pycc/data')
#import molecules as mol
from molecules import *


#def test_polar_h2o_cc_pvdz():
"""H2O cc-pVDZ"""
psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13,
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
