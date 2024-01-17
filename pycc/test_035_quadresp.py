"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4

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
F 0.0000000 0.0000000  -0.087290008493927
H 0.0000000 0.0000000 1.645494724632280
units bohr
no_reorient
symmetry c1
"""

psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12
})
mol = psi4.geometry(hf)
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
omega2 = 0.0656

resp.pert_quadresp(omega1, omega2)
resp.hyperpolar()
