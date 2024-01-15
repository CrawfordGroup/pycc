"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import psi4

from ccwfn import ccwfn
from cchbar import cchbar
from cclambda import cclambda
from ccdensity import ccdensity

geom = psi4.geometry("""
O
H 1 1.8084679
H 1 1.8084679 2 104.5
units bohr
symmetry c1 
no_reorient
""")

psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'STO-3G',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12
})
mol = psi4.geometry(geom)
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

omega = 0.0656
omega = 0.14238

resp.pert_quadresp(omega1, omega2)
resp.hyperpolar()
