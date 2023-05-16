"""
Test basic PNO-CCD energy
"""
# Import package, test suite, and other packages as needed
import psi4
from ccwfn import ccwfn
from lccwfn import lccwfn
from data.molecules import *
from cchbar import cchbar

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13,
                  'diis': 1})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 100
e_conv = 1e-12
r_conv = 1e-10
   
#simulation code of pno-ccd
ccd_sim = ccwfn(rhf_wfn, model='CCD',local='PNO', local_cutoff=1e-11,it2_opt=False,filter=True)
eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)
hbar_sim = cchbar(ccd_sim)
#pno-ccd
lccd = ccwfn(rhf_wfn,model='CCD', local='PNO', local_cutoff=1e-11,it2_opt=False)
elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
lhbar = cchbar(lccd)  

print("eccd_sim",eccd_sim)
print("elccd", elccd)
assert(abs(eccd_sim - elccd) < 1e-12) 

#lhbar.Debug._read_t2()