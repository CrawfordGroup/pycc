"""
Test basic PNO-CCD energy
"""
# Import package, test suite, and other packages as needed
import psi4
from ccwfn import ccwfn
from lccwfn import lccwfn
from data.molecules import *

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
r_conv = 1e-12
   
#simulation code of pno-ccd
ccsd_sim = ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-11,it2_opt=False,filter=True)
eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

#pno-ccd
lccsd = ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-11,it2_opt=False)
elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter) 

print("eccd_sim",eccsd_sim)
print("elccd", elccsd)
assert(abs(eccsd_sim - elccsd) < 1e-12) 

