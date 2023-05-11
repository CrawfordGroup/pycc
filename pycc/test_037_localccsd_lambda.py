"""
Test basic PNO-CCD energy
"""
# Import package, test suite, and other packages as needed
import psi4
from ccwfn import ccwfn
from lccwfn import lccwfn
from data.molecules import *
from cchbar import cchbar
from cclambda import cclambda

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '6-311G',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13,
                  'diis': 1})
mol = psi4.geometry(moldict["(H2)_7"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 200
e_conv = 1e-12
r_conv = 1e-12
   
#simulation code of pno-ccsd
ccsd_sim = ccwfn(rhf_wfn, model='CCSD',local='PNO++', local_cutoff=1e-6,it2_opt=False,filter=True)
#eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
#hbar_sim = cchbar(ccsd_sim)
#cclambda_sim = cclambda(ccsd_sim, hbar_sim)
#l_ccsd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter) #maxiter)

#pno-ccsd
#lccsd = ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-5,it2_opt=False)
#elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
#lhbar = cchbar(lccsd)  
#lcclambda = cclambda(lccsd, lhbar)
#l_lccsd = lcclambda.solve_llambda(e_conv, r_conv, maxiter) # maxiter

#print("eccsd_sim",eccsd_sim)
#print("elccsd", elccsd)
#assert(abs(eccsd_sim - elccsd) < 1e-12) 

#print("l_ccsd_sim",l_ccsd_sim)
#print("l_lccsd", l_lccsd)
#assert(abs(l_ccsd_sim - l_lccsd) < 1e-12)

#lhbar.Debug._read_t2()
