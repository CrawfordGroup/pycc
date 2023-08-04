"""
Test basic PNO-CCSD, PNO++-CCSD, PAO-CCSD energies
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import sys
sys.path.append("/Users/jattakumi/pycc/pycc/data")
from molecules import *

"""BUTANE PNO-CCSD Test"""
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'aug-cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-09,
                  'd_convergence': 1e-09,
                  'r_convergence': 1e-09,
                  'diis': 1})
mol = psi4.geometry(moldict["butane"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 100
e_conv = 1e-12
r_conv = 1e-12

#simulation code of pno-ccsd
#ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-5,it2_opt=False,filter=True)
#eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

#pno-ccsd
lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-5,it2_opt=False)
elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

#assert(abs(eccsd_sim - elccsd) < 1e-12)
