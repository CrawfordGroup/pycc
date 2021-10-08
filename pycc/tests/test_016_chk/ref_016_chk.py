"""
Test checkpointing for rt propagation
Script to generate the 0-5.1au checkpoint and
the "full" propagation reference data
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pickle as pk
import numpy as np
from pycc.rt.integrators import rk2
from pycc.rt.lasers import gaussian_laser
from pycc.data.molecules import *

# run cc
psi4.set_memory('600MB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8,
                  'r_convergence': 1e-8,
                  'diis': 8})
mol = psi4.geometry(moldict["H2"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
e_conv = 1e-8
r_conv = 1e-8
cc = pycc.ccenergy(rhf_wfn)
ecc = cc.solve_ccsd(e_conv, r_conv)
hbar = pycc.cchbar(cc)
cclambda = pycc.cclambda(cc, hbar)
lecc = cclambda.solve_lambda(e_conv, r_conv)
ccdensity = pycc.ccdensity(cc, cclambda)

# narrow Gaussian pulse
F_str = 0.001
sigma = 0.01
center = 0.05
V = gaussian_laser(F_str, 0, sigma, center=center)

# RTCC
# use tf = 5 to generate checkpoint file
h = 0.1
ti = 0
#tf = 5
tf = 10
rtcc = pycc.rtcc(cc,cclambda,ccdensity,V,magnetic=True,kick='z')
y0 = rtcc.collect_amps(cc.t1,cc.t2,cclambda.l1,cclambda.l2).astype('complex128')
ODE = rk2(h)

# use first option to only generate checkpoint file
#ret = rtcc.propagate(ODE, y0, tf, ti=ti, ref=False, chk=True)
ret, ret_t = rtcc.propagate(ODE, y0, tf, ti=ti, ref=False, chk=False, tchk=1)

# comment these out when only saving the checkpoint file
with open('output.pk','wb') as of:
    pk.dump(ret,of,pk.HIGHEST_PROTOCOL)
with open('t_out.pk','wb') as ampf:
    pk.dump(ret_t,ampf,pk.HIGHEST_PROTOCOL)
