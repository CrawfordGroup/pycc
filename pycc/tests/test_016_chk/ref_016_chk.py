"""
Test checkpointing for rt propagation
Script to generate the 0-5.1au checkpoint and
the "full" propagation reference data

To re-generate reference data, first run full = True, then full = False
"""
full = False 

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
if full:
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
else:
    rhf_wfn = psi4.core.Wavefunction.from_file('ref_wfn')
e_conv = 1e-8
r_conv = 1e-8
cc = pycc.ccwfn(rhf_wfn)
ecc = cc.solve_cc(e_conv, r_conv)
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
if full:
    tf = 10
else:
    tf = 5
rtcc = pycc.rtcc(cc,cclambda,ccdensity,V,magnetic=True,kick='z')
y0 = rtcc.collect_amps(cc.t1,cc.t2,cclambda.l1,cclambda.l2).astype('complex128')
ODE = rk2(h)

# if full: run entire propagation, save amps / props / rhf wfn
if full:
    ret, ret_t = rtcc.propagate(ODE, y0, tf, ti=ti, ref=False, chk=True, tchk=1,
            cfile="chk_full.pk", ofile="output_full.pk",tfile="t_out_full.pk")
else:
    ret, ret_t = rtcc.propagate(ODE, y0, tf, ti=ti, ref=False, chk=True, tchk=1,
            cfile="chk_5.pk")
