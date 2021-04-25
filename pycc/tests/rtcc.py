"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import sys
sys.path.insert(0, '../data')
import molecules as mol
from lasers import gaussian_laser
sys.path.insert(0, '../')
import ode as ode

# Psi4 Setup
psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
memory = 2
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13,
                  'diis': 1})
mol = psi4.geometry(mol.moldict["He"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

## Set up initial (t=0) amplitudes
maxiter = 75
e_conv = 1e-13
r_conv = 1e-13
cc = pycc.ccenergy(rhf_wfn)
ecc = cc.solve_ccsd(e_conv, r_conv, maxiter)
hbar = pycc.cchbar(cc)
maxiter = 75
max_diis = 8
cclambda = pycc.cclambda(cc, hbar)
lecc = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)
ccdensity = pycc.ccdensity(cc, cclambda)
ecc_test = ccdensity.compute_energy()
print("ECC from density       = %20.15f" % ccdensity.ecc)
print("ECC from wave function = %20.15f" % cc.ecc)

# Gaussian pulse (a.u.)
F_str = 0.01
omega = 2.874
sigma = 0.446994
center = 2.5
V = gaussian_laser(F_str, omega, sigma, center)

print("no = %d   nv = %d   len1 = %d  len2 = %d" % (cc.no, cc.nv, cc.no*cc.nv, cc.no*cc.no*cc.nv*cc.nv))

axis = 2  # z-axis for field
t0 = 0
tf = 2
h = 0.01
rtcc = pycc.rtcc(cc, cclambda, V, axis, t0, tf, h, ode.RK4())
step = rtcc.RK.step()
y = next(step)
print(y)
