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
from lasers import sine_square_laser
from lasers import gaussian_laser
#sys.path.insert(0, '../')
#import ode as myode
import numpy as np
from scipy.integrate import complex_ode as ode
import time as timer

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
mol = psi4.geometry(mol.moldict["LiH"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
enuc = mol.nuclear_repulsion_energy()
print("Enuc = %20.15f" % enuc)

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

epsi4 = psi4.gradient('CCSD')

print("Starting RTCC propagation...")
time_init = timer.time()

# Sine squared pulse (a.u.)
F_str = 0.05
omega = 0.05
tprime = 5.0
V = sine_square_laser(F_str, omega, tprime)

t0 = 0
tf = 1.0
h = 0.01
rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2).astype('complex128')
ODE = ode(rtcc.f).set_integrator('vode',atol=1e-13,rtol=1e-13)
ODE.set_initial_value(y0, t0)
t = t0
t1, t2, l1, l2 = rtcc.extract_amps(y0)
mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)
ecc0 = rtcc.lagrangian(t, t1, t2, l1, l2)
time = [t0]
dip_x = [mu_x]
dip_y = [mu_y]
dip_z = [mu_z]
energy = [ecc0+rhf_e-enuc]
print("Time(s)                  Energy (a.u.)                               Z-Dipole (a.u.)     ")
print("%7.2f  %20.15f + %20.15fi  %20.15f + %20.15fi" % (t, ecc0.real+rhf_e-enuc, ecc0.imag, mu_z.real, mu_z.imag))

while ODE.successful() and ODE.t < tf:
    y = ODE.integrate(ODE.t+h)
    t = ODE.t
    t1, t2, l1, l2 = rtcc.extract_amps(y)
    mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)
    ecc = rtcc.lagrangian(t, t1, t2, l1, l2)
    time.append(t)
    dip_x.append(mu_x)
    dip_y.append(mu_y)
    dip_z.append(mu_z)
    energy.append(ecc+rhf_e-enuc)
    print("%7.2f  %20.15f + %20.15fi  %20.15f + %20.15fi" % (t, ecc.real+rhf_e-enuc, ecc.imag, mu_z.real, mu_z.imag))

#np.savez("lih_cc-pvdz_F_str=0.05_omega=0.05_tightconv.npz", time_points=time, energy=energy, dip_x=dip_x, dip_y=dip_y, dip_z=dip_z)

print("\nRTCC propagation over %.2f a.u. completed in %.3f seconds.\n" % (tf-t0, timer.time() - time_init))
