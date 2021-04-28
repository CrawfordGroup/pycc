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
sys.path.insert(0, '../')
import ode as myode
import numpy as np
from scipy.integrate import ode

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
F_str = 1.0
omega = 2.87
tprime = 5.0
V = sine_square_laser(F_str, omega, tprime)

axis = 2  # z-axis for field
t0 = 0
tf = 10.0
h = 0.01
rtcc = pycc.rtcc(cc, cclambda, ccdensity, V, axis)
y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2).astype('complex128')
ODE = ode(rtcc.f).set_integrator('zvode')
ODE.set_initial_value(y0, t0)
t = t0
t1, t2, l1, l2 = rtcc.extract_amps(y0)
dip0 = rtcc.dipole(t1, t2, l1, l2)
ecc0 = rtcc.energy(t, t1, t2, l1, l2)
time = [t0]
dip_z = [dip0]
energy = [ecc0]
print("Time(s)                  Energy (a.u.)                               Z-Dipole (a.u.)     ")
print("%7.2f  %20.15f + %20.15fi  %20.15f + %20.15fi" % (t, ecc0.real+rhf_e, ecc0.imag, dip0.real, dip0.imag))

while ODE.successful() and ODE.t < tf:
    y = ODE.integrate(ODE.t+h)
    t = ODE.t
    t1, t2, l1, l2 = rtcc.extract_amps(y)
    dip = rtcc.dipole(t1, t2, l1, l2)
    ecc = rtcc.energy(t, t1, t2, l1, l1)
    time.append(t)
    dip_z.append(dip)
    energy.append(ecc+rhf_e)
    print("%7.2f  %20.15f + %20.15fi  %20.15f + %20.15fi" % (t, ecc.real+rhf_e, ecc.imag, dip.real, dip.imag))

#np.savez("helium_cc-pvdz_F_str=10.0_omega=2.87.npz", time_points=time, energy=energy, dip_x=dip_z, dip_y=dip_z, dip_z=dip_z)
