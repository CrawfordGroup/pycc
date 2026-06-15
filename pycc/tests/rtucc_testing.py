"""
Real-time UCCSD propagation driver. Mirrors full_rtcc_code.py structure.
Uses UCCWfn (uccwfn.py) + rtcc_ucc (rt/rtcc_ucc.py) from PyCC rt-uccsd branch.

Computes:
  - Autocorrelation function C(t) = <Psi(0)|Psi(t)>
  - Dipole moment (once ucc_onepdm is available from Ajay)
  - UCC BCH energy at each time step

Output: HDF5 file with all time-series data.
"""

import psi4
import pycc
import numpy as np
import h5py

from pycc.rt.lasers import delta_pulse_laser
from pycc.rt.integrators import rk4
from pycc.data.molecules import *
from pycc.uccwfn import make_ucc_fns
from pycc.rt.rtcc_ucc import rtcc_ucc

one_ev = 27.211386245988468

# Psi4 stuff
psi4.core.clean()
psi4.set_memory('5 GiB')
psi4.core.set_output_file('h2o_uccsd_rt.dat', True)
psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'pk',
    'mp2_type': 'conv',
    'freeze_core': 'false',
    'e_convergence': 1e-13,
    'd_convergence': 1e-13,
    'r_convergence': 1e-13,
    'diis': 1
})

mol = psi4.geometry(moldict["H2O_HEK"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

# Ground-state CCSD (provides integrals + initial T amplitudes)
e_conv = 1e-12
r_conv = 1e-12

cc = pycc.ccwfn(rhf_wfn, model='CCSD')
ecc = cc.solve_cc(e_conv, r_conv)
print(f"CCSD energy: {ecc:.10f}")

# Build UCC energy + residuals functions from Ajay's SeQuant equations
energy_fn, residuals_fn = make_ucc_fns(cc)

# Laser: delta pulse kick along z
F_str  = 0.01
center = 0.01
V      = delta_pulse_laser(F_str, center)
tprime = center + 0.01 - center + 0.01  # time after pulse is off, book-keeping

# RT-UCCSD Setup
phase = 0 + 0j
t0, tf, h = 0, 50, 0.01
num_steps  = int(tf / h) + 1
t_values   = np.linspace(t0, tf, num_steps)

rt = rtcc_ucc(cc, V, energy_fn, residuals_fn, kick='z')
y0 = rt.collect_amps(cc.t1, cc.t2, phase)
y  = y0.copy()
ODE = rk4(h)

# Initial values
t1_0, t2_0, ph_0 = rt.extract_amps(y0)
ecc0 = rt.energy(t0, t1_0, t2_0)

# Preallocate arrays
energy_list = np.zeros(num_steps, dtype=complex)
ac          = np.zeros(num_steps, dtype=complex)
ac_1        = np.zeros(num_steps, dtype=complex)
field       = np.zeros(num_steps, dtype=complex)
phase_list  = np.zeros(num_steps, dtype=complex)

# Initialize t=0
energy_list[0] = ecc0
ac[0]          = rt.autocorrelation(y0, y0)
field[0]       = V(t0)
phase_list[0]  = ph_0

# RT-UCCSD Propagation Loop
check    = False
y_wanted = None

for i in range(1, num_steps):
    t = t_values[i]
    y = ODE(rt.f, t, y)

    t1, t2, phase = rt.extract_amps(y)

    # Energy
    ecc = rt.energy(t, t1, t2)
    energy_list[i] = ecc

    # Autocorrelation C(t) = <Psi(0)|Psi(t)>
    ac[i] = rt.autocorrelation(y0, y)

    # Autocorrelation from tprime
    if not check and abs(t - tprime) < h:
        y_wanted = y.copy()
        check    = True
    if check:
        ac_1[i] = rt.autocorrelation(y_wanted, y)

    # Field + phase
    field[i]      = V(t)
    phase_list[i] = phase


    if i % 1000 == 0:
        print(f"  t = {t:.2f}  |C(t)|^2 = {abs(ac[i])**2:.6f}  E = {ecc.real:.10f}")

print("RT-UCCSD Propagation Completed Successfully!")

# Save to HDF5
outfile = "water_uccsd_rt_50_ccpvdz.h5"
with h5py.File(outfile, "w") as h5:
    grp = h5.create_group("UCC")
    grp.create_dataset("energy",    data=energy_list)
    grp.create_dataset("ac",        data=ac)
    grp.create_dataset("ac_1",      data=ac_1)
    grp.create_dataset("field",     data=field)
    grp.create_dataset("phase",     data=phase_list)
    grp.create_dataset("t_values",  data=t_values)

print(f"Data saved to {outfile}")