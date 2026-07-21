"""
test_ucc_gs.py
==============
Sanity check for RT-UCCSD:
  - Converges UCC amplitudes via BCH2 residuals
  - Evaluates BCH4 energy on converged amplitudes
  - Compares to Ajay's MPQC reference
  - No external field, one RK4 step
  - Checks |C(t)|^2 ~ 1 and amplitude stationarity - todo
"""

import psi4
import pycc
import numpy as np
import time
import cProfile
import pstats
import io

from pycc.rt.integrators import rk4
from pycc.data.molecules import *
from pycc.uccwfn import make_ucc_fns, UCCWfn
from pycc.rt.rtcc_ucc import rtcc_ucc

# ------------------------------------------------------------------
# Psi4 Setup
# ------------------------------------------------------------------
psi4.core.clean()
psi4.set_memory('2 GiB')
psi4.core.set_output_file('test_ucc_gs.dat', True)
psi4.set_options({
    'basis': 'aug-cc-pvdz',
    'scf_type': 'pk',
    'freeze_core': 'false',
    'e_convergence': 1e-13,
    'd_convergence': 1e-13,
    'r_convergence': 1e-13,
    'diis': 1
})

mol = psi4.geometry("""
He 0 0 0 
units bohr
""")
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

# Ground-state CCSD (PyCC)
e_conv = 1e-12
r_conv = 1e-12

t_ccsd_start = time.time()
cc = pycc.ccwfn(rhf_wfn, model='CCSD')
ecc = cc.solve_cc(e_conv, r_conv, maxiter=75)
t_ccsd_end = time.time()
print(f"CCSD wall time: {t_ccsd_end - t_ccsd_start:.2f} s")

# Converge UCC amplitudes via BCH2 residuals, then eval BCH4 energy
t_ucc_start = time.time()
ucc, energy_fn, residuals_fn = make_ucc_fns(cc, e_conv=1e-10, r_conv=1e-10)
t_ucc_end = time.time()
print(f"UCCSD wall time: {t_ucc_end - t_ucc_start:.2f} s")
print(f"  T1 norm (PyCC UCC)      : {np.linalg.norm(ucc.t1):.14f}")
print(f"  T2 norm (PyCC UCC)      : {np.linalg.norm(ucc.t2):.14f}")
print(f"  T1 norm (Ajay MPQC)     : 0.0191111210") #ccpvtz
print(f"  T2 norm (Ajay MPQC)     : 0.2129270336") #ccpvtz

# Update cc with UCC-converged amplitudes for RT propagation
cc.t1 = ucc.t1
cc.t2 = ucc.t2

# Reference from Ajay's MPQC. ccpvtz
ajay_bch4 = -0.0707223905

print()
print("=" * 55)
print(f"  SCF energy              : {rhf_e:.14f}")
print(f"  CCSD correlation        : {ecc:.14f}")
print(f"  UCC BCH4 (converged)    : {ucc.last_energy:.14f}")
print(f"  Ajay BCH4 (MPQC)        : {ajay_bch4:.14f}")
print(f"  Diff (UCC - Ajay)       : {ucc.last_energy - ajay_bch4:.2e}")
print(f"  Diff (UCC - CCSD)       : {ucc.last_energy - ecc:.2e}")
print("=" * 55)

# # ------------------------------------------------------------------
# # Profile a single energy call
# # ------------------------------------------------------------------
# V  = lambda t: 0.0
# h  = 0.01
# t0 = 0.0

# rt  = rtcc_ucc(cc, V, energy_fn, residuals_fn, kick=None)
# y0  = rt.collect_amps(cc.t1, cc.t2, 0.0 + 0.0j)
# ODE = rk4(h)

# t1_0, t2_0, ph_0 = rt.extract_amps(y0)

# print()
# print("=" * 55)
# print("  Profiling single energy() call")
# print("=" * 55)

# pr = cProfile.Profile()
# pr.enable()
# e_ucc_0 = rt.energy(t0, t1_0, t2_0)
# pr.disable()

# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
# ps.print_stats(30)
# print(s.getvalue())

# print(f"  UCC energy : {e_ucc_0.real:.10f}")

# # ------------------------------------------------------------------
# # One RK4 step
# # ------------------------------------------------------------------
# y1 = ODE(rt.f, t0, y0)
# t1_1, t2_1, ph_1 = rt.extract_amps(y1)

# e_ucc_1 = rt.energy(t0 + h, t1_1, t2_1)
# dt1     = np.max(np.abs(t1_1 - t1_0))
# dt2     = np.max(np.abs(t2_1 - t2_0))

# print()
# print(f"  --- t = {h} (one RK4 step, no field) ---")
# print(f"  UCC energy              : {e_ucc_1.real:.10f}")
# print(f"  Energy change           : {abs(e_ucc_1 - e_ucc_0):.2e}  (should be 0)")
# print(f"  Max |delta t1|          : {dt1:.2e}  (should be 0)")
# print(f"  Max |delta t2|          : {dt2:.2e}  (should be 0)")
# print()
# print("Done.")
