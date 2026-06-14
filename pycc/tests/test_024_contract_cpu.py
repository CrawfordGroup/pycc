"""
Test RT-CCSD propagation with the RK4 integrator on water (CPU/NumPy path).
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pycc
import pytest
from pycc.rt.integrators import rk4
from pycc.rt.lasers import gaussian_laser
from ..data.molecules import *


def test_rtcc_water_cc_pvdz(rhf_wfn):
    """H2O cc-pVDZ, RT-CCSD propagated with RK4 to t=0.1 a.u."""
    e_conv = 1e-13
    r_conv = 1e-13

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    cc = pycc.ccwfn(wfn)
    ecc = cc.solve_cc(e_conv, r_conv)

    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)

    ccdensity = pycc.ccdensity(cc, cclambda)

    # Gaussian pulse (a.u.)
    F_str = 0.1
    omega = 0
    sigma = 0.01
    center = 0.05
    V = gaussian_laser(F_str, omega, sigma, center)

    # RT-CC setup
    phase = 0
    t0 = 0.0
    tf = 0.1
    h = 0.01
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase).astype('complex128')
    y = y0
    ODE = rk4(h)

    # Propagate t0 -> tf with a fixed number of RK4 steps. Driving the loop with a
    # `while t < tf` bound would overshoot by one step here: ten 0.01 increments
    # sum to 0.09999999999999999 < 0.1 in floating point, triggering an extra step.
    nsteps = int(round((tf - t0) / h))
    t = t0
    for step in range(nsteps):
        y = ODE(rtcc.f, t, y)
        t = t0 + (step + 1) * h
        t1, t2, l1, l2, phase = rtcc.extract_amps(y)
        mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)

    # Reference is mu_z(t=tf) computed with this code (RK4, h=0.01). The RT-CCSD
    # propagation is deterministic and reproduces to ~1e-10 across platforms
    # (cf. test_006). The F_str=0.1 pulse drives ~7e-4 of dipole motion over the
    # run, so matching to 1e-10 genuinely tests the time evolution.
    mu_z_ref = -0.078669702343987763
    assert abs(mu_z_ref - mu_z.real) < 1e-10
    # <mu_z> is a real observable; the propagated imaginary part is numerical noise.
    assert abs(mu_z.imag) < 1e-7
