"""
Test RT-CCSD propagation with the RK4 integrator on water (torch "GPU" path).
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pycc
import pytest
from pycc.rt.integrators import rk4
from pycc.rt.lasers import gaussian_laser
from ..data.molecules import *

# Skip the whole module if PyTorch is absent, and bind `torch` for use below.
# (Without torch installed the GPU path can't run; with CPU-only torch it runs
# on CPU, which is enough to exercise the torch code path in CI.)
torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_rtcc_water_cc_pvdz(rhf_wfn):
    """H2O cc-pVDZ, RT-CCSD propagated with RK4 to t=0.1 a.u. on the torch device.

    Mirrors test_024 (CPU/NumPy) and compares against the same reference, so it
    checks that the torch path reproduces the NumPy propagation.
    """
    e_conv = 1e-13
    r_conv = 1e-13

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    cc = pycc.ccwfn(wfn, device='GPU')
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
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase).type(torch.complex128)
    y = y0
    ODE = rk4(h)

    # Propagate t0 -> tf with a fixed number of RK4 steps (a `while t < tf` bound
    # overshoots by one step here -- see test_024).
    nsteps = int(round((tf - t0) / h))
    t = t0
    for step in range(nsteps):
        y = ODE(rtcc.f, t, y)
        t = t0 + (step + 1) * h
        t1, t2, l1, l2, phase = rtcc.extract_amps(y)
        mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)

    # Same reference as test_024 (mu_z(t=tf) from this code, RK4, h=0.01): the torch
    # path must reproduce the NumPy propagation to ~1e-10.
    mu_z_ref = -0.078669702343987763
    assert abs(mu_z_ref - mu_z.real) < 1e-10
    # <mu_z> is a real observable; the propagated imaginary part is numerical noise.
    assert abs(mu_z.imag) < 1e-7
