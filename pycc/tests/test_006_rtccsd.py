"""
Test RT-CCSD propagation on He atom.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from scipy.integrate import complex_ode as ode
from pycc.rt.lasers import sine_square_laser
from ..data.molecules import *

def test_rtcc_he_cc_pvdz():
    """He cc-pVDZ"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["He"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-13
    r_conv = 1e-13

    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)

    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)

    ccdensity = pycc.ccdensity(cc, cclambda)

    # Sine squared pulse (a.u.)
    F_str = 1.0
    omega = 2.87
    tprime = 5.0
    V = sine_square_laser(F_str, omega, tprime)

    # RT-CC Setup
    phase = 0
    t0 = 0
    tf = 1.0
    h = 0.01
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase).astype('complex128')
    ODE = ode(rtcc.f).set_integrator('vode',atol=1e-13,rtol=1e-13)
    ODE.set_initial_value(y0, t0)

    t1, t2, l1, l2, phase = rtcc.extract_amps(y0)
    mu0_x, mu0_y, mu0_z = rtcc.dipole(t1, t2, l1, l2)
    ecc0 = rtcc.lagrangian(t0, t1, t2, l1, l2)

    mu_z_ref = 0.008400738202694  # a.u.

    while ODE.successful() and ODE.t < tf:
        y = ODE.integrate(ODE.t+h)
        t = ODE.t
        t1, t2, l1, l2, phase = rtcc.extract_amps(y)
        mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)
        ecc = rtcc.lagrangian(t, t1, t2, l1, l2)

    print(mu_z)
    assert (abs(mu_z_ref - mu_z.real) < 1e-10)

