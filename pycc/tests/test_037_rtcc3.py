"""
Test RT-CC3 propagation on water molecule.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from pycc.rt.integrators import rk4
from pycc.rt.lasers import qrcw_laser
from ..data.molecules import *

def test_rtcc_he_cc_pvdz():
    """H2O cc-pVDZ"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O_Teach"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-12
    r_conv = 1e-12

    cc = pycc.ccwfn(rhf_wfn, model='CC3', real_time=True)
    ecc = cc.solve_cc(e_conv, r_conv)

    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)

    ccdensity = pycc.ccdensity(cc, cclambda)

    # quadratic ramped continuous wave (QRCW)
    F_str = 0.002
    omega = 0.078
    nr = 1
    V = qrcw_laser(F_str, omega, nr)

    # RT-CC Setup
    phase = 0
    t0 = 0
    tf = 0.05
    h = 0.01
    t = t0
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V, kick='x')
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase)
    y = y0
    ODE = rk4(h)
    t1, t2, l1, l2, phase = rtcc.extract_amps(y0)
    mu0_x, mu0_y, mu0_z = rtcc.dipole(t1, t2, l1, l2)

    mu_z_ref = -0.0859645691 #CFOUR

    while t < tf:
        y = ODE(rtcc.f, t, y)
        t += h
        t1, t2, l1, l2, phase = rtcc.extract_amps(y)
        mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2, real_time=True)
        
    print(mu_z)
    assert (abs(mu_z_ref - mu_z.real) < 1e-10)

