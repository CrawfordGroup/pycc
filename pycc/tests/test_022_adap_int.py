"""
Test RT-CCSD propagation with Cash-Karp integrator (adaptive) on water molecule.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pycc
import pytest
from pycc.rt.integrators import ck 
from pycc.rt.lasers import gaussian_laser
from ..data.molecules import *


def test_rtcc_water_cc_pvdz():
    """H2O cc-pVDZ"""
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
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-13
    r_conv = 1e-13

    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)

    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)

    ccdensity = pycc.ccdensity(cc, cclambda)

    # Gaussian pulse (a.u.)
    F_str = 0.01
    omega = 0
    sigma = 0.01
    center = 0.05
    V = gaussian_laser(F_str, omega, sigma, center)

    # RT-CC Setup
    phase = 0
    t0 = 0
    tf = 1
    h = 0.01
    t = t0
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase).astype('complex128')
    y = y0
    # Setting for the adaptive integrator
    maxiter = 10
    yconv = 1e-7
    ODE = ck(maxiter, yconv)
    t1, t2, l1, l2, phase = rtcc.extract_amps(y0)
    mu0_x, mu0_y, mu0_z = rtcc.dipole(t1, t2, l1, l2)
    ecc0 = rtcc.lagrangian(t0, t1, t2, l1, l2)

    # For saving data at each time step.
    """
    dip_x = []
    dip_y = []
    dip_z = []
    time_points = []
    dip_x.append(mu0_x)
    dip_y.append(mu0_y)
    dip_z.append(mu0_z)
    time_points.append(t)
    """
    
    while t < tf:
        (y, h_old, h) = ODE(rtcc.f, t, y, h)
        t += h_old
        t1, t2, l1, l2, phase = rtcc.extract_amps(y)
        mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)
        ecc = rtcc.lagrangian(t, t1, t2, l1, l2)
        """
        dip_x.append(mu_x)
        dip_y.append(mu_y)
        dip_z.append(mu_z)
        time_points.append(t)
        """
        
    print(mu_z)
    mu_z_ref = -0.0780067603267549 # computed by removing SCF from original ref
    assert (abs(mu_z_ref - mu_z.real) < 1e-3)
    
    #return (dip_x, dip_y, dip_z, time_points)

#dip = test_rtcc_water_cc_pvdz()
#np.savez('h2o_F_0.01_h_0.01_t_1_rk4', dip_x=dip[0], dip_y=dip[1], dip_z=dip[2], time_points=dip[3])



