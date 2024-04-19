"""
Test RT-CCSD propagation with RK4 integrator on water molecule.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pycc
import pytest
from pycc.rt.integrators import rk4 
from pycc.rt.lasers import gaussian_laser
from ..data.molecules import *

def test_rtcc_water_cc_pvdz():
    """H2O cc-pVDZ"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-7
    r_conv = 1e-7
    
    # The precision for the calculation (single-precision (sp)
    # /double-precision (dp)) can be specified.
    # Option: 'SP', 'DP'
    # Default: 'DP'
    #cc = pycc.ccwfn(rhf_wfn)
    cc = pycc.ccwfn(rhf_wfn, precision='SP')
    ecc = cc.solve_cc(e_conv, r_conv)
    
    # Check CCSD energy
    epsi4 = -0.223910018703551
    assert (abs(epsi4 - ecc) < 1e-7)    
    
    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    
    # Check CCSD pseudo energy
    lepsi4 = -0.219688229733875
    assert (abs(lepsi4 - lecc) < 1e-7)

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
    tf = 0.01
    h = 0.01
    t = t0
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase)
    y = y0
    ODE = rk4(h)
    t1, t2, l1, l2, phase = rtcc.extract_amps(y0)
    mu0_x, mu0_y, mu0_z = rtcc.dipole(t1, t2, l1, l2)
    ecc0 = rtcc.lagrangian(t0, t1, t2, l1, l2)
   
    # Check dipole moment 
    mu0_z_ref = -0.0780069121607703 # computed by removing SCF from original ref
    assert(abs(mu0_z_ref - mu0_z) < 1e-6)
    
    while t < tf:
        y = ODE(rtcc.f, t, y)
        t += h 
        t1, t2, l1, l2, phase = rtcc.extract_amps(y)
        mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)
        ecc = rtcc.lagrangian(t, t1, t2, l1, l2)
        
    print(mu_z)
   
    # Check the dipole value at time step 1
    mu_z_ref = -0.0780069121607703 # computed by removing SCF from original ref
    assert (abs(mu_z_ref - mu_z.real) < 1e-6)    

