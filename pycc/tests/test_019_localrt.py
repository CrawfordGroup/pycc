"""
Test RT-CCSD with local correlation simulation.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from pycc.rt.lasers import gaussian_laser
from pycc.rt.integrators import rk4
from ..data.molecules import *

def test_rtpno():
    """H2O RT-LPNO"""
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
    mol = psi4.geometry(moldict["H2O"]+"\nnoreorient\nnocom")
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-13
    r_conv = 1e-13
    max_diis = 8

    cc = pycc.ccwfn(rhf_wfn, local='LPNO', local_cutoff=1e-5)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter, max_diis)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    ccdensity = pycc.ccdensity(cc, cclambda)

    # Narrow Gaussian pulse 
    F_str = 0.001 
    omega = 0
    sigma = 0.01
    center = 0.05
    V = gaussian_laser(F_str, omega, sigma, center=center)

    # RT-CC Setup
    t0 = 0
    tf = 0.5
    h = 0.02
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2).astype('complex128')
    ODE = rk4(h)
    
    # Propagate
    ret = rtcc.propagate(ODE, y0, tf, ti=t0, ref=False)

    # check
    ref = {'ecc': (-84.21331867964982+5.150113844027836e-17j), 
           'mu_x': (-5.106207675194608e-05+3.641896201488591e-12j), 
           'mu_y': (-5.001589647321558e-05-1.7454256152926208e-12j), 
           'mu_z': (-0.06905411136458055-9.326692043866045e-12j)}
    for prop in ret['0.50']:
        assert (abs(ret['0.50'][prop] - ref[prop]) < 1e-10) 

def test_rtpao():
    """H2O RT-PAO"""
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
    mol = psi4.geometry(moldict["H2O"]+"\nnoreorient\nnocom")
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-13
    r_conv = 1e-13
    max_diis = 8

    cc = pycc.ccwfn(rhf_wfn, local='PAO', local_cutoff=1e-2)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter, max_diis)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    ccdensity = pycc.ccdensity(cc, cclambda)

    # Narrow Gaussian pulse 
    F_str = 0.001 
    omega = 0
    sigma = 0.01
    center = 0.05
    V = gaussian_laser(F_str, omega, sigma, center=center)

    # RT-CC Setup
    t0 = 0
    tf = 0.5
    h = 0.02
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2).astype('complex128')
    ODE = rk4(h)
    
    # Propagate
    ret = rtcc.propagate(ODE, y0, tf, ti=t0, ref=False)

    # check
    ref = {'ecc': (-84.21540972042284+2.958562073841837e-16j), 
           'mu_x': (-4.9877171488058465e-05-2.5885461834321234e-12j), 
           'mu_y': (-4.707816563388062e-05-1.566080244561177e-11j), 
           'mu_z': (-0.07830379603787624-1.1680833071320664e-11j)}
    for prop in ret['0.50']:
        assert (abs(ret['0.50'][prop] - ref[prop]) < 1e-10) 
