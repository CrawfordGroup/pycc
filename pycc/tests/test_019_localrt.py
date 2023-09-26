"""
Test RT-CCSD with local correlation simulation.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import os
from pycc.rt.lasers import gaussian_laser
from pycc.rt.integrators import rk4
from ..data.molecules import *
from pytest import fixture
from distutils import dir_util

@fixture
def datadir(tmpdir, request):
    '''
    from: https://stackoverflow.com/a/29631801
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    else:
        raise FileNotFoundError("Test folder not found.")

    return tmpdir

def test_rtpno(datadir):
    """H2O RT-PNO"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1,
                      'local_convergence': 1.e-13})
    mol = psi4.geometry(moldict["H2O"]+"\nnoreorient\nnocom")
    ref_dir = str(datadir.join(f"wfn.npy"))
    rhf_wfn = psi4.core.Wavefunction.from_file(ref_dir)
    
    maxiter = 200
    e_conv = 1e-13
    r_conv = 1e-13
    max_diis = 8

    cc = pycc.ccwfn(rhf_wfn, local='PNO', local_cutoff=1e-5, filter=True)
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
    phase = 0
    t0 = 0
    tf = 0.5
    h = 0.02
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase).astype('complex128')
    ODE = rk4(h)
    
    # Propagate
    ret = rtcc.propagate(ODE, y0, tf, ti=t0, ref=False)

    # check
    ref = {'ecc': (-84.21331867940133+4.925945912792495e-17j), 
           'mu_x': (-5.106207671158796e-05+3.641896436116718e-12j), 
           'mu_y': (-5.001503722097678e-05-1.7436592314191415e-12j), 
           'mu_z': (-0.06905411053873889-9.328439713393588e-12j)}
    for prop in ret['0.50']:
        assert (abs(ret['0.50'][prop] - ref[prop]) < 1e-8) 

def test_rtpao(datadir):
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
                      'diis': 1,
                      'local_convergence': 1.e-13})
    mol = psi4.geometry(moldict["H2O"]+"\nnoreorient\nnocom")
    ref_dir = str(datadir.join(f"wfn.npy"))
    rhf_wfn = psi4.core.Wavefunction.from_file(ref_dir)

    maxiter = 75
    e_conv = 1e-13
    r_conv = 1e-13
    max_diis = 8

    cc = pycc.ccwfn(rhf_wfn, local='PAO', local_cutoff=1e-2, filter=True)
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
    phase = 0
    t0 = 0
    tf = 0.5
    h = 0.02
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, phase).astype('complex128')
    ODE = rk4(h)
    
    # Propagate
    ret = rtcc.propagate(ODE, y0, tf, ti=t0, ref=False)

    # check
    ref = {'ecc': (-84.21540972040579+2.9584453421137937e-16j), 
           'mu_x': (-4.987717148832141e-05-2.5885460555301484e-12j), 
           'mu_y': (-4.707786986481166e-05-1.5660290548026362e-11j), 
           'mu_z': (-0.0783037960868978-1.168135844689433e-11j)}
    for prop in ret['0.50']:
        assert (abs(ret['0.50'][prop] - ref[prop]) < 1e-8) 
