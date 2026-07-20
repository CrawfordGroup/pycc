"""
Test basic PNO-CCD, PNO++-CCD, PAO-CCD energies
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

@pytest.mark.slow
def test_pno_ccd(rhf_wfn):
    """H2O PNO-CCD Test"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pno-ccd
    ccd_sim = pycc.ccwfn(wfn, model='CCD',local='PNO', local_cutoff=1e-5,it2_opt=False,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno-ccd
    lccd = pycc.ccwfn(wfn,model='CCD', local='PNO', local_cutoff=1e-5,it2_opt=False)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12)

@pytest.mark.slow
def test_pnopp_ccd(rhf_wfn):
    """H2O PNO++ CCD Test"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "6-31G", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pno++-ccd
    ccd_sim = pycc.ccwfn(wfn, model='CCD',local='PNO++', local_cutoff=1e-7,it2_opt=False,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno++-ccd
    lccd = pycc.ccwfn(wfn,model='CCD', local='PNO++', local_cutoff=1e-7,it2_opt=False)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12)

@pytest.mark.slow
def test_pno_ccd_opt(rhf_wfn):
    """H2O PNO-CCD with Optimized Initial T2 Amplitudes"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pno-ccd
    ccd_sim = pycc.ccwfn(wfn, model='CCD',local='PNO', local_cutoff=1e-7,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno-ccd
    lccd = pycc.ccwfn(wfn,model='CCD', local='PNO', local_cutoff=1e-7)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12)

@pytest.mark.slow
def test_pao_ccd_opt(rhf_wfn):
    """H2O PAO-CCD with Optimized Initial T2 Amplitudes"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pao-ccd
    ccd_sim = pycc.ccwfn(wfn, model='CCD',local='PAO', local_cutoff=2e-2,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pao-ccd
    lccd = pycc.ccwfn(wfn,model='CCD', local='PAO', local_cutoff=2e-2)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12)
