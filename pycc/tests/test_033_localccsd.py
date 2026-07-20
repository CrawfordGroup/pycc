"""
Test basic PNO-CCSD, PNO++-CCSD, PAO-CCSD energies
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

@pytest.mark.slow
def test_pno_ccsd(rhf_wfn):
    """H2O PNO-CCSD Test"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pno-ccsd
    ccsd_sim = pycc.ccwfn(wfn, model='CCSD',local='PNO', local_cutoff=1e-5,it2_opt=False,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno-ccsd
    lccsd = pycc.ccwfn(wfn,model='CCSD', local='PNO', local_cutoff=1e-5,it2_opt=False)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-12)

@pytest.mark.slow
def test_pnopp_ccsd(rhf_wfn):
    """H2O PNO++ CCSD Test"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "6-31G", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pno++-ccsd
    ccsd_sim = pycc.ccwfn(wfn, model='CCSD',local='PNO++', local_cutoff=1e-7,it2_opt=False,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno++-ccsd
    lccsd = pycc.ccwfn(wfn,model='CCSD', local='PNO++', local_cutoff=1e-7,it2_opt=False)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-12)

@pytest.mark.slow
def test_pno_ccsd_opt(rhf_wfn):
    """H2O PNO-CCSD with Optimized Initial T2 Amplitudes"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pno-ccsd
    ccsd_sim = pycc.ccwfn(wfn, model='CCSD',local='PNO', local_cutoff=1e-5,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno-ccsd
    lccsd = pycc.ccwfn(wfn,model='CCSD', local='PNO', local_cutoff=1e-5)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-12)

@pytest.mark.slow
def test_pao_ccsd_opt(rhf_wfn):
    """H2O PAO-CCSD with Optimized Initial T2 Amplitudes"""
    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)

    #simulation code of pao-ccsd
    ccsd_sim = pycc.ccwfn(wfn, model='CCSD',local='PAO', local_cutoff=2e-2,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pao-ccsd
    lccsd = pycc.ccwfn(wfn,model='CCSD', local='PAO', local_cutoff=2e-2)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-12)
