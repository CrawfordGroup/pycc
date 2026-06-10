"""
Test basic PNO++-CCSD energy
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_pnopp_ccsd(rhf_wfn):
    """H2O PNO++-CCSD Test"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false", diis=8,
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)
    ccsd = pycc.ccwfn(wfn, local='PNO++', local_cutoff=1e-7, it2_opt=False)
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)
    
    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)
    
    esim = -0.216064367834782
    lsim = -0.211938482158711 
   
    assert (abs(esim - eccsd) < 1e-7)
    assert (abs(lsim - lccsd) < 1e-7)   

def test_pnopp_ccsd_opt(rhf_wfn):
    """H2O PNO++-CCSD with Optimized Initial T2 Amplitudes"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false", diis=8,
                  e_convergence=1e-13, d_convergence=1e-13, r_convergence=1e-13)
    ccsd = pycc.ccwfn(wfn, local='PNO++', local_cutoff=1e-9)
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)

    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)

    esim = -0.223695614177277
    lsim = -0.219502543226770   
 
    assert (abs(esim - eccsd) < 1e-7)
    assert (abs(lsim - lccsd) < 1e-7)
