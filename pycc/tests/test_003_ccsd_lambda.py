"""
Test CCSD Lambda equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_lambda_ccsd_h2o(rhf_wfn):
    """H2O STO-3G"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "STO-3G")
    ccsd = pycc.ccwfn(wfn)
    eccsd = ccsd.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv)
    epsi4 = -0.070616830152761
    lpsi4 = -0.068826452648939
    assert (abs(epsi4 - eccsd) < 1e-11)
    assert (abs(lpsi4 - lccsd) < 1e-11)

    # cc-pVDZ basis set
    wfn = rhf_wfn("H2O", "cc-pVDZ")
    ccsd = pycc.ccwfn(wfn)
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv)
    epsi4 = -0.222029814166783
    lpsi4 = -0.217838951550509
    assert (abs(epsi4 - eccsd) < 1e-11)
    assert (abs(lpsi4 - lccsd) < 1e-11)
