"""
Test CC2 equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_cc2_h2o(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='CC2')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.215857544656
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    lcc_psi4 = -0.215765740373555
    assert(abs(lcc - lcc_psi4) < 1e-11)

    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc = ccdensity.compute_energy()
    assert (abs(epsi4 - ecc) < 1e-11)

def test_cc2_h2(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2", "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='CC2')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.026445902512140185
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    lcc_psi4 = -0.026443139737993
    assert(abs(lcc - lcc_psi4) < 1e-11)

    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc = ccdensity.compute_energy()
    assert (abs(epsi4 - ecc) < 1e-11)