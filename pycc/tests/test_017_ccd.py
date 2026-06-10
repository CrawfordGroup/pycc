"""
Test CCD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_ccd_h2o(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='CCD')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.222559319034  # Checked against CFOUR
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    lcc_cfour = -0.218758826700
    assert (abs(lcc - lcc_cfour) < 1e-11)

    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc = ccdensity.compute_energy()
    assert (abs(epsi4 - ecc) < 1e-11)
