"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
from pycc.ccwfn import HAS_EINSUMS

@pytest.mark.skipif(not HAS_EINSUMS, reason="Einsums not installed")
def test_ccsd_h2o(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    # STO-3G basis set
    wfn = rhf_wfn("H2O", "STO-3G")
    ccsd = pycc.ccwfn(wfn, einsums=True)
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.070616830152761
    assert (abs(epsi4 - eccsd) < 1e-11)

    # cc-pVDZ basis set
    wfn = rhf_wfn("H2O", "cc-pVDZ")
    ccsd = pycc.ccwfn(wfn, einsums=True)
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.222029814166783
    assert (abs(epsi4 - eccsd) < 1e-11)
