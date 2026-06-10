"""
Test CC3 equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np
from pycc.ccwfn import HAS_EINSUMS

# H2O/cc-pVDZ
@pytest.mark.skipif(not HAS_EINSUMS, reason="Einsums not installed")
def test_cc3_h2o(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O_Teach", "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='CC3', einsums=True)
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.227888246840310
    ecfour = -0.2278882468404231
    assert (abs(epsi4 - ecc) < 1e-11)

