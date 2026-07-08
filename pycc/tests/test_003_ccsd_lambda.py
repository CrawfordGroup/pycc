"""
Test CCSD Lambda equation solution using various molecule test cases, for both the
spin-adapted spatial (RHF) and spin-orbital (UHF/ROHF) kernels
(docs/archive/ENHANCEMENT_PLAN_2026-06.md).
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

# OH doublet at 1.83 bohr (verbatim for both codes); CFOUR Lambda pseudoenergies
# (cc-pVDZ, all-electron, semicanonical -- CFOUR PRINT=2 reports the total Lambda
# pseudoenergy, defined as PyCC's 1/4 <ij||ab> l2_ijab).
OH_BOHR = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.83
units bohr
no_com
no_reorient
symmetry c1
"""
CFOUR_UHF_LAMBDA = -0.162890511230
CFOUR_ROHF_LAMBDA = -0.163565638102


def _lambda(wfn, **cckwargs):
    cc = pycc.CCwfn(wfn, **cckwargs)
    cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    return pycc.cclambda(cc, hbar).solve_lambda(e_conv=1e-11, r_conv=1e-11)


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


def test_so_lambda_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital Lambda (forced) reproduces the spin-adapted spatial Lambda
    pseudoenergy on a closed shell, isolating the spin-orbital HBAR + Lambda kernel."""
    wfn = rhf_wfn("H2O", "STO-3G")
    l_spatial = _lambda(wfn)
    l_so = _lambda(wfn, orbital_basis="spinorbital")
    assert abs(l_so - l_spatial) < 1e-10


def test_uhf_lambda_vs_cfour(uhf_wfn):
    """Open-shell .OH cc-pVDZ: spin-orbital UHF Lambda pseudoenergy vs CFOUR."""
    wfn = uhf_wfn(OH_BOHR, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    lcc = _lambda(wfn)
    assert abs(lcc - CFOUR_UHF_LAMBDA) < 1e-10


def test_rohf_lambda_vs_cfour(rohf_wfn):
    """Open-shell .OH cc-pVDZ: spin-orbital ROHF Lambda pseudoenergy vs CFOUR
    (semicanonical, as CFOUR uses)."""
    wfn = rohf_wfn(OH_BOHR, "cc-pVDZ", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)
    lcc = _lambda(wfn)
    assert abs(lcc - CFOUR_ROHF_LAMBDA) < 1e-10
