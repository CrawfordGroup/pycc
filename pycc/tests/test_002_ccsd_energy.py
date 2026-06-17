"""
Test CCSD equation solution using various molecule test cases.

Covers both the spin-adapted spatial (RHF) kernel and the spin-orbital (UHF)
kernel selected by orbital_basis (docs/ENHANCEMENT_PLAN_2026-06.md, phase 3).
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

# Open-shell doublet (hydroxyl radical) for the UHF CCSD checks.
OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""

def test_ccsd_h2o(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    # STO-3G basis set
    wfn = rhf_wfn("H2O", "STO-3G")
    ccsd = pycc.ccwfn(wfn)
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.070616830152761
    assert (abs(epsi4 - eccsd) < 1e-11)

    # cc-pVDZ basis set
    wfn = rhf_wfn("H2O", "cc-pVDZ")
    ccsd = pycc.ccwfn(wfn)
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.222029814166783
    assert (abs(epsi4 - eccsd) < 1e-11)


def test_so_ccsd_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital CCSD (forced) reproduces spin-adapted spatial CCSD on a closed
    shell, isolating the spin-orbital residual kernel."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    e_spatial = pycc.CCwfn(wfn, frozen_core=False).solve_cc(e_conv=1e-11, r_conv=1e-11)

    so = pycc.CCwfn(wfn, frozen_core=False, orbital_basis="spinorbital")
    assert so.orbital_basis == "spinorbital"
    e_so = so.solve_cc(e_conv=1e-11, r_conv=1e-11)

    assert abs(e_so - e_spatial) < 1e-10


def test_uccsd_oh(uhf_wfn):
    """Open-shell .OH cc-pVDZ, all-electron UCCSD vs Psi4's UCCSD."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    cc = pycc.CCwfn(wfn, frozen_core=False)
    assert cc.orbital_basis == "spinorbital"
    eccsd = cc.solve_cc(e_conv=1e-11, r_conv=1e-11)

    psi4.energy('ccsd')
    ref = psi4.variable('CCSD CORRELATION ENERGY')

    assert abs(eccsd - ref) < 1e-10


def test_uccsd_oh_frzc(uhf_wfn):
    """Open-shell .OH cc-pVDZ, frozen-core UCCSD vs Psi4's UCCSD."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)

    eccsd = pycc.CCwfn(wfn).solve_cc(e_conv=1e-11, r_conv=1e-11)

    psi4.energy('ccsd')
    ref = psi4.variable('CCSD CORRELATION ENERGY')

    assert abs(eccsd - ref) < 1e-10
