"""
Test CCSD(T) energies for the spatial (RHF) and spin-orbital (UHF) kernels
(docs/archive/ENHANCEMENT_PLAN_2026-06.md, phase 4).
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
from ..cctriples import t_vikings, t_vikings_inverted, t_tjl

# Open-shell doublet (hydroxyl radical) for the UHF CCSD(T) checks.
OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""

def test_ccsd_t_h2o(rhf_wfn):
    """H2O cc-pVDZ"""
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "STO-3G")
    cc = pycc.ccwfn(wfn, model='ccsd(t)')
    eccsd = cc.solve_cc(e_conv,r_conv,maxiter)
    et_vik_ijk = t_vikings(cc)
    et_vik_abc = t_vikings_inverted(cc)
    et_tjl = t_tjl(cc)
    epsi4 = -0.000099957499645
    assert (abs(epsi4 - et_vik_ijk) < 1e-11)
    assert (abs(epsi4 - et_vik_abc) < 1e-11)
    assert (abs(epsi4 - et_tjl) < 1e-11)

    wfn = rhf_wfn("H2O", "cc-pVDZ")
    cc = pycc.ccwfn(wfn, model='ccsd(t)')
    eccsd = cc.solve_cc(e_conv,r_conv,maxiter)
    et_vik_ijk = t_vikings(cc)
    et_vik_abc = t_vikings_inverted(cc)
    et_tjl = t_tjl(cc)
    epsi4 = -0.003861236558801
    assert (abs(epsi4 - et_vik_ijk) < 1e-11)
    assert (abs(epsi4 - et_vik_abc) < 1e-11)
    assert (abs(epsi4 - et_tjl) < 1e-11)


def test_so_ccsd_t_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital CCSD(T) (forced) reproduces spin-adapted spatial CCSD(T) on a
    closed shell, isolating the spin-orbital (T) driver."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    e_spatial = pycc.CCwfn(wfn, model="CCSD(T)", frozen_core=False).solve_cc(
        e_conv=1e-11, r_conv=1e-11)

    so = pycc.CCwfn(wfn, model="CCSD(T)", frozen_core=False, orbital_basis="spinorbital")
    assert so.orbital_basis == "spinorbital"
    e_so = so.solve_cc(e_conv=1e-11, r_conv=1e-11)

    assert abs(e_so - e_spatial) < 1e-10


def test_uccsd_t_oh(uhf_wfn):
    """Open-shell .OH cc-pVDZ, all-electron UCCSD(T) vs Psi4's UCCSD(T)."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    cc = pycc.CCwfn(wfn, model="CCSD(T)", frozen_core=False)
    assert cc.orbital_basis == "spinorbital"
    eccsd_t = cc.solve_cc(e_conv=1e-11, r_conv=1e-11)

    psi4.energy('ccsd(t)')
    ref = psi4.variable('CCSD(T) CORRELATION ENERGY')

    assert abs(eccsd_t - ref) < 1e-10


def test_uccsd_t_oh_frzc(uhf_wfn):
    """Open-shell .OH cc-pVDZ, frozen-core UCCSD(T) vs Psi4's UCCSD(T)."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)

    eccsd_t = pycc.CCwfn(wfn, model="CCSD(T)").solve_cc(e_conv=1e-11, r_conv=1e-11)

    psi4.energy('ccsd(t)')
    ref = psi4.variable('CCSD(T) CORRELATION ENERGY')

    assert abs(eccsd_t - ref) < 1e-10
