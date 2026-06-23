"""
Test CC3 equation solution using various molecule test cases, for both the
spatial (RHF) and spin-orbital (UHF) kernels (docs/ENHANCEMENT_PLAN_2026-06.md,
phase 4b).
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np

# Open-shell doublet (hydroxyl radical) for the UHF CC3 checks. CC3 rebuilds the
# triples every iteration, so a small basis keeps the spin-orbital runs quick.
OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""

# H2O/cc-pVDZ
def test_cc3_h2o(rhf_wfn):
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O_Teach", "cc-pVDZ", freeze_core="false")
    cc = pycc.ccwfn(wfn, model='CC3')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.227888246840310
    ecfour = -0.2278882468404231
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    print("lcc: ", lcc)
    lcc_cfour = -0.2233231845185215 
    assert(abs(lcc - lcc_cfour) < 1e-11)

    ccdensity = pycc.ccdensity(cc, cclambda)
    # no laser
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, None, magnetic = False)
    
    CFOUR = [0, 0, 0.7703875967] # CFOUR total dipole (CC3 + SCF + nuclear)
    scf = wfn.variable('SCF DIPOLE') # PSI4 reference dipole (SCF + nuclear)
    ref = CFOUR - scf # Final reference: CC3 only

    mu_x, mu_y, mu_z = rtcc.dipole(cc.t1, cc.t2, cclambda.l1, cclambda.l2)

    assert (abs(ref[1] - np.real(mu_y)) < 1E-10)
    assert (abs(ref[2] - np.real(mu_z)) < 1E-10)


def test_so_cc3_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital CC3 (forced) reproduces spin-adapted spatial CC3 on a closed
    shell, isolating the spin-orbital CC3 kernel."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    e_spatial = pycc.CCwfn(wfn, model="CC3", frozen_core=False).solve_cc(
        e_conv=1e-11, r_conv=1e-11)

    so = pycc.CCwfn(wfn, model="CC3", frozen_core=False, orbital_basis="spinorbital")
    assert so.orbital_basis == "spinorbital"
    e_so = so.solve_cc(e_conv=1e-11, r_conv=1e-11)

    assert abs(e_so - e_spatial) < 1e-10


def test_so_cc3_lambda_equals_spatial_rhf(rhf_wfn):
    """Spin-orbital CC3 Lambda (forced) reproduces the spin-adapted spatial CC3
    Lambda pseudoenergy on a closed shell -- keystone for the spin-orbital CC3
    Lambda kernel. The spatial value is itself CFOUR-pinned in test_cc3_h2o."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    def _lcc(basis):
        cc = pycc.CCwfn(wfn, model="CC3", frozen_core=False, orbital_basis=basis)
        cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
        hbar = pycc.cchbar(cc)
        lam = pycc.cclambda(cc, hbar)
        return lam.solve_lambda(e_conv=1e-11, r_conv=1e-11)

    l_spatial = _lcc("spatial")
    l_so = _lcc("spinorbital")
    assert abs(l_so - l_spatial) < 1e-10


def test_so_cc3_equals_spatial_rhf_frzc(rhf_wfn):
    """Frozen-core CC3 (closed shell): spin-orbital reproduces spin-adapted spatial, and
    both match Psi4's CC3. Validates the frozen-core triples indexing -- the spin-orbital
    Hamiltonian is now full-MO (the active occupied no longer starts at index 0), so the
    triples kernels must slice the Hamiltonian (``ERI[o,o,v,v][j,k]``) rather than index it
    with loop variables (``ERI[j,k,v,v]``)."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)

    e_spatial = pycc.CCwfn(wfn, model="CC3").solve_cc(e_conv=1e-11, r_conv=1e-11)

    so = pycc.CCwfn(wfn, model="CC3", orbital_basis="spinorbital")
    assert so.orbital_basis == "spinorbital" and so.nfzc > 0
    e_so = so.solve_cc(e_conv=1e-11, r_conv=1e-11)

    assert abs(e_so - e_spatial) < 1e-10
    psi4.energy('cc3')
    assert abs(e_spatial - psi4.variable('CC3 CORRELATION ENERGY')) < 1e-10


def test_so_cc3_lambda_equals_spatial_rhf_frzc(rhf_wfn):
    """Frozen-core CC3 Lambda (closed shell): spin-orbital reproduces spin-adapted spatial
    -- keystone for the frozen-core CC3 Lambda triples indexing."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)

    def _lcc(basis):
        cc = pycc.CCwfn(wfn, model="CC3", orbital_basis=basis)
        cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
        hbar = pycc.cchbar(cc)
        lam = pycc.cclambda(cc, hbar)
        return lam.solve_lambda(e_conv=1e-11, r_conv=1e-11)

    assert abs(_lcc("spinorbital") - _lcc("spatial")) < 1e-10


def test_ucc3_oh(uhf_wfn):
    """Open-shell .OH 6-31G, all-electron UCC3 vs Psi4's UCC3."""
    wfn = uhf_wfn(OH, "6-31G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    cc = pycc.CCwfn(wfn, model="CC3", frozen_core=False)
    assert cc.orbital_basis == "spinorbital"
    ecc3 = cc.solve_cc(e_conv=1e-11, r_conv=1e-11)

    psi4.energy('cc3')
    ref = psi4.variable('CC3 CORRELATION ENERGY')

    assert abs(ecc3 - ref) < 1e-10
    # Frozen-core CC3 is validated on a closed shell by
    # test_so_cc3_equals_spatial_rhf_frzc / _lambda_frzc (energy and Lambda, vs Psi4 and
    # the spin-orbital==spatial keystone). The full-MO spin-orbital Hamiltonian means the
    # active occupied no longer starts at index 0, so the triples kernels slice the
    # Hamiltonian rather than indexing it with loop variables. Open-shell frozen-core UCC3
    # is not separately retested (same triples indexing; the iterative CC3 solve is the
    # costliest in the suite), and frozen-core UCCSD(T) is covered by test_005.

