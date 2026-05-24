"""
Test EOM-CCSD solver.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np
import sys

np.set_printoptions(precision=10, linewidth=300, threshold=sys.maxsize, suppress=True)

# H2O/cc-pVDZ
def test_eomccsd_h2o():
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O_Teach"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    hbar = pycc.cchbar(cc)
    eom = pycc.cceom(cc, hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7

    guess = 'hbar_ss'
    eom_E_guess_1, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "right")

    guess = 'cis'
    eom_E_guess_2, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "right")

    guess = 'unit'
    eom_E_guess_3, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "right")

    psi_E = [0.2464015742, 0.3136327374, 0.3543763364]
    assert(abs(eom_E_guess_1[0] - psi_E[0]) < 1e-5)
    assert(abs(eom_E_guess_2[1] - psi_E[1]) < 1e-5)
    assert(abs(eom_E_guess_3[2] - psi_E[2]) < 1e-5)

# Test frozen core
def test_eomccsd_h2o_fc():
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O_Teach"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    hbar = pycc.cchbar(cc)
    eom = pycc.cceom(cc, hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7

    guess = 'hbar_ss'
    eom_E_guess_1, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "left")

    guess = 'cis'
    eom_E_guess_2, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "left")

    guess = 'unit'
    eom_E_guess_3, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "left")

    psi_E = [0.2463657776, 0.3135919025, 0.3543900976]
    assert(abs(eom_E_guess_1[0] - psi_E[0]) < 1e-5)
    assert(abs(eom_E_guess_2[1] - psi_E[1]) < 1e-5)
    assert(abs(eom_E_guess_3[2] - psi_E[2]) < 1e-5)


def test_eomccsd_c2h4_fc():
    # Psi4 Setup
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry("""
C            0.000000000000     0.667203595356     0.000000000000
C            0.000000000000    -0.667196404644     0.000000000000
H           -0.931693962629     1.221303595356     0.000000000000
H            0.931693962629     1.221203595356     0.000000000000
H            0.931693962629    -1.221296404644     0.000000000000
H           -0.931693962629    -1.221296404644     0.000000000000
units angstrom
""")
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    hbar = pycc.cchbar(cc)
    eom = pycc.cceom(cc, hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7

    guess = 'hbar_ss'
    eom_E_guess_1, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "right")

    guess = 'cis'
    eom_E_guess_2, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "right")

    guess = 'unit'
    eom_E_guess_3, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess, "right")

    psi4.set_options({'cceom__e_convergence': e_conv,
                      'cceom__r_convergence': r_conv,
                      'roots_per_irrep': [N],
                      'restart': 'false'})

    psi_E = [0.3260065052, 0.3298285196, 0.3345771530]
    assert(abs(eom_E_guess_1[0] - psi_E[0]) < 1e-5)
    assert(abs(eom_E_guess_2[1] - psi_E[1]) < 1e-5)
    assert(abs(eom_E_guess_3[2] - psi_E[2]) < 1e-5)

