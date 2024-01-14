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
    eom = pycc.cceom(hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7

    guess = 'hbar_ss'
    eom_E_guess_1, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    guess = 'cis'
    eom_E_guess_2, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    guess = 'unit'
    eom_E_guess_3, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    psi4.set_options({'cceom__e_convergence': e_conv,
                      'cceom__r_convergence': r_conv,
                      'roots_per_irrep': [N]})
    psi4.energy('eom-ccsd')
    for i in range(N):
        var_str = "CCSD ROOT {} CORRELATION ENERGY".format(i + 1)
        psi_E = psi4.core.variable(var_str)
        assert(abs(eom_E_guess_1[i] - psi_E) < 1e-5)
        assert(abs(eom_E_guess_2[i] - psi_E) < 1e-5)
        assert(abs(eom_E_guess_3[i] - psi_E) < 1e-5)

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
    eom = pycc.cceom(hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7

    guess = 'hbar_ss'
    eom_E_guess_1, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    guess = 'cis'
    eom_E_guess_2, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    guess = 'unit'
    eom_E_guess_3, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    psi4.set_options({'cceom__e_convergence': e_conv,
                      'cceom__r_convergence': r_conv,
                      'roots_per_irrep': [N],
                      'restart': 'false'})
    psi4.energy('eom-ccsd')
    for i in range(N):
        var_str = "CCSD ROOT {} CORRELATION ENERGY".format(i + 1)
        psi_E = psi4.core.variable(var_str)
        assert(abs(eom_E_guess_1[i] - psi_E) < 1e-5)
        assert(abs(eom_E_guess_2[i] - psi_E) < 1e-5)
        assert(abs(eom_E_guess_3[i] - psi_E) < 1e-5)

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
    mol = psi4.geometry("pubchem:ethylene")
    mol.reset_point_group("c1")
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    hbar = pycc.cchbar(cc)
    eom = pycc.cceom(hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7

    guess = 'hbar_ss'
    eom_E_guess_1, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    guess = 'cis'
    eom_E_guess_2, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    guess = 'unit'
    eom_E_guess_3, _ = eom.solve_eom(N, e_conv, r_conv, maxiter, guess)

    psi4.set_options({'cceom__e_convergence': e_conv,
                      'cceom__r_convergence': r_conv,
                      'roots_per_irrep': [N],
                      'restart': 'false'})
    psi4.energy('eom-ccsd')
    for i in range(N):
        var_str = "CCSD ROOT {} CORRELATION ENERGY".format(i + 1)
        psi_E = psi4.core.variable(var_str)
        assert(abs(eom_E_guess_1[i] - psi_E) < 1e-5)
        assert(abs(eom_E_guess_2[i] - psi_E) < 1e-5)
        assert(abs(eom_E_guess_3[i] - psi_E) < 1e-5)

