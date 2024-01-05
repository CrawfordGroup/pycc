"""
Test EOM-CCSD solver.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

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
#    epsi4 = -0.227888246840310
#    ecfour = -0.2278882468404231
#    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)

    eom = pycc.cceom(hbar)

    N = 3
    maxiter = 75
    e_conv = 1e-7
    r_conv = 1e-7
    guess = 'HBAR_SS'
    eom.solve_eom(N, e_conv, r_conv, maxiter, guess)
