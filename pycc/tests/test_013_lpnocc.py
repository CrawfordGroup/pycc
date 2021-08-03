"""
Test basic LPNO-CCSD energy and Lambda code
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_lpno_ccsd():
    """H2O LPNO-CCSD Test"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    ccsd = pycc.ccwfn(rhf_wfn, local='LPNO', local_cutoff=1e-5)
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)

    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)

    epsi4 = -0.221156413159674
    lpsi4 = -0.217144045119535

    assert (abs(epsi4 - eccsd) < 1e-7)
    assert (abs(lpsi4 - lccsd) < 1e-7)
