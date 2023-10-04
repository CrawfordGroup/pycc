
"""
Test CCSD density equations using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_density_ccsd_h2o():
    """H2O"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    
    ccsd = pycc.ccwfn(rhf_wfn)
    eccsd = ccsd.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.:wqcclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv)
    epsi4 = -0.070616830152761
    lpsi4 = -0.068826452648939
    ccdensity = pycc.ccdensity(ccsd, cclambda)
    ecc_density = ccdensity.compute_energy()
    assert (abs(epsi4 - eccsd) < 1e-11)
    assert (abs(lpsi4 - lccsd) < 1e-11)
    assert (abs(epsi4 - ecc_density) < 1e-11)

    psi4.set_options({'basis': 'cc-pVDZ'})
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    ccsd = pycc.ccwfn(rhf_wfn)
    eccsd = ccsd.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv)
    epsi4 = -0.222029814166783
    lpsi4 = -0.217838951550509
    ccdensity = pycc.ccdensity(ccsd, cclambda)
    ecc_density = ccdensity.compute_energy()
    assert (abs(epsi4 - eccsd) < 1e-11)
    assert (abs(lpsi4 - lccsd) < 1e-11)
    assert (abs(epsi4 - ecc_density) < 1e-11)
