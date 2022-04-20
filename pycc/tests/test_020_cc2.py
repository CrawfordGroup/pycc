"""
Test CC2 equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_cc2_h2o():
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
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn, model='CC2')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.215857544656
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    lcc_psi4 = -0.215765740373555
    assert(abs(lcc - lcc_psi4) < 1e-11)

    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc = ccdensity.compute_energy()
    assert (abs(epsi4 - ecc) < 1e-11)