"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_ccsd_h2o():
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
    ccsd = pycc.ccwfn(rhf_wfn, model='CCSD')
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.070616830152761  
    assert (abs(epsi4 - eccsd) < 1e-11)

    # cc-pVDZ basis set
    psi4.set_options({'basis': 'cc-pVDZ'})
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    ccsd = pycc.ccwfn(rhf_wfn)
    eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.222029814166783
    assert (abs(epsi4 - eccsd) < 1e-11)
