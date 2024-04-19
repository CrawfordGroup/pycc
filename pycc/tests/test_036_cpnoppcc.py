"""
Test basic CPNO++-CCSD energy
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_cpnopp_ccsd():
    """H2O CPNO++-CCSD Test"""
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
                      'diis': 8})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    ccsd = pycc.ccwfn(rhf_wfn, local='CPNO++', local_cutoff=1e-7, it2_opt=False)
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)
    
    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)
    
    #Ruhee's ccsd_lpno code 
    esim = -0.22303320613504354
    lsim = -0.21890326836263854
   
    assert (abs(esim - eccsd) < 1e-7)
    assert (abs(lsim - lccsd) < 1e-7)   

def test_pnopp_ccsd_opt():
    """H2O CPNO++-CCSD with Optimized Initial T2 Amplitudes"""
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
                      'diis': 8})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    ccsd = pycc.ccwfn(rhf_wfn, local='CPNO++', local_cutoff=1e-7)
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)

    hbar = pycc.cchbar(ccsd)
    cclambda = pycc.cclambda(ccsd, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)

    #Comparing against simulation code 
    esim = -0.223866820104919 
    lsim = -0.21966259490352782 
 
    assert (abs(esim - eccsd) < 1e-7)
    assert (abs(lsim - lccsd) < 1e-7)
