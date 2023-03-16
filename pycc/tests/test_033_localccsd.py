"""
Test basic PNO-CCSD, PNO++-CCSD, PAO-CCSD energies
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_pno_ccsd():
    """H2O PNO-CCSD Test"""
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
    
    maxiter = 100
    e_conv = 1e-7
    r_conv = 1e-7
    
    #simulation code of pno-ccsd
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-5,it2_opt=False,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
    
    #pno-ccsd
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-5,it2_opt=False)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter) 

    assert(abs(eccsd_sim - elccsd) < 1e-7) 

def test_pnopp_ccsd():
    """H2O PNO++ CCSD Test"""
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '6-31G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["(H2O)_2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 100
    e_conv = 1e-7
    r_conv = 1e-7

    #simulation code of pno++-ccsd
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO++', local_cutoff=1e-7,it2_opt=False,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno++-ccsd
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO++', local_cutoff=1e-7,it2_opt=False)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-7) 

def test_pno_ccsd_opt():
    """H2O PNO-CCSD with Optimized Initial T2 Amplitudes"""
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

    maxiter = 100
    e_conv = 1e-7
    r_conv = 1e-7

    #simulation code of pno-ccsd
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-5,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno-ccsd
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-5)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-7)

def test_pao_ccsd_opt():
    """H2O PAO-CCSD with Optimized Initial T2 Amplitudes"""
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

    maxiter = 100
    e_conv = 1e-7
    r_conv = 1e-7

    #simulation code of pao-ccsd
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PAO', local_cutoff=2e-2,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pao-ccsd
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PAO', local_cutoff=2e-2)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccsd_sim - elccsd) < 1e-7)
