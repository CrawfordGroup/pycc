"""
Test basic local lambda CCSD energies(local = PNO, PNO++, PAO)
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc 
import pytest 
from ..data.molecules import *
 
def test_pno_lambda_ccsd():
    """H2O PNO-Lambda CCSD Test"""    
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '3-21G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    maxiter = 200
    e_conv = 1e-12
    r_conv = 1e-12
       
    #simulation code of pno-ccsd lambda
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-6,it2_opt=False,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccsd_sim)
    cclambda_sim = pycc.cclambda(ccsd_sim, hbar_sim)
    l_ccsd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)
    
    #pno-ccsd lambda
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-6,it2_opt=False)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccsd)  
    lcclambda = pycc.cclambda(lccsd, lhbar)
    l_lccsd = lcclambda.solve_llambda(e_conv, r_conv, maxiter) 
    
    assert(abs(l_ccsd_sim - l_lccsd) < 1e-12)

def test_pnopp_lambda_ccsd():
    """H2O PNO++-Lambda CCSD Test"""
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '3-21G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    maxiter = 200
    e_conv = 1e-12
    r_conv = 1e-12
   
    #simulation code of pno++-ccsd lambda
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO++', local_cutoff=1e-6,it2_opt=False,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccsd_sim)
    cclambda_sim = pycc.cclambda(ccsd_sim, hbar_sim)
    l_ccsd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)
    
    #pno++-ccsd lambda
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO++', local_cutoff=1e-6,it2_opt=False)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccsd)
    lcclambda = pycc.cclambda(lccsd, lhbar)
    l_lccsd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
    
    assert(abs(l_ccsd_sim - l_lccsd) < 1e-12)

def test_pno_lambda_ccsd_opt():
    """H2O PNO-Lambda CCSD Test with Optimized Intiial T2 Amplitudes"""
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '3-21G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 200
    e_conv = 1e-12
    r_conv = 1e-12

    #simulation code of pno-ccsd lambda
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-6,it2_opt=True,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccsd_sim)
    cclambda_sim = pycc.cclambda(ccsd_sim, hbar_sim)
    l_ccsd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)

    #pno-ccsd lambda
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-6,it2_opt=True)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccsd)
    lcclambda = pycc.cclambda(lccsd, lhbar)
    l_lccsd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
    
    assert(abs(l_ccsd_sim - l_lccsd) < 1e-12)

def test_pao_lambda_ccsd_opt():
    """H2O PAO-Lambda CCSD Test with Optimized Initial T2 Amplitudes"""
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '3-21G',
                      'scf_type': 'pk', 
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})  
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 200
    e_conv = 1e-12
    r_conv = 1e-12

    #simulation code of pao-ccsd    
    ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PAO', local_cutoff=1e-6,it2_opt=True,filter=True)
    eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccsd_sim)  
    cclambda_sim = pycc.cclambda(ccsd_sim, hbar_sim) 
    l_ccsd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)

    #pao-ccsd
    lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PAO', local_cutoff=1e-6,it2_opt=True)
    elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccsd)
    lcclambda = pycc.cclambda(lccsd, lhbar)
    l_lccsd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
    
    assert(abs(l_ccsd_sim - l_lccsd) < 1e-12)
