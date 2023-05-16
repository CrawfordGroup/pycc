"""
Test basic local lambda CCD energies(local = PNO, PNO++, PAO)
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc 
import pytest 
from ..data.molecules import * 

def test_pno_lambda_ccd():
    """H2O PNO-Lambda CCD Test"""    
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
       
    #simulation code of pno-ccd lambda
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PNO', local_cutoff=1e-6,it2_opt=False,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccd_sim)
    cclambda_sim = pycc.cclambda(ccd_sim, hbar_sim)
    l_ccd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)
    
    #pno-ccd lambda
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PNO', local_cutoff=1e-6,it2_opt=False)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccd)  
    lcclambda = pycc.cclambda(lccd, lhbar)
    l_lccd = lcclambda.solve_llambda(e_conv, r_conv, maxiter) 
    
    assert(abs(l_ccd_sim - l_lccd) < 1e-12)

def test_pnopp_lambda_ccd():
    """H2O PNO++-Lambda CCD Test"""
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
   
    #simulation code of pno++-ccd lambda
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PNO', local_cutoff=1e-6,it2_opt=False,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccd_sim)
    cclambda_sim = pycc.cclambda(ccd_sim, hbar_sim)
    l_ccd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)
    
    #pno++-ccd lambda
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PNO', local_cutoff=1e-6,it2_opt=False)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccd)
    lcclambda = pycc.cclambda(lccd, lhbar)
    l_lccd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
    
    assert(abs(l_ccd_sim - l_lccd) < 1e-12)

def test_pno_lambda_ccd_opt():
    """H2O PNO-Lambda CCD Test with Optimized Intiial T2 Amplitudes"""
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

    #simulation code of pno-ccd lambda
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PNO', local_cutoff=1e-6,it2_opt=True,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccd_sim)
    cclambda_sim = pycc.cclambda(ccd_sim, hbar_sim)
    l_ccd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)

    #pno-ccd lambda
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PNO', local_cutoff=1e-6,it2_opt=True)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccd)
    lcclambda = pycc.cclambda(lccd, lhbar)
    l_lccd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
    
    assert(abs(l_ccd_sim - l_lccd) < 1e-12)

def test_pao_lambda_ccd_opt():
    """H2O PAO-Lambda CCD Test with Optimized Initial T2 Amplitudes"""
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

    #simulation code of pao-ccd lambda   
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PAO', local_cutoff=1e-6,it2_opt=True,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)
    hbar_sim = pycc.cchbar(ccd_sim)  
    cclambda_sim = pycc.cclambda(ccd_sim, hbar_sim) 
    l_ccd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)

    #pao-ccd lambda
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PAO', local_cutoff=1e-6,it2_opt=True)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.cchbar(lccd)
    lcclambda = pycc.cclambda(lccd, lhbar)
    l_lccd = lcclambda.solve_llambda(e_conv, r_conv, maxiter)
    
    assert(abs(l_ccd_sim - l_lccd) < 1e-12)
