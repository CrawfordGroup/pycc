"""
Test basic PNO-CCD, PNO++-CCD, PAO-CCD energies
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_pno_ccd():
    """H2O PNO-CCD Test"""
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
    e_conv = 1e-12
    r_conv = 1e-12
    
    #simulation code of pno-ccd
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PNO', local_cutoff=1e-5,it2_opt=False,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)
    
    #pno-ccd
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PNO', local_cutoff=1e-5,it2_opt=False)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter) 

    assert(abs(eccd_sim - elccd) < 1e-12) 

def test_pnopp_ccd():
    """H2O PNO++ CCD Test"""
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
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 100
    e_conv = 1e-12
    r_conv = 1e-12

    #simulation code of pno++-ccd
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PNO++', local_cutoff=1e-7,it2_opt=False,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno++-ccd
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PNO++', local_cutoff=1e-7,it2_opt=False)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12) 

def test_pno_ccd_opt():
    """H2O PNO-CCD with Optimized Initial T2 Amplitudes"""
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
    e_conv = 1e-12
    r_conv = 1e-12

    #simulation code of pno-ccd
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PNO', local_cutoff=1e-7,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pno-ccd
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PNO', local_cutoff=1e-7)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12)

def test_pao_ccd_opt():
    """H2O PAO-CCD with Optimized Initial T2 Amplitudes"""
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
    e_conv = 1e-12
    r_conv = 1e-12

    #simulation code of pao-ccd
    ccd_sim = pycc.ccwfn(rhf_wfn, model='CCD',local='PAO', local_cutoff=2e-2,filter=True)
    eccd_sim = ccd_sim.solve_cc(e_conv, r_conv, maxiter)

    #pao-ccd
    lccd = pycc.ccwfn(rhf_wfn,model='CCD', local='PAO', local_cutoff=2e-2)
    elccd = lccd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)

    assert(abs(eccd_sim - elccd) < 1e-12)
