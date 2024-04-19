"""
Test CCSD quadratic response function.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import * 

def test_ccsd_SHG():
    """ H2O Second Harmonic Generation CCSD Test """
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'scf_type': 'pk',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    mol = psi4.geometry(moldict["H2O_D"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    e_conv = 1e-12
    r_conv = 1e-12
    
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)
    
    resp = pycc.ccresponse(density)
    
    # SHG frequencies 
    omega1 = 0.0656
    omega2 = 0.0656
    
    resp.pert_quadresp(omega1, omega2)
    SHG = resp.hyperpolar()
    
    #Dalton 
    H2O_SHG = -19.7591180824

    # Other system's Dalton SHG value, all validated    #PyCC SHG values
    #CO_SHG = -32.4314012752                     #CO = -32.431401275449  
    #HF_SHG = 10.23985062319                     #HF = 10.239850624779
    #HCl_SHG = -21.348258444480                  #HCl = -21.348258543702
    
    assert(abs(SHG - H2O_SHG) < 1e-7) 
   
def test_ccsd_OR():
    """ CO Optical Refractivity CCSD Test """
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'scf_type': 'pk',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    mol = psi4.geometry(moldict["CO"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    
    e_conv = 1e-12
    r_conv = 1e-12
    
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)
    
    resp = pycc.ccresponse(density)
    
    #OR frequencies
    omega1 = -0.0656
    omega2 = 0.0656
    
    resp.pert_quadresp(omega1, omega2)
    OR = resp.hyperpolar()
    
    #Dalton 
    CO_OR = -29.3364710172

    # Other system's Dalton OR value, all validated    #PyCC OR values
    #H2O_OR = -17.257079066                     #H2O = 17.257079081719
    #HF_OR = 9.6113353644                       #HF = 9.611335367514
    #HCl_OR = -19.34209144399                   #HCl = -19.342091493471

    assert(abs(OR - CO_OR) < 1e-7) 
