# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_PNO_ccsd_SHG():
    psi4.core.clean
    psi4.set_memory('16 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'scf_type': 'pk',
                      'freeze_core':'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12
    })
    mol = psi4.geometry(moldict["(H2)_2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    #Convergence and maximum iteration  
    e_conv = 1e-12
    r_conv = 1e-12
    maxiter = 1000

    #simulation code
    cc = pycc.ccwfn(rhf_wfn, local="PNO++", local_mos = 'BOYS',  local_cutoff=1e-07, filter=True)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)

    resp = pycc.ccresponse(density)

    # SHG frequencies 
    omega1 = 0.0656
    omega2 = 0.0656

    resp.pert_quadresp(omega1, omega2, e_conv, r_conv)
    SHG = resp.hyperpolar()

    #PNO
    lcc = pycc.ccwfn(rhf_wfn, local = 'PNO++', local_mos = 'BOYS', local_cutoff = 1e-07, filter=False)
    lecc = lcc.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.lcchbar(lcc)
    lcc_lambda = pycc.lcclambda(lcc, lhbar)
    llecc = lcc_lambda.solve_llambda(e_conv, r_conv)

    omega1 = 0.0656
    omega2 = 0.0656
    
    lresp = pycc.lccresponse(lcc, lcc_lambda)
    lresp.pert_lquadresp(omega1, omega2, e_conv, r_conv, maxiter)
    Beta_abc, PNO_SHG  = lresp.lhyperpolar()
    
    #Dalton 
    #H2O_SHG = -19.7591180824

    #simulation code 
    ### Untruncated ###
    # Other system's Dalton SHG value, all validated    #PyCC SHG values
    #CO_SHG = -32.4314012752                     #CO = -32.431401275449  
    #HF_SHG = 10.23985062319                     #HF = 10.239850624779
    #HCl_SHG = -21.348258444480                  #HCl = -21.348258543702

    assert(abs(SHG - PNO_SHG) < 1e-7)
