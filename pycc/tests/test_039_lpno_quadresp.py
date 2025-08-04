# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_PNO_ccsd_SHG():
    
    h2o = """
    O -1.5167088799 -0.0875022822  0.0744338901
    H -0.5688047242  0.0676402012 -0.0936613229
    H -1.9654552961  0.5753254158 -0.4692384530
    symmetry c1
    """

    psi4.core.clean
    psi4.set_memory('16 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz',
                      'scf_type': 'pk',
                      'freeze_core':'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'local_maxiter': 10000, 
    })
    mol = psi4.geometry(moldict["(H2O)_2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    #Convergence and maximum iteration  
    e_conv = 1e-10
    r_conv = 1e-10
    maxiter = 1000

    #simulation code
    cc = pycc.ccwfn(rhf_wfn, local="PNO++", local_mos = 'BOYS',  local_cutoff=1e-05, filter=True)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)

    resp = pycc.ccresponse(density)

    # SHG frequencies 
    omega1 = 0.0428
    omega2 = 0.0428

    resp.pert_quadresp(omega1, omega2, e_conv, r_conv)
    SHG = resp.hyperpolar()

    #PNO
    lcc = pycc.ccwfn(rhf_wfn, local = 'PNO++', local_mos = 'BOYS', local_cutoff = 1e-05, filter=False)
    lecc = lcc.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
    lhbar = pycc.lcchbar(lcc)
    lcc_lambda = pycc.lcclambda(lcc, lhbar)
    llecc = lcc_lambda.solve_llambda(e_conv, r_conv)

    omega1 = 0.0656
    omega2 = 0.0656
    
    lresp = pycc.lccresponse(lcc, lcc_lambda)
    lresp.pert_lquadresp(omega1, omega2, e_conv, r_conv, maxiter)
    Beta_abc, PNO_SHG  = lresp.lhyperpolar()
    
    assert(abs(SHG - PNO_SHG) < 1e-7)
