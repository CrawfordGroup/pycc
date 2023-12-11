"""
Test CC3 equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
import numpy as np

# H2O/cc-pVDZ
def test_cc3_h2o():
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
    mol = psi4.geometry(moldict["H2O_Teach"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn, model='CC3')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.227888246840310
    ecfour = -0.2278882468404231
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    print("lcc: ", lcc)
    lcc_cfour = -0.2233231845185215 
    assert(abs(lcc - lcc_cfour) < 1e-11)

    ccdensity = pycc.ccdensity(cc, cclambda)
    # no laser
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, None, magnetic = False)
    
    CFOUR = [0, 0, 0.7703875967] # CFOUR total dipole (CC3 + SCF + nuclear)
    scf = rhf_wfn.variable('SCF DIPOLE') # PSI4 reference dipole (SCF + nuclear)
    ref = CFOUR - scf # Final reference: CC3 only

    mu_x, mu_y, mu_z = rtcc.dipole(cc.t1, cc.t2, cclambda.l1, cclambda.l2)

    assert (abs(ref[1] - np.real(mu_y)) < 1E-10)
    assert (abs(ref[2] - np.real(mu_z)) < 1E-10)

# H2/cc-pVDZ
"""
def test_cc3_h2():
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
    mol = psi4.geometry(moldict["H2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    cc = pycc.ccwfn(rhf_wfn, model='CC3')
    ecc = cc.solve_cc(e_conv,r_conv,maxiter)
    epsi4 = -0.034689283017250
    ecfour = -0.0346892830172550
    assert (abs(epsi4 - ecc) < 1e-11)

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    lcc_cfour = -0.0341034656430758
    assert(abs(lcc - lcc_cfour) < 1e-11)
"""

