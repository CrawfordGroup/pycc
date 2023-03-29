"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
from ..cctriples import t_vikings, t_vikings_inverted, t_tjl

def test_ccsd_t_h2o():
    """H2O cc-pVDZ"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    psi4.core.clean()

    etot = psi4.energy('CCSD(T)')
    et_psi4 = psi4.variable('(T) CORRECTION ENERGY')
    print("(T) correction from Psi: %20.15f" % (et_psi4))
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')
    print("CCSD(T) correlation energy from Psi: %20.15f" % (ecc_psi4))

    cc = pycc.ccwfn(rhf_wfn, model='ccsd(t)', dertype='first')
    eccsd = cc.solve_cc(e_conv,r_conv,maxiter, max_diis=0)
    et_tjl = t_tjl(cc)
    print("(T) correction from PyCC: %20.15f" % (et_tjl))
    assert (abs(et_psi4 - et_tjl) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis=0)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    print("CCSD(T) correlation energy from PyCC: %20.15f" % (ecc_density))
    assert (abs(ecc_psi4 - ecc_density) < 1e-11)

    psi4.core.clean()

#    psi4.set_options({'basis': 'cc-pVDZ'})
#    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
#    etot = psi4.energy('CCSD(T)')
#    et_psi4 = psi4.variable('(T) CORRECTION ENERGY')
#    print("(T) correction from Psi: %20.15f" % (et_psi4))
#    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')
#    print("CCSD(T) correlation energy from Psi: %20.15f" % (ecc_psi4))
#
#    cc = pycc.ccwfn(rhf_wfn, model='ccsd(t)', dertype='first')
#    eccsd = cc.solve_cc(e_conv,r_conv,maxiter)
#    et_tjl = t_tjl(cc)
#    print("(T) correction from PyCC: %20.15f" % (et_tjl))
#    assert (abs(et_psi4 - et_tjl) < 1e-11)
#    hbar = pycc.cchbar(cc)
#    cclambda = pycc.cclambda(cc, hbar)
#    lcc = cclambda.solve_lambda(e_conv, r_conv)
#    ccdensity = pycc.ccdensity(cc, cclambda)
#    ecc_density = ccdensity.compute_energy()
#    print("CCSD(T) correlation energy from PyCC: %20.15f" % (ecc_density))
#    assert (abs(ecc_psi4 - ecc_density) < 1e-11)

