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
    mol = psi4.geometry(
"""
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""
)
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    etot = psi4.energy('CCSD(T)')
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')

    cc = pycc.ccwfn(rhf_wfn, model='ccsd(t)', make_t3_density=True)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter, max_diis=0)
    assert (abs(ecc_psi4 - ecc) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis=0)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    eone = ccdensity.eone
    etwo = ccdensity.etwo

    lambda_psi4 = -0.069084521221746
    eone_psi4 = 0.104463374777302
    etwo_psi4 = -0.175243393781829
    assert (abs(lambda_psi4 - lcc) < 1e-11)
    assert (abs(eone_psi4 - eone) < 1e-11)
    assert (abs(etwo_psi4 - etwo) < 1e-11)

    psi4.core.clean()

    psi4.set_options({'basis': 'cc-pVDZ'})
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    etot = psi4.energy('CCSD(T)')
    ecc_psi4 = psi4.variable('CCSD(T) CORRELATION ENERGY')

    cc = pycc.ccwfn(rhf_wfn, model='ccsd(t)', make_t3_density=True)
    ecc = cc.solve_cc(e_conv, r_conv, maxiter)
    assert (abs(ecc_psi4 - ecc) < 1e-11)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    ccdensity = pycc.ccdensity(cc, cclambda)
    ecc_density = ccdensity.compute_energy()
    eone = ccdensity.eone
    etwo = ccdensity.etwo

    lambda_psi4 = -0.227199866607450
    eone_psi4 = 0.251210862963227
    etwo_psi4 = -0.479006477929931
    assert (abs(lambda_psi4 - lcc) < 1e-11)
    assert (abs(eone_psi4 - eone) < 1e-11)
    assert (abs(etwo_psi4 - etwo) < 1e-11)
