"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *
from ..cctriples import t_vikings, t_vikings_inverted, t_tjl
from pycc.ccwfn import HAS_EINSUMS

@pytest.mark.skipif(not HAS_EINSUMS, reason="Einsums not installed")
def test_ccsd_t_h2o():
    """H2O cc-pVDZ"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    cc = pycc.ccwfn(rhf_wfn, model='ccsd(t)', einsums=True)
    eccsd = cc.solve_cc(e_conv,r_conv,maxiter)
    et_vik_ijk = t_vikings(cc)
    et_vik_abc = t_vikings_inverted(cc)
    et_tjl = t_tjl(cc)
    epsi4 = -0.000099957499645
    print("E(T) = %20.15f (Vikings' IJK)" % (et_vik_ijk))
    print("E(T) = %20.15f (Vikings' ABC)" % (et_vik_abc))
    print("E(T) = %20.15f (Lee/Rendell IJK)" % (et_tjl))
    print("E(T) = %20.15f (Psi4)" % (epsi4))
    assert (abs(epsi4 - et_vik_ijk) < 1e-11)
    assert (abs(epsi4 - et_vik_abc) < 1e-11)
    assert (abs(epsi4 - et_tjl) < 1e-11)

    psi4.set_options({'basis': 'cc-pVDZ'})
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    cc = pycc.ccwfn(rhf_wfn, model='ccsd(t)', einsums=True)
    eccsd = cc.solve_cc(e_conv,r_conv,maxiter)
    et_vik_ijk = t_vikings(cc)
    et_vik_abc = t_vikings_inverted(cc)
    et_tjl = t_tjl(cc)
    epsi4 = -0.003861236558801
    print("E(T) = %20.15f (Vikings' IJK)" % (et_vik_ijk))
    print("E(T) = %20.15f (Vikings' ABC)" % (et_vik_abc))
    print("E(T) = %20.15f (Lee/Rendell IJK)" % (et_tjl))
    print("E(T) = %20.15f (Psi4)" % (epsi4))
    assert (abs(epsi4 - et_vik_ijk) < 1e-11)
    assert (abs(epsi4 - et_vik_abc) < 1e-11)
    assert (abs(epsi4 - et_tjl) < 1e-11)
