"""
Test basic LPNOPP-CCSD energy
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *


def test_lpnopp_ccsd():
    """H2O LPNOPP-CCSD Test"""
    # Psi4 Setup
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

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    max_diis = 8

    ccsd = pycc.ccwfn(rhf_wfn, local='LPNOpp', local_cutoff=0)
    eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)

    epsi4 = -0.2239100187035

    assert (abs(epsi4 - eccsd) < 1e-7)
