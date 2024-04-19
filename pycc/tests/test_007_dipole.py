"""
Test CCSD electric and magnetic dipole on H2 dimer.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import *

def test_dipole_h2_2_cc_pvdz():
    """H4 cc-pVDZ"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["(H2)_2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-13
    r_conv = 1e-13
    
    cc = pycc.ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)

    hbar = pycc.cchbar(cc)

    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)

    ccdensity = pycc.ccdensity(cc, cclambda)

    # no laser
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, None, magnetic = True)
    y0 = rtcc.collect_amps(cc.t1, cc.t2, cclambda.l1, cclambda.l2, ecc).astype('complex128')
    t1, t2, l1, l2, phase = rtcc.extract_amps(y0)

    ref = np.array([0, 0, -0.0007395036977002]) # computed by removing SCF from original ref

    mu_x, mu_y, mu_z = rtcc.dipole(t1, t2, l1, l2)

    assert (abs(ref[0] - mu_x) < 1E-10)
    assert (abs(ref[1] - mu_y) < 1E-10)
    assert (abs(ref[2] - mu_z) < 1E-10)

    ref = [0, 0, -2.3037968376087573E-5]
    m_x, m_y, m_z = rtcc.dipole(t1, t2, l1, l2, magnetic = True)

    assert (abs(ref[0]*1.0j - m_x) < 1E-10)
    assert (abs(ref[1]*1.0j - m_y) < 1E-10)
    assert (abs(ref[2]*1.0j - m_z) < 1E-10)
