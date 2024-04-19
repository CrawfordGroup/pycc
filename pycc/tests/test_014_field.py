"""
Test electric field generation 
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import *
from pycc.rt.lasers import gaussian_laser

def test_dipole_h2_2_field():
    """H2 dimer"""
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': '6-31G',
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

    # narrow Gaussian pulse
    F_str = 0.01
    sigma = 0.01
    center = 0.05
    V = gaussian_laser(F_str, 0, sigma, center=center)
    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V, magnetic = True)

    mints = psi4.core.MintsHelper(cc.ref.basisset())
    dipole_ints = mints.ao_dipole()
    m_ints = mints.ao_angular_momentum()
    C = np.asarray(cc.ref.Ca_subset("AO", "ACTIVE"))
    ref_mu = []
    ref_m = []
    for axis in range(3):
        ref_mu.append(C.T @ np.asarray(dipole_ints[axis]) @ C)
        ref_m.append(C.T @ (np.asarray(m_ints[axis])*-0.5) @ C)
        assert np.allclose(ref_mu[axis],rtcc.mu[axis])
        assert np.allclose(ref_m[axis]*1.0j,rtcc.m[axis])
    ref_mu_tot = sum(ref_mu)/np.sqrt(3.0)

    assert np.allclose(ref_mu_tot,rtcc.mu_tot)

    rtcc = pycc.rtcc(cc, cclambda, ccdensity, V, magnetic = True, kick="Y")

    mints = psi4.core.MintsHelper(cc.ref.basisset())
    dipole_ints = mints.ao_dipole()
    m_ints = mints.ao_angular_momentum()
    C = np.asarray(cc.ref.Ca_subset("AO", "ACTIVE"))
    ref_mu = []
    ref_m = []
    for axis in range(3):
        ref_mu.append(C.T @ np.asarray(dipole_ints[axis]) @ C)
        ref_m.append(C.T @ (np.asarray(m_ints[axis])*-0.5) @ C)
        assert np.allclose(ref_mu[axis],rtcc.mu[axis])
        assert np.allclose(ref_m[axis]*1.0j,rtcc.m[axis])

    assert np.allclose(ref_mu[1],rtcc.mu_tot)
