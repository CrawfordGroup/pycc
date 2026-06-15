"""
Test the RHF static dipole polarizability (HFwfn CPHF orbital response) against
Psi4's analytic CPHF polarizability.
"""

import psi4
import pycc
import numpy as np
from ..data.molecules import *


def test_cphf_polarizability_h2o(rhf_wfn):
    """H2O cc-pVDZ static dipole polarizability vs Psi4 CPHF (symmetry left on)."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-11, d_convergence=1e-11)
    alpha = pycc.HFwfn(wfn).polarizability()

    # Independent reference: Psi4's own analytic CPHF dipole polarizability on the
    # same system (properties() populates the DIPOLE POLARIZABILITY ** variables).
    psi4.properties('scf', properties=['DIPOLE_POLARIZABILITIES'])
    ax = 'XYZ'
    ref = np.array([[psi4.variable(f'DIPOLE POLARIZABILITY {ax[i]}{ax[j]}')
                     for j in range(3)] for i in range(3)])

    assert np.max(np.abs(alpha - ref)) < 1e-10
