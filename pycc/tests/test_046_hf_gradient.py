"""
Test the RHF analytic nuclear gradient (HFwfn) against Psi4's analytic SCF gradient.
"""

import psi4
import pycc
import numpy as np
import pytest
from ..data.molecules import *


def test_hf_gradient_h2o(rhf_wfn):
    """H2O cc-pVDZ RHF gradient vs Psi4 (molecular symmetry left on)."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-11, d_convergence=1e-11)
    grad = pycc.HFwfn(wfn).gradient()

    # Independent reference: Psi4's own analytic SCF gradient on the same system.
    ref = np.asarray(psi4.gradient('scf'))

    assert np.max(np.abs(grad - ref)) < 1e-11


def test_hf_gradient_frozen_core_ref(rhf_wfn):
    """A frozen-core reference still yields the correct all-electron HF gradient:
    HFwfn uses the full MO space regardless of the reference's freeze_core."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-11, d_convergence=1e-11)
    grad = pycc.HFwfn(wfn).gradient()

    ref = np.asarray(psi4.gradient('scf'))

    assert np.max(np.abs(grad - ref)) < 1e-11
