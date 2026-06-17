"""
Test the MP2 correlation energy (MPwfn) against Psi4's conventional MP2, for both
the spin-adapted spatial (RHF) path and the spin-orbital (UHF) path.
"""

import psi4
import pycc
import pytest
from ..data.molecules import *

# Open-shell doublet for the UHF (spin-orbital) MP2 checks.
OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""


def test_mp2_h2o(rhf_wfn):
    """H2O cc-pVDZ, all-electron MP2 vs Psi4's conventional MP2."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    emp2 = pycc.MPwfn(wfn).compute_energy()

    # Independent reference: Psi4's own conventional MP2 on the same system.
    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')

    assert abs(emp2 - ref) < 1e-10


def test_mp2_h2o_frzc(rhf_wfn):
    """H2O cc-pVDZ, frozen-core MP2 vs Psi4 (exercises the active-space counts)."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)
    emp2 = pycc.MPwfn(wfn).compute_energy()

    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')

    assert abs(emp2 - ref) < 1e-10


def test_ump2_oh(uhf_wfn):
    """Open-shell .OH cc-pVDZ, all-electron UMP2 via the spin-orbital MPwfn path."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)
    mp = pycc.MPwfn(wfn)
    assert mp.orbital_basis == "spinorbital"
    emp2 = mp.compute_energy()

    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')

    assert abs(emp2 - ref) < 1e-9


def test_ump2_oh_frzc(uhf_wfn):
    """Open-shell .OH cc-pVDZ, frozen-core UMP2 (exercises the spin-orbital active
    space) via the spin-orbital MPwfn path."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)
    emp2 = pycc.MPwfn(wfn).compute_energy()

    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')

    assert abs(emp2 - ref) < 1e-9
