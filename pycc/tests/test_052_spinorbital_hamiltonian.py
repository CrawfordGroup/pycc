"""
Step 1 of the spin-orbital enhancement (docs/archive/ENHANCEMENT_PLAN_2026-06.md): the
SpinOrbitalHamiltonian plus the Wavefunction base's orbital_basis dispatch.

Keystone: forcing the spin-orbital path on a closed-shell RHF reference must
reproduce the spin-adapted spatial MP2 energy (and Psi4's conventional MP2) to
machine precision. That isolates and validates the Hamiltonian fork -- block Fock,
antisymmetrized <pq||rs>, active-space counts -- before any open-shell physics. A
UHF check then confirms the open-shell Hamiltonian against Psi4's UMP2.

MP2 is computed inline here (t2 = <ij||ab>/D, E = 1/4 <ij||ab> t2) rather than via
MPwfn: the MPwfn spin-orbital branch is step 2. This keeps step 1 a pure test of the
base + Hamiltonian.
"""

import numpy as np
import psi4

import pycc
from pycc.wavefunction import Wavefunction

# Open-shell doublet for the UHF check (hydroxyl radical).
OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""


def _so_mp2(wfn):
    """Spin-orbital MP2 correlation energy from a base built on the spin-orbital
    Hamiltonian: t2 = <ij||ab> / D_ijab, E = 1/4 <ij||ab> t2_ijab."""
    o, v = wfn.o, wfn.v
    eps = np.diag(np.asarray(wfn.H.F))
    eo, ev = eps[o], eps[v]
    D = (eo.reshape(-1, 1, 1, 1) + eo.reshape(-1, 1, 1)
         - ev.reshape(-1, 1) - ev)
    oovv = np.asarray(wfn.H.ERI)[o, o, v, v]
    t2 = oovv / D
    return 0.25 * np.einsum('ijab,ijab->', oovv, t2)


def test_so_equals_spatial_rhf(rhf_wfn):
    """All-electron: spin-orbital MP2 (forced) == spatial MP2 == Psi4 RMP2."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    e_spatial = pycc.MPwfn(wfn).compute_energy()
    so = Wavefunction(wfn, orbital_basis="spinorbital", frozen_core=False)
    assert so.orbital_basis == "spinorbital"
    e_so = _so_mp2(so)

    assert abs(e_so - e_spatial) < 1e-12

    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')
    assert abs(e_so - ref) < 1e-10


def test_so_equals_spatial_rhf_frozen_core(rhf_wfn):
    """Frozen-core: same equivalence, exercising the active-space counts."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="true",
                  e_convergence=1e-12, d_convergence=1e-12)

    e_spatial = pycc.MPwfn(wfn).compute_energy()
    so = Wavefunction(wfn, orbital_basis="spinorbital", frozen_core=True)
    e_so = _so_mp2(so)

    assert abs(e_so - e_spatial) < 1e-12

    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')
    assert abs(e_so - ref) < 1e-10


def test_auto_dispatch_rhf_is_spatial(rhf_wfn):
    """A closed-shell RHF reference auto-selects the spatial path."""
    wfn = rhf_wfn("H2O", "cc-pVDZ")
    assert pycc.MPwfn(wfn).orbital_basis == "spatial"


def test_uhf_so_mp2(uhf_wfn):
    """Open-shell UHF auto-selects the spin-orbital path; SO-MP2 == Psi4 UMP2."""
    wfn = uhf_wfn(OH, "cc-pVDZ", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12)

    so = Wavefunction(wfn, frozen_core=False)  # auto -> spinorbital
    assert so.orbital_basis == "spinorbital"
    e_so = _so_mp2(so)

    psi4.energy('mp2')
    ref = psi4.variable('MP2 CORRELATION ENERGY')
    assert abs(e_so - ref) < 1e-10
