"""
Spin-orbital coupled cluster on a ROHF reference, via semicanonical orbitals
(docs/archive/ENHANCEMENT_PLAN_2026-06.md, phase 5).

SpinOrbitalHamiltonian semicanonicalizes the ROHF orbitals (diagonalizing the
occ-occ and vir-vir Fock blocks per spin), giving well-defined MP2/(T)/CC3
denominators; the residual occ-vir Fock block feeds the MP2 singles (t1 = f_ia/Dia).
Psi4's CCENERGY module semicanonicalizes a ROHF reference the same way, so its
reported MP2/CCSD/CCSD(T)/CC3 correlation energies are the cross-checks.
"""

import psi4
import pycc

# Open-shell doublet (hydroxyl radical).
OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""


def test_rohf_mp2_ccsd_ccsd_t_oh(rohf_wfn):
    """ROHF .OH cc-pVDZ: spin-orbital MP2, CCSD, and CCSD(T) vs Psi4 (semicanonical
    CCENERGY). A single energy('ccsd(t)') run populates all three reference variables
    (plain energy('ccsd') does not set MP2 CORRELATION ENERGY for ROHF)."""
    wfn = rohf_wfn(OH, "cc-pVDZ", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)

    mp = pycc.MPwfn(wfn)
    assert mp.orbital_basis == "spinorbital"
    emp2 = mp.compute_energy()
    eccsd = pycc.CCwfn(wfn, frozen_core=False).solve_cc(e_conv=1e-11, r_conv=1e-11)
    eccsd_t = pycc.CCwfn(wfn, model="CCSD(T)", frozen_core=False).solve_cc(
        e_conv=1e-11, r_conv=1e-11)

    psi4.energy('ccsd(t)')
    assert abs(emp2 - psi4.variable('MP2 CORRELATION ENERGY')) < 1e-10
    assert abs(eccsd - psi4.variable('CCSD CORRELATION ENERGY')) < 1e-10
    assert abs(eccsd_t - psi4.variable('CCSD(T) CORRELATION ENERGY')) < 1e-10


def test_rohf_cc3_oh(rohf_wfn):
    """ROHF .OH 6-31G: spin-orbital CC3 vs Psi4's ROHF-CC3 (small basis for the
    iterative spin-orbital solve)."""
    wfn = rohf_wfn(OH, "6-31G", freeze_core="false",
                   e_convergence=1e-12, d_convergence=1e-12)

    e = pycc.CCwfn(wfn, model="CC3", frozen_core=False).solve_cc(e_conv=1e-11, r_conv=1e-11)

    psi4.energy('cc3')
    ref = psi4.variable('CC3 CORRELATION ENERGY')

    assert abs(e - ref) < 1e-10
