"""
Spin-orbital EOM-CCSD (right-hand roots).

The spin-orbital sigma vectors (cceom._so_s_r1 / _so_s_r2) extend the RHF EOM-CCSD code to
UHF/ROHF references. Two validations, mirroring the spin-orbital idiom used elsewhere in the
suite:

1. Closed-shell keystone (iterative): a closed-shell RHF run through the spin-orbital path must
   reproduce every spatial (spin-adapted singlet) excitation energy, and additionally produce the
   lower-lying triplet roots that the spin-adapted path cannot represent.

2. Open-shell sigma vs psi4 (dense): for the OH radical (UHF) the spin-orbital sigma operator,
   diagonalized densely, reproduces psi4's UHF-EOM-CCSD roots. The dense check validates the sigma
   physics directly, independent of the iterative Davidson (whose robust resolution of the
   near-degenerate open-shell roots is a separate block-Davidson upgrade, PR2).
"""

import psi4
import pycc
import numpy as np


OH = """
0 2
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.969000
symmetry c1
"""

H2O = """
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
"""


def _eom(wfn, orbital_basis, frozen_core=False):
    cc = pycc.ccwfn(wfn, orbital_basis=orbital_basis, frozen_core=frozen_core)
    cc.solve_cc(1e-11, 1e-11, 75)
    return cc, pycc.cceom(cc, pycc.cchbar(cc))


def _dense_so_spectrum(eom):
    """Excitation energies from a dense diagonalization of the spin-orbital sigma operator
    (built column by column from _so_s_r1 / _so_s_r2), keeping the real, positive roots."""
    cc = eom.ccwfn
    no, nv = cc.no, cc.nv
    n1 = no * nv
    n = n1 + n1 * n1
    H = np.zeros((n, n))
    for k in range(n):
        vec = np.zeros(n); vec[k] = 1.0
        C1 = vec[:n1].reshape(no, nv)
        C2 = vec[n1:].reshape(no, no, nv, nv)
        H[:n1, k] = eom.s_r1(eom.hbar, C1, C2).ravel()
        H[n1:, k] = eom.s_r2(eom.hbar, C1, C2).ravel()
    E = np.linalg.eigvals(H)
    E = np.sort(E[np.abs(E.imag) < 1e-8].real)
    return E[E > 1e-3]


def test_so_eom_closed_shell_keystone(rhf_wfn):
    """Closed-shell H2O/STO-3G: the spin-orbital EOM spectrum contains every spatial (spin-adapted
    singlet) root, plus lower-lying triplets. The spatial singlets come from the (robust) iterative
    solver; the full spin-orbital spectrum from a dense diagonalization (the triplets interleave the
    singlets, so covering all singlets iteratively would need many roots)."""
    wfn = rhf_wfn(H2O, "STO-3G", freeze_core="false")
    _, eom_s = _eom(wfn, "spatial")
    _, eom_o = _eom(wfn, "spinorbital")

    E_spatial = np.sort(eom_s.solve_eom(3, 1e-6, 1e-6, 150, "cis", "right")[0].real)
    E_so = _dense_so_spectrum(eom_o)

    # every spatial singlet root appears in the spin-orbital spectrum
    for e in E_spatial:
        assert np.min(np.abs(E_so - e)) < 1e-5, (e, E_so[:12])
    # the spin-orbital path also finds lower roots (the triplets) the spatial path cannot
    assert E_so.min() < E_spatial.min() - 1e-3

    # the iterative spin-orbital solver reproduces the lowest (triplet) root
    E_iter = np.sort(eom_o.solve_eom(4, 1e-6, 1e-6, 150, "cis", "right")[0].real)
    assert abs(E_iter.min() - E_so.min()) < 1e-5


# psi4 UHF-EOM-CCSD roots for OH/STO-3G at this geometry (energy('eom-ccsd'), reference uhf)
PSI4_UHF_ROOTS = [0.224266, 0.404873]


def test_so_eom_sigma_vs_psi4_uhf(uhf_wfn):
    """Open-shell OH/UHF/STO-3G: the dense spin-orbital EOM spectrum reproduces psi4's
    UHF-EOM-CCSD roots (the sigma physics, independent of the iterative solver)."""
    wfn = uhf_wfn(OH, "STO-3G", freeze_core="false")
    _, eom = _eom(wfn, "spinorbital")
    E = _dense_so_spectrum(eom)

    for e in PSI4_UHF_ROOTS:
        assert np.min(np.abs(E - e)) < 1e-5, (e, E[:8])


def test_so_eom_open_shell_iterative(uhf_wfn):
    """Open-shell OH/UHF/STO-3G: the iterative block-Davidson spin-orbital EOM reproduces psi4's
    UHF-EOM-CCSD roots. This exercises the full production path (block Davidson + spin-orbital
    sigma) on the near-degenerate open-shell spectrum that a single-vector Davidson could not
    resolve."""
    wfn = uhf_wfn(OH, "STO-3G", freeze_core="false")
    _, eom = _eom(wfn, "spinorbital")
    E = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "right")[0].real)

    # psi4's roots (plus their spin-contamination partners) appear among the iterative roots
    for e in PSI4_UHF_ROOTS:
        assert np.min(np.abs(E - e)) < 1e-5, (e, E)


def test_so_eom_left_equals_right(rhf_wfn, uhf_wfn):
    """The spin-orbital left-hand sigma (_so_s_l1 / _so_s_l2) yields the same excitation energies
    as the right-hand sigma -- they are the left/right eigenvectors of the same (non-Hermitian)
    EOM operator -- for a closed-shell (H2O) and an open-shell (OH/UHF) reference."""
    for factory, mol in ((rhf_wfn, H2O), (uhf_wfn, OH)):
        wfn = factory(mol, "STO-3G", freeze_core="false")
        _, eom = _eom(wfn, "spinorbital")
        E_right = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "right")[0].real)
        E_left = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "left")[0].real)
        assert np.max(np.abs(E_left - E_right)) < 1e-6, (mol, E_left, E_right)


def test_so_eom_rohf(rohf_wfn):
    """Open-shell OH/ROHF/STO-3G (all-electron): the spin-orbital EOM on a semicanonical ROHF
    reference reproduces psi4's ROHF-EOM-CCSD roots, and left == right."""
    wfn = rohf_wfn(OH, "STO-3G", freeze_core="false")
    _, eom = _eom(wfn, "spinorbital")
    R = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "right")[0].real)
    L = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "left")[0].real)
    # psi4 ROHF-EOM-CCSD roots (energy('eom-ccsd'), reference rohf): the excitation energy is
    # (CC ROOT n TOTAL ENERGY) minus the minimum over roots -- for ROHF the ground state is not
    # root 0 (root 0 is a corr=0 placeholder), so subtracting root 0 would be wrong.
    for e in (0.224190, 0.403713):
        assert np.min(np.abs(R - e)) < 1e-5, (e, R)
    assert np.max(np.abs(L - R)) < 1e-6


def test_so_eom_frozen_core(uhf_wfn):
    """Frozen-core spin-orbital EOM (OH/UHF/STO-3G): reproduces psi4's frozen-core UHF-EOM-CCSD
    roots, is distinct from the all-electron result, and left == right."""
    wfn = uhf_wfn(OH, "STO-3G", freeze_core="true")
    _, eom = _eom(wfn, "spinorbital", frozen_core=True)
    R = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "right")[0].real)
    L = np.sort(eom.solve_eom(4, 1e-7, 1e-6, 200, "hbar_ss", "left")[0].real)
    for e in (0.224329, 0.404870):        # psi4 frozen-core UHF-EOM-CCSD roots
        assert np.min(np.abs(R - e)) < 1e-5, (e, R)
    assert np.max(np.abs(L - R)) < 1e-6
    # frozen core actually changes the roots (vs the all-electron 0.224266 / 0.404873)
    assert np.min(np.abs(R - 0.224266)) > 1e-5
