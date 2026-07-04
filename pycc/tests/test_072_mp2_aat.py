"""
MP2 atomic axial tensors (AATs) -- MPwfn.atomic_axial_tensors(), the density/wave-function-
overlap formulation (Krishnan, Shumberger & Crawford, in prep.), built on the diagonal
Born-Oppenheimer correction of Gauss, Tajti, Kallay, Stanton & Szalay, J. Chem. Phys. 125,
144111 (2006) [Eqs. (16), (18), (19)].

Reference: the independent analytic MP2-VCD implementation apyib
(https://github.com/bshumberger/apyib; Shumberger, Krishnan & Crawford), for (P)-hydrogen
peroxide / STO-3G at the manuscript geometry below (a.u.).  All-electron AND frozen-core,
both spin paths.
"""

import psi4
import pycc
import numpy as np


# (P)-hydrogen peroxide, manuscript geometry (a.u.); atom order H,H,O,O.
H2O2 = """
H -1.780954530308296   1.411647335546379   0.872055376436941
H  1.780954530308296  -1.411647335546379   0.872055376436941
O -1.371214332646589  -0.115525249760340  -0.054947416764017
O  1.371214332646589   0.115525249760340  -0.054947416764017
no_com
no_reorient
symmetry c1
units bohr
"""

# apyib reference AATs (electronic), indexed [3*atom + cart, beta].
AAT_REF = {
    'false': {(0, 0): -0.00452027, (0, 2): 0.02808062, (1, 1): -0.11644060,
              (2, 0): -0.16918159, (6, 2): -0.17323102, (7, 1): -0.18817788, (8, 0): 0.14146044},
    'true':  {(0, 0): -0.00452025, (0, 2): 0.02808068, (1, 1): -0.11644088,
              (2, 0): -0.16918189, (6, 2): -0.17326318, (7, 1): -0.18819783, (8, 0): 0.14143838},
}


# Water, fixed Cartesian frame (bohr), for the larger-basis (cc-pVDZ) AAT.  cc-pVDZ H2O2 AATs run
# ~100 s (too slow for the default suite); cc-pVDZ H2O is ~12 s.  The frame is locked (no_com,
# no_reorient) so pycc and the apyib reference below see the identical geometry.
WATER = """
O  0.000000000000  0.000000000000 -0.143225816552
H  0.000000000000  1.638036840407  1.136548822547
H  0.000000000000 -1.638036840407  1.136548822547
no_com
no_reorient
symmetry c1
units bohr
"""

# apyib reference electronic MP2 AAT for H2O / cc-pVDZ (independent implementation, the frame above);
# only the symmetry-allowed elements are nonzero.  pycc reproduces these (both spin paths) to ~1e-11.
AAT_REF_CCPVDZ = {
    (0, 1):  0.1344969130, (1, 0): -0.1531649831,
    (3, 1): -0.1289334907, (3, 2):  0.1522846408, (4, 0):  0.0925725477, (5, 0): -0.1534201482,
    (6, 1): -0.1289334907, (6, 2): -0.1522846408, (7, 0):  0.0925725477, (8, 0):  0.1534201482,
}


def _mpwfn(orbital_basis='spatial', freeze_core='false', geom=H2O2, basis='STO-3G'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp


def test_mp2_aat_all_electron():
    """All-electron spatial MP2 electronic AAT (SCF reference + correlation, via the pycc.aat
    facade) reproduces the apyib reference for (P)-H2O2/STO-3G."""
    P = np.asarray(pycc.aat(_mpwfn()).electronic).reshape(-1, 3)
    for (row, col), ref in AAT_REF['false'].items():
        assert abs(P[row, col] - ref) < 1e-6, (row, col, P[row, col], ref)


def test_mp2_aat_frozen_core():
    """Frozen-core spatial MP2 electronic AAT reproduces the independent apyib frozen-core
    reference (all-electron SCF reference + frozen-core correlation)."""
    P = np.asarray(pycc.aat(_mpwfn(freeze_core='true')).electronic).reshape(-1, 3)
    for (row, col), ref in AAT_REF['true'].items():
        assert abs(P[row, col] - ref) < 1e-6, (row, col, P[row, col], ref)


def test_mp2_aat_so_equals_spatial():
    """Spin-orbital MP2 correlation AAT == spin-adapted (the keystone), all-electron and
    frozen-core; the electronic total also matches the apyib reference."""
    for fc in ('false', 'true'):
        P = np.asarray(_mpwfn(freeze_core=fc).atomic_axial_tensors()).reshape(-1, 3)
        P_so = np.asarray(_mpwfn('spinorbital', fc).atomic_axial_tensors()).reshape(-1, 3)
        assert np.max(np.abs(P_so - P)) < 1e-9, (fc, np.max(np.abs(P_so - P)))
        E_so = np.asarray(pycc.aat(_mpwfn('spinorbital', fc)).electronic).reshape(-1, 3)
        for (row, col), ref in AAT_REF[fc].items():
            assert abs(E_so[row, col] - ref) < 1e-6, (fc, row, col, E_so[row, col], ref)


def test_mp2_aat_ccpvdz_vs_apyib():
    """Larger basis (cc-pVDZ): the electronic MP2 AAT (H2O -- a real virtual space with polarization
    functions and several virtuals per irrep, unlike STO-3G/H2O) reproduces the independent apyib
    reference, for BOTH the spin-adapted and the spin-orbital paths (so SO == spatial as well)."""
    P = np.asarray(pycc.aat(_mpwfn('spatial', 'false', WATER, 'cc-pVDZ')).electronic).reshape(-1, 3)
    P_so = np.asarray(pycc.aat(_mpwfn('spinorbital', 'false', WATER, 'cc-pVDZ')).electronic).reshape(-1, 3)
    for (row, col), ref in AAT_REF_CCPVDZ.items():
        assert abs(P[row, col] - ref) < 1e-6, ('spatial', row, col, P[row, col], ref)
        assert abs(P_so[row, col] - ref) < 1e-6, ('SO', row, col, P_so[row, col], ref)


def test_mp2_aat_gauge_invariance():
    """The MP2 correlation AAT is invariant to the orbital gauge of the redundant magnetic oo/vv
    response (non-canonical default vs canonical), all-electron and frozen-core, both spin paths.
    (The dropped SCF-reference block is itself gauge invariant, so the correlation is too.)"""
    for fc in ('false', 'true'):
        for ob in ('spatial', 'spinorbital'):
            mp = _mpwfn(ob, fc)
            nc = np.asarray(mp.atomic_axial_tensors(gauge='non-canonical'))
            ca = np.asarray(mp.atomic_axial_tensors(gauge='canonical'))
            assert np.max(np.abs(nc - ca)) < 1e-9, (fc, ob, np.max(np.abs(nc - ca)))


def test_mp2_aat_normalization():
    """MP2 normalization N < 1 and (closed shell) spin-orbital == spin-adapted."""
    N = _mpwfn()._mp2_normalization()
    assert 0.9 < N < 1.0
    assert abs(N - _mpwfn('spinorbital')._so_mp2_normalization()) < 1e-10
