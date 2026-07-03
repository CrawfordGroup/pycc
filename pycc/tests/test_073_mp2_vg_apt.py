"""
MP2 velocity-gauge (VG) atomic polar tensors -- MPwfn.velocity_dipole_derivatives(), the
density/wave-function-overlap formulation.  This is the atomic-axial-tensor machinery
(test_072_mp2_aat) with the magnetic-dipole operator replaced by the linear momentum p = -i nabla
and the Levi-Civita nuclear term replaced by the length-gauge Z_A delta.

Validation (mirrors the AAT suite; apyib provides no MP2 VG APT so the HF VG APT -- Amos, Jalkanen
& Stephens, JPC 92, 5571 (1988), pinned in test_071 -- is the correlation-limit reference):
  * regression values (all-electron and frozen-core) for (P)-H2O2 / STO-3G,
  * reduces to the HF VG APT as the correlation vanishes,
  * spin-orbital == spin-adapted (the keystone), machine precision,
  * orbital-gauge invariance (non-canonical default == canonical), machine precision.
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

# Regression VG APTs [3*atom + beta, alpha], self-consistent reference (validated against the
# HF VG APT in the correlation limit, and SO==spatial / gauge-invariance to machine precision).
VG_REF = {
    'false': {(0, 0): 0.95836464, (0, 1): 0.17156159, (0, 2): 0.10371825, (1, 1): 0.32086739,
              (2, 0): 0.04286371, (6, 0): 6.80543370, (6, 2): 0.16251338, (8, 0): 0.26002860},
    'true':  {(0, 0): 0.95836834, (0, 1): 0.17156195, (0, 2): 0.10371842, (1, 1): 0.32086827,
              (2, 0): 0.04286475, (6, 0): 6.80538456, (6, 2): 0.16250689, (8, 0): 0.26002293},
}


def _mpwfn(orbital_basis='spatial', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(H2O2)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp, wfn


def test_mp2_vg_apt_all_electron():
    """All-electron spatial MP2 VG APT reproduces the regression reference for (P)-H2O2/STO-3G."""
    P = np.asarray(_mpwfn()[0].velocity_dipole_derivatives()).reshape(-1, 3)
    for (row, col), ref in VG_REF['false'].items():
        assert abs(P[row, col] - ref) < 1e-6, (row, col, P[row, col], ref)


def test_mp2_vg_apt_frozen_core():
    """Frozen-core spatial MP2 VG APT reproduces the regression reference."""
    P = np.asarray(_mpwfn(freeze_core='true')[0].velocity_dipole_derivatives()).reshape(-1, 3)
    for (row, col), ref in VG_REF['true'].items():
        assert abs(P[row, col] - ref) < 1e-6, (row, col, P[row, col], ref)


def test_mp2_vg_apt_reduces_to_hf():
    """The MP2 VG APT reduces to the (Amos-pinned) HF VG APT as the correlation vanishes: the
    difference is the small MP2 correlation contribution."""
    mp, wfn = _mpwfn()
    VG = np.asarray(mp.velocity_dipole_derivatives())
    HF = np.asarray(pycc.HFwfn(wfn).velocity_dipole_derivatives())
    diff = np.max(np.abs(VG - HF))
    assert diff < 0.05, diff            # small correlation contribution
    assert diff > 1e-4                  # ... but nonzero (correlation is really present)


def test_mp2_vg_apt_so_equals_spatial():
    """Spin-orbital MP2 VG APT == spin-adapted (the keystone), all-electron and frozen-core."""
    for fc in ('false', 'true'):
        P = np.asarray(_mpwfn(freeze_core=fc)[0].velocity_dipole_derivatives())
        P_so = np.asarray(_mpwfn('spinorbital', fc)[0].velocity_dipole_derivatives())
        assert np.max(np.abs(P_so - P)) < 1e-9, (fc, np.max(np.abs(P_so - P)))


def test_mp2_vg_apt_gauge_invariance():
    """The VG APT is invariant to the orbital gauge of the redundant momentum oo/vv response
    (non-canonical default vs canonical), all-electron and frozen-core, both spin paths."""
    for fc in ('false', 'true'):
        for ob in ('spatial', 'spinorbital'):
            mp = _mpwfn(ob, fc)[0]
            nc = np.asarray(mp.velocity_dipole_derivatives(gauge='non-canonical'))
            ca = np.asarray(mp.velocity_dipole_derivatives(gauge='canonical'))
            assert np.max(np.abs(nc - ca)) < 1e-9, (fc, ob, np.max(np.abs(nc - ca)))
