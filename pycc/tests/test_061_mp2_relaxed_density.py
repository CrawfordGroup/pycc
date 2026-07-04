"""
MP2 relaxed (orbital-response) one-particle density and analytic nuclear gradient, both
the spin-orbital and the spin-adapted (closed-shell RHF) paths; docs/DERIVATIVES_PLAN_2026-06.md.

The relaxed density adds the orbital-relaxation (Z-vector) contribution to the
unrelaxed MP2 correlation density; the gradient assembles it with the cumulant 2-PDM
and energy-weighted density against the skeleton derivative integrals. Both follow the
CC gradient formulation (Gauss, Stanton & Bartlett, JCP 95, 2623 (1991)).

Validation:
  * relaxed MP2 dipole -Tr(D_relaxed mu) vs a finite field of (E_MP2 - E_SCF), frozen (see
    FF_CORR_MU_Z; the open-shell NH2 value is pinned to its 2-B1 ground state via NH2_OCC);
  * analytic MP2 gradient vs psi4.gradient('mp2');
  * keystone: the spin-adapted and spin-orbital gradients agree on a closed shell.
"""

import psi4
import pycc
import numpy as np

WATER = """
O
H 1 0.96
H 1 0.96 2 104.5
"""

# Open-shell UHF reference: NH2 (2-B1, bent).  Run in C2v with the ground-state occupation pinned
# (NH2_OCC) so the SCF cannot fall into the 2-A1 excited solution ~0.074 Eh higher that a poor guess
# ('core') otherwise reaches; the pinned 2-B1 is guess- and platform-independent, hence freezeable.
# Deliberately not OH (2-Pi), whose degenerate pi orbitals leave the UHF non-reproducible and the
# orbital Hessian near-singular.
NH2 = """
0 2
N
H 1 1.02
H 1 1.02 2 103.0
"""
NH2_OCC = {'docc': [3, 0, 0, 1], 'socc': [0, 0, 1, 0]}   # pin 2-B1 ground state (C2v irreps A1,A2,B1,B2)


# Frozen finite-field oracles.  The correlation dipole mu_z from the psi4 energy finite field
# (_ff_corr_dipole / _ff_total_dipole) is a disposable *external* oracle -- it re-derives, through
# psi4, a number the pycc analytic routes already compute -- so we freeze it once rather than re-run
# 4-8 psi4 energy evaluations per test.  Each value below was validated against the psi4 5-point
# field to ~1e-11; the analytic routes are cross-checked live by the explicit==relaxed and (for the
# gradient) SO==spatial keystones, which all land on these same numbers.  Regenerate with
# _ff_corr_dipole / _ff_total_dipole.  The open-shell NH2 value is made reproducible by pinning the
# 2-B1 ground-state occupation (NH2_OCC, C2v); without the pin its UHF is bistable (a 2-A1 solution
# ~0.074 Eh higher is reachable by a poor SCF guess such as 'core').
FF_CORR_MU_Z = {
    ('6-31G', False):  -0.0349952749,   # H2O/6-31G, all-electron
    ('cc-pVDZ', False): -0.0367852258,   # H2O/cc-pVDZ, all-electron
    ('6-31G', True):   -0.0351121565,   # H2O/6-31G, frozen core
}
FF_TOTAL_MU_Z_631 = 1.0003212210         # total (HF + correlation) MP2 dipole mu_z, H2O/6-31G
FF_CORR_MU_Z_UHF_NH2 = -0.0249262717     # NH2 (2-B1, C2v, pinned occ) / 6-31G, UHF-MP2 correlation


def _ff_corr_dipole(geom, basis, F=0.0005, freeze_core='false', reference='rhf', occ=None):
    """Relaxed correlation mu_z by a 5-point finite field of (E_MP2 - E_SCF)."""
    def e(model, Fz):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(geom)
        opt = {'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv', 'reference': reference,
               'freeze_core': freeze_core, 'e_convergence': 1e-12, 'd_convergence': 1e-12}
        if occ:
            opt.update(occ)
        if Fz:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole',
                        'perturb_dipole': [0.0, 0.0, Fz]})
        psi4.set_options(opt)
        return psi4.energy(model)

    def mu(model):
        return (-e(model, 2 * F) + 8 * e(model, F)
                - 8 * e(model, -F) + e(model, -2 * F)) / (12 * F)
    return mu('mp2') - mu('scf')


def _pycc_corr_dipole(geom, basis, orbital_basis='spinorbital', freeze_core='false', reference='rhf', occ=None):
    """PyCC relaxed-MP2 electronic correlation mu_z (spin-orbital or spin-adapted),
    via the user-API :meth:`MPwfn.relaxed_dipole`."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    opt = {'basis': basis, 'scf_type': 'pk', 'reference': reference,
           'freeze_core': freeze_core, 'e_convergence': 1e-12, 'd_convergence': 1e-12}
    if occ:
        opt.update(occ)
    psi4.set_options(opt)
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp.relaxed_dipole()[2]


def test_mp2_relaxed_dipole_631g():
    """Relaxed spin-orbital MP2 dipole (mu_z) vs the frozen finite-field oracle, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G')
               - FF_CORR_MU_Z[('6-31G', False)]) < 1e-8


def test_sa_mp2_relaxed_dipole_631g():
    """Relaxed spin-adapted (closed-shell RHF) MP2 dipole vs the frozen finite-field oracle, H2O/6-31G."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G', orbital_basis='spatial')
               - FF_CORR_MU_Z[('6-31G', False)]) < 1e-8


def test_mp2_relaxed_dipole_ccpvdz():
    """Relaxed MP2 dipole vs the frozen finite-field oracle, H2O/cc-pVDZ (C2v: polarization
    functions and A2-irrep MOs). Exercises symmetry-adapted MOs in the relaxed density."""
    assert abs(_pycc_corr_dipole(WATER, 'cc-pVDZ')
               - FF_CORR_MU_Z[('cc-pVDZ', False)]) < 1e-8


def test_ump2_relaxed_dipole_nh2_631g():
    """Open-shell UHF-MP2 relaxed correlation dipole (mu_z) vs the frozen finite-field oracle,
    NH2 (2-B1, C2v, pinned occupation) / 6-31G. The open-shell oracle for the spin-orbital relaxed
    density (Z-vector orbital response)."""
    assert abs(_pycc_corr_dipole(NH2, '6-31G', reference='uhf', occ=NH2_OCC)
               - FF_CORR_MU_Z_UHF_NH2) < 1e-8


# ---- explicit-derivative route (derivints.pdf): correlation dipole from the full
# CPHF-folded derivatives of f and <pq||rs> (CPHF.perturbed_fock / perturbed_eri),
# contracted with the *unrelaxed* densities -- an independent computation of the same
# correlation dipole, and the validation of the perturbed-derivative engine that the
# analytic MP2 polarizability will build on. Spin-orbital path.

def _pycc_corr_dipole_explicit(geom, basis, orbital_basis='spinorbital', freeze_core='false'):
    """PyCC relaxed-MP2 correlation mu_z via :meth:`MPwfn._corr_dipole_explicit`."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp._corr_dipole_explicit()[2]


def test_mp2_explicit_corr_dipole_631g():
    """Explicit-derivative SO MP2 correlation dipole vs the frozen finite-field oracle, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole_explicit(geom, '6-31G')
               - FF_CORR_MU_Z[('6-31G', False)]) < 1e-8


def test_sa_mp2_explicit_corr_dipole_631g():
    """Explicit-derivative spin-adapted (spatial) MP2 correlation dipole vs the frozen
    finite-field oracle, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole_explicit(geom, '6-31G', orbital_basis='spatial')
               - FF_CORR_MU_Z[('6-31G', False)]) < 1e-8


def test_mp2_explicit_equals_relaxed_631g():
    """Keystone: the explicit-derivative correlation dipole equals the relaxed-density
    route (same number, computed without the Z-vector / relaxed density), both bases, H2O/6-31G."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole_explicit(geom, '6-31G')
               - _pycc_corr_dipole(geom, '6-31G')) < 1e-10
    assert abs(_pycc_corr_dipole_explicit(geom, '6-31G', orbital_basis='spatial')
               - _pycc_corr_dipole(geom, '6-31G', orbital_basis='spatial')) < 1e-10


def test_mp2_explicit_corr_dipole_ccpvdz():
    """Explicit-derivative MP2 correlation dipole vs the frozen finite-field oracle, H2O/cc-pVDZ
    (C2v: polarization functions and A2-irrep MOs), both spin-orbital and spin-adapted."""
    assert abs(_pycc_corr_dipole_explicit(WATER, 'cc-pVDZ')
               - FF_CORR_MU_Z[('cc-pVDZ', False)]) < 1e-8
    assert abs(_pycc_corr_dipole_explicit(WATER, 'cc-pVDZ', orbital_basis='spatial')
               - FF_CORR_MU_Z[('cc-pVDZ', False)]) < 1e-8


def _pycc_corr_gradient_explicit_and_relaxed(geom, basis, orbital_basis='spinorbital',
                                             freeze_core='false'):
    """MP2 correlation nuclear gradient two ways: the explicit-derivative route
    (`_corr_gradient_explicit`) and the relaxed-density route (`gradient`, correlation-only)."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp._corr_gradient_explicit(), mp.gradient()


def test_mp2_explicit_corr_gradient_631g():
    """Keystone: the explicit-derivative SO MP2 correlation nuclear gradient equals the
    relaxed-density route, H2O/6-31G (C1) -- the nuclear analog of the field engine,
    exercising the full skeleton + CPHF response."""
    geom = WATER + "symmetry c1\n"
    g_explicit, g_relaxed = _pycc_corr_gradient_explicit_and_relaxed(geom, '6-31G')
    assert np.max(np.abs(g_explicit - g_relaxed)) < 1e-9


def test_sa_mp2_explicit_corr_gradient_631g():
    """Keystone: the explicit-derivative spin-adapted (spatial) MP2 correlation gradient
    equals the relaxed-density route, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    g_explicit, g_relaxed = _pycc_corr_gradient_explicit_and_relaxed(geom, '6-31G', 'spatial')
    assert np.max(np.abs(g_explicit - g_relaxed)) < 1e-9


def test_mp2_explicit_corr_gradient_ccpvdz():
    """Explicit-derivative MP2 correlation gradient == relaxed-density route, H2O/cc-pVDZ
    (C2v: polarization functions and A2-irrep MOs), both bases."""
    g_so_e, g_so_r = _pycc_corr_gradient_explicit_and_relaxed(WATER, 'cc-pVDZ')
    assert np.max(np.abs(g_so_e - g_so_r)) < 1e-9
    g_sa_e, g_sa_r = _pycc_corr_gradient_explicit_and_relaxed(WATER, 'cc-pVDZ', 'spatial')
    assert np.max(np.abs(g_sa_e - g_sa_r)) < 1e-9


def test_fc_sa_mp2_explicit_corr_dipole_631g():
    """Frozen-core spin-adapted (spatial) explicit-derivative MP2 correlation dipole vs the frozen
    finite-field oracle, H2O/6-31G (C1) -- the core<->active orbital response (the canonical
    d_x f_ij = 0 block of U) is what the explicit route needs for frozen core."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole_explicit(geom, '6-31G', orbital_basis='spatial', freeze_core='true')
               - FF_CORR_MU_Z[('6-31G', True)]) < 1e-8


def test_fc_sa_mp2_explicit_corr_gradient_631g():
    """Keystone: frozen-core spin-adapted explicit-derivative MP2 correlation gradient ==
    relaxed-density route, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    g_explicit, g_relaxed = _pycc_corr_gradient_explicit_and_relaxed(
        geom, '6-31G', orbital_basis='spatial', freeze_core='true')
    assert np.max(np.abs(g_explicit - g_relaxed)) < 1e-9


def test_fc_so_mp2_explicit_corr_dipole_631g():
    """Frozen-core spin-orbital explicit-derivative MP2 correlation dipole vs the frozen finite-field
    oracle, H2O/6-31G (C1). The core<->active response is built over MPwfn's own full-occupied SO
    space (no all-electron SO HFwfn to borrow -- different spin ordering)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole_explicit(geom, '6-31G', orbital_basis='spinorbital', freeze_core='true')
               - FF_CORR_MU_Z[('6-31G', True)]) < 1e-8


def test_fc_so_mp2_explicit_corr_gradient_631g():
    """Keystone: frozen-core spin-orbital explicit-derivative MP2 correlation gradient ==
    relaxed-density route, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    g_explicit, g_relaxed = _pycc_corr_gradient_explicit_and_relaxed(
        geom, '6-31G', orbital_basis='spinorbital', freeze_core='true')
    assert np.max(np.abs(g_explicit - g_relaxed)) < 1e-9


def _pycc_gradient(geom, basis, orbital_basis='spinorbital', freeze_core='false', reference='rhf', occ=None):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    opt = {'basis': basis, 'scf_type': 'pk', 'reference': reference,
           'freeze_core': freeze_core, 'e_convergence': 1e-12, 'd_convergence': 1e-12}
    if occ:
        opt.update(occ)
    psi4.set_options(opt)
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return np.asarray(pycc.gradient(mp).total)


def _psi4_mp2_gradient(geom, basis, reference='rhf', occ=None):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    opt = {'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv', 'reference': reference,
           'freeze_core': 'false', 'e_convergence': 1e-12, 'd_convergence': 1e-12}
    if occ:
        opt.update(occ)
    psi4.set_options(opt)
    return np.asarray(psi4.gradient('mp2'))


def test_mp2_gradient_631g():
    """Spin-orbital MP2 analytic nuclear gradient vs Psi4, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G')
                         - _psi4_mp2_gradient(geom, '6-31G'))) < 1e-8


def test_sa_mp2_gradient_631g():
    """Spin-adapted (closed-shell RHF) MP2 gradient vs Psi4, H2O/6-31G (C1)."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G', orbital_basis='spatial')
                         - _psi4_mp2_gradient(geom, '6-31G'))) < 1e-8


def test_ump2_gradient_nh2_631g():
    """Open-shell UHF-MP2 analytic nuclear gradient vs Psi4, NH2 (2-B1, C2v, pinned occupation) / 6-31G.

    The open-shell oracle for the spin-orbital Z-vector gradient (``_so_zvector``).  psi4 is the
    right check here rather than the explicit == relaxed identity: an open-shell UHF spin-orbital
    orbital Hessian carries a near-zero mode, so the single linear solve both internal routes run
    through is ill-conditioned and their difference is platform-dependent -- but the final gradient
    is orthogonal to that mode and reproduces psi4 to machine precision.  The 2-B1 occupation is
    pinned (NH2_OCC) so the SCF cannot fall into the 2-A1 excited solution; unlike OH (2-Pi) the
    reference is then reproducible."""
    assert np.max(np.abs(_pycc_gradient(NH2, '6-31G', reference='uhf', occ=NH2_OCC)
                         - _psi4_mp2_gradient(NH2, '6-31G', reference='uhf', occ=NH2_OCC))) < 1e-8


def test_mp2_gradient_spatial_vs_so_631g():
    """Keystone: the spin-adapted and spin-orbital MP2 gradients agree on a closed shell."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G', orbital_basis='spatial')
                         - _pycc_gradient(geom, '6-31G', orbital_basis='spinorbital'))) < 1e-10


def test_mp2_gradient_ccpvdz():
    """Spin-orbital MP2 gradient vs Psi4, H2O/cc-pVDZ (C2v: polarization + A2-irrep MOs)."""
    assert np.max(np.abs(_pycc_gradient(WATER, 'cc-pVDZ')
                         - _psi4_mp2_gradient(WATER, 'cc-pVDZ'))) < 1e-8


def test_sa_mp2_gradient_ccpvdz():
    """Spin-adapted MP2 gradient vs Psi4, H2O/cc-pVDZ (C2v: polarization + A2-irrep MOs)."""
    assert np.max(np.abs(_pycc_gradient(WATER, 'cc-pVDZ', orbital_basis='spatial')
                         - _psi4_mp2_gradient(WATER, 'cc-pVDZ'))) < 1e-8


# ---- frozen-core spin-adapted MP2 relaxed density and gradient ----
# The orbital response spans the full occupied space: a core<->active-occupied direct
# divide plus the occ-virtual Z-vector (with that coupling) over the full ndocc. The
# ground-truth oracle is the finite difference of PyCC's own frozen-core MP2 energy --
# Psi4's analytic frozen-core MP2 gradient is itself inconsistent with its energy at ~1e-5.

WATER_XYZ = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.814), ("H", 1.755, 0.0, -0.46)]  # bohr


def _fc_geom(disp=None):
    g = [list(a) for a in WATER_XYZ]
    if disp:
        i, c, h = disp
        g[i][1 + c] += h
    return ("units bohr\nnocom\nnoreorient\n"
            + "\n".join(f"{a[0]} {a[1]} {a[2]} {a[3]}" for a in g) + "\nsymmetry c1\n")


def _fc_mp2_total_energy(disp, basis):
    """PyCC total frozen-core MP2 energy (E_SCF + E_corr) at a displaced geometry."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(_fc_geom(disp))
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': 'true',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    escf, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis='spatial')
    return escf + mp.compute_energy()


def test_fc_mp2_relaxed_dipole_631g():
    """Frozen-core relaxed spin-adapted MP2 dipole (mu_z) vs the frozen finite-field oracle,
    H2O/6-31G (C1) -- validates the frozen-core relaxed 1-PDM (the core-active divide and the
    full-occ Z-vector)."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G', orbital_basis='spatial', freeze_core='true')
               - FF_CORR_MU_Z[('6-31G', True)]) < 1e-8


def test_fc_mp2_gradient_vs_energy_fd_631g():
    """Frozen-core spin-adapted MP2 nuclear gradient vs a 5-point finite difference of
    PyCC's own frozen-core MP2 total energy, H2O/6-31G (C1). This is the ground-truth
    oracle for the analytic gradient (Psi4's analytic frozen-core MP2 gradient disagrees
    with its own energy at ~7e-6, so it is *not* used here)."""
    basis = '6-31G'
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(_fc_geom())
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': 'true',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis='spatial')
    mp.compute_energy()
    assert mp.nfzc > 0                                   # frozen core is actually active
    gA = np.asarray(pycc.gradient(mp).total)             # total = SCF + correlation

    h = 2.0e-3
    gF = np.zeros((3, 3))
    for i in range(3):
        for c in range(3):
            e = [_fc_mp2_total_energy((i, c, k * h), basis) for k in (-2, -1, 1, 2)]
            gF[i, c] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * h)
    assert np.max(np.abs(gA - gF)) < 1e-8


# ---- SCF dipole (HFwfn.dipole) and the total MP2 dipole ----
# The reference dipole is kept separate from the correlation dipole (as for the gradient):
# the total MP2 dipole is HFwfn.dipole() + MPwfn.relaxed_dipole().

def _scf_dipole(geom, basis):
    """PyCC HFwfn.dipole() (spatial + spin-orbital) and Psi4's analytic SCF dipole."""
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    ref = np.asarray(psi4.variable('SCF DIPOLE'))
    return (pycc.HFwfn(wfn).dipole(),
            pycc.HFwfn(wfn, orbital_basis='spinorbital').dipole(), ref)


def test_hf_dipole_631g():
    """HFwfn.dipole() (spatial and spin-orbital) vs Psi4's analytic SCF dipole, H2O/6-31G."""
    d_spatial, d_so, ref = _scf_dipole(WATER + "symmetry c1\n", '6-31G')
    assert np.max(np.abs(d_spatial - ref)) < 1e-9
    assert np.max(np.abs(d_so - ref)) < 1e-9


def test_hf_dipole_ccpvdz():
    """HFwfn.dipole() vs Psi4's analytic SCF dipole, H2O/cc-pVDZ (C2v)."""
    d_spatial, d_so, ref = _scf_dipole(WATER, 'cc-pVDZ')
    assert np.max(np.abs(d_spatial - ref)) < 1e-9
    assert np.max(np.abs(d_so - ref)) < 1e-9


def _ff_total_dipole(geom, basis, model, F=0.0005):
    """z-component of the total ``model`` dipole via a 5-point finite field of its energy."""
    def e(Fz):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(geom)
        opt = {'basis': basis, 'scf_type': 'pk', 'mp2_type': 'conv',
               'freeze_core': 'false', 'e_convergence': 1e-12, 'd_convergence': 1e-12}
        if Fz:
            opt.update({'perturb_h': True, 'perturb_with': 'dipole',
                        'perturb_dipole': [0.0, 0.0, Fz]})
        psi4.set_options(opt)
        return psi4.energy(model)
    return (-e(2 * F) + 8 * e(F) - 8 * e(-F) + e(-2 * F)) / (12 * F)


def test_mp2_total_dipole_631g():
    """Total MP2 dipole = HFwfn.dipole() + MPwfn.relaxed_dipole() vs the frozen finite-field
    oracle of the total MP2 energy, H2O/6-31G (C1) -- validates the reference/correlation split."""
    geom = WATER + "symmetry c1\n"
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis='spinorbital')
    mp.compute_energy()
    total = np.asarray(pycc.dipole(mp).total)
    assert abs(total[2] - FF_TOTAL_MU_Z_631) < 1e-8


def test_fc_so_mp2_relaxed_dipole_631g():
    """Frozen-core relaxed spin-orbital MP2 dipole (mu_z) vs the frozen finite-field oracle,
    H2O/6-31G (C1). Exercises the full-MO spin-orbital Hamiltonian (the frozen core is now kept
    in H) and the spin-orbital frozen-core relaxed 1-PDM."""
    geom = WATER + "symmetry c1\n"
    assert abs(_pycc_corr_dipole(geom, '6-31G', orbital_basis='spinorbital', freeze_core='true')
               - FF_CORR_MU_Z[('6-31G', True)]) < 1e-8


def test_fc_mp2_gradient_spatial_vs_so_631g():
    """Keystone: the frozen-core spin-orbital and spin-adapted MP2 gradients agree on a
    closed shell (H2O/6-31G, C1). Both are the exact derivative of the MP2(fc) energy, so
    they must match to machine precision -- the cleanest frozen-core validation, and a check
    that the full-MO spin-orbital Hamiltonian carries the same core response as the spatial
    path."""
    geom = WATER + "symmetry c1\n"
    assert np.max(np.abs(_pycc_gradient(geom, '6-31G', orbital_basis='spatial', freeze_core='true')
                         - _pycc_gradient(geom, '6-31G', orbital_basis='spinorbital', freeze_core='true'))) < 1e-10
