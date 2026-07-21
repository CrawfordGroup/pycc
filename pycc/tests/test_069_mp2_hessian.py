"""
MP2 molecular (nuclear) Hessian (correlation contribution) via the 2n+1 route --
H_corr[Aa,Bb] = d^2 E_corr / dX_{Aa} dX_{Bb}; DERIVATIVES_PLAN_2026-06.md.

The nuclear-nuclear analog of the polarizability/APT: differentiate the relaxed nuclear gradient
w.r.t. a second nucleus, using only 3N first-order solves (the perturbed relaxed density /
energy-weighted density and U^Y) plus the full nuclear-nuclear second skeletons
f^{XY}/<pq||rs>^{XY}/S^{XY}. The reference (SCF) Hessian is separate (HFwfn.hessian, carrying the
nuclear-repulsion term); the total is their sum.

Validation (frozen analytic references; the FD recipe that validated them once is kept but not
re-run):
  * primary: correlation Hessian columns vs frozen references, each validated once against a
    7-point O(h^6) finite difference of the analytic correlation gradient (H = d(gradient)/dX, a
    1/h stencil ~1e-13);
  * symmetry H = H.T and the translational sum rule sum_B H_{Aa,Bb} = 0 (E_corr is
    translationally invariant) -- FD-free checks;
  * keystone: spin-orbital == spin-adapted (closed-shell RHF) correlation Hessian.

Geometry is Cartesian in bohr with no_com/no_reorient so displacing an atom keeps the frame
fixed and matches the analytic (bohr) integral derivatives. The FD recipe uses pycc's own analytic
gradient (Psi4's frozen-core MP2 gradient bug rules out its analytic derivatives). 6-31G only
(the shared 2n+1 machinery is already cc-pVDZ-tested via the polarizability/APT).
"""

import psi4
import pycc
import numpy as np


BASE = np.array([[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.814137],
                 [0.0, 1.756000, -0.454300]])
SYM = ['O', 'H', 'H']


def _geom(coords):
    s = "units bohr\nsymmetry c1\nno_com\nno_reorient\n"
    for sym, xyz in zip(SYM, coords):
        s += f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n"
    return s


def _mpwfn(coords, orbital_basis='spatial', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(_geom(coords))
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn, orbital_basis=orbital_basis)
    mp.compute_energy()
    return mp


def _gradfd_col(orbital_basis, atom, cart, freeze_core='false', h=0.002):
    """Regeneration recipe for HESS_COL_* (not run in the tests): one Hessian column (natom*3,)
    via a 7-point O(h^6) FD of the analytic correlation gradient (the first-order Z-vector
    gradient, independent of the 2n+1 Hessian code the frozen columns validate) over the nuclear
    coordinate (atom, cart)."""
    def g(delta):
        c = BASE.copy(); c[atom, cart] += delta
        return np.asarray(_mpwfn(c, orbital_basis, freeze_core).gradient())
    return ((-g(-3*h) + 9*g(-2*h) - 45*g(-h)
             + 45*g(h) - 9*g(2*h) + g(3*h)) / (60*h)).reshape(-1)


# ---- frozen analytic Hessian columns (each validated once vs the 7-point gradient FD to ~1e-13) ----
# Analytic 2n+1 columns H[:, atom*3+cart], frozen rather than re-run.  6-31G spatial.
HESS_COL_631G = {   # all-electron
    (0, 2): np.array([-2.8609251475071246e-17, -0.018635317906270498, -0.012363235197342884, 2.6372824139656578e-17, 0.004451384302276866, -0.003435965582463366, 2.236427335412826e-18, 0.014183933603993532, 0.01579920077980606]),
    (2, 1): np.array([-3.4521721063771614e-18, 0.0013858164195239618, 0.014183933603993861, -2.0975137392280085e-18, -0.004090433966428426, -0.006963531058088191, 5.549685845604445e-18, 0.0027046175469041265, -0.007220402545905623]),
}
HESS_COL_631G_FC = {   # frozen core
    (0, 2): np.array([-2.870963418482767e-17, -0.018630196424885136, -0.012248276520495104, 2.6330327847888316e-17, 0.004429445169139285, -0.0035551803844278626, 2.379306336937463e-18, 0.014200751255745755, 0.01580345690492303]),
    (2, 1): np.array([-3.4376089472059303e-18, 0.0012651673537513648, 0.014200751255746071, -2.1200800955361294e-18, -0.0040801896909060465, -0.006963192813133718, 5.557689042741037e-18, 0.0028150223371544193, -0.007237558442612311]),
}


def test_mp2_corr_hessian_symmetry_sum_rule_631g():
    """All-electron correlation Hessian: symmetric, and translationally invariant
    (sum over the second atom vanishes), H2O/6-31G -- FD-free checks."""
    H = _mpwfn(BASE, 'spatial').hessian()
    assert H.shape == (9, 9)
    assert np.max(np.abs(H - H.T)) < 1e-10
    assert np.max(np.abs(H.reshape(3, 3, 3, 3).sum(axis=2))) < 1e-10


def test_mp2_corr_hessian_gradfd_631g():
    """All-electron correlation Hessian columns vs their frozen references (validated once against
    a 7-point FD of the analytic gradient to ~1e-13), H2O/6-31G."""
    H = _mpwfn(BASE, 'spatial').hessian()
    for (atom, cart), ref in HESS_COL_631G.items():
        j = atom * 3 + cart
        assert np.max(np.abs(H[:, j] - ref)) < 1e-9


def test_so_mp2_corr_hessian_vs_spatial_631g():
    """Keystone (6-31G): closed-shell spin-orbital == spin-adapted correlation Hessian."""
    H_so = _mpwfn(BASE, 'spinorbital').hessian()
    H_sa = _mpwfn(BASE, 'spatial').hessian()
    assert np.max(np.abs(H_so - H_sa)) < 1e-11


# ---- frozen core (the nuclear-nuclear core<->active response) ----

def test_fc_mp2_corr_hessian_gradfd_631g():
    """Frozen-core correlation Hessian columns vs their frozen references (validated once against
    the 7-point gradient FD to ~1e-13), H2O/6-31G."""
    H = _mpwfn(BASE, 'spatial', freeze_core='true').hessian()
    for (atom, cart), ref in HESS_COL_631G_FC.items():
        j = atom * 3 + cart
        assert np.max(np.abs(H[:, j] - ref)) < 1e-9


def test_fc_mp2_corr_hessian_symmetry_sum_rule_631g():
    """Frozen-core correlation Hessian: symmetric and translationally invariant."""
    H = _mpwfn(BASE, 'spatial', freeze_core='true').hessian()
    assert np.max(np.abs(H - H.T)) < 1e-10
    assert np.max(np.abs(H.reshape(3, 3, 3, 3).sum(axis=2))) < 1e-10


def test_fc_so_mp2_corr_hessian_vs_spatial_631g():
    """Keystone (6-31G, frozen core): spin-orbital == spin-adapted correlation Hessian."""
    H_so = _mpwfn(BASE, 'spinorbital', freeze_core='true').hessian()
    H_sa = _mpwfn(BASE, 'spatial', freeze_core='true').hessian()
    assert np.max(np.abs(H_so - H_sa)) < 1e-11
