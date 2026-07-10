"""
Rotation-invariant spin-orbital (T) energy (``cctriples.t_invariant_so``).

A reference/testing instrument: it solves ``<nu3|([F,T3] + [V,T2])|0> = 0`` for an
explicitly-stored T3, carrying the full off-diagonal Fock via the iterative ``[F,T3]``
commutator (``F = diag(F)`` -> the denominator, ``F_offdiag`` -> the coupling).  The
resulting (T) energy is therefore invariant to occupied-occupied and virtual-virtual MO
rotations, exactly like the HF and CCSD energies -- unlike the canonical batched driver
(``t_vikings_so``), whose diagonal-denominator T3 assumes a diagonal Fock and so changes
under such a rotation.

Two guards:
  1. Canonical reference: ``t_invariant_so == t_vikings_so`` (to machine precision) and both
     equal Psi4's (T) correction -- i.e. the invariant driver reduces to standard (T) when
     the Fock is diagonal.
  2. Rotation invariance: after an arbitrary unitary rotation *within* the active-occupied
     and virtual spaces (the deep core is frozen -- rotating it into the valence would make
     the off-diagonal Fock overwhelm the diagonal-Fock Jacobi preconditioner; the invariance
     is exact regardless), with CCSD re-solved from scratch in the rotated basis, the
     invariant (T) is unchanged while the standard (T) shifts.  Pure CCSD is unchanged too
     (a sanity check on the re-solve).
"""

import psi4
import pycc
import numpy as np
from scipy.linalg import expm
from pycc.cctriples import t_invariant_so, t_vikings_so


# H2O, C1, bohr (the test_034 (T)-density frame).
GEOM = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""


def _so_ccsd(wfn):
    """Spin-orbital CCSD (pure -- no bundled (T)), converged tightly."""
    cc = pycc.ccwfn(wfn, model='ccsd', orbital_basis='spinorbital')
    cc.solve_cc(1e-12, 1e-12, 100)
    return cc


def _both_T(cc):
    """(standard batched (T), rotation-invariant (T)) from a solved CCSD wavefunction."""
    args = (cc.o, cc.v, cc.t1, cc.t2, cc.H.F, cc.H.ERI, cc.contract)
    return t_vikings_so(*args), t_invariant_so(*args)


def test_invariant_t_reduces_to_standard(rhf_wfn):
    """Canonical reference: invariant (T) == standard batched (T) == Psi4 (T)."""
    wfn = rhf_wfn(GEOM, "6-31G")                       # fixture default: freeze_core='true'
    psi4.set_options({'e_convergence': 1e-11, 'r_convergence': 1e-11})
    psi4.energy('CCSD(T)')
    et_psi4 = (psi4.variable('CCSD(T) CORRELATION ENERGY')
               - psi4.variable('CCSD CORRELATION ENERGY'))

    cc = _so_ccsd(wfn)
    std, inv = _both_T(cc)
    assert abs(std - inv) < 1e-11, (std, inv)          # reduces to standard (T)
    assert abs(inv - et_psi4) < 1e-9, (inv, et_psi4)   # external Psi4 anchor


def test_invariant_t_rotation_invariance(rhf_wfn):
    """Arbitrary active-space oo/vv rotation, CCSD re-solved: invariant (T) unchanged,
    standard (T) shifts, pure CCSD unchanged."""
    wfn = rhf_wfn(GEOM, "6-31G")
    cc = _so_ccsd(wfn)
    std_can, inv_can = _both_T(cc)
    o, v = cc.o, cc.v
    ccsd_can = cc._so_cc_energy(o, v, cc.H.F, cc.H.ERI, cc.t1, cc.t2)

    # arbitrary rotation within the active-occ and virtual blocks (identity on frozen core)
    F = np.asarray(cc.H.F)
    ERI = np.asarray(cc.H.ERI)
    nmo = F.shape[0]
    rng = np.random.default_rng(7)
    kappa = np.zeros((nmo, nmo))
    no_a, nv_a = o.stop - o.start, v.stop - v.start
    Ko = rng.standard_normal((no_a, no_a)); kappa[o, o] = 0.30 * (Ko - Ko.T)
    Kv = rng.standard_normal((nv_a, nv_a)); kappa[v, v] = 0.30 * (Kv - Kv.T)
    U = expm(kappa)
    Frot = U.T @ F @ U
    ERIrot = np.einsum('pi,qj,rk,sl,pqrs->ijkl', U, U, U, U, ERI, optimize=True)

    # re-solve CCSD from scratch in the rotated basis (overwrite the MO integrals + a fresh guess)
    cc_rot = pycc.ccwfn(wfn, model='ccsd', orbital_basis='spinorbital')
    cc_rot.H.F = Frot
    cc_rot.H.ERI = ERIrot
    eps = np.diag(Frot)
    Dia = eps[o][:, None] - eps[v][None, :]
    Dijab = (eps[o][:, None, None, None] + eps[o][None, :, None, None]
             - eps[v][None, None, :, None] - eps[v][None, None, None, :])
    cc_rot.t1 = Frot[o, v] / Dia
    cc_rot.t2 = ERIrot[o, o, v, v] / Dijab
    cc_rot.solve_cc(1e-12, 1e-12, 200)
    std_rot, inv_rot = _both_T(cc_rot)
    ccsd_rot = cc_rot._so_cc_energy(o, v, Frot, ERIrot, cc_rot.t1, cc_rot.t2)

    assert abs(ccsd_rot - ccsd_can) < 1e-10, (ccsd_can, ccsd_rot)   # pure CCSD: invariant
    assert abs(inv_rot - inv_can) < 1e-9, (inv_can, inv_rot)        # invariant (T): unchanged
    assert abs(std_rot - std_can) > 1e-6, (std_can, std_rot)        # standard (T): shifts
