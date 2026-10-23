"""
CCSD velocity-gauge atomic polar tensors -- CCderiv._correlation_velocity_dipole_derivatives
through the ``pycc.apt(..., gauge='velocity')`` facade.

Eq. (11) of ``docs/cc_vgapt_derivation.pdf`` is Eq. (39) of the AAT derivation with the magnetic
field replaced by a uniform vector potential, block for block, so this tensor shares every block
with the AAT (``CCderiv._mixed_overlap_tensor``) and differs only in the imaginary perturbation:
``Perturbation('vecpot', g)`` in place of ``Perturbation('magnetic', b)``.  The perturbing operator
is the electronic linear momentum ``pi^e = -i grad``, anti-Hermitian exactly as the magnetic dipole
is, so the orbital-orbital block is a commutator here too.

Anchors, in the order of the derivation's section 6:

* **two electrons**: CCSD is exact, so it must equal the validated CISD velocity-gauge APT
  (``test_080``) element by element.  This also fixes the ``-2`` prefactor, which the invariance
  tests cannot see;
* **the momentum-form sum rule**, Eq. (13), which shares no machinery with the tensor's assembly;
* **gauge invariance** in both conventions, and the spin-orbital/spatial keystone.

Not covered here: the origin-shift relation of Eq. (12), which needs electronic AATs at two gauge
origins and so waits on a way to move the gauge origin of the angular-momentum integrals.
"""

import numpy as np
import psi4
import pytest

import pycc
from pycc.cphf import Perturbation


H3P = """1 1
H  0.000000000000  0.000000000000  0.000000000000
H  1.700000000000  0.000000000000  0.000000000000
H  0.620000000000  1.430000000000  0.170000000000
symmetry c1
units bohr
"""

WATER_C1 = """
O  0.000000000000  0.000000000000 -0.129476880010
H  0.080000000000 -1.494186750504  1.027446024111
H -0.050000000000  1.394186750504  1.127446024111
symmetry c1
units bohr
"""


def _scf(geom, basis='STO-3G', freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-13, 'd_convergence': 1e-13})
    return psi4.energy('scf', return_wfn=True)[1]


def _ccderiv(geom, basis='STO-3G', freeze_core='false', model='CCSD', orbital_basis='spatial'):
    cc = pycc.ccwfn(_scf(geom, basis, freeze_core), model=model, orbital_basis=orbital_basis)
    cc.solve_cc(1e-13, 1e-13, 200)
    return pycc.CCderiv(cc)


def test_cc_vg_apt_two_electron_equals_cisd():
    """For two electrons CCSD is exact, so its correlation velocity-gauge APT must equal the
    validated CISD one.  This is what fixes the ``-2`` of ``E^(VG) = -(2 i e hbar/m) V``: without
    it the ratio is exactly -1/2, which no invariance test would reveal.

    6-31G, since H3+ in a minimal basis has a singular DIIS B matrix on some LAPACK builds."""
    wfn = _scf(H3P, basis='6-31G')

    cc = pycc.ccwfn(wfn, model='CCSD', orbital_basis='spatial')
    cc.solve_cc(1e-13, 1e-13, 200)
    V_cc = np.asarray(pycc.CCderiv(cc)._correlation_velocity_dipole_derivatives())

    ci = pycc.CIwfn(wfn, model='CISD', orbital_basis='spatial')
    ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=300)
    V_ci = np.asarray(pycc.CIderiv(ci)._correlation_velocity_dipole_derivatives())

    assert np.abs(V_ci).max() > 1e-4, np.abs(V_ci).max()
    assert np.abs(V_cc - V_ci).max() < 1e-10, np.abs(V_cc - V_ci).max()


@pytest.mark.parametrize('orbital_basis', ['spatial', 'spinorbital'])
def test_cc_vg_apt_momentum_sum_rule(orbital_basis):
    """Eq. (13): summing over atoms gives the vector-potential derivative of the momentum
    expectation value::

        sum_lambda 2 V = sum_pq D^A_pq (grad_alpha)_pq + sum_pq D_pq [grad_alpha, kappa^A]_pq

    Same bracket as the AAT's sum rule with A in place of H.  The right-hand side needs no nuclear
    response at all, so the two sides share no machinery."""
    d = _ccderiv(WATER_C1, orbital_basis=orbital_basis)
    c = d.contract
    grad = [np.real(-1.0j * np.asarray(d.ccwfn.H.p[a])) for a in range(3)]
    aD, aU = {}, {}
    for g in range(3):
        p = Perturbation('vecpot', g)
        aD[g] = d._aat_perturbed_opdm(p)
        aU[g] = d._cc_perturbed_ints(p)[2]
    D0 = np.asarray(d._unrelaxed_densities()[0])
    V = d._mixed_overlap_tensor('vecpot', 'non-canonical')     # the tensor without the -2

    checked = 0
    for a in range(3):
        for g in range(3):
            lhs = 2.0 * V[:, a, g].sum()
            rhs = (c('pq,pq->', aD[g], grad[a])
                   + c('pq,pq->', D0, grad[a] @ aU[g] - aU[g] @ grad[a]))
            assert abs(lhs - rhs) < 1e-10, (a, g, lhs, rhs)
            if abs(lhs) > 1e-6:
                checked += 1
    assert checked >= 3, checked


def test_cc_vg_apt_gauge_invariance():
    """Invariant to the redundant vecpot oo/vv response and, separately, to the perturbed-MO gauge
    that governs the nuclear perturbations."""
    for fc in ('false', 'true'):
        d = _ccderiv(WATER_C1, freeze_core=fc)
        nc = np.asarray(d._correlation_velocity_dipole_derivatives('non-canonical'))
        ca = np.asarray(d._correlation_velocity_dipole_derivatives('canonical'))
        assert np.abs(nc - ca).max() < 1e-9, (fc, np.abs(nc - ca).max())

        dn = pycc.CCderiv(d.ccwfn)
        dn._gauge_override = 'canonical'
        assert np.abs(nc - np.asarray(
            dn._correlation_velocity_dipole_derivatives())).max() < 1e-9, fc


def test_cc_vg_apt_spinorbital_equals_spatial():
    """The closed-shell keystone, as for the AAT."""
    sp = np.asarray(_ccderiv(WATER_C1, orbital_basis='spatial')
                    ._correlation_velocity_dipole_derivatives())
    so = np.asarray(_ccderiv(WATER_C1, orbital_basis='spinorbital')
                    ._correlation_velocity_dipole_derivatives())
    assert np.abs(sp).max() > 1e-3, np.abs(sp).max()
    assert np.abs(sp - so).max() < 1e-10, np.abs(sp - so).max()


def test_cc_vg_apt_through_the_facade_and_vcd():
    """``pycc.apt(..., gauge='velocity')`` and ``pycc.vcd(..., velocity_gauge=True)`` both reach it;
    before this existed the latter died with an AttributeError."""
    d = _ccderiv(WATER_C1)
    P = pycc.apt(d, gauge='velocity')
    assert np.abs(np.asarray(P.correlation)).max() > 1e-4
    assert np.allclose(np.asarray(P.total),
                       np.asarray(P.nuclear) + np.asarray(P.reference) + np.asarray(P.correlation))

    res = pycc.vcd(d, velocity_gauge=True)
    for key in ('rotatory_strengths', 'rotatory_strengths_velocity',
                'rotatory_strengths_origin_independent'):
        assert res.get(key) is not None, key
        assert np.isfinite(np.asarray(res[key])).all(), key


def test_cc_vg_apt_ccsdt_raises():
    """CCSD only, as for the AAT."""
    cc_t = pycc.ccwfn(_scf(WATER_C1), model='CCSD(T)', orbital_basis='spatial',
                      make_t3_density=True)
    cc_t.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(NotImplementedError, match='CCSD only'):
        pycc.CCderiv(cc_t)._correlation_velocity_dipole_derivatives()
