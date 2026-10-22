"""
CCSD atomic axial tensors (AATs, for VCD) -- CCderiv._correlation_aat through the pycc.aat
facade: Eq. (39) of the CC-VCD derivation, the amplitude-amplitude, amplitude-orbital and
orbital-orbital blocks built from the perturbed T and Lambda amplitudes and the half-derivative
overlaps.  Spatial orbitals, CCSD only.

There is no external CC AAT reference, so the anchors here are exact identities rather than
tabulated numbers (docs/cc_aat_validation_plan.md):

* **two electrons**: CCSD is exact, so the CCSD AAT must equal the validated CISD AAT element by
  element.  This is the anchor that fixes the overall sign and prefactor;
* **translational sum rule**: summing over atoms must reproduce the field derivative of the
  momentum expectation value, which shares no machinery with the AAT assembly;
* **gauge invariance**, independently in the magnetic orbital gauge and the perturbed-MO gauge.

The orbital-orbital block is a commutator of the two half-derivative overlaps, not the form the
nuclear side's symmetry suggests: the nuclear overlap is antisymmetric while the magnetic one is
symmetric once the ``i`` is stripped.  The sum rule and both gauge tests are what catch that, so
they guard a real error rather than a hypothetical one.

Both orbital bases, checked against each other by the usual closed-shell keystone.
"""

import numpy as np
import psi4
import pytest

import pycc
from pycc.cphf import Perturbation


GAUGES = ('non-canonical', 'canonical')

# Two electrons, distorted C1 so the tensor is non-trivial.  A diatomic is useless here: its
# correlation AAT vanishes by symmetry (HeH+ gives 1e-19).
H3P = """1 1
H  0.000000000000  0.000000000000  0.000000000000
H  1.700000000000  0.000000000000  0.000000000000
H  0.620000000000  1.430000000000  0.170000000000
symmetry c1
units bohr
"""

# Distorted C1 water, for the identities that need more than two electrons.
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


def _ccderiv(geom, basis='STO-3G', freeze_core='false', model='CCSD',
             orbital_basis='spatial'):
    """Converged CC wavefunction wrapped in its derivative driver."""
    cc = pycc.ccwfn(_scf(geom, basis, freeze_core), model=model, orbital_basis=orbital_basis)
    cc.solve_cc(1e-13, 1e-13, 200)
    return pycc.CCderiv(cc)


def test_cc_aat_two_electron_equals_cisd():
    """For two electrons CCSD is exact, so its correlation AAT must equal the validated CISD one.

    Fixes the overall sign (the magnetic operator enters as Im(m), so the sign is a convention the
    invariance tests cannot see) and the Eq. (39) prefactor.  The CISD side carries its
    intermediate-normalization correction and the CC side needs none, so this also checks the
    normalization result."""
    wfn = _scf(H3P)

    cc = pycc.ccwfn(wfn, model='CCSD', orbital_basis='spatial')
    e_cc = cc.solve_cc(1e-13, 1e-13, 200)
    A_cc = np.asarray(pycc.CCderiv(cc)._correlation_aat())

    ci = pycc.CIwfn(wfn, model='CISD', orbital_basis='spatial')
    e_ci = ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=250)
    A_ci = np.asarray(pycc.CIderiv(ci)._correlation_aat())

    assert abs(float(e_cc) - float(e_ci)) < 1e-12, (e_cc, e_ci)   # CCSD == CISD for 2 electrons
    assert np.abs(A_ci).max() > 1e-4, np.abs(A_ci).max()          # the reference is non-trivial
    assert np.abs(A_cc - A_ci).max() < 1e-10, np.abs(A_cc - A_ci).max()


@pytest.mark.parametrize('orbital_basis', ['spatial', 'spinorbital'])
def test_cc_aat_translational_sum_rule(orbital_basis):
    """Summing the AAT over atoms gives the field derivative of the momentum expectation value::

        sum_A 2 I^A_{alpha,beta} = sum_pq D^H_pq <p|grad_alpha|q>
                                 + sum_pq D_pq [grad_alpha, U^H]_pq

    The two sides share no machinery: the left needs all 3N displacement responses through the four
    sectors, the right only D, D^H, U^H and the gradient integrals, with no nuclear response at
    all.  It therefore tests every block and the nonlinear terms of the perturbed amplitude
    equations at once."""
    d = _ccderiv(WATER_C1, orbital_basis=orbital_basis)
    c = d.contract
    grad = [np.real(-1.0j * np.asarray(d.ccwfn.H.p[a])) for a in range(3)]   # i stripped, as H.m
    magD, magU = {}, {}
    for b in range(3):
        p = Perturbation('magnetic', b)
        magD[b] = d._aat_perturbed_opdm(p)
        magU[b] = d._cc_perturbed_ints(p)[2]
    D0 = np.asarray(d._unrelaxed_densities()[0])
    aat = np.asarray(d._correlation_aat())

    checked = 0
    for a in range(3):
        for b in range(3):
            lhs = 2.0 * aat[:, a, b].sum()
            rhs = (c('pq,pq->', magD[b], grad[a])
                   + c('pq,pq->', D0, grad[a] @ magU[b] - magU[b] @ grad[a]))
            assert abs(lhs - rhs) < 1e-10, (a, b, lhs, rhs)
            if abs(lhs) > 1e-6:
                checked += 1
    assert checked >= 3, checked      # the rule must be exercised on non-vanishing elements


def test_cc_aat_translational_zero_sums():
    """Each perturbed quantity separately sums to zero over a rigid translation, and the
    half-derivative overlaps sum to minus the gradient integrals.  These localize a failure of the
    sum rule to one ingredient."""
    d = _ccderiv(WATER_C1)
    natom = d.wfn.derivatives.natom
    for a in range(3):
        sums, M = None, None
        grad = np.real(-1.0j * np.asarray(d.ccwfn.H.p[a]))
        for A in range(natom):
            p = Perturbation('nuclear', (A, a))
            rec = [np.asarray(x) for x in d._cc_amplitude_record(p)]
            rec.append(np.asarray(d._aat_perturbed_opdm(p)))
            rec.append(np.asarray(d._cc_perturbed_ints(p)[2]))
            half = np.asarray(d.wfn.derivatives.overlap_half(A)[a]).T
            if sums is None:
                sums = [x.copy() for x in rec]
                M = rec[-1] + half
            else:
                for s, x in zip(sums, rec):
                    s += x
                M = M + rec[-1] + half
        for name, s in zip(('dt1', 'dt2', 'dl1', 'dl2', 'dD', 'U'), sums):
            assert np.abs(s).max() < 1e-10, (a, name, np.abs(s).max())
        assert np.abs(M + grad).max() < 1e-10, (a, np.abs(M + grad).max())


@pytest.mark.parametrize('orbital_basis', ['spatial', 'spinorbital'])
def test_cc_aat_magnetic_gauge_invariance(orbital_basis):
    """Invariant to the gauge of the redundant magnetic oo/vv response, all-electron and
    frozen-core.  With a frozen core the two choices genuinely differ."""
    for fc in ('false', 'true'):
        d = _ccderiv(WATER_C1, freeze_core=fc, orbital_basis=orbital_basis)
        nc = np.asarray(d.aat(orbital_gauge='non-canonical').correlation)
        ca = np.asarray(d.aat(orbital_gauge='canonical').correlation)
        assert np.abs(nc - ca).max() < 1e-9, (fc, np.abs(nc - ca).max())


def test_cc_aat_perturbed_mo_gauge_invariance():
    """Invariant to the **perturbed-MO** gauge, which governs the *real* (nuclear) perturbations --
    a different knob from the magnetic orbital gauge above, which governs the imaginary ones.
    CCSD is oo/vv-invariant, so canonical and non-canonical perturbed orbitals must agree."""
    for fc in ('false', 'true'):
        d = _ccderiv(WATER_C1, freeze_core=fc)
        nc = np.asarray(d.aat().correlation)
        dc = pycc.CCderiv(d.ccwfn)
        dc._gauge_override = 'canonical'
        ca = np.asarray(dc.aat().correlation)
        assert np.abs(nc - ca).max() < 1e-9, (fc, np.abs(nc - ca).max())


def test_cc_aat_spinorbital_equals_spatial():
    """The house keystone: for a closed-shell reference the spin-orbital and spin-adapted paths must
    agree.  Everything the AAT needs is basis-agnostic except the half-derivative overlaps, so this
    is what guards that dispatch."""
    sp = np.asarray(_ccderiv(WATER_C1, orbital_basis='spatial')._correlation_aat())
    so = np.asarray(_ccderiv(WATER_C1, orbital_basis='spinorbital')._correlation_aat())
    assert np.abs(sp).max() > 1e-3, np.abs(sp).max()
    assert np.abs(sp - so).max() < 1e-10, np.abs(sp - so).max()


def test_cc_aat_shares_perturbed_amplitudes_with_the_hessian():
    """The perturbed amplitudes are memoized in the DerivStore, so the AAT reads back what a
    Hessian on the same driver already solved instead of repeating it."""
    d = _ccderiv(WATER_C1)
    store = d.wfn.derivatives.store
    # the group stores one dataset per member, so the ctx carries the member name
    member = lambda p, n: store.has('ccamp', p, d._cc_amplitude_ctx(p, 'non-canonical') + (n,))
    p00 = Perturbation('nuclear', (0, 0))
    assert not member(p00, 'dt1')
    pycc.hessian(d)
    for A in range(d.wfn.derivatives.natom):
        for cart in range(3):
            p = Perturbation('nuclear', (A, cart))
            for n in ('dt1', 'dt2', 'dl1', 'dl2'):
                assert member(p, n), (A, cart, n)


def test_cc_aat_ccsdt_raises():
    """CCSD only: Eq. (39) is derived for the (1 + Lambda) e^{-T} bra without (T), so CCSD(T) raises
    rather than returning a wrong number.  Both orbital bases are supported."""
    cc_t = pycc.ccwfn(_scf(WATER_C1), model='CCSD(T)', orbital_basis='spatial',
                      make_t3_density=True)
    cc_t.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(NotImplementedError, match='CCSD only'):
        pycc.CCderiv(cc_t)._correlation_aat()
