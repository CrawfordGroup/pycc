"""
CCSD atomic axial tensors (AATs, for VCD) -- CCderiv._correlation_aat through the pycc.aat
facade: Eq. (39) of the CC-VCD derivation, the amplitude-amplitude, amplitude-orbital and
orbital-orbital blocks built from the perturbed T and Lambda amplitudes and the half-derivative
overlaps.  CCSD only, both orbital bases.

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

# Two electrons AND genuinely C1, on the H2O2 skeleton the production cases use: a central H-H
# with one terminal H on each, twisted.  A triatomic cannot be C1, since three nuclei are always
# coplanar and that plane is a mirror plane.  Internal coordinates (angstrom/degrees):
#     r(H1-H2) = 1.100   r(H1-H3) = 0.950   r(H2-H4) = 1.200
#     a(H3-H1-H2) = a(H4-H2-H1) = 100.0     d(H3-H1-H2-H4) = 113.7
# The UNEQUAL terminal distances are what make it C1: with them equal the structure is C2, which
# psi4 confirms, so the distortion is load-bearing rather than decorative.  HOMO-LUMO gap 0.57 Eh.
H4_2PLUS = """2 1
H   0.511219079100 -0.217582036777 -1.059858587625
H   0.511219079100 -0.217582036777  1.018840150378
H   0.511219079100  1.550384055673 -1.371598710702
H  -1.533657237299 -1.115219982120  1.412617147949
symmetry c1
units bohr
no_com
no_reorient
"""

# Cs, not C1 (triatomic).  no_com / no_reorient keep psi4 from rotating the mirror plane onto a
# coordinate plane, which would zero 15 of the 27 components; it also pins the AAT's gauge origin.
WATER_CS = """
O  0.000000000000  0.000000000000 -0.129476880010
H  0.080000000000 -1.494186750504  1.027446024111
H -0.050000000000  1.394186750504  1.127446024111
symmetry c1
units bohr
no_com
no_reorient
"""

# Genuinely C1: unequal O-H lengths and an asymmetric dihedral remove the C2 of ordinary H2O2.
HOOH_C1 = """
O  0.000000000000  1.317000000000  0.000000000000
O  0.000000000000 -1.317000000000  0.000000000000
H  1.585000000000  1.663000000000  0.982000000000
H -1.471000000000 -1.738000000000  1.058000000000
symmetry c1
units bohr
no_com
no_reorient
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


@pytest.mark.parametrize('basis', ['6-31G', 'cc-pVDZ'])
def test_cc_aat_two_electron_equals_cisd(basis):
    """For two electrons CCSD is exact, so its correlation AAT must equal the validated CISD one.

    Fixes the overall sign (the magnetic operator enters as Im(m), so the sign is a convention the
    invariance tests cannot see) and the Eq. (39) prefactor.  The CISD side carries its
    intermediate-normalization correction and the CC side needs none, so this also checks the
    normalization result.

    Four nuclei on the H2O2 skeleton with unequal terminal distances, so genuinely C1 (equal ones
    give C2) and every component is exercised; a triatomic is Cs at best.

    Two bases: 6-31G, and cc-pVDZ for polarization functions and a real virtual space.  The
    identity also holds at STO-3G for this geometry, unlike the H3+ case it replaced, whose
    minimal-basis DIIS matrix was singular enough that Linux LAPACK raised where macOS did not."""
    wfn = _scf(H4_2PLUS, basis=basis)

    cc = pycc.ccwfn(wfn, model='CCSD', orbital_basis='spatial')
    e_cc = cc.solve_cc(1e-13, 1e-13, 200)
    A_cc = np.asarray(pycc.CCderiv(cc)._correlation_aat())

    ci = pycc.CIwfn(wfn, model='CISD', orbital_basis='spatial')
    e_ci = ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=250)
    A_ci = np.asarray(pycc.CIderiv(ci)._correlation_aat())

    assert abs(float(e_cc) - float(e_ci)) < 1e-12, (e_cc, e_ci)   # CCSD == CISD for 2 electrons
    assert np.abs(A_ci).max() > 1e-4, np.abs(A_ci).max()          # the reference is non-trivial
    assert np.abs(A_cc - A_ci).max() < 1e-10, np.abs(A_cc - A_ci).max()


# The spatial legs run on the C1 H2O2 in 6-31G: 36 independent components and a real valence
# space, and these are the numbers quoted for the validation.  The spin-orbital leg stays on
# water/STO-3G because it costs 2 s there against 67 s on H2O2, and its job is only to show
# that code path satisfies the identity -- `..._spinorbital_equals_spatial` pins the two
# against each other directly.
@pytest.mark.parametrize('orbital_basis,freeze_core,geom,basis',
                         [('spatial', 'false', HOOH_C1, '6-31G'),
                          ('spatial', 'true', HOOH_C1, '6-31G'),
                          ('spinorbital', 'false', WATER_CS, 'STO-3G')],
                         ids=['H2O2-spatial-AE', 'H2O2-spatial-FC', 'H2O-SO-AE'])
def test_cc_aat_translational_sum_rule(orbital_basis, freeze_core, geom, basis):
    """Summing the AAT over atoms gives the field derivative of the momentum expectation value::

        sum_A 2 I^A_{alpha,beta} = sum_pq D^H_pq <p|grad_alpha|q>
                                 + sum_pq D_pq [grad_alpha, U^H]_pq

    The two sides share no machinery: the left needs all 3N displacement responses through the four
    sectors, the right only D, D^H, U^H and the gradient integrals, with no nuclear response at
    all.  It therefore tests every block and the nonlinear terms of the perturbed amplitude
    equations at once.

    Run frozen-core as well: it is the only **value** anchor that survives a frozen core, since the
    two-electron identity has no core to freeze, and it is not obvious a priori that the core /
    active partition preserves a rule whose derivation rests on every basis function moving with
    its atom."""
    d = _ccderiv(geom, basis=basis, orbital_basis=orbital_basis, freeze_core=freeze_core)
    c = d.contract
    grad = [np.real(-1.0j * np.asarray(d.ccwfn.H.p[a])) for a in range(3)]   # i stripped, as H.m
    magD, magU = {}, {}
    for b in range(3):
        p = Perturbation('magnetic', b)
        magD[b] = d._aat_perturbed_opdm(p)
        magU[b] = d._cc_perturbed_ints(p)[1]
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
    d = _ccderiv(HOOH_C1, basis='6-31G')
    natom = d.wfn.derivatives.natom
    for a in range(3):
        sums, M = None, None
        grad = np.real(-1.0j * np.asarray(d.ccwfn.H.p[a]))
        for A in range(natom):
            p = Perturbation('nuclear', (A, a))
            rec = [np.asarray(x) for x in d._cc_amplitude_record(p)]
            rec.append(np.asarray(d._aat_perturbed_opdm(p)))
            rec.append(np.asarray(d._cc_perturbed_ints(p)[1]))
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


def test_cc_aat_size_intensivity():
    """Fragment A's AAT is unchanged by a distant, non-interacting fragment B.

    CC is size extensive, so with ``T = T_A + T_B`` and ``Lambda = Lambda_A + Lambda_B`` every term
    of Eq. (39) contracts something localized on A with something additive, and B drops out.  CISD
    is not, and fails this on the identical geometry by 4.4e-2 relative, which is what gives the
    test teeth.  Both fragments carry ``no_com`` / ``no_reorient``: A must sit at identical
    absolute coordinates in the two runs, since the AAT depends on the gauge origin.

    **Why 6-31G and why helium**, which looks like an under-powered choice and is deliberate.  A
    polarizable B leaves a real R^-2 tail: B's magnetically perturbed density acquires a dipole
    growing like R (from the (1/2) R x p term of the operator about a gauge origin on A), whose
    field at A falls as R/R^3.  That is physics of the common gauge origin, not a size-intensivity
    failure, and it would swamp this assertion -- measured 3.5e-6 at 40 bohr in cc-pVDZ and 1.3e-5
    in aug-cc-pVDZ, decaying as R^-2.00 out to 1500 bohr and scaling with alpha(B).  Helium in
    6-31G has **exactly zero** dipole polarizability (s functions only), so that tail is absent and
    the test isolates size intensivity alone.  **Upgrading the basis here will make this fail for
    an innocent reason**; test the tail separately if it is wanted."""
    A_mono = np.asarray(_ccderiv(HOOH_C1, basis='6-31G')._correlation_aat())
    scale = np.abs(A_mono).max()
    assert scale > 1e-3, scale

    supermolecule = HOOH_C1.replace('units bohr',
                                    'He 0.000000000000 0.000000000000 40.000000000000\nunits bohr')
    A_super = np.asarray(_ccderiv(supermolecule, basis='6-31G')._correlation_aat())
    dev = np.abs(A_super[:4] - A_mono).max()
    assert dev < 1e-8 * scale, (dev, dev / scale)


@pytest.mark.parametrize('orbital_basis', ['spatial', 'spinorbital'])
def test_cc_aat_magnetic_gauge_invariance(orbital_basis):
    """Invariant to the gauge of the redundant magnetic oo/vv response, all-electron and
    frozen-core.  With a frozen core the two choices genuinely differ."""
    for fc in ('false', 'true'):
        d = _ccderiv(WATER_CS, freeze_core=fc, orbital_basis=orbital_basis)
        nc = np.asarray(d.aat(orbital_gauge='non-canonical').correlation)
        ca = np.asarray(d.aat(orbital_gauge='canonical').correlation)
        assert np.abs(nc - ca).max() < 1e-9, (fc, np.abs(nc - ca).max())


def test_cc_aat_perturbed_mo_gauge_invariance():
    """Invariant to the **perturbed-MO** gauge, which governs the *real* (nuclear) perturbations --
    a different knob from the magnetic orbital gauge above, which governs the imaginary ones.
    CCSD is oo/vv-invariant, so canonical and non-canonical perturbed orbitals must agree."""
    for fc in ('false', 'true'):
        d = _ccderiv(WATER_CS, freeze_core=fc)
        nc = np.asarray(d.aat().correlation)
        dc = pycc.CCderiv(d.ccwfn)
        dc._gauge_override = 'canonical'
        ca = np.asarray(dc.aat().correlation)
        assert np.abs(nc - ca).max() < 1e-9, (fc, np.abs(nc - ca).max())


def test_cc_aat_amplitude_block_vs_response_code():
    """The amplitude-amplitude block, with orbitals frozen and the electric dipole in place of the
    nuclear displacement, is the zero-frequency slope of the CC linear response function for the
    electric and magnetic dipoles -- which ``ccresponse`` computes independently for optical
    rotation.  This checks the perturbed Lambda equations, quadratic terms included, against
    trusted code.

    ``G'(omega)`` is already the odd-in-omega part of ``<<mu; m>>``, so ``G'(omega)/omega`` is the
    central-difference slope ``[LR(+omega) - LR(-omega)] / 2 omega``.  ``G'(0) = 0`` analytically,
    so this is a genuine difference quotient, and its error carries only **even** powers of omega.
    Each Richardson level removes the leading one, and the residuals follow that exactly::

        one frequency        5.1e-04 at omega = 0.005, scaling as omega^2
        Richardson level 1   9.7e-07, scaling as omega^4
        Richardson level 2   8.1e-11, scaling as omega^6

    Four frequencies and two levels are used here.  **The tolerance is still 1e-8, looser than the
    1e-13 of the identity tests in this file, and that is inherent**: only the frequency derivative
    is numerical -- ``G'(omega)`` itself is analytic at each frequency -- but a difference quotient
    is a difference quotient.  No convergence floor has been reached, so tightening further is a
    matter of adding frequencies, not of fixing anything.

    The sign is opposite by convention, not defect: ``linear_response(a, b)`` evaluates
    ``Tr(d_b D . a)``, so the perturbation is ``m`` and the observable is ``mu``, the reverse of
    this block's ordering, and reversing an antisymmetric quantity flips it.  Test 1 pins the AAT's
    absolute sign independently."""
    d = _ccderiv(HOOH_C1, basis='6-31G')
    zero_eri = np.zeros_like(np.asarray(d.ccwfn.H.ERI))
    zero_L = np.zeros_like(np.asarray(d.ccwfn.H.L))

    def bare(op):
        """Perturbed amplitudes and multipliers for a bare operator: orbitals frozen."""
        dt1, dt2 = d._perturbed_amplitudes(op, zero_eri, zero_L, d.hbar, omega=0.0)
        dl1, dl2 = d._perturbed_lambda(op, zero_eri, zero_L, dt1, dt2, d.hbar, d.cclambda,
                                       omega=0.0)
        return dt1, dt2, dl1, dl2

    mu, mm = d._perturbation_ints('mu'), d._perturbation_ints('m')
    R_mu, R_m = [bare(mu[i]) for i in range(3)], [bare(mm[j]) for j in range(3)]
    block = np.zeros((3, 3))
    for i, (t1a, t2a, l1a, l2a) in enumerate(R_mu):
        for j, (t1b, t2b, l1b, l2b) in enumerate(R_m):
            block[i, j] = (d._biorthogonal_overlap(l1a, l2a, t1b, t2b)
                           - d._biorthogonal_overlap(l1b, l2b, t1a, t2a))
    nz = np.abs(block) > 1e-8 * np.abs(block).max()
    assert nz.sum() == 9, nz.sum()            # C1: every element should be live

    ratios = [np.asarray(d._correlation_optical_rotation(w))[nz] / (w * block[nz])
              for w in (0.020, 0.010, 0.005, 0.0025)]
    lvl1 = [(4.0 * ratios[k + 1] - ratios[k]) / 3.0 for k in range(3)]      # removes omega^2
    lvl2 = [(16.0 * lvl1[k + 1] - lvl1[k]) / 15.0 for k in range(2)]       # removes omega^4
    assert np.abs(lvl2[-1] + 1.0).max() < 1e-8, np.abs(lvl2[-1] + 1.0).max()


def test_cc_aat_spinorbital_equals_spatial():
    """The house keystone: for a closed-shell reference the spin-orbital and spin-adapted paths must
    agree.  Everything the AAT needs is basis-agnostic except the half-derivative overlaps, so this
    is what guards that dispatch."""
    for fc in ('false', 'true'):
        sp = np.asarray(_ccderiv(WATER_CS, orbital_basis='spatial', freeze_core=fc)
                        ._correlation_aat())
        so = np.asarray(_ccderiv(WATER_CS, orbital_basis='spinorbital', freeze_core=fc)
                        ._correlation_aat())
        assert np.abs(sp).max() > 1e-3, (fc, np.abs(sp).max())
        assert np.abs(sp - so).max() < 1e-10, (fc, np.abs(sp - so).max())


def test_cc_aat_shares_perturbed_amplitudes_with_the_hessian():
    """The perturbed amplitudes are memoized in the DerivStore, so the AAT reads back what a
    Hessian on the same driver already solved instead of repeating it."""
    d = _ccderiv(WATER_CS)
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
    cc_t = pycc.ccwfn(_scf(WATER_CS), model='CCSD(T)', orbital_basis='spatial',
                      make_t3_density=True)
    cc_t.solve_cc(1e-12, 1e-12, 100)
    with pytest.raises(NotImplementedError, match='CCSD only'):
        pycc.CCderiv(cc_t)._correlation_aat()
