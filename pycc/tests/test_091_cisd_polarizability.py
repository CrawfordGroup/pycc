"""
CISD static electric-dipole polarizability (correlation contribution), full 3x3 tensor -
pycc.polarizability(pycc.CIderiv(ci)).correlation, the base 2n+1 machinery driven by CIderiv's
density hooks - validated against a finite field of the ANALYTIC relaxed correlation dipole,
alpha[:, b] = d mu / dF_b: the same oracle as CC's _findiff_alpha and MP2's _dipfd_alpha_diag,
at MP2's stencil order (7-point O(h^6), h = 0.002) and tight convergence at every field point.
The dipole uses only the first-order machinery (unrelaxed densities + equilibrium Z-vector), the
polarizability the 2n+1 second-derivative machinery (perturbed densities + perturbed Z-vector),
so their agreement - here to 1e-10 - confirms the two analytic derivations are mutually
consistent to well below any physically relevant scale.
"""

import psi4
import pycc
import numpy as np


WATER = """
O  0.00000  0.00000  0.00000
H  0.00000  1.43121 -1.10664
H  0.00000 -1.43121 -1.10664
symmetry c1
units bohr
no_com
no_reorient
"""

H = 0.002
_CACHE = {}


def _solve(field=(0.0, 0.0, 0.0), freeze_core='false'):
    key = (tuple(round(f, 10) for f in field), freeze_core)
    if key in _CACHE:
        return _CACHE[key]
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(WATER)
    opt = {'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': freeze_core,
           'e_convergence': 1e-14, 'd_convergence': 1e-14}
    if any(field):
        opt.update({'perturb_h': True, 'perturb_with': 'dipole', 'perturb_dipole': list(field)})
    psi4.set_options(opt)
    _, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD')
    ci.solve_ci(e_conv=1e-14, r_conv=1e-13, maxiter=200)
    _CACHE[key] = ci
    return ci


def _alpha_analytic(freeze_core='false'):
    key = ('analytic', freeze_core)
    if key not in _CACHE:
        _CACHE[key] = np.asarray(
            pycc.polarizability(pycc.CIderiv(_solve(freeze_core=freeze_core))).correlation)
    return _CACHE[key]


def _dipfd_alpha_full(freeze_core='false'):
    """Full 3x3 alpha = -d(mu_rel)/d(lambda) via the 7-point O(h^6) central first-derivative
    stencil of the analytic CISD relaxed correlation dipole under psi4's perturb_dipole (+lambda.mu
    convention - see the module docstring for the sign)."""
    def mu(axis, Fval):
        field = [0.0, 0.0, 0.0]
        field[axis] = Fval
        return np.asarray(pycc.CIderiv(_solve(tuple(field), freeze_core)).dipole().correlation)
    alpha = np.zeros((3, 3))
    for b in range(3):
        alpha[:, b] = -(-mu(b, -3 * H) + 9 * mu(b, -2 * H) - 45 * mu(b, -H)
                        + 45 * mu(b, H) - 9 * mu(b, 2 * H) + mu(b, 3 * H)) / (60 * H)
    return alpha



class _CIderivCanonicalMO(pycc.CIderiv):
    """CIderiv forced onto the CANONICAL perturbed-MO route.

    ``CorrelatedDerivs.perturbed_mo_gauge`` is a property, not a per-call argument: it returns
    ``'canonical'`` only for CCSD(T) (whose (T) kernels are not oo/vv-rotation invariant) and
    ``'non-canonical'`` otherwise, so CISD never takes the canonical route in production.  
    But the CISD energy IS invariant to occ-occ and virt-virt rotations, so both must give the
    same property.  Subclassing here lets the tests below assert that.
    """

    @property
    def perturbed_mo_gauge(self):
        return 'canonical'

def test_cisd_polarizability_vs_dipole_fd_full_631g():
    """Analytic full correlation polarizability tensor vs the 7-point finite field of the
    analytic relaxed correlation dipole, every element, SIGNED, to 1e-10."""
    diff = np.max(np.abs(_alpha_analytic() - _dipfd_alpha_full()))
    assert diff < 1e-10, (diff, _alpha_analytic(), _dipfd_alpha_full())


def test_cisd_polarizability_symmetric_631g():
    """The correlation polarizability is symmetric (alpha_ab = alpha_ba). FD-free physics check."""
    alpha = _alpha_analytic()
    assert np.max(np.abs(alpha - alpha.T)) < 1e-10, alpha


def test_cisd_polarizability_off_diagonal_zero_by_symmetry_631g():
    """H2O/6-31G is C2v (z along C2, molecular plane yz): the correlation polarizability must be
    diagonal. FD-free physics check."""
    alpha = _alpha_analytic()
    assert abs(alpha[0, 1]) < 1e-9
    assert abs(alpha[0, 2]) < 1e-9
    assert abs(alpha[1, 2]) < 1e-9


def test_cisd_polarizability_vs_dipole_fd_full_631g_frozen_core():
    """Frozen-core analytic correlation polarizability vs the same 7-point finite field of the
    analytic relaxed correlation dipole, every element, SIGNED.

    The FD is its own oracle, so the frozen-core case needs no tabulated reference: the dipole
    side uses only the first-order machinery (unrelaxed densities + equilibrium Z-vector,
    frozen-core aware via P_co) while the polarizability side uses the 2n+1 perturbed Z-vector
    with the perturbed core<->active divide.  Their agreement exercises the frozen-core
    field-perturbation path end to end."""
    ana = _alpha_analytic('true')
    fd = _dipfd_alpha_full('true')
    assert np.max(np.abs(ana - fd)) < 1e-10, (np.max(np.abs(ana - fd)), ana, fd)


def test_cisd_polarizability_symmetric_631g_frozen_core():
    """The frozen-core correlation polarizability is symmetric. FD-free physics check."""
    alpha = _alpha_analytic('true')
    assert np.max(np.abs(alpha - alpha.T)) < 1e-10, alpha


def test_cisd_polarizability_perturbed_mo_gauge_invariance_631g():
    """The correlation polarizability is invariant to the perturbed-MO gauge (non-canonical
    default vs the canonical dependent-pair route), all-electron and frozen-core."""
    for fc in ('false', 'true'):
        ci = _solve(freeze_core=fc)
        nc = np.asarray(pycc.CIderiv(ci).polarizability().correlation)
        ca = np.asarray(_CIderivCanonicalMO(ci).polarizability().correlation)
        assert np.max(np.abs(nc - ca)) < 1e-9, (fc, np.max(np.abs(nc - ca)))
