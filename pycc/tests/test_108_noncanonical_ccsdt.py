"""Non-canonical perturbed MOs for CCSD(T) first and second derivatives.

CCSD(T) defaults to the *canonical* perturbed-MO gauge, where the oo/vv orbital response is
carried by the dependent-pair rotations `P_oo`/`P_vv` and only the **diagonal** of the (T)
`Doo`/`Dvv` is needed.  That divide amplifies as `1/(eps_m - eps_n)`, so an accidental
near-degeneracy corrupts open-shell second derivatives
(`docs/openshell_ccsdt_second_derivative_defect.md`).

The non-canonical gauge has no such divide, but needs three ingredients the canonical route does
not (`docs/noncanonical_ccsdt_design_plan.md`):

  A  off-diagonal (T) `Doo`/`Dvv`              `t3_density_offdiag`, `so_t3_density_offdiag`
  B  off-diagonal *perturbed* (T) `Doo`/`Dvv`  `dt3_density_offdiag`, `so_dt3_density_offdiag`
  C  off-diagonal `d_x f` coupling in the perturbed T3   `t3_full_coupled`, `so_t3_full_coupled`

A and B are the Table ccsdt_1pdm / ccsdt_deriv_1pdm expressions of
`docs/cc_gradients_orbital_response.tex` with the restriction to the diagonal lifted; the free
index sits on the `Lambda_3` multiplier for vv and on the connected amplitude for oo.  The density
is **not** symmetric, so that choice is not cosmetic -- the opposite one gives the transpose, and
only gauge invariance distinguishes them.  C replaces the `diag(df)` denominator response with the
full bracket; it is direct, not iterative, because the *unperturbed* Fock stays diagonal.

The gauge is selected per driver with `_gauge_override`, since `perturbed_mo_gauge` still returns
`'canonical'` for CCSD(T).  Nothing below runs in production as a result; these tests are the only
coverage this code has.

Oracles (xcfour 2.1, REFERENCE=UHF, MULTIPLICITY=3, SCF_CONV=13, CC_CONV=12, LINEQ_CONV=12):
gradient GRD, Hessian FCMFINAL, APT DIPDER (all VIB=EXACT), polarizability POLAR-POLARSCF
(PROP=SECOND_ORDER); correlation is always the CCSD(T) file minus the SCF file.  All print 10
decimals, so ~1e-10 is their floor.  CH2's 6-31G is transcribed from CFOUR's GENBAS below: psi4's
built-in 6-31G differs by digit rounding and reproduces CFOUR's SCF energy only to 1.4e-8, against
8.1e-13 for the transcription.  cc-pVDZ needs no transcription (psi4's cc-pVDZ is CFOUR's PVDZ).
"""
import os
import contextlib

import numpy as np
import psi4
import pytest

import pycc
from pycc import cctriples
from .conftest import record
from .test_083_ccsdt_gradient import CH2_CFOUR_FRAME
from .test_089_ccsdt_hessian import (_cfour_wfn, _pvdz_wfn, WATER,
                                     CFOUR_HESS_T_WATER, CFOUR_HESS_T_WATER_PVDZ)
from .test_087_ccsdt_apt import CFOUR_APT_T_WATER_631G, CFOUR_APT_T_WATER_PVDZ

# CFOUR's GENBAS 6-31G for C and H, transcribed so the AO basis matches CFOUR to all printed
# digits (verified: SCF agrees to 8.1e-13, against 1.4e-8 for psi4's built-in 6-31G).
CFOUR_631G_CH = """cartesian
****
H     0
S   3   1.00
     18.7311370             0.0334946
      2.8253944             0.2347270
      0.6401217             0.8137573
S   1   1.00
      0.1612778             1.0000000
****
C     0
S   6   1.00
   3047.5248800             0.0018347
    457.3695180             0.0140373
    103.9486850             0.0688426
     29.2101553             0.2321844
      9.2866630             0.4679413
      3.1639270             0.3623120
S   3   1.00
      7.8682724            -0.1193324
      1.8812885            -0.1608542
      0.5442493             1.1434564
S   1   1.00
      0.1687145             1.0000000
P   3   1.00
      7.8682724             0.0689991
      1.8812885             0.3164240
      0.5442493             0.7443083
P   1   1.00
      0.1687145             1.0000000
****
"""

CFOUR_GRAD_T_CH2_631G = np.array(
    [[0.0000000000,  0.0000000000, -0.0096892326],
     [0.0000000000,  0.0127224625,  0.0048446163],
     [0.0000000000, -0.0127224625,  0.0048446163]])

CFOUR_ALPHA_T_CH2_631G = np.array(
    [[0.0065469441,  0.0000000000,  0.0000000000],
     [0.0000000000, -0.2367417595,  0.0000000000],
     [0.0000000000,  0.0000000000, -0.0236263342]])

CFOUR_APT_T_CH2_631G = np.array(
    [[[0.0056150390,  0.0000000000,  0.0000000000],
      [0.0000000000,  0.0061646646,  0.0000000000],
      [0.0000000000,  0.0000000000, -0.0077654424]],
     [[-0.0028075195, 0.0000000000,  0.0000000000],
      [0.0000000000, -0.0030823322, -0.0048364931],
      [0.0000000000, -0.0001168493,  0.0038827213]],
     [[-0.0028075195, 0.0000000000,  0.0000000000],
      [0.0000000000, -0.0030823322,  0.0048364931],
      [0.0000000000,  0.0001168493,  0.0038827213]]])

CFOUR_HESS_T_CH2_631G = np.array(
    [[-0.0121599100, 0.0000000000, 0.0000000000, 0.0060799550, 0.0000000000, 0.0000000000,
      0.0060799550, 0.0000000000, 0.0000000000],
     [0.0000000000, -0.0023591296, 0.0000000000, 0.0000000000, 0.0011795649, -0.0020836260,
      0.0000000000, 0.0011795649, 0.0020836260],
     [0.0000000000, 0.0000000000, -0.0167983371, 0.0000000000, -0.0014961158, 0.0083991685,
      0.0000000000, 0.0014961158, 0.0083991685],
     [0.0060799550, 0.0000000000, 0.0000000000, -0.0064344419, 0.0000000000, 0.0000000000,
      0.0003544870, 0.0000000000, 0.0000000000],
     [0.0000000000, 0.0011795649, -0.0014961158, 0.0000000000, -0.0018080031, 0.0017898709,
      0.0000000000, 0.0006284383, -0.0002937551],
     [0.0000000000, -0.0020836260, 0.0083991685, 0.0000000000, 0.0017898709, -0.0071510740,
      0.0000000000, 0.0002937551, -0.0012480945],
     [0.0060799550, 0.0000000000, 0.0000000000, 0.0003544870, 0.0000000000, 0.0000000000,
      -0.0064344419, 0.0000000000, 0.0000000000],
     [0.0000000000, 0.0011795649, 0.0014961158, 0.0000000000, 0.0006284383, 0.0002937551,
      0.0000000000, -0.0018080031, -0.0017898709],
     [0.0000000000, 0.0020836260, 0.0083991685, 0.0000000000, -0.0002937551, -0.0012480945,
      0.0000000000, -0.0017898709, -0.0071510740]])

CFOUR_APT_T_CH2_PVDZ = np.array(
    [[[-0.0144591212, 0.0000000000,  0.0000000000],
      [0.0000000000, -0.0105997925,  0.0000000000],
      [0.0000000000,  0.0000000000, -0.0121819541]],
     [[0.0072295606,  0.0000000000,  0.0000000000],
      [0.0000000000,  0.0052998962,  0.0016572553],
      [0.0000000000, -0.0008204854,  0.0060909770]],
     [[0.0072295606,  0.0000000000,  0.0000000000],
      [0.0000000000,  0.0052998962, -0.0016572553],
      [0.0000000000,  0.0008204854,  0.0060909770]]])

CFOUR_HESS_T_CH2_PVDZ = np.array(
    [[-0.0074676191, 0.0000000000, 0.0000000000, 0.0037338095, 0.0000000000, 0.0000000000,
      0.0037338095, 0.0000000000, 0.0000000000],
     [0.0000000000, 0.0021067536, 0.0000000000, 0.0000000000, -0.0010533768, -0.0020354921,
      0.0000000000, -0.0010533768, 0.0020354921],
     [0.0000000000, 0.0000000000, -0.0157400719, 0.0000000000, -0.0016097011, 0.0078700360,
      0.0000000000, 0.0016097011, 0.0078700360],
     [0.0037338095, 0.0000000000, 0.0000000000, -0.0044943878, 0.0000000000, 0.0000000000,
      0.0007605782, 0.0000000000, 0.0000000000],
     [0.0000000000, -0.0010533768, -0.0016097011, 0.0000000000, 0.0001196889, 0.0018225966,
      0.0000000000, 0.0009336879, -0.0002128955],
     [0.0000000000, -0.0020354921, 0.0078700360, 0.0000000000, 0.0018225966, -0.0061297595,
      0.0000000000, 0.0002128955, -0.0017402765],
     [0.0037338095, 0.0000000000, 0.0000000000, 0.0007605782, 0.0000000000, 0.0000000000,
      -0.0044943878, 0.0000000000, 0.0000000000],
     [0.0000000000, -0.0010533768, 0.0016097011, 0.0000000000, 0.0009336879, 0.0002128955,
      0.0000000000, 0.0001196889, -0.0018225966],
     [0.0000000000, 0.0020354921, 0.0078700360, 0.0000000000, -0.0002128955, -0.0017402765,
      0.0000000000, -0.0018225966, -0.0061297595]])

# CFOUR POLAR - POLARSCF, yy component only (docs/openshell_ccsdt_second_derivative_defect.md).
CFOUR_ALPHA_YY_T_CH2_PVDZ = 0.04267621340


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------
def _ch2(basis, gauge):
    """CCSD(T)/UHF spin-orbital driver for triplet CH2 in the requested gauge.

    A fresh wavefunction per call is required, not an optimisation choice:
    :meth:`CCderiv._ensure_t3_offdiag` replaces ``cc.Doo``/``cc.Dvv`` in place, so a wavefunction
    shared between gauges would be contaminated by whichever ran first.
    """
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_memory('16 GB')
        psi4.core.set_output_file('/dev/null', False)
        psi4.geometry(CH2_CFOUR_FRAME)
        if basis == '6-31G':
            psi4.basis_helper(CFOUR_631G_CH, name='cfour631gch')
            name = 'cfour631gch'
        else:
            name = basis
        psi4.set_options({'basis': name, 'scf_type': 'pk', 'reference': 'uhf',
                          'e_convergence': 1e-13, 'd_convergence': 1e-13, 'maxiter': 300})
        _, wfn = psi4.energy('scf', return_wfn=True)
        cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital', make_t3_density=True)
        cc.solve_cc(1e-13, 1e-13, 300)
        d = pycc.CCderiv(cc)
        if gauge is not None:
            d._gauge_override = gauge
    return d


def _water(basis, gauge, freeze_core='false'):
    """CCSD(T)/RHF spatial driver for water; see :func:`_ch2` on the fresh-wavefunction rule."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        wfn = (_cfour_wfn if basis == '6-31G' else _pvdz_wfn)(WATER, freeze_core)
        cc = pycc.ccwfn(wfn, model='CCSD(T)', make_t3_density=True)
        cc.solve_cc(1e-13, 1e-13, 200)
        d = pycc.CCderiv(cc)
        if gauge is not None:
            d._gauge_override = gauge
    return d


def _perturbation(shape_t1, shape_t2, shape_eri, antisym):
    """A fixed pseudo-random perturbation (dt1, dt2, df, deri).

    Deliberately not a physical perturbation: it exercises the product-rule machinery over a
    general direction, which is a stronger test of the construction than any single field or
    displacement, and it needs no wavefunction response.
    """
    rng = np.random.default_rng(0)
    dt1 = rng.normal(size=shape_t1) * 1e-2
    dt2 = rng.normal(size=shape_t2) * 1e-2
    df = rng.normal(size=shape_eri[:2]) * 1e-2
    df = df + df.T
    deri = rng.normal(size=shape_eri) * 1e-3
    if antisym:
        deri = deri - deri.swapaxes(0, 1) - deri.swapaxes(2, 3) + deri.swapaxes(0, 1).swapaxes(2, 3)
    else:
        deri = 0.5 * (deri + deri.transpose(1, 0, 3, 2))
    return dt1, dt2, df, deri


# --------------------------------------------------------------------------------------------
# A and B: the off-diagonal blocks must reproduce the existing diagonals exactly
# --------------------------------------------------------------------------------------------
def test_so_t3_density_offdiag_reproduces_diagonal():
    """Piece A, spin-orbital: widening the contraction must not disturb the diagonal.

    The cheapest regression on A -- it catches a wrong prefactor or a transposed index pairing
    before any property is computed.  It cannot catch the *transpose* of the off-diagonal blocks,
    which reduce to the same diagonal; gauge invariance of the gradient does that.
    """
    cc = _ch2('STO-3G', None).ccwfn
    _, dens = cctriples.so_t3_density(cc.o, cc.v, cc.no, cc.nv, cc.t1, cc.t2,
                                      cc.H.F, cc.H.ERI, cc.contract)
    Doo, Dvv = cctriples.so_t3_density_offdiag(cc.o, cc.v, cc.no, cc.nv, cc.t1, cc.t2,
                                               cc.H.F, cc.H.ERI, cc.contract)
    assert np.max(np.abs(np.diag(Doo) - np.diag(np.asarray(dens['Doo'])))) < 1e-15
    assert np.max(np.abs(np.diag(Dvv) - np.diag(np.asarray(dens['Dvv'])))) < 1e-15
    # and the off-diagonal elements are not trivially zero
    assert np.max(np.abs(Doo - np.diag(np.diag(Doo)))) > 1e-8
    assert np.max(np.abs(Dvv - np.diag(np.diag(Dvv)))) > 1e-8


def test_t3_density_offdiag_reproduces_diagonal_spatial():
    """Piece A, spatial.  See :func:`test_so_t3_density_offdiag_reproduces_diagonal`."""
    cc = _water('6-31G', None).ccwfn
    _, dens = cctriples.t3_density(cc.o, cc.v, cc.no, cc.nv, cc.t1, cc.t2,
                                   cc.H.F, cc.H.ERI, cc.H.L, cc.contract)
    Doo, Dvv = cctriples.t3_density_offdiag(cc.o, cc.v, cc.no, cc.nv, cc.t1, cc.t2,
                                            cc.H.F, cc.H.ERI, cc.contract)
    assert np.max(np.abs(np.diag(Doo) - np.diag(np.asarray(dens['Doo'])))) < 1e-15
    assert np.max(np.abs(np.diag(Dvv) - np.diag(np.asarray(dens['Dvv'])))) < 1e-15
    assert np.max(np.abs(Doo - np.diag(np.diag(Doo)))) > 1e-8
    assert np.max(np.abs(Dvv - np.diag(np.diag(Dvv)))) > 1e-8


def test_so_dt3_density_offdiag_reproduces_diagonal():
    """Piece B, spin-orbital: same check on the perturbed blocks, under a general perturbation."""
    cc = _ch2('STO-3G', None).ccwfn
    F, ERI = np.asarray(cc.H.F), np.asarray(cc.H.ERI)
    t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
    dt1, dt2, df, deri = _perturbation(t1.shape, t2.shape, ERI.shape, antisym=True)
    base = cctriples.so_dt3_density(cc.o, cc.v, cc.no, cc.nv, t1, t2, dt1, dt2,
                                    F, df, ERI, deri, cc.contract)
    dDoo, dDvv = cctriples.so_dt3_density_offdiag(cc.o, cc.v, cc.no, cc.nv, t1, t2, dt1, dt2,
                                                  F, df, ERI, deri, cc.contract)
    assert np.max(np.abs(np.diag(dDoo) - np.diag(np.asarray(base['Doo'])))) < 1e-15
    assert np.max(np.abs(np.diag(dDvv) - np.diag(np.asarray(base['Dvv'])))) < 1e-15


def test_dt3_density_offdiag_reproduces_diagonal_spatial():
    """Piece B, spatial.  See :func:`test_so_dt3_density_offdiag_reproduces_diagonal`."""
    cc = _water('6-31G', None).ccwfn
    F, ERI = np.asarray(cc.H.F), np.asarray(cc.H.ERI)
    t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
    L = 2.0 * ERI - ERI.swapaxes(2, 3)
    dt1, dt2, df, deri = _perturbation(t1.shape, t2.shape, ERI.shape, antisym=False)
    dL = 2.0 * deri - deri.swapaxes(2, 3)
    base = cctriples.dt3_density(cc.o, cc.v, cc.no, cc.nv, t1, t2, dt1, dt2,
                                 F, df, ERI, deri, L, dL, cc.contract)
    dDoo, dDvv = cctriples.dt3_density_offdiag(cc.o, cc.v, cc.no, cc.nv, t1, t2, dt1, dt2,
                                               F, df, ERI, deri, cc.contract)
    assert np.max(np.abs(np.diag(dDoo) - np.diag(np.asarray(base['Doo'])))) < 1e-15
    assert np.max(np.abs(np.diag(dDvv) - np.diag(np.asarray(base['Dvv'])))) < 1e-15


# --------------------------------------------------------------------------------------------
# C: the full d_x f bracket must collapse to the batched diag(df) form when df IS diagonal
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('route', ['spatial', 'spinorbital'])
def test_t3_full_coupled_reduces_to_batched_for_diagonal_df(route):
    """Piece C: with a diagonal ``df`` the off-diagonal bracket vanishes term by term, so the
    coupled amplitudes must reproduce the batched ones across *every* intermediate -- not just
    ``Doo``/``Dvv``, since C also reaches ``Dov``, the two-particle blocks and the Lambda sources
    ``S1``/``S2``.  With a general ``df`` they must differ, or the coupling is not being applied.
    """
    so = route == 'spinorbital'
    cc = (_ch2('STO-3G', None) if so else _water('6-31G', None)).ccwfn
    F, ERI = np.asarray(cc.H.F), np.asarray(cc.H.ERI)
    t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
    dt1, dt2, dfg, deri = _perturbation(t1.shape, t2.shape, ERI.shape, antisym=so)
    args = (cc.o, cc.v, cc.no, cc.nv, t1, t2, dt1, dt2, F)
    for df, same in ((np.diag(np.diag(dfg)), True), (dfg, False)):
        if so:
            batched = cctriples.so_dt3_density(*args, df, ERI, deri, cc.contract)
            t3f = cctriples.so_t3_full_coupled(*args, df, ERI, deri, cc.contract)
            coupled = cctriples.so_dt3_density(*args, df, ERI, deri, cc.contract, t3_full=t3f)
        else:
            L, dL = 2.0 * ERI - ERI.swapaxes(2, 3), 2.0 * deri - deri.swapaxes(2, 3)
            batched = cctriples.dt3_density(*args, df, ERI, deri, L, dL, cc.contract)
            t3f = cctriples.t3_full_coupled(*args, df, ERI, deri, cc.contract)
            coupled = cctriples.dt3_density(*args, df, ERI, deri, L, dL, cc.contract, t3_full=t3f)
        dev = max(float(np.max(np.abs(np.asarray(coupled[k]) - np.asarray(batched[k]))))
                  for k in batched)
        if same:
            assert dev < 1e-15, f"diagonal df must reduce to the batched form (got {dev:.2e})"
        else:
            assert dev > 1e-8, f"general df must NOT reduce to the batched form (got {dev:.2e})"


# --------------------------------------------------------------------------------------------
# Open-shell, CFOUR-anchored.  6-31G runs in CI; cc-pVDZ is marked slow.
# --------------------------------------------------------------------------------------------
def _check_ch2(prop, basis, ref, tol, label):
    d = _ch2(basis, 'non-canonical')
    got = np.asarray(getattr(d, prop)().correlation)
    dev = float(np.max(np.abs(got - np.asarray(ref).reshape(got.shape))))
    record(label, 'CCSD(T)', 'all-electron', 'cfour', dev, molecule='CH2',
           note='spin-orbital, non-canonical perturbed MOs', ao_basis=basis,
           limited_by='oracle-print')
    assert dev < tol, (dev, got)
    return got


def test_so_ccsdt_gradient_noncanonical_cfour_ch2_631g():
    """Open-shell CCSD(T) gradient in the non-canonical gauge vs CFOUR GRD.  Needs piece A only."""
    _check_ch2('gradient', '6-31G', CFOUR_GRAD_T_CH2_631G, 1e-9, 'Gradient')


def test_so_ccsdt_polarizability_noncanonical_cfour_ch2_631g():
    """Open-shell CCSD(T) polarizability in the non-canonical gauge vs CFOUR POLAR-POLARSCF."""
    _check_ch2('polarizability', '6-31G', CFOUR_ALPHA_T_CH2_631G, 1e-9, 'Polarizability')


def test_so_ccsdt_apt_noncanonical_cfour_ch2_631g():
    """Open-shell CCSD(T) length-gauge APT in the non-canonical gauge vs CFOUR DIPDER."""
    _check_ch2('apt', '6-31G', CFOUR_APT_T_CH2_631G, 1e-9, 'APT')


def test_so_ccsdt_hessian_noncanonical_cfour_ch2_631g():
    """Open-shell CCSD(T) Hessian in the non-canonical gauge vs CFOUR FCMFINAL."""
    _check_ch2('hessian', '6-31G', CFOUR_HESS_T_CH2_631G, 1e-9, 'Hessian')


def test_so_ccsdt_gradient_gauge_invariance_ch2_631g():
    """The two gauges must give the same gradient.

    Sharper than the CFOUR comparison: it is free of the oracle's 10-decimal floor, and it is what
    fixes the *transpose* of the off-diagonal (T) density, which the diagonal checks cannot see.
    CH2/6-31G has no near-degeneracy, so the canonical result is itself trustworthy here.
    """
    g_can = np.asarray(_ch2('6-31G', None).gradient().correlation)
    g_non = np.asarray(_ch2('6-31G', 'non-canonical').gradient().correlation)
    assert np.max(np.abs(g_non - g_can)) < 1e-12, (g_can, g_non)


@pytest.mark.slow
@pytest.mark.parametrize('prop,ref,label', [
    ('hessian', CFOUR_HESS_T_CH2_PVDZ, 'Hessian'),
    ('apt', CFOUR_APT_T_CH2_PVDZ, 'APT'),
])
def test_so_ccsdt_second_derivative_noncanonical_cfour_ch2_pvdz(prop, ref, label):
    """CH2/cc-pVDZ: the system that exposed the defect.

    Its alpha virtual space carries an accidental cross-irrep near-degeneracy (gap 3.5e-5), which
    the canonical dependent-pair divide amplifies by ~3e4.  The canonical Hessian and APT miss
    CFOUR by 4.7e-10 and 4.4e-9 here; the non-canonical route reaches the oracle's floor.  Slow
    (~4-7 min each): the abc loop runs at nv = 40 and the coupled amplitudes are materialised.
    """
    _check_ch2(prop, 'cc-pVDZ', ref, 5e-10, label)


@pytest.mark.slow
def test_so_ccsdt_polarizability_noncanonical_cfour_ch2_pvdz():
    """CH2/cc-pVDZ polarizability, plus the conditioning the canonical gauge cannot deliver.

    The defect document's target was "the conditioning of the non-canonical row together with
    CFOUR's value".  Canonical gives +4.5e-08 with a 4.4e-12 tensor asymmetry and two unconverged
    solves, and is not even reproducible run to run (+3.4e-08 on one run, +4.5e-08 on another).
    """
    a = np.asarray(_ch2('cc-pVDZ', 'non-canonical').polarizability().correlation).reshape(3, 3)
    dev = abs(a[1, 1] - CFOUR_ALPHA_YY_T_CH2_PVDZ)
    record('Polarizability', 'CCSD(T)', 'all-electron', 'cfour', dev, molecule='CH2',
           note='spin-orbital, non-canonical perturbed MOs', ao_basis='cc-pVDZ',
           limited_by='oracle-print')
    assert dev < 1e-9, a[1, 1]
    assert np.max(np.abs(a - a.T)) < 1e-14, a      # canonical: 4.4e-12


# --------------------------------------------------------------------------------------------
# Closed-shell: the canonical gauge is healthy here, so these test gauge invariance
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('prop,ref', [('hessian', CFOUR_HESS_T_WATER),
                                      ('apt', CFOUR_APT_T_WATER_631G)])
def test_ccsdt_second_derivative_noncanonical_cfour_water_631g_spatial(prop, ref):
    """Spatial route, water/6-31G.  With no near-degeneracy the canonical gauge is already
    correct, so what this establishes is that the non-canonical route reproduces it."""
    got = np.asarray(getattr(_water('6-31G', 'non-canonical'), prop)().correlation)
    assert np.max(np.abs(got - np.asarray(ref).reshape(got.shape))) < 1e-9, got


@pytest.mark.slow
@pytest.mark.parametrize('prop,ref', [('hessian', CFOUR_HESS_T_WATER_PVDZ),
                                      ('apt', CFOUR_APT_T_WATER_PVDZ)])
def test_ccsdt_second_derivative_noncanonical_cfour_water_pvdz_spatial(prop, ref):
    """Spatial route, water/cc-pVDZ.  See the 6-31G case; slow for the larger virtual space."""
    got = np.asarray(getattr(_water('cc-pVDZ', 'non-canonical'), prop)().correlation)
    assert np.max(np.abs(got - np.asarray(ref).reshape(got.shape))) < 1e-9, got
