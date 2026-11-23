"""Anchored batching of the non-canonical perturbed (T) amplitudes against full materialisation.

Under non-canonical perturbed MOs the perturbed T3 acquires an off-diagonal `d_x f` coupling
whose two halves are batchable in opposite orders, so no single batching produces `dt3` complete
(`docs/batching_triples.md`).  The drivers use the **anchored** scheme, which anchors each
non-local half on the last two labels of the offending triple and deposits into three targets,
costing `O(N^4)` storage instead of the `O(N^6)` of materialising the amplitudes.

The materialised routines (`so_t3_full_coupled`, `t3_full_coupled`, and the `*_offdiag` density
builders that slice from them) are the reference the anchored scheme was developed against.  They
are no longer on any driver path, so these tests exist to keep them exercised and to catch drift
between the two implementations; they are the only machine-precision evidence that the anchored
scheme is correct, everything else being bounded by a CFOUR print floor.

The perturbation is a fixed pseudo-random direction rather than a physical one.  That is
deliberate: it exercises the product-rule machinery over a general `df` with nonzero oo/vv
off-diagonal blocks, which is exactly the regime the coupling exists for, and it needs no
wavefunction response.
"""
import contextlib
import os

import numpy as np
import psi4
import pytest

import pycc
from pycc import cctriples
from .test_083_ccsdt_gradient import CH2_CFOUR_FRAME


def _spinorbital_wfn():
    """Triplet CH2 / STO-3G, UHF.  Small enough that both implementations run in seconds."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_memory('4 GB')
        psi4.core.set_output_file('/dev/null', False)
        psi4.geometry(CH2_CFOUR_FRAME)
        psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'reference': 'uhf',
                          'e_convergence': 1e-13, 'd_convergence': 1e-13, 'maxiter': 300})
        _, wfn = psi4.energy('scf', return_wfn=True)
        cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital', make_t3_density=True)
        cc.solve_cc(1e-13, 1e-13, 300)
    return cc


def _spatial_wfn():
    """Water / 6-31G, RHF, in a C1 geometry."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_memory('4 GB')
        psi4.core.set_output_file('/dev/null', False)
        psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': 'false',
                          'e_convergence': 1e-12, 'd_convergence': 1e-12})
        psi4.geometry("0 1\nO 0.0 0.0 0.0\nH 0.0 0.0 1.83\nH 0.0 1.75 -0.42\n"
                      "units bohr\nno_com\nno_reorient\nsymmetry c1\n")
        _, wfn = psi4.energy('scf', return_wfn=True)
        cc = pycc.ccwfn(wfn, model='CCSD(T)', make_t3_density=True)
        cc.solve_cc(1e-12, 1e-12, 200)
    return cc


def _perturbation(cc, antisym, seed=0):
    """A fixed pseudo-random (dt1, dt2, df, deri) with df off-diagonal in oo and vv."""
    F, ERI = np.asarray(cc.H.F), np.asarray(cc.H.ERI)
    t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
    rng = np.random.default_rng(seed)
    dt1 = rng.normal(size=t1.shape) * 1e-2
    dt2 = rng.normal(size=t2.shape) * 1e-2
    df = rng.normal(size=F.shape) * 1e-2
    df = df + df.T
    deri = rng.normal(size=ERI.shape) * 1e-3
    if antisym:
        deri = deri - deri.swapaxes(0, 1) - deri.swapaxes(2, 3) + deri.swapaxes(0, 1).swapaxes(2, 3)
    else:
        deri = 0.5 * (deri + deri.transpose(1, 0, 3, 2))
    return (cc.o, cc.v, cc.no, cc.nv, t1, t2, dt1, dt2, F, df, ERI, deri)


@pytest.mark.parametrize('route', ['spinorbital', 'spatial'])
def test_anchored_matches_materialised(route):
    """Every perturbed (T) intermediate from the anchored passes must match the materialised
    implementation to machine precision, under a general off-diagonal ``df``.

    This is the test that caught the one real error in the scheme: the consumers are *not* linear
    in ``dt3``.  Each is a product rule whose two halves are e.g. ``dt3c . t2[j,k]`` and
    ``t3c . dt2[j,k]``, and only the first carries ``dt3``; the first is deposited once per anchor
    (three times per target), the second once per target.  ``Doo`` and ``Dvv`` are the only blocks
    whose halves both carry ``dt3``, so they stay exact while the other blocks break, which makes
    the failure easy to misread as success.  Assert on all of them.
    """
    so = route == 'spinorbital'
    cc = _spinorbital_wfn() if so else _spatial_wfn()
    args = _perturbation(cc, antisym=so)
    con = cc.contract

    if so:
        t3f = cctriples.so_t3_full_coupled(*args, con)
        ref = cctriples.so_dt3_density(*args, con, t3_full=t3f)
        refDoo, refDvv = cctriples.so_dt3_density_offdiag(*args, con, t3_full=t3f)
        got = cctriples.so_dt3_density_pass1(*args, con)
        gotDoo = cctriples.so_dt3_doo_pass2(*args, con)
    else:
        ERI = np.asarray(cc.H.ERI)
        L = 2.0 * ERI - ERI.swapaxes(2, 3)
        dL = 2.0 * args[11] - args[11].swapaxes(2, 3)
        t3f = cctriples.t3_full_coupled(*args, con)
        ref = cctriples.dt3_density(*args, L, dL, con, t3_full=t3f)
        refDoo, refDvv = cctriples.dt3_density_offdiag(*args, con, t3_full=t3f)
        got = cctriples.dt3_density_pass1(*args, L, dL, con)
        gotDoo = cctriples.dt3_doo_pass2(*args, con)

    for key in ref:
        a = np.asarray(got[key])
        b = np.asarray(refDvv) if key == 'Dvv' else np.asarray(ref[key])
        if key == 'Doo':                      # pass 1 carries the diagonal; pass 2 the full block
            a, b = np.diag(a), np.diag(np.asarray(refDoo))
        assert np.max(np.abs(a - b)) < 1e-14, (key, np.max(np.abs(a - b)))
        assert np.max(np.abs(b)) > 1e-12, (key, 'reference is trivially zero')

    gotDoo, refDoo = np.asarray(gotDoo), np.asarray(refDoo)
    assert np.max(np.abs(gotDoo - refDoo)) < 1e-14, np.max(np.abs(gotDoo - refDoo))
    # the off-diagonal elements are the whole point of the exercise, so they must be present
    assert np.max(np.abs(gotDoo - np.diag(np.diag(gotDoo)))) > 1e-12


@pytest.mark.parametrize('route', ['spinorbital', 'spatial'])
def test_anchored_passes_agree_on_the_doo_diagonal(route):
    """The two passes compute the ``Doo`` diagonal by completely different routes -- pass 1 from
    occupied anchors, pass 2 from virtual anchors -- so their agreement is a free internal check
    that needs no reference implementation at all."""
    so = route == 'spinorbital'
    cc = _spinorbital_wfn() if so else _spatial_wfn()
    args = _perturbation(cc, antisym=so)
    con = cc.contract
    if so:
        p1 = cctriples.so_dt3_density_pass1(*args, con)
        p2 = cctriples.so_dt3_doo_pass2(*args, con)
    else:
        ERI = np.asarray(cc.H.ERI)
        L = 2.0 * ERI - ERI.swapaxes(2, 3)
        dL = 2.0 * args[11] - args[11].swapaxes(2, 3)
        p1 = cctriples.dt3_density_pass1(*args, L, dL, con)
        p2 = cctriples.dt3_doo_pass2(*args, con)
    d1, d2 = np.diag(np.asarray(p1['Doo'])), np.diag(np.asarray(p2))
    assert np.max(np.abs(d1 - d2)) < 1e-14, np.max(np.abs(d1 - d2))
    assert np.max(np.abs(d2)) > 1e-12


@pytest.mark.parametrize('route', ['spinorbital', 'spatial'])
def test_coupling_reduces_to_the_batched_form_for_diagonal_df(route):
    """With a diagonal ``df`` the off-diagonal bracket vanishes term by term, so the anchored
    passes must reproduce the *canonical* batched routines, which carry only ``diag(df)``.  This
    pins the coupling's construction independently of the materialised reference."""
    so = route == 'spinorbital'
    cc = _spinorbital_wfn() if so else _spatial_wfn()
    args = list(_perturbation(cc, antisym=so))
    args[9] = np.diag(np.diag(args[9]))               # df -> diagonal only
    args = tuple(args)
    con = cc.contract
    if so:
        batched = cctriples.so_dt3_density(*args, con)          # canonical path, diag(df)
        got = cctriples.so_dt3_density_pass1(*args, con)
    else:
        ERI = np.asarray(cc.H.ERI)
        L = 2.0 * ERI - ERI.swapaxes(2, 3)
        dL = 2.0 * args[11] - args[11].swapaxes(2, 3)
        batched = cctriples.dt3_density(*args, L, dL, con)
        got = cctriples.dt3_density_pass1(*args, L, dL, con)
    for key in batched:
        if key in ('Doo', 'Dvv'):                     # anchored returns these full, not diagonal
            a = np.diag(np.asarray(got[key])) if key == 'Dvv' else np.diag(np.asarray(got[key]))
            b = np.diag(np.asarray(batched[key]))
        else:
            a, b = np.asarray(got[key]), np.asarray(batched[key])
        assert np.max(np.abs(a - b)) < 1e-14, (key, np.max(np.abs(a - b)))
