"""DerivStore: the persistent HDF5-backed store for four-index derivative tensors.

The store is **mandatory and always disk-backed**: four-index derivative quantities belong on
disk, not in memory, because a run generates far too many of them for a RAM cache to pay.  There
is no in-memory mode and no fallback, so a missing h5py is an error rather than a silent switch
to unbounded memoization.

The derivative suite exercises the on-disk path end-to-end throughout.  These tests add focused
coverage for the store itself: the round-trip (single and grouped), the h5py requirement, and that
a correlated Hessian really does drive tensors through the store on both orbital-basis routes.
"""
import os

import numpy as np
import pytest

from pycc.derivatives import DerivStore
from pycc.exceptions import PyCCError


def test_roundtrip_disk():
    """First call builds and persists, second reads back exactly, close removes the file."""
    store = DerivStore()
    try:
        calls = {"n": 0}

        def build():
            calls["n"] += 1
            return np.arange(24, dtype=float).reshape(2, 3, 4)

        a = store.get_or_compute("q", ("nuc", 0), build, ctx=(1, True))
        assert calls["n"] == 1
        assert store.has("q", ("nuc", 0), ctx=(1, True))
        b = store.get_or_compute("q", ("nuc", 0), build, ctx=(1, True))     # hit: builder not re-run
        assert calls["n"] == 1
        assert np.array_equal(a, b)                                          # exact round-trip
        assert store._file and os.path.exists(store._file)                  # disk file was created
        path = store._file
    finally:
        store.close()
    assert not os.path.exists(path)                                         # removed on close
    assert store._f is None


def test_group_roundtrip_disk():
    """Grouped store: differing-shape components built once, read back exactly, all-or-nothing."""
    store = DerivStore()
    try:
        calls = {"n": 0}

        def build():
            calls["n"] += 1
            return (np.ones((2, 2)), np.arange(8.0).reshape(2, 2, 2), np.full((3,), 2.0))

        names = ("dDrel", "dGam", "dI")
        r1 = store.get_or_compute_group("resp", ("nuc", 1), build, names, ctx=("sp",))
        assert calls["n"] == 1
        r2 = store.get_or_compute_group("resp", ("nuc", 1), build, names, ctx=("sp",))  # full hit
        assert calls["n"] == 1
        for x, y in zip(r1, r2):
            assert np.array_equal(x, y)
    finally:
        store.close()


def test_requires_h5py(monkeypatch):
    """No h5py -> a readable error, NOT a silent in-memory store.  The store holds nmo^4 tensors;
    memoizing them in RAM instead would trade a bounded disk cache for an unbounded memory one,
    i.e. an out-of-memory kill in place of an error the user can act on."""
    import importlib.util as iu
    import pycc.derivatives as D

    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec",
                        lambda name, *a, **k: None if name == "h5py" else real(name, *a, **k))
    with pytest.raises(PyCCError, match="h5py is required"):
        D.DerivStore()


@pytest.mark.parametrize("orbital_basis", ["spatial", "spinorbital"])
def test_hessian_drives_tensors_through_the_store(rhf_wfn, orbital_basis):
    """A correlated Hessian routes its four-index derivative tensors through the on-disk store on
    both orbital-basis routes, which the store keys distinctly (``eri`` vs ``so_eri`` in the eri1
    context, and the ``'sp'``/``'so'`` tag on the perturbed-response records)."""
    import pycc

    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false")
    cc = pycc.ccwfn(wfn, orbital_basis=orbital_basis)
    cc.solve_cc(1e-10, 1e-10, 100)
    deriv = pycc.CCderiv(cc)
    store = deriv.wfn.derivatives.store                                     # shared per-wfn store

    H = np.asarray(deriv.hessian().correlation)
    assert H.shape[0] == H.shape[1]
    assert np.max(np.abs(H - H.T)) < 1e-10                                  # symmetric

    assert store._file and os.path.exists(store._file)                      # disk path exercised
    keys = sorted(store._f.keys())
    tag = "so_eri" if orbital_basis == "spinorbital" else "eri"
    eri1 = [k for k in keys if k.startswith("eri1")]
    assert eri1, keys
    assert all(f"__{tag}__" in k for k in eri1), eri1                       # basis-distinct keying
    store.close()
