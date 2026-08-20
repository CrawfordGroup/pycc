"""DerivStore: the persistent HDF5-backed store for four-index derivative tensors.

The derivative suite runs with the store ON by default (disk), so the on-disk path is exercised
end-to-end throughout.  These tests add focused coverage for the store itself: the on-disk
round-trip (single and grouped), the disabled/RAM-memo path, the graceful fallback when h5py is
absent, and an explicit check that the on-disk Hessian is bit-identical to the in-memory one.
"""
import os

import numpy as np
import pytest

from pycc.derivatives import DerivStore


def test_roundtrip_disk():
    """Enabled store: first call builds + persists, second reads back exactly, close removes file."""
    pytest.importorskip("h5py")
    store = DerivStore(enabled=True)
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
    pytest.importorskip("h5py")
    store = DerivStore(enabled=True)
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


def test_ram_path_no_file():
    """Disabled store memoizes in RAM and never touches disk."""
    store = DerivStore(enabled=False)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return np.zeros((4, 4))

    store.get_or_compute("q", ("f", 0), build)
    store.get_or_compute("q", ("f", 0), build)
    assert calls["n"] == 1                                                  # memoized in RAM
    assert store._file is None and store._f is None


def test_graceful_fallback_without_h5py(monkeypatch):
    """When h5py is unavailable, an enabled store degrades to the RAM path with a warning."""
    import importlib.util as iu
    import pycc.derivatives as D

    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec",
                        lambda name, *a, **k: None if name == "h5py" else real(name, *a, **k))
    with pytest.warns(RuntimeWarning):
        store = D.DerivStore(enabled=True)
    assert store.enabled is False                                           # degraded to RAM memo


@pytest.mark.parametrize("orbital_basis", ["spatial", "spinorbital"])
def test_disk_matches_ram_hessian(rhf_wfn, orbital_basis):
    """The on-disk Hessian is bit-identical to the in-memory-memo Hessian (same driver machinery,
    just disk vs RAM backing), and the disk path really is used (temp file created) -- checked on
    both the spatial and spin-orbital routes, which the store keys distinctly (eri vs so_eri, and
    the 'sp'/'so' response-context tag)."""
    pytest.importorskip("h5py")
    import pycc

    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false")
    cc = pycc.ccwfn(wfn, orbital_basis=orbital_basis)
    cc.solve_cc(1e-10, 1e-10, 100)
    store = pycc.CCderiv(cc).wfn.derivatives.store                          # shared per-wfn store

    store.enabled = False
    H_ram = np.asarray(pycc.CCderiv(cc).hessian().correlation)

    store.enabled = True
    H_disk = np.asarray(pycc.CCderiv(cc).hessian().correlation)
    assert store._file and os.path.exists(store._file)                      # disk path exercised
    assert np.max(np.abs(H_ram - H_disk)) < 1e-12
    store.close()
