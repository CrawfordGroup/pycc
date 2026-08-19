"""DerivStore scratch-directory guardrail: filesystem-type detection and the RAM-backed
(``tmpfs``) warning.  A store directory on a RAM-backed filesystem (common on HPC nodes, where
``/tmp`` is often a ``tmpfs``) silently turns the on-disk derivative cache into a memory hog; the
store must warn so the user redirects it to real disk."""

import warnings

import pytest

import pycc.derivatives as D
from pycc.exceptions import PyCCWarning


# A synthetic Linux mount table: root on ext4, a RAM-backed /ramdisk, a nested real fs under it.
# (Mount points avoid /tmp, which is a symlink to /private/tmp on macOS and would be rewritten by
# the realpath() in _filesystem_type, defeating the test off Linux.)
_MOUNTS = (
    "sysfs /sys sysfs rw 0 0\n"
    "/dev/sda1 / ext4 rw 0 0\n"
    "tmpfs /ramdisk tmpfs rw 0 0\n"
    "/dev/sda7 /localscratch ext4 rw 0 0\n"
    "vaster:/owl/scratch /netscratch lustre rw 0 0\n"
    "/dev/sdb1 /ramdisk/real ext4 rw 0 0\n"      # a real fs mounted *under* the tmpfs /ramdisk
)


def _fstype(tmp_path, target):
    """Run _filesystem_type against the synthetic mount table for an absolute target path."""
    mounts = tmp_path / "mounts"
    mounts.write_text(_MOUNTS)
    return D._filesystem_type(target, _mounts=str(mounts))


def test_filesystem_type_longest_prefix(tmp_path):
    """The fstype is that of the LONGEST matching mount point, so nesting resolves correctly."""
    assert _fstype(tmp_path, "/data/user") == "ext4"              # falls through to root
    assert _fstype(tmp_path, "/ramdisk/pycc_deriv_abc.h5") == "tmpfs"
    assert _fstype(tmp_path, "/ramdisk/real/x.h5") == "ext4"      # real fs under /ramdisk (longer prefix)
    assert _fstype(tmp_path, "/localscratch/crawdad") == "ext4"
    assert _fstype(tmp_path, "/netscratch/crawdad") == "lustre"
    assert _fstype(tmp_path, "/ramdiskfoo/x") == "ext4"          # not under /ramdisk (no false match)


def test_filesystem_type_missing_table_returns_none():
    """A missing/unreadable mount table -> None (unknown), so callers do not warn."""
    assert D._filesystem_type("/anything", _mounts="/no/such/mounts/file") is None


def _store_warnings(monkeypatch, fake_fstype):
    """Create a DerivStore whose scratch resolves to ``fake_fstype`` and return its RAM-backed
    warnings (the temp .h5 is cleaned up)."""
    monkeypatch.setattr(D, "_filesystem_type", lambda *a, **k: fake_fstype)
    store = D.DerivStore(enabled=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store._ensure()
        return [w for w in caught
                if issubclass(w.category, PyCCWarning) and "RAM-backed" in str(w.message)]
    finally:
        if store._f is not None:
            store._f.close()
        import os
        if store._file and os.path.exists(store._file):
            os.remove(store._file)


@pytest.mark.skipif(D.DerivStore(enabled=True).enabled is False,
                    reason="h5py unavailable; DerivStore uses the in-memory path (no scratch file)")
def test_store_warns_on_ram_backed_scratch(monkeypatch):
    """The store warns (once) when its scratch directory is a RAM-backed filesystem."""
    assert len(_store_warnings(monkeypatch, "tmpfs")) == 1


@pytest.mark.skipif(D.DerivStore(enabled=True).enabled is False,
                    reason="h5py unavailable; DerivStore uses the in-memory path (no scratch file)")
def test_store_silent_on_real_or_unknown_scratch(monkeypatch):
    """No warning when the scratch is a real filesystem, or when the type is unknown (non-Linux)."""
    assert _store_warnings(monkeypatch, "lustre") == []
    assert _store_warnings(monkeypatch, "ext4") == []
    assert _store_warnings(monkeypatch, None) == []
