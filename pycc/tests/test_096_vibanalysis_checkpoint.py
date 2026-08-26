"""
Vibrational-analysis plumbing: the property-checkpoint archive (:mod:`pycc.checkpoint`) and the
one-call spectrum drivers (:func:`pycc.ir` / :func:`pycc.vcd`).

Covers the behaviour, not the physics (the frequency/IR/VCD numbers are anchored in
``test_095_vcd``):

* a checkpoint round-trips the property tensors at full precision -- re-analyzing from the ``.npz``
  reproduces the direct-component analysis bit-for-bit;
* ``pycc.ir`` gives the same result from a live driver and from the checkpoint it wrote;
* ``save_checkpoint`` is harvest-only (it never computes) -- an empty or missing-property request
  raises;
* the drivers degrade gracefully on an incomplete checkpoint: a missing APT warns and falls back
  to a frequencies-only analysis, while a missing Hessian raises.
"""

import os
import io
import contextlib

import psi4
import pycc
import numpy as np


WATER = """
O  0.000000000000  0.000000000000 -0.129476880010
H  0.000000000000 -1.494186750504  1.027446024111
H  0.000000000000  1.494186750504  1.027446024111
symmetry c1
units bohr
"""


def _mp2_deriv(rhf_wfn):
    wfn = rhf_wfn(WATER, "STO-3G", freeze_core="false")
    mp = pycc.MPwfn(wfn, quiet=True)
    mp.compute_energy()
    return pycc.MPderiv(mp)


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def test_checkpoint_roundtrip(rhf_wfn, tmp_path):
    """Saving Hessian + APT and re-analyzing from the ``.npz`` reproduces the direct-component
    harmonic analysis exactly (full float64 precision, molecule/frame/masses intact)."""
    d = _mp2_deriv(rhf_wfn)
    H = np.asarray(_quiet(lambda: pycc.hessian(d)).total)
    apt = np.asarray(_quiet(lambda: pycc.apt(d)).total)
    direct = _quiet(lambda: pycc.harmonic_analysis(d.wfn.ref.molecule(), H, apt=apt))

    path = str(tmp_path / "water.npz")
    pycc.save_checkpoint(d, path)                 # harvests the two recorded properties
    fromfile = _quiet(lambda: pycc.harmonic_analysis(path))

    assert np.array_equal(direct['frequencies'], fromfile['frequencies'])
    assert np.array_equal(direct['ir_intensities'], fromfile['ir_intensities'])
    ck = pycc.load_checkpoint(path)
    assert sorted(ck.components) == ['apt', 'hessian']
    assert ck.method == 'MP2' and ck.reference == 'RHF'
    psi4.core.clean()


def test_vibanalysis_apt_route_invariant(rhf_wfn):
    """The ir/vcd drivers build the length-gauge APT via the '2n+1-nuclear' route (reusing the
    Hessian's just-cached 3N nuclear responses, cheaper in this workflow); the resulting spectrum
    must be identical to the standalone-default '2n+1-field' route.  Guards the vibanalysis route
    switch: (i) the two routes give the same APT, and (ii) ir()/vcd() IR intensities and VCD
    rotatory strengths are unchanged by it."""
    d = _mp2_deriv(rhf_wfn)
    spec = _quiet(lambda: pycc.ir(d))                      # ir() uses '2n+1-nuclear' internally
    mol = d.wfn.ref.molecule()
    H = np.asarray(_quiet(lambda: pycc.hessian(d)).total)  # cached from ir()
    apt_field = np.asarray(_quiet(lambda: pycc.apt(d, route='2n+1-field')).total)
    apt_nuclear = np.asarray(_quiet(lambda: pycc.apt(d, route='2n+1-nuclear')).total)

    # (i) the two routes give the same APT tensor
    assert np.allclose(apt_nuclear, apt_field, atol=1e-12), np.max(np.abs(apt_nuclear - apt_field))

    # (ii) ir() spectrum (nuclear route) == the explicit field-route spectrum
    spec_field = _quiet(lambda: pycc.harmonic_analysis(mol, H, apt=apt_field))
    assert np.allclose(spec['frequencies'], spec_field['frequencies'], atol=1e-8)
    assert np.allclose(spec['ir_intensities'], spec_field['ir_intensities'], rtol=1e-8, atol=1e-10)

    # (ii, VCD) vcd() rotatory strengths (nuclear-route APT) == the field-route ones (AAT unchanged)
    vcd_spec = _quiet(lambda: pycc.vcd(d))
    aat = np.asarray(_quiet(lambda: pycc.aat(d)).total)
    vcd_field = _quiet(lambda: pycc.harmonic_analysis(mol, H, apt=apt_field, aat=aat))
    assert np.allclose(vcd_spec['rotatory_strengths'], vcd_field['rotatory_strengths'],
                       rtol=1e-8, atol=1e-10)
    psi4.core.clean()


def test_ir_driver_live_vs_checkpoint(rhf_wfn, tmp_path):
    """``pycc.ir`` from a live driver and from the checkpoint it wrote give identical frequencies
    and IR intensities."""
    d = _mp2_deriv(rhf_wfn)
    path = str(tmp_path / "water_ir.npz")
    live = _quiet(lambda: pycc.ir(d, checkpoint=path))
    reloaded = _quiet(lambda: pycc.ir(path))     # str source -> re-analysis, no compute
    assert np.array_equal(live['frequencies'], reloaded['frequencies'])
    assert np.array_equal(live['ir_intensities'], reloaded['ir_intensities'])
    psi4.core.clean()


def test_save_checkpoint_is_harvest_only(rhf_wfn, tmp_path):
    """``save_checkpoint`` never computes: nothing recorded -> error; a requested-but-uncomputed
    property -> error."""
    d = _mp2_deriv(rhf_wfn)
    path = str(tmp_path / "x.npz")

    with np.testing.assert_raises(ValueError):
        pycc.save_checkpoint(d, path)            # nothing computed yet

    _quiet(lambda: pycc.hessian(d))              # record only the Hessian
    with np.testing.assert_raises(ValueError):
        pycc.save_checkpoint(d, path, ['hessian', 'apt'])   # apt never computed
    psi4.core.clean()


def test_ir_driver_incomplete_checkpoint(rhf_wfn, tmp_path):
    """On a checkpoint missing the APT, ``ir`` warns and runs a frequencies-only analysis; on one
    missing the Hessian it raises."""
    d = _mp2_deriv(rhf_wfn)

    # Hessian-only checkpoint: ir warns, proceeds, ir_intensities is None.
    hpath = str(tmp_path / "hess_only.npz")
    _quiet(lambda: pycc.hessian(d))
    pycc.save_checkpoint(d, hpath)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = pycc.ir(hpath)
    assert res['ir_intensities'] is None
    assert "WARNING" in buf.getvalue() and "APT" in buf.getvalue()

    # APT-only checkpoint (no Hessian): ir raises.
    d2 = _mp2_deriv(rhf_wfn)
    apath = str(tmp_path / "apt_only.npz")
    _quiet(lambda: pycc.apt(d2))
    pycc.save_checkpoint(d2, apath)
    with np.testing.assert_raises(ValueError):
        _quiet(lambda: pycc.ir(apath))
    psi4.core.clean()
