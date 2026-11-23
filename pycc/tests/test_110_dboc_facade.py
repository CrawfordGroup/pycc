"""The ``pycc.dboc`` facade entry.

The DBOC is not an energy derivative: like the AAT and the velocity-gauge APT it is an overlap of
wavefunction derivatives, and `CIderiv.dboc` reuses the AAT's sectors with both perturbations
nuclear.  So it belongs in the same facade as those, with two differences from the energy-derivative
entries: the value is a scalar, and there is no nuclear block (as for the polarizability).

The driver methods return the *total* electronic DBOC, so the facade's ``correlation`` is that
total minus the Hartree-Fock value -- the decomposition CFOUR's ``method - SCF`` recipe assumes.
Frozen core is a no-op for the reference, so the subtraction holds either way.

Reference values are the frozen-core water/cc-pVDZ numbers already asserted by test_101 (CISD),
test_102 (HF), test_103 (MP1), test_106 (MP2 order-consistent) and test_107 (CCSD); this file does
not re-validate the physics, it checks that the facade reproduces the driver methods exactly and
decomposes correctly.
"""
import contextlib
import os

import numpy as np
import psi4
import pytest

import pycc
from pycc.properties import CM_PER_EH

WATER = "0 1\nO\nH 1 0.95623\nH 1 0.95623 2 104.25\nsymmetry c1\n"


@pytest.fixture(scope='module')
def water_scf():
    """Frozen-core water / cc-pVDZ RHF, shared by every case in this file."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_memory('4 GB')
        psi4.core.set_output_file('/dev/null', False)
        psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk', 'freeze_core': 'true',
                          'e_convergence': 1e-12, 'd_convergence': 1e-12})
        psi4.geometry(WATER)
        _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _driver(wfn, kind):
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        if kind == 'hf':
            return pycc.HFwfn(wfn)
        if kind == 'mp':
            return pycc.MPderiv(pycc.MPwfn(wfn))
        if kind == 'ci':
            ci = pycc.CIwfn(wfn)
            ci.solve_ci(e_conv=1e-12, r_conv=1e-12)
            return pycc.CIderiv(ci)
        cc = pycc.ccwfn(wfn, model='CCSD')
        cc.solve_cc(1e-12, 1e-12, 200)
        return pycc.CCderiv(cc)


@pytest.mark.parametrize('kind,variant,method,ref_cm', [
    ('hf', None, 'dboc', 600.367450),
    ('mp', 'mp1', 'dboc_mp1', 609.630785),
    ('mp', 'order-consistent', 'dboc_mp2_oc', 614.643924),
    ('ci', None, 'dboc', 615.101001),
    ('cc', None, 'dboc', 617.255437),
])
def test_dboc_facade_matches_the_driver_method(water_scf, kind, variant, method, ref_cm):
    """The facade total must equal the driver method exactly, and match the suite's reference."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        pc = pycc.dboc(_driver(water_scf, kind), variant)
        direct = float(getattr(_driver(water_scf, kind), method)())
    assert float(pc.total) == pytest.approx(direct, abs=1e-14)
    assert float(pc.total) * CM_PER_EH == pytest.approx(ref_cm, abs=1e-4)


@pytest.mark.parametrize('kind,variant,corr_expected', [
    ('hf', None, False),                 # Hartree-Fock: correlation is identically zero
    ('mp', 'mp1', True),
    ('ci', None, True),
])
def test_dboc_facade_decomposition(water_scf, kind, variant, corr_expected):
    """No nuclear block; ``reference`` is the Hartree-Fock DBOC; ``correlation`` is the remainder."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        pc = pycc.dboc(_driver(water_scf, kind), variant)
        hf = float(pycc.HFwfn(water_scf).dboc())
    assert float(pc.nuclear) == 0.0
    assert float(pc.reference) == pytest.approx(hf, abs=1e-14)
    assert float(pc.total) == pytest.approx(float(pc.reference) + float(pc.correlation), abs=1e-14)
    if corr_expected:
        assert abs(float(pc.correlation)) > 1e-6
    else:
        assert float(pc.correlation) == 0.0


def test_dboc_facade_guards(water_scf):
    """A variant only means something for MPderiv; unknown variants and bare wavefunctions raise."""
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        ci = _driver(water_scf, 'ci')
        mp = _driver(water_scf, 'mp')
    with pytest.raises(ValueError, match="applies only to MPderiv"):
        pycc.dboc(ci, 'mp1')
    with pytest.raises(ValueError, match="unknown DBOC variant"):
        pycc.dboc(mp, 'nonsense')
    with pytest.raises(TypeError, match="derivative driver"):
        pycc.dboc(pycc.MPwfn(water_scf))


def test_dboc_is_exported():
    """It must be reachable as ``pycc.dboc``, like the other facade entries."""
    assert 'dboc' in pycc.__all__
    assert callable(pycc.dboc)
