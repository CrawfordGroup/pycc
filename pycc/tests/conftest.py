"""Shared pytest fixtures for the PyCC test suite.

Centralizes the Psi4 boilerplate (scratch memory, output file, SCF options,
geometry + SCF call) that was previously copy-pasted into every test module.

A test now obtains its reference wavefunction through the ``rhf_wfn`` factory::

    def test_ccsd_h2o(rhf_wfn):
        wfn = rhf_wfn("H2O", "STO-3G")
        ccsd = pycc.ccwfn(wfn)
        ...

The ``pycc.ccwfn(...)`` call (model, local, ... kwargs) legitimately
varies per test and intentionally stays in the test body.
"""

import os
from distutils import dir_util

import psi4
import pytest

from pycc.data.molecules import moldict

# Psi4 SCF options shared by essentially every test in the suite. Individual
# tests override any of these (most commonly 'basis' and 'freeze_core') by
# passing keyword arguments to the rhf_wfn factory.
DEFAULT_SCF_OPTIONS = {
    'scf_type': 'pk',
    'mp2_type': 'conv',
    'freeze_core': 'true',
    'e_convergence': 1e-12,
    'd_convergence': 1e-12,
    'r_convergence': 1e-12,
    'diis': 1,
}


@pytest.fixture(autouse=True)
def _psi4_clean_state():
    """Autouse: reset Psi4's global options before every test (and clean scratch after).

    A leftover option -- most importantly a non-default ``reference`` left by an earlier
    test -- otherwise leaks into any later test that drives Psi4 itself rather than going
    through the ``rhf_wfn``/``uhf_wfn``/``rohf_wfn`` factories (which already clear options
    via :func:`psi4_environment`). This runs for *every* test regardless, so order can
    never matter. (The factory path's own ``clean_options`` is now redundant but harmless.)
    """
    psi4.core.clean_options()
    yield
    psi4.core.clean()


@pytest.fixture
def psi4_environment():
    """Put Psi4 in a clean, quiet state around a test.

    Sets scratch memory, redirects output to ``output.dat``, and clears global
    options so settings do not leak between test modules; cleans scratch files
    on teardown. Not autouse: it only affects tests that opt in, either by
    requesting it directly or (more commonly) via the ``rhf_wfn`` factory, so
    tests that manage their own Psi4 state are untouched.
    """
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.core.clean_options()
    yield
    psi4.core.clean()


@pytest.fixture
def rhf_wfn(psi4_environment):
    """Factory fixture that builds an RHF reference wavefunction.

    Returns a callable so a single test can build several references (e.g. the
    same molecule in different bases)::

        def test_x(rhf_wfn):
            wfn  = rhf_wfn("H2O", "STO-3G")
            wfn2 = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false")

    Parameters
    ----------
    molecule : str
        Either a key into ``pycc.data.molecules.moldict`` (e.g. "H2O",
        "H2O_Teach") or a literal Psi4 geometry string (for the one-off
        geometries some tests define inline).
    basis : str, optional
        Psi4 basis-set name. Defaults to "cc-pVDZ", the suite's most common basis.
    geom_extra : str, optional
        Extra lines appended to the geometry block, e.g. ``"\\nnoreorient\\nnocom"``
        or ``"\\nsymmetry c1"``.
    **options
        Additional Psi4 options, overriding (or extending) DEFAULT_SCF_OPTIONS.

    Returns
    -------
    psi4.core.Wavefunction
        The converged RHF wavefunction (the ``return_wfn=True`` object).
    """
    def _build(molecule, basis="cc-pVDZ", geom_extra="", **options):
        opts = {**DEFAULT_SCF_OPTIONS, 'basis': basis, **options}
        psi4.set_options(opts)
        geometry = moldict.get(molecule, molecule)  # moldict key, else literal geometry
        psi4.geometry(geometry + geom_extra)
        _, wfn = psi4.energy('SCF', return_wfn=True)
        return wfn
    return _build


@pytest.fixture
def uhf_wfn(psi4_environment):
    """Factory fixture that builds a UHF reference wavefunction.

    Same interface as :func:`rhf_wfn`, but sets ``reference = 'uhf'`` so open-shell
    species (radicals, etc.) can be used to exercise the spin-orbital path. The
    geometry should carry its own charge/multiplicity line (e.g. ``"0 2\\n..."``).
    """
    def _build(molecule, basis="cc-pVDZ", geom_extra="", **options):
        opts = {**DEFAULT_SCF_OPTIONS, 'basis': basis, 'reference': 'uhf', **options}
        psi4.set_options(opts)
        geometry = moldict.get(molecule, molecule)  # moldict key, else literal geometry
        psi4.geometry(geometry + geom_extra)
        _, wfn = psi4.energy('SCF', return_wfn=True)
        return wfn
    return _build


@pytest.fixture
def rohf_wfn(psi4_environment):
    """Factory fixture that builds an ROHF reference wavefunction.

    Same interface as :func:`rhf_wfn`, but sets ``reference = 'rohf'`` for open-shell
    species. The spin-orbital path semicanonicalizes the ROHF orbitals internally.
    The geometry should carry its own charge/multiplicity line (e.g. ``"0 2\\n..."``).
    """
    def _build(molecule, basis="cc-pVDZ", geom_extra="", **options):
        opts = {**DEFAULT_SCF_OPTIONS, 'basis': basis, 'reference': 'rohf', **options}
        psi4.set_options(opts)
        geometry = moldict.get(molecule, molecule)  # moldict key, else literal geometry
        psi4.geometry(geometry + geom_extra)
        _, wfn = psi4.energy('SCF', return_wfn=True)
        return wfn
    return _build


@pytest.fixture
def datadir(tmpdir, request):
    '''
    from: https://stackoverflow.com/a/29631801
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    else:
        raise FileNotFoundError("Test folder not found.")

    return tmpdir

# CISD analytic-derivative tests (test_079-082): shared geometry + builder

# (P)-hydrogen peroxide, CISD-VCD validation geometry (Bohr); atom order H,H,O,O.
CISD_H2O2_GEOM = """
H -1.938703940499  1.630081390557  0.408551539155
H  1.938703940499 -1.630081390557  0.408551534619
O -1.364718227128 -0.236825138192  0.408556429577
O  1.364718227128  0.236825138192  0.408556416160
units bohr
no_com
no_reorient
"""


@pytest.fixture
def cisd_h2o2(psi4_environment):
    """A solved spatial CISD wavefunction for (P)-H2O2 at the CISD-VCD validation
    geometry, at the requested basis.  All-electron by default; ``freeze_core=True``
    freezes the two oxygen 1s orbitals (the frozen-core derivative path)."""
    def _make(basis='sto-3g', freeze_core=False):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.geometry(CISD_H2O2_GEOM)
        psi4.set_options({'basis': basis, 'scf_type': 'pk', 'freeze_core': freeze_core,
                          'e_convergence': 1e-13, 'd_convergence': 1e-13})
        _, wfn = psi4.energy('scf', return_wfn=True)
        import pycc
        ci = pycc.CIwfn(wfn, model='CISD', orbital_basis='spatial')
        ci.solve_ci(e_conv=1e-13, r_conv=1e-13, maxiter=150)
        return ci
    return _make


# ---------------------------------------------------------------------------
# Validation-table instrumentation
# ---------------------------------------------------------------------------
# The tests already compute the achieved deviation from their oracle and then
# discard it when the assertion passes.  ``record()`` keeps it, so the manuscript's
# supplementary validation table reports *measured* agreement rather than the
# assertion tolerance (an upper bound somebody chose).
#
# Opt-in: nothing is written unless PYCC_VALIDATION_LOG names an output file.
# Rows accumulate in memory and are flushed once at session end, so a partial or
# interrupted run still leaves whatever it reached.
#
#   PYCC_VALIDATION_LOG=/tmp/validation.json pytest pycc/tests/test_090*.py
#
# ``oracle`` is a short key into the table's footnoted oracle list (see
# docs/validation_table.py); ``limited_by`` marks a row whose agreement is set by
# the reference rather than by PyCC -- CFOUR's printed precision, or a finite
# difference's truncation error.

import atexit as _atexit
import json as _json

_VALIDATION_ROWS = []


def record(property, method, basis, oracle, achieved, *,
           reference=None, note=None, limited_by=None, molecule=None, ao_basis=None):
    """Record one measured agreement for the supplementary validation table.

    Parameters
    ----------
    property : str   e.g. 'Hessian', 'APT (length gauge)', 'AAT'
    method   : str   e.g. 'CCSD', 'CCSD(T)', 'MP2', 'CISD'
    basis    : str   the all-electron / frozen-core label for this case
    ao_basis : str, optional -- the AO basis set ('6-31G', 'cc-pVDZ', 'STO-3G').
               Pass it explicitly; it is reported as its own column in the
               supplementary validation table.
    oracle   : str   short key: 'cfour', 'fd-energy', 'fd-dipole', 'internal',
                     'identity', 'psi4'
    achieved : float measured max|pycc - reference|
    limited_by : str, optional -- 'oracle-print' or 'fd-truncation' when the
                 number reflects the reference's precision, not PyCC's error.
    """
    if not os.environ.get('PYCC_VALIDATION_LOG'):
        return
    _VALIDATION_ROWS.append({
        'property': property, 'method': method, 'basis': basis,
        'ao_basis': ao_basis,
        'molecule': molecule, 'oracle': oracle, 'achieved': float(achieved),
        'reference': reference, 'note': note, 'limited_by': limited_by,
    })


@_atexit.register
def _flush_validation_rows():
    path = os.environ.get('PYCC_VALIDATION_LOG')
    if not path or not _VALIDATION_ROWS:
        return
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as fh:
                existing = _json.load(fh)
        except (ValueError, OSError):
            existing = []
    with open(path, 'w') as fh:
        _json.dump(existing + _VALIDATION_ROWS, fh, indent=1)


def current_test_context():
    """Molecule / orbital-basis / frozen-core context parsed from the running test's name.

    pytest exports ``PYTEST_CURRENT_TEST`` as e.g.
    ``pycc/tests/test_090_ccsd_hessian.py::test_ccsd_hessian_hof_spinorbital_fc (call)``.
    Instrumenting a shared ``_check``-style helper therefore needs no change at the call
    sites: the molecule and route are already encoded in the test name.
    """
    raw = os.environ.get('PYTEST_CURRENT_TEST', '')
    name = raw.split('::')[-1].split(' ')[0] if raw else ''
    low = name.lower()
    mol = None
    for tag, label in (('h2op', 'H2O+'), ('c2h4p', 'C2H4+'), ('hof', 'HOF'),
                       ('h2o2', 'H2O2'), ('water', 'H2O'), ('ch4', 'CH4'),
                       ('h2o', 'H2O'), ('nh2', 'NH2')):
        if tag in low:
            mol = label
            break
    if 'spinorbital' in low or '_so' in low or low.endswith('_so'):
        route = 'spin-orbital'
    elif 'spatial' in low:
        route = 'spatial'
    else:
        route = None
    fc = 'frozen core' if ('_fc' in low or 'frozen' in low) else ('all-electron' if route else None)
    return {'test': name, 'molecule': mol, 'route': route, 'frozen_core': fc}
