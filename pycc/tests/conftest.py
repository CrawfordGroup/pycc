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
