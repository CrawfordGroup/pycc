"""
checkpoint.py: a full-precision property archive for a single derivative-driver point.

A checkpoint bundles everything computed on one ``CCderiv`` / ``MPderiv`` / ``CIderiv`` object --
the molecule (at its exact frame and masses), the basis/method/energy context, and the analytic
property tensors (each as its ``nuclear + reference + correlation`` breakdown) -- into a single
``.npz`` file at full float64 precision.  The expensive part of a calculation is producing those
tensors; a checkpoint lets the cheap downstream analysis (vibrational, VCD, thermochemistry, ...)
run later without repeating the CC solve.  See :func:`~pycc.vibanalysis.harmonic_analysis`, which
accepts a checkpoint directly.
"""

from __future__ import annotations

import json
from typing import Any, List

import numpy as np

#: Checkpoint schema version stored in the ``meta`` blob; bump on any incompatible layout change.
FORMAT_VERSION = 1


def _reference_label(scf_wfn: Any) -> str:
    """RHF / ROHF / UHF from the psi4 SCF wavefunction's spin structure."""
    if scf_wfn.same_a_b_orbs() and scf_wfn.same_a_b_dens():
        return 'RHF'
    if scf_wfn.same_a_b_orbs():
        return 'ROHF'
    return 'UHF'


def _correlation_energy(wfn: Any):
    """The correlation energy stored on the pycc wavefunction (``ecc`` / ``emp2`` / ``eci``), or
    ``None`` if not found."""
    for attr in ('ecc', 'emp2', 'eci'):
        e = getattr(wfn, attr, None)
        if e is not None:
            return float(e)
    return None


def save_checkpoint(source: Any, path: str, properties: List[str] = None) -> str:
    """Harvest the property results already computed on ``source`` and write them, with the
    molecular/method context, to a ``.npz`` checkpoint at full precision.

    ``save_checkpoint`` does **not** compute anything: the property facades (``pycc.hessian(d)``,
    ``pycc.apt(d)``, ...) record their :class:`~pycc.properties.PropertyComponents` on the
    wavefunction as they run (with their normal reports printed), and this collects them.  So the
    workflow is: call the dispatchers you want (seeing their output), then save.

    Parameters
    ----------
    source : CCderiv | MPderiv | CIderiv | wavefunction
        the driver or wavefunction whose recorded results to harvest (resolved to the underlying
        wavefunction, where the facades record).
    path : str
        output file (``.npz`` is appended if absent).
    properties : list of str, optional
        which recorded properties to store; ``None`` (default) stores every property recorded so
        far.  A named property that has not been computed yet is an error.

    Returns
    -------
    str
        the path actually written.
    """
    from .properties import _wavefunction

    wfn = _wavefunction(source)
    results = getattr(wfn, '_property_results', {})
    if properties is None:
        names = list(results.keys())
    else:
        names = list(properties)
        missing = [n for n in names if n not in results]
        if missing:
            raise ValueError(
                "save_checkpoint: these properties have not been computed on this wavefunction "
                "yet: %s.  Call the dispatcher(s) first, e.g. pycc.hessian(d) / pycc.apt(d)." % missing)
    if not names:
        raise ValueError(
            "save_checkpoint: no property results recorded on this wavefunction.  Compute some "
            "first (e.g. pycc.hessian(d), pycc.apt(d)); the facades record their results and "
            "save_checkpoint harvests them.")

    ref = wfn.ref
    mol = ref.molecule()
    natom = mol.natom()

    arrays = {
        'geometry': np.asarray(mol.geometry(), dtype=float),                 # (natom, 3), bohr
        'atomic_numbers': np.array([mol.Z(i) for i in range(natom)], dtype=float),
        'masses': np.array([mol.mass(i) for i in range(natom)], dtype=float),  # u
    }

    # Per-property arrays keyed by list index (so a parametrized name like
    # 'polarizability(omega=0.07)' never has to be a .npz entry name); real names live in meta.
    property_meta = {}
    for i, name in enumerate(names):
        pc = results[name]
        arrays['p%d__nuclear' % i] = np.asarray(pc.nuclear, dtype=float)
        arrays['p%d__reference' % i] = np.asarray(pc.reference, dtype=float)
        arrays['p%d__correlation' % i] = np.asarray(pc.correlation, dtype=float)
        if getattr(pc, 'origin', None) is not None:
            property_meta[name] = {'origin': list(pc.origin)}

    ecorr = _correlation_energy(wfn)
    meta = {
        'format_version': FORMAT_VERSION,
        'method': getattr(wfn, 'method', None),
        'orbital_basis': getattr(wfn, 'orbital_basis', None),
        'basis': ref.basisset().name(),
        'reference': _reference_label(ref),
        'nfzc': int(getattr(wfn, 'nfzc', 0)),
        'scf_energy': float(wfn.eref),
        'correlation_energy': ecorr,
        'total_energy': (float(wfn.eref) + ecorr) if ecorr is not None else None,
        'charge': int(mol.molecular_charge()),
        'multiplicity': int(mol.multiplicity()),
        'symbols': [mol.symbol(i) for i in range(natom)],
        'properties': names,
        'property_meta': property_meta,
    }
    arrays['meta'] = np.array(json.dumps(meta))

    if not path.endswith('.npz'):
        path += '.npz'
    np.savez(path, **arrays)
    return path


class Checkpoint:
    """A loaded checkpoint (see :func:`load_checkpoint`).

    Exposes the molecular/method context as attributes (``method``, ``basis``, ``reference``,
    ``orbital_basis``, ``nfzc``, ``scf_energy``, ``correlation_energy``, ``total_energy``,
    ``charge``, ``multiplicity``), the geometry data (``geometry`` bohr, ``atomic_numbers``,
    ``masses`` u, ``symbols``), the full ``PropertyComponents`` per property in ``components``,
    and convenience total-tensor accessors (``hessian``, ``apt``, ``apt_velocity``, ``aat``,
    ``gradient``, ``polarizability``, ``dipole``) that return the ``.total`` array or ``None`` if absent.
    ``molecule`` rebuilds the psi4 molecule lazily at the stored frame and masses.
    """

    _TOTALS = ('gradient', 'hessian', 'apt', 'apt_velocity', 'polarizability', 'dipole', 'aat')

    def __init__(self, meta: dict, geometry: np.ndarray, atomic_numbers: np.ndarray,
                 masses: np.ndarray, components: dict) -> None:
        self.meta = meta
        self.geometry = geometry
        self.atomic_numbers = atomic_numbers
        self.masses = masses
        self.symbols = meta['symbols']
        self.components = components
        for k in ('method', 'basis', 'reference', 'orbital_basis', 'nfzc', 'scf_energy',
                  'correlation_energy', 'total_energy', 'charge', 'multiplicity'):
            setattr(self, k, meta.get(k))
        self._molecule = None

    def __getattr__(self, name: str):
        # convenience total-tensor accessors, resolved only if not a real attribute
        if name in Checkpoint._TOTALS:
            comps = self.__dict__.get('components', {})
            return comps[name].total if name in comps else None
        raise AttributeError(name)

    def get(self, name: str):
        """The full :class:`~pycc.properties.PropertyComponents` for a stored property."""
        return self.components[name]

    @property
    def molecule(self):
        """The psi4 molecule, rebuilt lazily at the stored Cartesian frame (bohr, no reorient/COM)
        with the stored masses set explicitly (so isotopic labels survive the round trip)."""
        if self._molecule is None:
            import psi4
            lines = ["%d %d" % (self.charge, self.multiplicity)]
            for s, xyz in zip(self.symbols, self.geometry):
                lines.append("%s %.15f %.15f %.15f" % (s, xyz[0], xyz[1], xyz[2]))
            lines += ["units bohr", "no_com", "no_reorient", "symmetry c1"]
            mol = psi4.geometry("\n".join(lines))
            mol.update_geometry()
            for i, m in enumerate(self.masses):
                mol.set_mass(i, float(m))
            self._molecule = mol
        return self._molecule


def load_checkpoint(path: str) -> Checkpoint:
    """Load a checkpoint written by :func:`save_checkpoint` into a :class:`Checkpoint`."""
    from .properties import PropertyComponents

    data = np.load(path, allow_pickle=False)
    meta = json.loads(data['meta'].item())

    components = {}
    for i, name in enumerate(meta['properties']):
        origin = meta.get('property_meta', {}).get(name, {}).get('origin')
        components[name] = PropertyComponents(
            nuclear=data['p%d__nuclear' % i],
            reference=data['p%d__reference' % i],
            correlation=data['p%d__correlation' % i],
            origin=tuple(origin) if origin is not None else None,
        )
    return Checkpoint(meta, data['geometry'], data['atomic_numbers'], data['masses'], components)
