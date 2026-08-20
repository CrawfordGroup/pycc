"""
derivatives.py: lazy MO-basis derivative-integral provider.

A thin, memory-conscious wrapper around Psi4's MintsHelper MO derivative-integral
routines (mo_*_deriv1/2). It serves the *skeleton* (fixed-MO-coefficient) first and
second derivatives of the one- and two-electron integrals, the overlap half-derivatives
(for AATs), and the dipole derivatives (for APTs), in the MO basis -- consistent with
PyCC's reference-implementation, MO-basis derivative-property formulations.

Block-label interface
---------------------
Both the spatial and the spin-orbital methods select the MO block(s) to transform into
by *label* rather than by passing coefficient matrices: ``'o'`` (occupied), ``'v'``
(virtual), or ``'all'`` (the full MO space, the default). The provider owns all the MO
bookkeeping -- the spatial methods slice the base's symmetry-handled ``self.wfn.C`` (a
single irrep block in global energy order, so symmetry stays on), while the spin-orbital
``so_*`` methods spin-block the spatial MO derivatives (built in the semicanonical MO
gauge from the spin-orbital Hamiltonian's ``Ca``/``Cb`` + ``spin``/``spat`` maps), so the
spin-orbital integrals live in the same MO gauge the spin-orbital densities do. The call
sites are therefore parallel between the two paths, e.g. ``d.core(atom, 'o', 'o')`` and
``d.so_core(atom, 'o', 'o')``.

Memory discipline
-----------------
One-electron derivatives are small (3*N_atom * n**2), so they are served per atom
directly. Two-electron derivatives (3*N_atom * n**4) are the heavy class: they are
computed one atom at a time (via :meth:`eri` / :meth:`iter_eri`), so the caller contracts
and discards each atom's block rather than ever materializing all 3*N_atom of them; a
caller wanting only the occupied block passes ``'o'`` to keep the transform at n_occ**4.

Lives on the Wavefunction base (lazy ``self.derivatives``); it depends only on base state
(the basis set, molecule, and MO coefficients).
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
import warnings
from typing import Any, List, Iterator, Tuple

import psi4
import numpy as np

from .exceptions import PyCCWarning


#: Persistent derivative-tensor store defaults.  Enabled by default (opt-out); set env var
#: ``PYCC_DERIV_STORE=0`` (or ``derivatives.DERIV_STORE_ENABLED = False``) to disable, and
#: ``PYCC_DERIV_STORE_DIR`` for the scratch directory.  The default follows ``tempfile`` (i.e.
#: ``$TMPDIR``, else ``/tmp``); if that resolves to a RAM-backed ``tmpfs`` the store consumes memory
#: instead of disk, so :meth:`DerivStore._ensure` warns (see :func:`_filesystem_type`).
DERIV_STORE_ENABLED = os.environ.get('PYCC_DERIV_STORE', '1') != '0'
DERIV_STORE_DIR = os.environ.get('PYCC_DERIV_STORE_DIR') or None

#: Filesystem types that live in RAM -- a store/scratch dir on one of these consumes memory, not disk.
_RAM_BACKED_FS = ('tmpfs', 'ramfs', 'devtmpfs')


def _filesystem_type(path: str, _mounts: str = '/proc/mounts'):
    """Best-effort filesystem type of the mount containing ``path``, via ``/proc/mounts`` (Linux).
    Returns the fstype string (e.g. ``'tmpfs'``, ``'ext4'``, ``'lustre'``) for the longest mount
    point that is a prefix of ``path``, or ``None`` when it cannot be determined (non-Linux, or any
    error) -- callers treat ``None`` as "unknown, do not warn".  ``_mounts`` is the mount table to
    read (overridable in tests)."""
    try:
        target = os.path.realpath(path)
        best_mp, best_type = '', None
        with open(_mounts) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp = parts[1].replace('\\040', ' ')     # /proc/mounts octal-escapes spaces
                if target == mp or target.startswith(mp.rstrip('/') + '/'):
                    if len(mp) >= len(best_mp):
                        best_mp, best_type = mp, parts[2]
        return best_type
    except Exception:
        return None


class DerivStore:
    """Persistent HDF5-backed store for four-index derivative tensors, keyed by
    ``(quantity, perturbation, context)``.

    Owned by :class:`Derivatives` (one per wavefunction), so a derivative tensor computed for one
    property is read back by any later property on the same wavefunction rather than recomputed --
    disk reads (~0.1 s for an ``nmo^4`` block) are far cheaper than re-derivation (seconds) and keep
    the tensors off RAM.  The HDF5 file is created lazily on first write and removed on
    :meth:`close` / deletion.  ``ctx`` is a hashable capturing everything the tensor depends on
    beyond the perturbation (e.g. ``(ncore, canonical)`` for the perturbed ERI derivative, the
    perturbed-MO gauge for amplitudes/densities)."""

    def __init__(self, enabled: bool = True, path: str = None) -> None:
        if enabled and importlib.util.find_spec('h5py') is None:
            # No h5py: degrade to the in-memory memo path rather than fail.  Correctness is
            # unchanged (RAM memo is bit-identical to the disk store); only the off-RAM /
            # cross-property-persistence benefit is lost until h5py is installed.
            warnings.warn("h5py not available; DerivStore is using in-memory memoization instead of "
                          "the on-disk store (install h5py for the off-RAM / cross-property benefit).",
                          RuntimeWarning, stacklevel=2)
            enabled = False
        self.enabled = enabled
        self._dir = path
        self._f = None                                  # lazy h5py.File
        self._file = None                               # temp-file path
        self._ram: dict = {}                            # in-memory memo when disabled

    def _ensure(self):
        if self._f is None:
            import h5py
            fd, self._file = tempfile.mkstemp(suffix='.h5', prefix='pycc_deriv_', dir=self._dir)
            os.close(fd)
            scratch = os.path.dirname(self._file)       # the directory tempfile actually used
            fstype = _filesystem_type(scratch)
            if fstype in _RAM_BACKED_FS:
                warnings.warn(
                    "PyCC's derivative-tensor cache directory %r is a RAM-backed filesystem (%s): "
                    "the cache will consume memory instead of disk and can exhaust the machine.  Set "
                    "PYCC_DERIV_STORE_DIR and TMPDIR to a real filesystem with free space."
                    % (scratch, fstype),
                    PyCCWarning, stacklevel=3)
            self._f = h5py.File(self._file, 'w')
        return self._f

    @staticmethod
    def _name(quantity, pert, ctx) -> str:
        raw = f"{quantity}|{pert!r}|{ctx!r}"
        return re.sub(r'[^A-Za-z0-9_.-]', '_', raw)     # HDF5-safe dataset name

    def get_or_compute(self, quantity, pert, builder, ctx=()):
        """Return the tensor for ``(quantity, pert, ctx)``: on a hit read it back, else call
        ``builder()`` and persist the result.  The store always memoizes -- to disk when enabled
        (off RAM, persistent across property calls), to an in-memory dict when disabled (matching
        the pre-store RAM caches).  Bit-identical either way."""
        name = self._name(quantity, pert, ctx)
        if not self.enabled:
            if name not in self._ram:
                self._ram[name] = np.asarray(builder())
            return self._ram[name]
        f = self._ensure()
        dset = f.get(name)
        if dset is not None:
            return dset[()]
        arr = np.asarray(builder())
        f.create_dataset(name, data=arr)
        return arr

    def get_or_compute_group(self, quantity, pert, builder, names, ctx=()):
        """Memoize a group of arrays produced together by a single ``builder()`` call (e.g. the
        three components of a perturbed-response record, of differing shapes).  ``names`` labels the
        components; ``builder()`` returns a tuple of arrays aligned with ``names``.  On a full hit
        every component is read back; otherwise ``builder()`` runs **once** and each component is
        persisted.  Returns the tuple.  All-or-nothing on the group, so a partial write never yields
        a stale mix."""
        full = [self._name(quantity, pert, tuple(ctx) + (n,)) for n in names]
        if not self.enabled:
            if all(n in self._ram for n in full):
                return tuple(self._ram[n] for n in full)
            vals = tuple(np.asarray(v) for v in builder())
            for n, v in zip(full, vals):
                self._ram[n] = v
            return vals
        f = self._ensure()
        if all(n in f for n in full):
            return tuple(f[n][()] for n in full)
        vals = tuple(np.asarray(v) for v in builder())
        for n, v in zip(full, vals):
            if n in f:
                del f[n]
            f.create_dataset(n, data=v)
        return vals

    def has(self, quantity, pert, ctx=()) -> bool:
        name = self._name(quantity, pert, ctx)
        return name in self._ram if not self.enabled else (self._f is not None and name in self._f)

    def close(self) -> None:
        self._ram.clear()
        if self._f is not None:
            self._f.close()
            self._f = None
        if self._file is not None:
            try:
                os.remove(self._file)
            except OSError:
                pass
            self._file = None

    def __del__(self):
        self.close()


def _complete_deriv2(chem: np.ndarray) -> np.ndarray:
    r"""Recover the complete symmetric two-electron second derivative from Psi4's raw output.

    For ``atom1 != atom2`` Psi4's ``mo_tei_deriv2`` returns *twice* the upper-triangular
    (canonically ordered) center-pair second derivatives of ``(mu nu|la si)`` and none of the
    lower-triangular ones.  The doubling makes it correct when contracted against the
    bra<->ket-symmetric two-particle density (the molecular Hessian), but the raw array is not
    itself the symmetric integral: it violates ``(mu nu|la si) = (la si|mu nu)`` by an O(1)
    amount.  Averaging over the bra<->ket swap supplies the missing lower-triangular pairs and
    cancels the factor of two, giving the true symmetric integral::

        (mu nu|la si)^(XY) = 0.5 * [ raw + raw.transpose(2, 3, 0, 1) ]

    Applied in chemist order (``chem``); the caller converts to physicist afterwards.  See
    ``docs/derivative_integral_permutational_symmetry.tex``.
    """
    return 0.5 * (chem + chem.transpose(2, 3, 0, 1))


class Derivatives(object):
    r"""MO-basis derivative-integral provider built on Psi4's MintsHelper.

    Parameters
    ----------
    wfn : Wavefunction
        provides the basis set (``wfn.H.basisset``), the molecule, the MO coefficients
        (``wfn.C``, or the spin-orbital Hamiltonian's ``Ca``/``Cb``), and the occupied /
        virtual block ranges (``wfn.o`` / ``wfn.v``).

    Notes
    -----
    Each ``deriv1`` call returns the three Cartesian (x, y, z) derivatives w.r.t. one
    ``atom``; ``deriv2`` calls return the nine (cart1, cart2) pairs (indexed
    ``cart1*3 + cart2``) for an ``(atom1, atom2)`` pair. Block labels ('o'/'v'/'all',
    default 'all') select the MO block(s) to transform into.

    A parenthesized superscript ``(X)`` (``X`` a nuclear Cartesian) denotes the *skeleton*
    derivative -- the partial derivative of the integral at *fixed* MO coefficients --
    transformed into the requested MO block(s)::

        A^(X)_pq... = C_mu,p C_nu,q ... (d A_mu,nu... / dX)

    .. math::

        A^{(X)}_{pq\cdots} = C^{\mu}_{p} C^{\nu}_{q}\cdots\,\frac{\partial A_{\mu\nu\cdots}}{\partial X}

    The orbital-response (CPHF) contribution is added downstream by the caller, not here.
    Second derivatives ``A^{(XY)}`` replace the single partial by :math:`\partial^2/\partial X\,\partial Y`.
    """

    def __init__(self, wfn: Any) -> None:
        """Bind the wavefunction, build the MintsHelper on its basis set, cache the
        molecule/atom count, and initialize the one-atom LRU for the heavy first-derivative
        two-electron MO transforms (see :meth:`_eri_cached`)."""
        self.wfn = wfn
        self.mints = psi4.core.MintsHelper(wfn.H.basisset)
        self.mol = wfn.ref.molecule()
        self.natom = self.mol.natom()
        # 1-atom LRU for the heavy first-derivative MO two-electron transforms (eri/so_eri):
        # hold only the most recent atom's blocks so an atom-outer sweep reuses a block across its
        # Cartesians and its several callers, without growing the deliberate one-atom footprint.
        self._d1_atom: Any = None
        self._d1_cache: dict = {}
        # Accumulating cache of nuclear-nuclear second-derivative skeletons, keyed by the canonical
        # atom pair (see :meth:`nuclear_hessian_skeletons`); persists for the whole molecular Hessian.
        self._d2int: dict = {}
        # Persistent, perturbation-keyed disk store for the large first-derivative four-index tensors
        # (skeleton ERI^(X), perturbed 2-PDM, perturbed amplitudes/Lambda, perturbed ERI deriv),
        # shared across property calls on this wavefunction (see :class:`DerivStore`).
        self.store = DerivStore(enabled=DERIV_STORE_ENABLED, path=DERIV_STORE_DIR)

    # ---- nuclear repulsion ----

    def nuclear_repulsion(self) -> np.ndarray:
        r"""Nuclear-repulsion-energy gradient, shape (natom, 3): the first derivatives of
        the nuclear repulsion E_nuc = sum_{A<B} Z_A Z_B / R_AB::

            dE_nuc / dX_A = -sum_{B!=A} Z_A Z_B (X_A - X_B) / R_AB^3

        .. math::

            E_\mathrm{nuc} = \sum_{A<B} \frac{Z_A Z_B}{R_{AB}},
            \qquad \frac{\partial E_\mathrm{nuc}}{\partial X_A}
                = -\sum_{B \ne A} Z_A Z_B\,\frac{X_A - X_B}{R_{AB}^{3}}
        """
        return np.asarray(self.mol.nuclear_repulsion_energy_deriv1())

    def nuclear_repulsion2(self) -> np.ndarray:
        r"""Nuclear-repulsion-energy Hessian, shape ``(3*natom, 3*natom)`` indexed
        ``(atom1*3 + cart1, atom2*3 + cart2)``: the second derivatives of the nuclear
        repulsion E_nuc = sum_{A<B} Z_A Z_B / R_AB::

            d2 E_nuc / dX_A dY_B

        .. math::

            E_\mathrm{nuc} = \sum_{A<B} \frac{Z_A Z_B}{R_{AB}},
            \qquad \frac{\partial^2 E_\mathrm{nuc}}{\partial X_A\,\partial Y_B}
        """
        return np.asarray(self.mol.nuclear_repulsion_energy_deriv2())

    # ---- spatial one-electron ----

    def overlap(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Overlap (S^(X)) skeleton derivatives for ``atom``: 3 (x, y, z) arrays,
        transformed into MO blocks (b1, b2)::

            S^(X)_pq = C_mu,p (dS_mu,nu / dX) C_nu,q

        .. math::

            S^{(X)}_{pq} = C^{\mu}_{p}\,\frac{\partial S_{\mu\nu}}{\partial X}\,C^{\nu}_{q}
        """
        return [np.asarray(m) for m in
                self.mints.mo_oei_deriv1("OVERLAP", atom, self._mo(b1), self._mo(b2))]

    def core(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Core one-electron (kinetic + potential) ``h^(X)`` skeleton derivatives for
        ``atom``: 3 (x, y, z) arrays::

            h^(X)_pq = C_mu,p (d(T+V)_mu,nu / dX) C_nu,q

        .. math::

            h^{(X)}_{pq} = C^{\mu}_{p}\,\frac{\partial (T+V)_{\mu\nu}}{\partial X}\,C^{\nu}_{q}
        """
        C1, C2 = self._mo(b1), self._mo(b2)
        T = self.mints.mo_oei_deriv1("KINETIC", atom, C1, C2)
        V = self.mints.mo_oei_deriv1("POTENTIAL", atom, C1, C2)
        return [np.asarray(t) + np.asarray(v) for t, v in zip(T, V)]

    def overlap_half(self, atom: int, b1: str = 'all', b2: str = 'all',
                     side: str = "LEFT") -> List[np.ndarray]:
        r"""Overlap half-derivatives for ``atom`` (``side`` = 'LEFT'/'RIGHT'): 3 arrays.
        Only the bra (``side='LEFT'``) or ket AO basis function is differentiated -- the MO
        coefficients and the other side are held fixed -- so the result is not symmetric in
        p, q and the two halves sum to the full overlap derivative,
        S^(X)_pq = S^(X)(LEFT)_pq + S^(X)(RIGHT)_pq. Used by the AAT machinery::

            S^(X)(LEFT)_pq = C_mu,p <d chi_mu / dX | chi_nu> C_nu,q

        .. math::

            S^{X,\mathrm{L}}_{pq} = C^{\mu}_{p}\,
                \Big\langle \tfrac{\partial \chi_\mu}{\partial X}\,\Big|\,\chi_\nu \Big\rangle\,C^{\nu}_{q}
        """
        return [np.asarray(m) for m in
                self.mints.mo_overlap_half_deriv1(side, atom, self._mo(b1), self._mo(b2))]

    def dipole(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Electric-dipole skeleton derivatives for ``atom``: 9 arrays, the
        d(mu_alpha)/d(X_beta) blocks (3 dipole components alpha x 3 Cartesians beta),
        for the APT machinery::

            mu^(X_beta)_pq,alpha = C_mu,p (d mu_alpha,mu,nu / dX_beta) C_nu,q

        .. math::

            \mu^{(X_\beta)}_{pq,\alpha} = C^{\mu}_{p}\,
                \frac{\partial (\mu_\alpha)_{\mu\nu}}{\partial X_\beta}\,C^{\nu}_{q}

        The AO-basis derivatives (``ao_elec_dip_deriv1``) are transformed into the MO
        block here rather than via ``mo_elec_dip_deriv1`` -- the latter segfaults on the
        linux build of Psi4 1.10.1 (the mo_oei/mo_tei deriv routines are unaffected)."""
        C1, C2 = self._mo(b1, as_array=True), self._mo(b2, as_array=True)
        return [C1.T @ np.asarray(m) @ C2 for m in self.mints.ao_elec_dip_deriv1(atom)]

    # ---- spatial two-electron ----

    def eri(self, atom: int, b1: str = 'all', b2: str = 'all',
            b3: str = 'all', b4: str = 'all') -> List[np.ndarray]:
        r"""Two-electron (ERI) skeleton derivatives for ``atom``: 3 (x, y, z) arrays, in
        physicist notation ``<pq|rs>^(X)`` (matching the stored integrals and
        :meth:`so_eri`, so callers need no transpose)::

            <pq|rs>^(X) = (pr|qs)^(X),
            (pr|qs)^(X) = C_mu,p C_nu,r (d(mu,nu|lam,sig) / dX) C_lam,q C_sig,s

        .. math::

            \langle pq|rs\rangle^{(X)} = (pr|qs)^{(X)}
                = C^{\mu}_{p} C^{\nu}_{r}\,
                \frac{\partial (\mu\nu|\lambda\sigma)}{\partial X}\,C^{\lambda}_{q} C^{\sigma}_{s}

        Psi4's ``mo_tei_deriv1`` returns the chemist integral ``(pq|rs)^(X)``; the
        ``swapaxes(1, 2)`` converts it to physicist order (as in :meth:`so_eri`, minus the
        spin-orbital antisymmetrization).

        Cached one atom at a time (:meth:`_eri_cached`) so an atom-outer sweep never holds
        more than one atom's block yet reuses the dominant ``nmo^4`` transform across the
        atom's Cartesians and its several callers."""
        return self._eri_cached(atom, ('eri', b1, b2, b3, b4), lambda: [
            np.asarray(m).swapaxes(1, 2) for m in self.mints.mo_tei_deriv1(
                atom, self._mo(b1), self._mo(b2), self._mo(b3), self._mo(b4))])

    def iter_eri(self, b1: str = 'all', b2: str = 'all', b3: str = 'all',
                 b4: str = 'all') -> Iterator[Tuple[int, List[np.ndarray]]]:
        """Yield ``(atom, [Ex, Ey, Ez])`` one atom at a time -- the lazy, non-
        materializing way to sweep the 2-e derivatives."""
        for atom in range(self.natom):
            yield atom, self.eri(atom, b1, b2, b3, b4)

    def ao_eri1(self, atom: int) -> List[np.ndarray]:
        r"""Raw **AO-basis** two-electron *first* derivative integrals for ``atom``: 3 arrays
        ``(mu nu|la si)^(X)`` (chemist order, un-transformed, indexed by Cartesian).  The
        first-derivative analogue of :meth:`ao_eri2`, and the AO source that :meth:`eri` transforms
        to MO.

        The memory-lean route for the gradient skeleton: hold these 3 ``nao**4`` AO blocks and
        contract them against a back-transformed effective 2-PDM (``Gamma_eff^AO``), instead of
        transforming the integrals to MO per atom (the ``mo_tei_deriv1`` transform of :meth:`eri`).
        Unlike ``ao_tei_deriv2``, ``ao_tei_deriv1`` is returned complete and fully permutationally
        symmetric (8-fold, including the bra<->ket and ket-pair swaps), so no completion
        (:func:`_complete_deriv2`) or ket reorder is needed before contraction."""
        out = []
        for m in self.mints.ao_tei_deriv1(atom):
            a = np.asarray(m)
            if a.ndim == 2:                          # (nbf*nbf) x (nbf*nbf) -> (nbf,nbf,nbf,nbf)
                n = int(round(a.shape[0] ** 0.5))
                a = a.reshape(n, n, n, n)
            out.append(a)
        return out

    # ---- spatial second derivatives (Hessian skeleton) ----

    def overlap2(self, atom1: int, atom2: int, b1: str = 'all',
                 b2: str = 'all') -> List[np.ndarray]:
        r"""Second overlap skeleton derivatives ``S^{(XY)}`` for the ``(atom1, atom2)`` pair:
        9 arrays, the (cart1, cart2) blocks indexed ``cart1*3 + cart2``::

            S^(XY)_pq = C_mu,p (d2 S_mu,nu / dX dY) C_nu,q

        .. math::

            S^{(XY)}_{pq} = C^{\mu}_{p}\,\frac{\partial^2 S_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{q}
        """
        return [np.asarray(m) for m in
                self.mints.mo_oei_deriv2("OVERLAP", atom1, atom2, self._mo(b1), self._mo(b2))]

    def core2(self, atom1: int, atom2: int, b1: str = 'all',
              b2: str = 'all') -> List[np.ndarray]:
        r"""Second core one-electron (kinetic + potential) derivatives ``h^{(XY)}`` for the
        ``(atom1, atom2)`` pair: 9 arrays, indexed ``cart1*3 + cart2``::

            h^(XY)_pq = C_mu,p (d2(T+V)_mu,nu / dX dY) C_nu,q

        .. math::

            h^{(XY)}_{pq} = C^{\mu}_{p}\,\frac{\partial^2 (T+V)_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{q}
        """
        C1, C2 = self._mo(b1), self._mo(b2)
        T = self.mints.mo_oei_deriv2("KINETIC", atom1, atom2, C1, C2)
        V = self.mints.mo_oei_deriv2("POTENTIAL", atom1, atom2, C1, C2)
        return [np.asarray(t) + np.asarray(v) for t, v in zip(T, V)]

    def eri2(self, atom1: int, atom2: int, b1: str = 'all', b2: str = 'all',
             b3: str = 'all', b4: str = 'all') -> List[np.ndarray]:
        r"""Second two-electron (ERI) derivatives for the ``(atom1, atom2)`` pair: 9 arrays,
        indexed ``cart1*3 + cart2``, in physicist notation ``<pq|rs>^(XY)`` (matching the
        stored integrals and :meth:`so_eri2`, so callers need no transpose)::

            <pq|rs>^(XY) = (pr|qs)^(XY),
            (pr|qs)^(XY) = C_mu,p C_nu,r (d2(mu,nu|lam,sig) / dX dY) C_lam,q C_sig,s

        .. math::

            \langle pq|rs\rangle^{(XY)} = (pr|qs)^{(XY)}
                = C^{\mu}_{p} C^{\nu}_{r}\,
                \frac{\partial^2 (\mu\nu|\lambda\sigma)}{\partial X\,\partial Y}\,C^{\lambda}_{q} C^{\sigma}_{s}

        The returned array is the *complete symmetric* integral: Psi4's raw ``mo_tei_deriv2``
        is completed over the bra<->ket swap by :func:`_complete_deriv2` (in chemist order),
        then ``swapaxes(1, 2)`` converts to physicist.  This matches :meth:`so_eri2`, so both
        accessors return integrals that already satisfy the electron-exchange symmetry and no
        caller need symmetrize.

        The Hessian skeleton needs only the occupied block, so callers pass ``'o'``
        (n_occ**4 per pair)."""
        return [_complete_deriv2(np.asarray(m)).swapaxes(1, 2) for m in self.mints.mo_tei_deriv2(
            atom1, atom2, self._mo(b1), self._mo(b2), self._mo(b3), self._mo(b4))]

    def ao_eri2(self, atom1: int, atom2: int) -> List[np.ndarray]:
        r"""Raw **AO-basis** two-electron second-derivative integrals for the ``(atom1, atom2)``
        pair: 9 arrays ``(mu nu|la si)^(XY)`` (chemist order, un-transformed, indexed
        ``cart1*3 + cart2``).  The AO analogue of the block that :meth:`eri2` transforms and
        completes.

        The memory-lean route for the nuclear-Hessian skeleton: hold these 9 ``nao**4`` AO blocks
        for a pair and transform ONE Cartesian pair at a time to the MO integral
        (:meth:`eri2_mo_component`), instead of materializing all 9 MO blocks at once (as
        ``mo_tei_deriv2`` does internally, holding AO + MO together).  Psi4 still builds all 9 AO
        blocks up front, so the AO floor is 9*``nao**4``; the saving is on the MO side (1 transient
        block instead of 9)."""
        out = []
        for m in self.mints.ao_tei_deriv2(atom1, atom2):
            a = np.asarray(m)
            if a.ndim == 2:                          # (nbf*nbf) x (nbf*nbf) -> (nbf,nbf,nbf,nbf)
                n = int(round(a.shape[0] ** 0.5))
                a = a.reshape(n, n, n, n)
            out.append(a)
        return out

    def eri2_mo_component(self, ao_chem: np.ndarray) -> np.ndarray:
        r"""Transform ONE raw AO chemist second-derivative block ``(mu nu|la si)^(XY)`` (from
        :meth:`ao_eri2`) into the physicist MO integral ``<pq|rs>^(XY)`` for that Cartesian pair.

        The full ``C`` (all MOs) is applied to each index by four BLAS-backed quarter transforms::

            (pq|rs)^(XY) = C_mu,p C_nu,q C_la,r C_si,s (mu nu|la si)^(XY)

        .. math::

            (pq|rs)^{(XY)} = C^{\mu}_{p} C^{\nu}_{q} C^{\lambda}_{r} C^{\sigma}_{s}\,
                (\mu\nu|\lambda\sigma)^{(XY)}

        then :func:`_complete_deriv2` supplies the bra<->ket average (Psi4's raw output is
        upper-triangular-doubled) and ``swapaxes(1, 2)`` converts chemist -> physicist -- the two
        steps :meth:`eri2` applies to ``mo_tei_deriv2``'s output.

        The AO ket pair is swapped (``transpose(0, 1, 3, 2)``) to reproduce ``mo_tei_deriv2``
        exactly.  This is NOT optional: ``mo_eri_helper`` transposes the ket pair internally, and
        while :func:`_complete_deriv2` averages the *bra<->ket* swap it does NOT symmetrize the
        *ket* pair, so ``complete()`` of the un-swapped transform is ket-asymmetric.  That cancels
        only against a ket-symmetric 2-PDM (CISD/MP2, which symmetrize their density) -- CCSD's
        cumulant ``Gam`` is not ket-symmetric, so the un-swapped block gives a wrong Hessian
        (~9e-3).  With the swap, this is bit-identical to :meth:`eri2`, method-independent.

        :func:`_complete_deriv2` is likewise required: although the raw integral is correct under a
        *symmetric-2-PDM* contraction (the ``Gam`` term), the skeleton also builds ``f^(XY)`` from
        the occupied trace (a Fock build, not a symmetric-density contraction), which the raw
        integral gets wrong (~4e-2)."""
        C = np.asarray(self.wfn.C)                    # AO x MO (all)
        c = self.wfn.contract
        t = np.asarray(ao_chem).transpose(0, 1, 3, 2)  # ket-pair layout -> mo_tei_deriv2 order
        t = c('mnls,mp->pnls', t, C)                  # nao^3 * nmo
        t = c('pnls,nq->pqls', t, C)                  # nao^2 * nmo^2
        t = c('pqls,lr->pqrs', t, C)                  # nao   * nmo^3
        t = c('pqrs,so->pqro', t, C)                  # (pq|rs) chemist, all MO
        return _complete_deriv2(t).swapaxes(1, 2)     # complete over bra<->ket, then physicist

    # ---- spin-orbital one-electron (spin-blocked from the spatial MO derivatives) ----

    def so_overlap(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital overlap ``S^(X)`` derivatives for ``atom``: 3 (x, y, z) arrays.
        Block-diagonal in spin (OVERLAP via :meth:`so_oei`)::

            S^(X)_pq = delta(spin_p, spin_q) C_mu,pbar (dS_mu,nu / dX) C_nu,qbar

        .. math::

            S^{(X)}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial S_{\mu\nu}}{\partial X}\,C^{\nu}_{\bar q}
        """
        return self.so_oei(atom, "OVERLAP", b1, b2)

    def so_oei(self, atom: int, kind: str, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital one-electron derivative integral (block-diagonal in spin): 3
        (x, y, z) arrays. ``kind`` is 'OVERLAP'/'KINETIC'/'POTENTIAL'.

        A spin-orbital one-electron integral vanishes unless bra and ket share a spin, so
        the derivative is block-diagonal: each same-spin block is the spatial MO derivative
        (in the semicanonical alpha/beta gauge), placed at that spin's positions::

            A^(X)_pq = delta(spin_p, spin_q) C_mu,pbar (dA_mu,nu / dX) C_nu,qbar

        .. math::

            A^{(X)}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial A_{\mu\nu}}{\partial X}\,C^{\nu}_{\bar q}

        with :math:`\sigma_p` the spin of spin-orbital p and :math:`\bar p` its spatial
        orbital. (:meth:`so_core` = KINETIC + POTENTIAL; :meth:`so_overlap` = OVERLAP.)
        """
        n1, a1, b1p, Ca1, Cb1 = self._so_mo(b1)
        n2, a2, b2p, Ca2, Cb2 = self._so_mo(b2)
        Da = self.mints.mo_oei_deriv1(kind, atom, Ca1, Ca2)
        Db = self.mints.mo_oei_deriv1(kind, atom, Cb1, Cb2)
        out = []
        for c in range(3):
            M = np.zeros((n1, n2))
            M[np.ix_(a1, a2)] = np.asarray(Da[c])
            M[np.ix_(b1p, b2p)] = np.asarray(Db[c])
            out.append(M)
        return out

    def so_core(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital core (kinetic + potential) ``h^(X)`` derivatives for ``atom``: 3
        (x, y, z) arrays. Block-diagonal in spin (KINETIC + POTENTIAL via :meth:`so_oei`)::

            h^(X)_pq = delta(spin_p, spin_q) C_mu,pbar (d(T+V)_mu,nu / dX) C_nu,qbar

        .. math::

            h^{(X)}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial (T+V)_{\mu\nu}}{\partial X}\,C^{\nu}_{\bar q}
        """
        T = self.so_oei(atom, "KINETIC", b1, b2)
        V = self.so_oei(atom, "POTENTIAL", b1, b2)
        return [t + v for t, v in zip(T, V)]

    def so_overlap_half(self, atom: int, b1: str = 'all', b2: str = 'all',
                        side: str = "LEFT") -> List[np.ndarray]:
        r"""Spin-orbital overlap half-derivatives ``<phi^(X)_p | phi_q>`` (block-diagonal in
        spin): 3 arrays. Bra perturbed, ket unperturbed (``side='LEFT'``); not symmetric.
        Used by the spin-orbital AAT machinery::

            S^(X)(LEFT)_pq = delta(spin_p, spin_q) C_mu,pbar <d chi_mu / dX | chi_nu> C_nu,qbar

        .. math::

            S^{X,\mathrm{L}}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\Big\langle \tfrac{\partial \chi_\mu}{\partial X}\,\Big|\,\chi_\nu \Big\rangle\,C^{\nu}_{\bar q}
        """
        n1, a1, b1p, Ca1, Cb1 = self._so_mo(b1)
        n2, a2, b2p, Ca2, Cb2 = self._so_mo(b2)
        La = self.mints.mo_overlap_half_deriv1(side, atom, Ca1, Ca2)
        Lb = self.mints.mo_overlap_half_deriv1(side, atom, Cb1, Cb2)
        out = []
        for c in range(3):
            M = np.zeros((n1, n2))
            M[np.ix_(a1, a2)] = np.asarray(La[c])
            M[np.ix_(b1p, b2p)] = np.asarray(Lb[c])
            out.append(M)
        return out

    def so_dipole(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital MO electric-dipole skeleton derivatives for ``atom``: 9 arrays
        indexed ``alpha*3 + beta`` (dipole component alpha x Cartesian beta),
        block-diagonal in spin. AO route then spin-blocked, as in :meth:`dipole`::

            mu^(X_beta)_pq,alpha = delta(spin_p, spin_q) C_mu,pbar (d mu_alpha,mu,nu / dX_beta) C_nu,qbar

        .. math::

            \mu^{(X_\beta)}_{pq,\alpha} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial (\mu_\alpha)_{\mu\nu}}{\partial X_\beta}\,C^{\nu}_{\bar q}
        """
        n1, a1, b1p, Ca1, Cb1 = self._so_mo(b1)
        n2, a2, b2p, Ca2, Cb2 = self._so_mo(b2)
        npCa1, npCa2 = np.asarray(Ca1), np.asarray(Ca2)
        npCb1, npCb2 = np.asarray(Cb1), np.asarray(Cb2)
        aod = self.mints.ao_elec_dip_deriv1(atom)   # 9 x (nao, nao)
        out = []
        for c in range(9):
            ao = np.asarray(aod[c])
            M = np.zeros((n1, n2))
            M[np.ix_(a1, a2)] = npCa1.T @ ao @ npCa2
            M[np.ix_(b1p, b2p)] = npCb1.T @ ao @ npCb2
            out.append(M)
        return out

    # ---- spin-orbital two-electron ----

    def _so_eri_blocks(self, blocks):
        """Per-index ``(size, [(alpha_pos, Ca), (beta_pos, Cb)])`` selectors for the four
        block labels of a spin-orbital ERI -- the shared spin-blocking bookkeeping for
        :meth:`so_eri` / :meth:`so_eri2`."""
        info = [self._so_mo(b) for b in blocks]
        shape = tuple(x[0] for x in info)
        sel = [[(x[1], x[3]), (x[2], x[4])] for x in info]   # [alpha:(pos,C), beta:(pos,C)]
        return shape, sel

    def so_eri(self, atom: int, b1: str = 'all', b2: str = 'all',
               b3: str = 'all', b4: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital antisymmetrized two-electron derivatives ``<pq||rs>^(X)`` for
        ``atom``: 3 (x, y, z) arrays. Spin-blocks the spatial chemist derivative integrals
        over the spin-conserving combinations ``(s12, s34)``, converts to physicist
        notation (``<pq|rs> = (pr|qs)``, the ``swapaxes(1,2)``), and antisymmetrizes over
        the ket (the ``phys - phys.swapaxes(2,3)``)::

            <pq||rs>^(X) = <pq|rs>^(X) - <pq|sr>^(X),   <pq|rs>^(X) = (pr|qs)^(X)  (chemist)

        .. math::

            \langle pq\Vert rs\rangle^{(X)} = \langle pq|rs\rangle^{(X)} - \langle pq|sr\rangle^{(X)},
            \qquad \langle pq|rs\rangle^{(X)} = (pr|qs)^{(X)}

        Cached one atom at a time (:meth:`_eri_cached`): the four spin-block
        ``mo_tei_deriv1`` transforms are the dominant cost and are otherwise re-run by every
        caller for the atom."""
        def compute():
            shape, sel = self._so_eri_blocks((b1, b2, b3, b4))
            chem = [np.zeros(shape) for _ in range(3)]
            for s12 in (0, 1):
                p1, C1 = sel[0][s12]
                p2, C2 = sel[1][s12]
                if not (p1.size and p2.size):
                    continue
                for s34 in (0, 1):
                    p3, C3 = sel[2][s34]
                    p4, C4 = sel[3][s34]
                    if not (p3.size and p4.size):
                        continue
                    G = self.mints.mo_tei_deriv1(atom, C1, C2, C3, C4)
                    for c in range(3):
                        chem[c][np.ix_(p1, p2, p3, p4)] = np.asarray(G[c])
            out = []
            for ch in chem:
                phys = ch.swapaxes(1, 2)
                out.append(phys - phys.swapaxes(2, 3))
            return out
        return self._eri_cached(atom, ('so_eri', b1, b2, b3, b4), compute)

    # ---- spin-orbital second derivatives ----

    def so_overlap2(self, atom1: int, atom2: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital second overlap derivatives ``S^{(XY)}`` for the ``(atom1, atom2)``
        pair: 9 arrays. Block-diagonal in spin (OVERLAP via :meth:`so_oei2`)::

            S^(XY)_pq = delta(spin_p, spin_q) C_mu,pbar (d2 S_mu,nu / dX dY) C_nu,qbar

        .. math::

            S^{(XY)}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial^2 S_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{\bar q}
        """
        return self.so_oei2(atom1, atom2, "OVERLAP", b1, b2)

    def so_oei2(self, atom1: int, atom2: int, kind: str, b1: str = 'all',
                b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital one-electron *second* derivative (block-diagonal in spin) for the
        ``(atom1, atom2)`` pair: 9 arrays indexed ``cart1*3 + cart2``::

            A^(XY)_pq = delta(spin_p, spin_q) C_mu,pbar (d2 A_mu,nu / dX dY) C_nu,qbar

        .. math::

            A^{(XY)}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial^2 A_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{\bar q}

        (:meth:`so_core2` = KINETIC + POTENTIAL; :meth:`so_overlap2` = OVERLAP.)
        """
        n1, a1, b1p, Ca1, Cb1 = self._so_mo(b1)
        n2, a2, b2p, Ca2, Cb2 = self._so_mo(b2)
        Da = self.mints.mo_oei_deriv2(kind, atom1, atom2, Ca1, Ca2)
        Db = self.mints.mo_oei_deriv2(kind, atom1, atom2, Cb1, Cb2)
        out = []
        for c in range(9):
            M = np.zeros((n1, n2))
            M[np.ix_(a1, a2)] = np.asarray(Da[c])
            M[np.ix_(b1p, b2p)] = np.asarray(Db[c])
            out.append(M)
        return out

    def so_core2(self, atom1: int, atom2: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital second core (kinetic + potential) derivatives ``h^{(XY)}`` for the
        ``(atom1, atom2)`` pair: 9 arrays. Block-diagonal in spin (KINETIC + POTENTIAL via
        :meth:`so_oei2`)::

            h^(XY)_pq = delta(spin_p, spin_q) C_mu,pbar (d2(T+V)_mu,nu / dX dY) C_nu,qbar

        .. math::

            h^{(XY)}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial^2 (T+V)_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{\bar q}
        """
        T = self.so_oei2(atom1, atom2, "KINETIC", b1, b2)
        V = self.so_oei2(atom1, atom2, "POTENTIAL", b1, b2)
        return [t + v for t, v in zip(T, V)]

    def so_eri2(self, atom1: int, atom2: int, b1: str = 'all', b2: str = 'all',
                b3: str = 'all', b4: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital antisymmetrized two-electron *second* derivatives ``<pq||rs>^{(XY)}``
        for the ``(atom1, atom2)`` pair: 9 arrays indexed ``cart1*3 + cart2``::

            <pq||rs>^(XY) = <pq|rs>^(XY) - <pq|sr>^(XY),   <pq|rs>^(XY) = (pr|qs)^(XY)  (chemist)

        .. math::

            \langle pq\Vert rs\rangle^{(XY)} = \langle pq|rs\rangle^{(XY)} - \langle pq|sr\rangle^{(XY)},
            \qquad \langle pq|rs\rangle^{(XY)} = (pr|qs)^{(XY)}

        The chemist integral is completed over the bra<->ket swap by :func:`_complete_deriv2`
        (Psi4's raw ``mo_tei_deriv2`` is twice the upper-triangular center-pairs -- see that
        helper) before conversion to physicist and ket antisymmetrization; the completion
        assumes matching bra/ket block pairs (``b1,b2`` == ``b3,b4``), as in the occupied-block
        Hessian use. All four spin combinations are built independently."""
        shape, sel = self._so_eri_blocks((b1, b2, b3, b4))
        chem = [np.zeros(shape) for _ in range(9)]
        for s12 in (0, 1):
            p1, C1 = sel[0][s12]
            p2, C2 = sel[1][s12]
            if not (p1.size and p2.size):
                continue
            for s34 in (0, 1):
                p3, C3 = sel[2][s34]
                p4, C4 = sel[3][s34]
                if not (p3.size and p4.size):
                    continue
                G = self.mints.mo_tei_deriv2(atom1, atom2, C1, C2, C3, C4)
                for c in range(9):
                    chem[c][np.ix_(p1, p2, p3, p4)] = np.asarray(G[c])
        out = []
        for ch in chem:
            phys = _complete_deriv2(ch).swapaxes(1, 2)   # complete over bra<->ket, then physicist
            out.append(phys - phys.swapaxes(2, 3))
        return out

    def so_eri2_mo_component(self, ao_chem: np.ndarray) -> np.ndarray:
        r"""Build ONE spin-orbital block ``<pq||rs>^(XY)`` for a Cartesian pair from one raw
        *spatial* AO chemist second-derivative block (from :meth:`ao_eri2`) -- the spin-orbital
        analogue of :meth:`eri2_mo_component`, reproducing one component of :meth:`so_eri2`.

        Memory-lean route for the spin-orbital Hessian skeleton: hold the 9 *spatial* AO blocks
        (``nao**4``) and build one SO block (``(2 nmo)**4``) at a time, instead of the 9 SO blocks
        :meth:`so_eri2` materializes together (``9 * 16 * nmo**4``).  Each of the four same-spin
        combinations is the spatial transform of the AO block placed at that spin's positions; then
        :func:`_complete_deriv2` completes the bra<->ket swap, ``swapaxes(1, 2)`` -> physicist, and
        ``- swapaxes(2, 3)`` antisymmetrizes -- exactly the steps of :meth:`so_eri2`.

        Unlike the spin-adapted :meth:`eri2_mo_component`, the AO ket pair IS swapped
        (``transpose(0, 1, 3, 2)``) to match ``mo_tei_deriv2``'s convention, so each spin block is
        bit-identical to :meth:`so_eri2`'s: the trailing ``- swapaxes(2, 3)`` antisymmetrization
        makes the ket order matter here (a straight transform would flip the sign of the exchange
        term)."""
        c = self.wfn.contract
        ao = np.asarray(ao_chem).transpose(0, 1, 3, 2)   # ket-pair layout -> mo_tei_deriv2 order
        shape, sel = self._so_eri_blocks(('all', 'all', 'all', 'all'))
        ch = np.zeros(shape)
        for s12 in (0, 1):
            p1, C1 = sel[0][s12]
            p2, C2 = sel[1][s12]
            if not (p1.size and p2.size):
                continue
            C1a, C2a = np.asarray(C1), np.asarray(C2)
            for s34 in (0, 1):
                p3, C3 = sel[2][s34]
                p4, C4 = sel[3][s34]
                if not (p3.size and p4.size):
                    continue
                C3a, C4a = np.asarray(C3), np.asarray(C4)
                t = c('mnls,mp->pnls', ao, C1a)          # spatial transform for this spin combo
                t = c('pnls,nq->pqls', t, C2a)
                t = c('pqls,lr->pqrs', t, C3a)
                t = c('pqrs,so->pqro', t, C4a)
                ch[np.ix_(p1, p2, p3, p4)] = t           # place into the SO layout
        phys = _complete_deriv2(ch).swapaxes(1, 2)
        return phys - phys.swapaxes(2, 3)

    # ---- nuclear-nuclear skeleton second-derivative integrals (for the 2n+1 molecular Hessian) ----

    def nuclear_hessian_skeletons(self, a1: int, a2: int, cache: bool = True) -> dict:
        r"""Cached nuclear-nuclear skeleton second-derivative integrals for the atom pair
        ``(a1, a2)``: the 9 ``(cart1, cart2)`` blocks of the core Hamiltonian ``h^{XY}``, the
        overlap ``S^{XY}``, and the two-electron ``<pq||rs>^{XY}`` (in the basis's ERI
        convention). The ``mo_*_deriv2`` calls -- ``mo_tei_deriv2`` especially -- are shared
        across a pair's 3x3 Cartesian blocks, so compute once per atom pair (not per coordinate
        pair). Returns ``{'core','overlap','eri'}`` -> lists of 9 arrays (indexed ``c1*3+c2``)::

            core -> h^(XY)_pq,   overlap -> S^(XY)_pq,   eri -> <pq||rs>^(XY)

        .. math::

            h^{(XY)}_{pq}, \qquad S^{(XY)}_{pq}, \qquad \langle pq\Vert rs\rangle^{(XY)}

        The mixed second derivative is symmetric under the atom-pair swap
        (``d^2/dA_i dB_j = d^2/dB_j dA_i``), so the cache is keyed on the canonical ``(min, max)``
        pair; a reversed request ``(a2, a1)`` returns the stored arrays with the ``3x3`` Cartesian
        grid transposed (``comp c1*3+c2 -> c2*3+c1``) -- no tensor is copied. Only the upper
        triangle of atom pairs is stored and computed (halving both). For a diagonal pair
        (``a1 == a2``) the two derivatives are on one atom, so only the 6 ``c1 <= c2`` Cartesian
        components are unique and the 3 lower-triangle components alias their partners.

        ``cache=False`` computes the block and returns it WITHOUT storing it in ``_d2int`` -- for the
        atom-pair-outer Hessian assembly, which visits each unique pair once and discards it, so the
        (natom^2-scale) accumulating cache would only waste memory.
        """
        lo, hi = (a1, a2) if a1 <= a2 else (a2, a1)
        key = (lo, hi)
        blk = self._d2int.get(key) if cache else None
        if blk is None:
            so = self.wfn.orbital_basis == 'spinorbital'
            if so:
                core = [np.asarray(m) for m in self.so_core2(lo, hi)]
                overlap = [np.asarray(m) for m in self.so_overlap2(lo, hi)]
                eri = [np.asarray(m) for m in self.so_eri2(lo, hi)]     # <pq||rs>^{XY} (antisym)
            else:
                core = [np.asarray(m) for m in self.core2(lo, hi)]
                overlap = [np.asarray(m) for m in self.overlap2(lo, hi)]
                eri = [np.asarray(m) for m in self.eri2(lo, hi)]        # physicist <pq|rs>^{XY}
            blk = {'core': core, 'overlap': overlap, 'eri': eri}
            if lo == hi:                        # same atom: comp (c1,c2) == (c2,c1), alias the dupes
                for arrs in blk.values():
                    for c1 in range(3):
                        for c2 in range(c1):
                            arrs[c1 * 3 + c2] = arrs[c2 * 3 + c1]
            if cache:
                self._d2int[key] = blk
        if a1 <= a2:
            return blk
        # reversed request: d^2/d(a1,c1)d(a2,c2) is the stored d^2/d(lo,c2)d(hi,c1) -- transpose comps
        return {name: [arrs[c2 * 3 + c1] for c1 in range(3) for c2 in range(3)]
                for name, arrs in blk.items()}

    # ---- MO block selection & caching (private helpers) ----

    def _mo(self, block: str, as_array: bool = False):
        """Spatial MO coefficients (AO x block) for a block label ('o'/'v'/'all').
        Returns a Psi4 ``Matrix`` (what the mints deriv routines expect) by default, or the
        raw NumPy array when ``as_array=True`` (used by :meth:`dipole` for its AO->MO
        matmul)."""
        C = np.asarray(self.wfn.C)
        if block == 'o':
            C = C[:, :self.wfn.no]
        elif block == 'v':
            C = C[:, self.wfn.no:]
        # 'all' -> the full C
        return C if as_array else psi4.core.Matrix.from_array(C)

    def _so_mo(self, block: str):
        """For a spin-orbital block label, return ``(n, a, b, Ca, Cb)``: the block size,
        the alpha/beta spin-orbital positions *within the block*, and the alpha/beta
        semicanonical MOs (Psi4 matrices) pre-sliced to that block's spatial columns (so
        the mints transforms return arrays already in block order)."""
        H = self.wfn.H
        spin = np.asarray(H.spin)
        spat = np.asarray(H.spat)
        idx = {'o': self.wfn.o, 'v': self.wfn.v}.get(block, slice(None))
        spin_b = spin[idx]
        spat_b = spat[idx]
        a = np.where(spin_b == 0)[0]
        b = np.where(spin_b == 1)[0]
        Ca = psi4.core.Matrix.from_array(np.asarray(H.Ca)[:, spat_b[a]])
        Cb = psi4.core.Matrix.from_array(np.asarray(H.Cb)[:, spat_b[b]])
        return len(spin_b), a, b, Ca, Cb

    def _eri_cached(self, atom: int, key, compute):
        """Return ``compute()`` for ``(atom, key)`` from the 1-atom cache, evicting the
        previous atom on change. ``key`` distinguishes the transform variant (eri/so_eri) and
        the MO blocks. The dominant cost is ``psi4.core.mo_tei_deriv1`` (the ``nmo^4`` MO
        transform), which every caller for a given atom otherwise re-runs; this reuses it
        across the atom's three Cartesians and callers. The cached arrays are treated
        read-only (callers already build new arrays via swapaxes/arithmetic).

        When the persistent :class:`DerivStore` is enabled the 3-Cartesian result is memoized
        there instead (per atom + transform/blocks) -- off RAM and reused across property calls;
        otherwise the 1-atom LRU (evict-on-atom-change) is used."""
        if self.store.enabled:
            stack = self.store.get_or_compute('eri1', atom, lambda: np.asarray(compute()), ctx=key)
            return list(stack)
        if atom != self._d1_atom:
            self._d1_atom = atom
            self._d1_cache = {}
        if key not in self._d1_cache:
            self._d1_cache[key] = compute()
        return self._d1_cache[key]
