"""
derivatives.py: lazy MO-basis derivative-integral provider.

A thin, memory-conscious wrapper around Psi4's MintsHelper MO derivative-integral
routines (mo_*_deriv1). It serves the *skeleton* (fixed-MO-coefficient) first
derivatives of the one- and two-electron integrals, the overlap half-derivatives
(for AATs), and the dipole derivatives (for APTs), in the MO basis -- consistent
with PyCC's reference-implementation, MO-basis derivative-property formulations.

The caller supplies the MO-coefficient blocks to transform into. Those should be
the base's symmetry-handled ``self.C`` (or a slice of it): it is a single irrep
block in global energy order, so derivative integrals work with molecular symmetry
left on -- no need to drop to C1.

Memory discipline
-----------------
One-electron derivatives are small (3*N_atom * nmo**2), so they are served per
atom directly. Two-electron derivatives (3*N_atom * nmo**4) are the heavy class:
they are computed one atom at a time (via :meth:`eri` / :meth:`iter_eri`), so the
caller contracts and discards each atom's block rather than ever materializing all
3*N_atom of them.

Created and owned by HFwfn for now; it depends only on base state (the basis set,
molecule, and MO coefficients), so it can be promoted to the Wavefunction base when
MP2/CI/CC derivative code needs it too.
"""

from __future__ import annotations

from typing import Any, List, Iterator, Tuple

import psi4
import numpy as np


class Derivatives(object):
    """MO-basis derivative-integral provider built on Psi4's MintsHelper.

    Parameters
    ----------
    wfn : Wavefunction
        provides the basis set (``wfn.H.basisset``) and the molecule.

    Notes
    -----
    Each ``deriv1`` call returns the three Cartesian (x, y, z) derivatives with
    respect to the coordinates of a single ``atom``; the returned arrays are NumPy
    arrays in the MO block defined by the supplied coefficient matrices.
    """

    def __init__(self, wfn: Any) -> None:
        self.mints = psi4.core.MintsHelper(wfn.H.basisset)
        self.mol = wfn.ref.molecule()
        self.natom = self.mol.natom()

    # ---- one-electron (small) ----
    def overlap(self, atom: int, C1, C2) -> List[np.ndarray]:
        """Overlap (Sx) derivatives for ``atom``: list of 3 (x, y, z) arrays."""
        return [np.asarray(m) for m in self.mints.mo_oei_deriv1("OVERLAP", atom, C1, C2)]

    def core(self, atom: int, C1, C2) -> List[np.ndarray]:
        """Core one-electron (kinetic + potential) hx derivatives for ``atom``."""
        T = self.mints.mo_oei_deriv1("KINETIC", atom, C1, C2)
        V = self.mints.mo_oei_deriv1("POTENTIAL", atom, C1, C2)
        return [np.asarray(t) + np.asarray(v) for t, v in zip(T, V)]

    def overlap_half(self, atom: int, C1, C2, side: str = "LEFT") -> List[np.ndarray]:
        """Overlap half-derivatives for ``atom`` (``side`` = 'LEFT' or 'RIGHT');
        list of 3 arrays. Used by the AAT machinery."""
        return [np.asarray(m) for m in self.mints.mo_overlap_half_deriv1(side, atom, C1, C2)]

    def dipole(self, atom: int, C1, C2) -> List[np.ndarray]:
        """Electric-dipole derivatives for ``atom``: list of 9 arrays, the
        d(mu_alpha)/d(X_beta) blocks (3 dipole components x 3 Cartesians). Used by
        the APT machinery.

        The AO-basis derivatives (``ao_elec_dip_deriv1``) are transformed into the
        (C1, C2) MO block here rather than via ``mo_elec_dip_deriv1`` -- the latter
        segfaults on the linux build of Psi4 1.10.1 (the mo_oei/mo_tei deriv routines
        are unaffected). The two routes are numerically identical."""
        npC1 = np.asarray(C1)
        npC2 = np.asarray(C2)
        return [npC1.T @ np.asarray(m) @ npC2
                for m in self.mints.ao_elec_dip_deriv1(atom)]

    # ---- second derivatives (Hessian skeleton) ----
    def overlap2(self, atom1: int, atom2: int, C1, C2) -> List[np.ndarray]:
        """Second overlap derivatives ``S^{XY}`` for the ``(atom1, atom2)`` pair: a
        list of 9 arrays, the (cart1, cart2) Cartesian-pair blocks indexed
        ``cart1*3 + cart2``."""
        return [np.asarray(m)
                for m in self.mints.mo_oei_deriv2("OVERLAP", atom1, atom2, C1, C2)]

    def core2(self, atom1: int, atom2: int, C1, C2) -> List[np.ndarray]:
        """Second core one-electron (kinetic + potential) derivatives ``h^{XY}`` for
        the ``(atom1, atom2)`` pair: 9 arrays, indexed ``cart1*3 + cart2``."""
        T = self.mints.mo_oei_deriv2("KINETIC", atom1, atom2, C1, C2)
        V = self.mints.mo_oei_deriv2("POTENTIAL", atom1, atom2, C1, C2)
        return [np.asarray(t) + np.asarray(v) for t, v in zip(T, V)]

    def eri2(self, atom1: int, atom2: int, C1, C2, C3, C4) -> List[np.ndarray]:
        """Second two-electron (ERI) derivatives for the ``(atom1, atom2)`` pair: 9
        arrays, indexed ``cart1*3 + cart2``. Chemist/Mulliken notation ``(pq|rs)``,
        as for :meth:`eri`. The Hessian skeleton needs only the occupied block, so
        callers pass ``Cocc`` (no**4 per pair) rather than the full MO space."""
        return [np.asarray(m)
                for m in self.mints.mo_tei_deriv2(atom1, atom2, C1, C2, C3, C4)]

    def nuclear_repulsion2(self) -> np.ndarray:
        """Nuclear-repulsion-energy Hessian, shape ``(3*natom, 3*natom)`` indexed
        ``(atom1*3 + cart1, atom2*3 + cart2)``."""
        return np.asarray(self.mol.nuclear_repulsion_energy_deriv2())

    # ---- two-electron (heavy: lazy, per atom) ----
    def eri(self, atom: int, C1, C2, C3, C4) -> List[np.ndarray]:
        """Two-electron (ERI) derivatives for ``atom``: list of 3 (x, y, z) arrays.
        Computed on demand so a caller iterating over atoms never holds more than
        one atom's block at a time."""
        return [np.asarray(m) for m in self.mints.mo_tei_deriv1(atom, C1, C2, C3, C4)]

    def iter_eri(self, C1, C2, C3, C4) -> Iterator[Tuple[int, List[np.ndarray]]]:
        """Yield ``(atom, [Ex, Ey, Ez])`` one atom at a time -- the lazy, non-
        materializing way to sweep the 2-e derivatives."""
        for atom in range(self.natom):
            yield atom, self.eri(atom, C1, C2, C3, C4)

    # ---- nuclear repulsion ----
    def nuclear_repulsion(self) -> np.ndarray:
        """Nuclear-repulsion-energy gradient, shape (natom, 3)."""
        return np.asarray(self.mol.nuclear_repulsion_energy_deriv1())
