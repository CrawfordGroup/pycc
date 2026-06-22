"""
derivatives.py: lazy MO-basis derivative-integral provider.

A thin, memory-conscious wrapper around Psi4's MintsHelper MO derivative-integral
routines (mo_*_deriv1). It serves the *skeleton* (fixed-MO-coefficient) first
derivatives of the one- and two-electron integrals, the overlap half-derivatives
(for AATs), and the dipole derivatives (for APTs), in the MO basis -- consistent
with PyCC's reference-implementation, MO-basis derivative-property formulations.

For the spatial path the caller supplies the MO-coefficient blocks to transform into.
Those should be the base's symmetry-handled ``self.C`` (or a slice of it): it is a
single irrep block in global energy order, so derivative integrals work with molecular
symmetry left on -- no need to drop to C1.

For the spin-orbital path the ``so_*`` methods take just an atom: they spin-block the
spatial MO derivative integrals (built in the semicanonical MO gauge from the
spin-orbital Hamiltonian's ``Ca``/``Cb`` + ``spin``/``spat`` maps), mirroring the
spin-orbital Hamiltonian's own integral construction, so the derivative integrals live
in the same MO gauge the spin-orbital densities do.

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
        self.wfn = wfn
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

    # ---- spin-orbital (spin-blocked from the spatial MO derivatives) ----
    def _so_spin_blocks(self):
        """``(nmo, a, b, sa, sb, Ca, Cb)`` from the spin-orbital Hamiltonian: the
        spin-orbital count, the alpha/beta spin-orbital index arrays and their spatial
        indices, and the semicanonical alpha/beta MOs (as Psi4 matrices)."""
        H = self.wfn.H
        spin = np.asarray(H.spin)
        spat = np.asarray(H.spat)
        a = np.where(spin == 0)[0]
        b = np.where(spin == 1)[0]
        Ca = psi4.core.Matrix.from_array(H.Ca)
        Cb = psi4.core.Matrix.from_array(H.Cb)
        return spin.shape[0], a, b, spat[a], spat[b], Ca, Cb

    def so_oei(self, atom: int, kind: str) -> List[np.ndarray]:
        """Spin-orbital one-electron derivative integral (block-diagonal in spin) for
        ``atom``: 3 (x, y, z) ``nmo x nmo`` arrays. ``kind`` is 'OVERLAP'/'KINETIC'/
        'POTENTIAL'."""
        nmo, a, b, sa, sb, Ca, Cb = self._so_spin_blocks()
        Da = self.mints.mo_oei_deriv1(kind, atom, Ca, Ca)
        Db = self.mints.mo_oei_deriv1(kind, atom, Cb, Cb)
        out = []
        for c in range(3):
            M = np.zeros((nmo, nmo))
            M[np.ix_(a, a)] = np.asarray(Da[c])[np.ix_(sa, sa)]
            M[np.ix_(b, b)] = np.asarray(Db[c])[np.ix_(sb, sb)]
            out.append(M)
        return out

    def so_core(self, atom: int) -> List[np.ndarray]:
        """Spin-orbital core (kinetic + potential) ``h^x`` derivatives for ``atom``."""
        T = self.so_oei(atom, "KINETIC")
        V = self.so_oei(atom, "POTENTIAL")
        return [t + v for t, v in zip(T, V)]

    def so_overlap(self, atom: int) -> List[np.ndarray]:
        """Spin-orbital overlap ``S^x`` derivatives for ``atom``."""
        return self.so_oei(atom, "OVERLAP")

    def so_eri(self, atom: int) -> List[np.ndarray]:
        """Spin-orbital antisymmetrized two-electron derivatives ``<pq||rs>^x`` for
        ``atom``: 3 (x, y, z) ``nmo^4`` arrays. Spin-blocks the spatial chemist
        derivative integrals, converts to physicist notation, and antisymmetrizes --
        the derivative analogue of the spin-orbital Hamiltonian's ERI build."""
        nmo, a, b, sa, sb, Ca, Cb = self._so_spin_blocks()
        AA = self.mints.mo_tei_deriv1(atom, Ca, Ca, Ca, Ca)
        BB = self.mints.mo_tei_deriv1(atom, Cb, Cb, Cb, Cb)
        AB = self.mints.mo_tei_deriv1(atom, Ca, Ca, Cb, Cb)
        out = []
        for c in range(3):
            chem = np.zeros((nmo, nmo, nmo, nmo))
            chem[np.ix_(a, a, a, a)] = np.asarray(AA[c])[np.ix_(sa, sa, sa, sa)]
            chem[np.ix_(b, b, b, b)] = np.asarray(BB[c])[np.ix_(sb, sb, sb, sb)]
            chem[np.ix_(a, a, b, b)] = np.asarray(AB[c])[np.ix_(sa, sa, sb, sb)]
            chem[np.ix_(b, b, a, a)] = np.asarray(AB[c]).transpose(2, 3, 0, 1)[np.ix_(sb, sb, sa, sa)]
            phys = chem.swapaxes(1, 2)
            out.append(phys - phys.swapaxes(2, 3))
        return out

    def so_dipole(self, atom: int) -> List[np.ndarray]:
        """Spin-orbital MO electric-dipole derivatives ``d(mu_alpha)/d(X_atom,beta)`` for
        ``atom``: 9 ``nmo x nmo`` arrays indexed ``alpha*3 + beta`` (block-diagonal in
        spin). As in :meth:`dipole`, the AO derivatives (``ao_elec_dip_deriv1``) are
        transformed here (the ``mo_elec_dip_deriv1`` route segfaults on linux Psi4
        1.10.1); here each spin block is transformed with the semicanonical MOs."""
        nmo, a, b, sa, sb, _, _ = self._so_spin_blocks()
        Ca = np.asarray(self.wfn.H.Ca)
        Cb = np.asarray(self.wfn.H.Cb)
        aod = self.mints.ao_elec_dip_deriv1(atom)   # 9 x (nao, nao)
        out = []
        for c in range(9):
            ao = np.asarray(aod[c])
            M = np.zeros((nmo, nmo))
            M[np.ix_(a, a)] = (Ca.T @ ao @ Ca)[np.ix_(sa, sa)]
            M[np.ix_(b, b)] = (Cb.T @ ao @ Cb)[np.ix_(sb, sb)]
            out.append(M)
        return out

    def so_overlap_half(self, atom: int, side: str = "LEFT") -> List[np.ndarray]:
        """Spin-orbital overlap half-derivatives ``<phi^X_p | phi_q>`` (block-diagonal in
        spin) for ``atom``: 3 (x, y, z) ``nmo x nmo`` arrays. The bra is the perturbed
        orbital, the ket the unperturbed (``side='LEFT'``); the matrix is not symmetric.
        Used by the spin-orbital AAT machinery (which takes the ov block)."""
        nmo, a, b, sa, sb, Ca, Cb = self._so_spin_blocks()
        La = self.mints.mo_overlap_half_deriv1(side, atom, Ca, Ca)
        Lb = self.mints.mo_overlap_half_deriv1(side, atom, Cb, Cb)
        out = []
        for c in range(3):
            M = np.zeros((nmo, nmo))
            M[np.ix_(a, a)] = np.asarray(La[c])[np.ix_(sa, sa)]
            M[np.ix_(b, b)] = np.asarray(Lb[c])[np.ix_(sb, sb)]
            out.append(M)
        return out

    # ---- spin-orbital second derivatives (Hessian skeleton; occupied block) ----
    def _so_occ_blocks(self):
        """Occupied spin-orbital spin split for the Hessian skeleton: ``(no, a, b,
        Cocc_a, Cocc_b)`` where ``a``/``b`` are the alpha/beta positions within the
        occupied block and ``Cocc_a``/``Cocc_b`` the corresponding occupied alpha/beta
        MOs (as Psi4 matrices, pre-sliced to the occupied spatial columns -- so the
        deriv2 transforms return arrays already in occupied-block order)."""
        H = self.wfn.H
        o = self.wfn.o
        spin_o = np.asarray(H.spin)[o]
        spat_o = np.asarray(H.spat)[o]
        a = np.where(spin_o == 0)[0]
        b = np.where(spin_o == 1)[0]
        Cocc_a = psi4.core.Matrix.from_array(np.asarray(H.Ca)[:, spat_o[a]])
        Cocc_b = psi4.core.Matrix.from_array(np.asarray(H.Cb)[:, spat_o[b]])
        return self.wfn.no, a, b, Cocc_a, Cocc_b

    def so_oei2(self, kind: str, atom1: int, atom2: int) -> List[np.ndarray]:
        """Spin-orbital second one-electron derivative (occupied block, block-diagonal
        in spin) for the ``(atom1, atom2)`` pair: 9 ``(no, no)`` arrays indexed
        ``cart1*3 + cart2``. ``kind`` is 'OVERLAP'/'KINETIC'/'POTENTIAL'."""
        no, a, b, Ca, Cb = self._so_occ_blocks()
        Da = self.mints.mo_oei_deriv2(kind, atom1, atom2, Ca, Ca)
        Db = self.mints.mo_oei_deriv2(kind, atom1, atom2, Cb, Cb)
        out = []
        for c in range(9):
            M = np.zeros((no, no))
            M[np.ix_(a, a)] = np.asarray(Da[c])
            M[np.ix_(b, b)] = np.asarray(Db[c])
            out.append(M)
        return out

    def so_core2(self, atom1: int, atom2: int) -> List[np.ndarray]:
        """Spin-orbital second core (kinetic + potential) derivatives ``h^{XY}`` (occupied
        block) for the ``(atom1, atom2)`` pair: 9 ``(no, no)`` arrays."""
        T = self.so_oei2("KINETIC", atom1, atom2)
        V = self.so_oei2("POTENTIAL", atom1, atom2)
        return [t + v for t, v in zip(T, V)]

    def so_overlap2(self, atom1: int, atom2: int) -> List[np.ndarray]:
        """Spin-orbital second overlap derivatives ``S^{XY}`` (occupied block) for the
        ``(atom1, atom2)`` pair: 9 ``(no, no)`` arrays."""
        return self.so_oei2("OVERLAP", atom1, atom2)

    def so_eri2(self, atom1: int, atom2: int) -> List[np.ndarray]:
        """Spin-orbital second antisymmetrized two-electron derivatives ``<ij||kl>^{XY}``
        (occupied block) for the ``(atom1, atom2)`` pair: 9 ``(no^4)`` arrays indexed
        ``cart1*3 + cart2``. Spin-blocked from the occupied-block chemist deriv2 integrals
        then antisymmetrized, as in :meth:`so_eri`.

        Psi4's ``mo_tei_deriv2(A, B)`` does not satisfy the two-electron integral's
        electron-exchange symmetry ``(pq|rs) = (rs|pq)`` term by term -- a single (A, B)
        call gives one ordering of the mixed derivative ``d^2/dXA dXB`` -- so the chemist
        integral is symmetrized over the bra<->ket swap here. That restores the missing
        symmetry, which (because the geometric derivative of a symmetric integral is
        symmetric) also makes the result invariant under the atom-pair swap the molecular
        Hessian requires. The cross-spin ``ab``/``ba`` blocks are therefore built from
        independent ``(aa|bb)`` and ``(bb|aa)`` deriv2 calls rather than as transposes of
        one another (only with both does the symmetrization see the true pair); the
        same-spin ``aa``/``bb`` blocks need it too (a no-op for their energy trace but it
        restores the tensor symmetry)."""
        no, a, b, Ca, Cb = self._so_occ_blocks()
        AA = self.mints.mo_tei_deriv2(atom1, atom2, Ca, Ca, Ca, Ca)
        BB = self.mints.mo_tei_deriv2(atom1, atom2, Cb, Cb, Cb, Cb)
        AB = self.mints.mo_tei_deriv2(atom1, atom2, Ca, Ca, Cb, Cb)
        BA = self.mints.mo_tei_deriv2(atom1, atom2, Cb, Cb, Ca, Ca)
        out = []
        for c in range(9):
            chem = np.zeros((no, no, no, no))
            chem[np.ix_(a, a, a, a)] = np.asarray(AA[c])
            chem[np.ix_(b, b, b, b)] = np.asarray(BB[c])
            chem[np.ix_(a, a, b, b)] = np.asarray(AB[c])
            chem[np.ix_(b, b, a, a)] = np.asarray(BA[c])
            chem = 0.5 * (chem + chem.transpose(2, 3, 0, 1))   # enforce (pq|rs) = (rs|pq)
            phys = chem.swapaxes(1, 2)
            out.append(phys - phys.swapaxes(2, 3))
        return out

    # ---- nuclear repulsion ----
    def nuclear_repulsion(self) -> np.ndarray:
        """Nuclear-repulsion-energy gradient, shape (natom, 3)."""
        return np.asarray(self.mol.nuclear_repulsion_energy_deriv1())
