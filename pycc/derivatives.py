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

from typing import Any, List, Iterator, Tuple

import psi4
import numpy as np


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

    A superscript ``X`` (a nuclear Cartesian) denotes the *skeleton* derivative -- the
    partial derivative of the integral at *fixed* MO coefficients -- transformed into the
    requested MO block(s)::

        A^X_pq... = C_mu,p C_nu,q ... (d A_mu,nu... / dX)

    .. math::

        A^{X}_{pq\cdots} = C^{\mu}_{p} C^{\nu}_{q}\cdots\,\frac{\partial A_{\mu\nu\cdots}}{\partial X}

    The orbital-response (CPHF) contribution is added downstream by the caller, not here.
    Second derivatives ``A^{XY}`` replace the single partial by :math:`\partial^2/\partial X\,\partial Y`.
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
        r"""Overlap (S^X) skeleton derivatives for ``atom``: 3 (x, y, z) arrays,
        transformed into MO blocks (b1, b2)::

            S^X_pq = C_mu,p (dS_mu,nu / dX) C_nu,q

        .. math::

            S^{X}_{pq} = C^{\mu}_{p}\,\frac{\partial S_{\mu\nu}}{\partial X}\,C^{\nu}_{q}
        """
        return [np.asarray(m) for m in
                self.mints.mo_oei_deriv1("OVERLAP", atom, self._mo(b1), self._mo(b2))]

    def core(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Core one-electron (kinetic + potential) ``h^X`` skeleton derivatives for
        ``atom``: 3 (x, y, z) arrays::

            h^X_pq = C_mu,p (d(T+V)_mu,nu / dX) C_nu,q

        .. math::

            h^{X}_{pq} = C^{\mu}_{p}\,\frac{\partial (T+V)_{\mu\nu}}{\partial X}\,C^{\nu}_{q}
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
        S^X_pq = S^X(LEFT)_pq + S^X(RIGHT)_pq. Used by the AAT machinery::

            S^X(LEFT)_pq = C_mu,p <d chi_mu / dX | chi_nu> C_nu,q

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

            \mu^{X_\beta}_{pq,\alpha} = C^{\mu}_{p}\,
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
        chemist notation ``(pq|rs)^X``::

            (pq|rs)^X = C_mu,p C_nu,q (d(mu,nu|lam,sig) / dX) C_lam,r C_sig,s

        .. math::

            (pq|rs)^{X} = C^{\mu}_{p} C^{\nu}_{q}\,
                \frac{\partial (\mu\nu|\lambda\sigma)}{\partial X}\,C^{\lambda}_{r} C^{\sigma}_{s}

        Cached one atom at a time (:meth:`_eri_cached`) so an atom-outer sweep never holds
        more than one atom's block yet reuses the dominant ``nmo^4`` transform across the
        atom's Cartesians and its several callers."""
        return self._eri_cached(atom, ('eri', b1, b2, b3, b4), lambda: [
            np.asarray(m) for m in self.mints.mo_tei_deriv1(
                atom, self._mo(b1), self._mo(b2), self._mo(b3), self._mo(b4))])

    def iter_eri(self, b1: str = 'all', b2: str = 'all', b3: str = 'all',
                 b4: str = 'all') -> Iterator[Tuple[int, List[np.ndarray]]]:
        """Yield ``(atom, [Ex, Ey, Ez])`` one atom at a time -- the lazy, non-
        materializing way to sweep the 2-e derivatives."""
        for atom in range(self.natom):
            yield atom, self.eri(atom, b1, b2, b3, b4)

    # ---- spatial second derivatives (Hessian skeleton) ----

    def overlap2(self, atom1: int, atom2: int, b1: str = 'all',
                 b2: str = 'all') -> List[np.ndarray]:
        r"""Second overlap skeleton derivatives ``S^{XY}`` for the ``(atom1, atom2)`` pair:
        9 arrays, the (cart1, cart2) blocks indexed ``cart1*3 + cart2``::

            S^XY_pq = C_mu,p (d2 S_mu,nu / dX dY) C_nu,q

        .. math::

            S^{XY}_{pq} = C^{\mu}_{p}\,\frac{\partial^2 S_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{q}
        """
        return [np.asarray(m) for m in
                self.mints.mo_oei_deriv2("OVERLAP", atom1, atom2, self._mo(b1), self._mo(b2))]

    def core2(self, atom1: int, atom2: int, b1: str = 'all',
              b2: str = 'all') -> List[np.ndarray]:
        r"""Second core one-electron (kinetic + potential) derivatives ``h^{XY}`` for the
        ``(atom1, atom2)`` pair: 9 arrays, indexed ``cart1*3 + cart2``::

            h^XY_pq = C_mu,p (d2(T+V)_mu,nu / dX dY) C_nu,q

        .. math::

            h^{XY}_{pq} = C^{\mu}_{p}\,\frac{\partial^2 (T+V)_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{q}
        """
        C1, C2 = self._mo(b1), self._mo(b2)
        T = self.mints.mo_oei_deriv2("KINETIC", atom1, atom2, C1, C2)
        V = self.mints.mo_oei_deriv2("POTENTIAL", atom1, atom2, C1, C2)
        return [np.asarray(t) + np.asarray(v) for t, v in zip(T, V)]

    def eri2(self, atom1: int, atom2: int, b1: str = 'all', b2: str = 'all',
             b3: str = 'all', b4: str = 'all') -> List[np.ndarray]:
        r"""Second two-electron (ERI) derivatives for the ``(atom1, atom2)`` pair: 9 arrays,
        indexed ``cart1*3 + cart2``, in chemist notation::

            (pq|rs)^XY = C_mu,p C_nu,q (d2(mu,nu|lam,sig) / dX dY) C_lam,r C_sig,s

        .. math::

            (pq|rs)^{XY} = C^{\mu}_{p} C^{\nu}_{q}\,
                \frac{\partial^2 (\mu\nu|\lambda\sigma)}{\partial X\,\partial Y}\,C^{\lambda}_{r} C^{\sigma}_{s}

        The Hessian skeleton needs only the occupied block, so callers pass ``'o'``
        (n_occ**4 per pair)."""
        return [np.asarray(m) for m in self.mints.mo_tei_deriv2(
            atom1, atom2, self._mo(b1), self._mo(b2), self._mo(b3), self._mo(b4))]

    # ---- spin-orbital one-electron (spin-blocked from the spatial MO derivatives) ----

    def so_overlap(self, atom: int, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital overlap ``S^X`` derivatives for ``atom``: 3 (x, y, z) arrays.
        Block-diagonal in spin (OVERLAP via :meth:`so_oei`)::

            S^X_pq = delta(spin_p, spin_q) C_mu,pbar (dS_mu,nu / dX) C_nu,qbar

        .. math::

            S^{X}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial S_{\mu\nu}}{\partial X}\,C^{\nu}_{\bar q}
        """
        return self.so_oei(atom, "OVERLAP", b1, b2)

    def so_oei(self, atom: int, kind: str, b1: str = 'all', b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital one-electron derivative integral (block-diagonal in spin): 3
        (x, y, z) arrays. ``kind`` is 'OVERLAP'/'KINETIC'/'POTENTIAL'.

        A spin-orbital one-electron integral vanishes unless bra and ket share a spin, so
        the derivative is block-diagonal: each same-spin block is the spatial MO derivative
        (in the semicanonical alpha/beta gauge), placed at that spin's positions::

            A^X_pq = delta(spin_p, spin_q) C_mu,pbar (dA_mu,nu / dX) C_nu,qbar

        .. math::

            A^{X}_{pq} = \delta_{\sigma_p \sigma_q}\,
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
        r"""Spin-orbital core (kinetic + potential) ``h^X`` derivatives for ``atom``: 3
        (x, y, z) arrays. Block-diagonal in spin (KINETIC + POTENTIAL via :meth:`so_oei`)::

            h^X_pq = delta(spin_p, spin_q) C_mu,pbar (d(T+V)_mu,nu / dX) C_nu,qbar

        .. math::

            h^{X}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial (T+V)_{\mu\nu}}{\partial X}\,C^{\nu}_{\bar q}
        """
        T = self.so_oei(atom, "KINETIC", b1, b2)
        V = self.so_oei(atom, "POTENTIAL", b1, b2)
        return [t + v for t, v in zip(T, V)]

    def so_overlap_half(self, atom: int, b1: str = 'all', b2: str = 'all',
                        side: str = "LEFT") -> List[np.ndarray]:
        r"""Spin-orbital overlap half-derivatives ``<phi^X_p | phi_q>`` (block-diagonal in
        spin): 3 arrays. Bra perturbed, ket unperturbed (``side='LEFT'``); not symmetric.
        Used by the spin-orbital AAT machinery::

            S^X(LEFT)_pq = delta(spin_p, spin_q) C_mu,pbar <d chi_mu / dX | chi_nu> C_nu,qbar

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

            \mu^{X_\beta}_{pq,\alpha} = \delta_{\sigma_p \sigma_q}\,
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
        r"""Spin-orbital antisymmetrized two-electron derivatives ``<pq||rs>^X`` for
        ``atom``: 3 (x, y, z) arrays. Spin-blocks the spatial chemist derivative integrals
        over the spin-conserving combinations ``(s12, s34)``, converts to physicist
        notation (``<pq|rs> = (pr|qs)``, the ``swapaxes(1,2)``), and antisymmetrizes over
        the ket (the ``phys - phys.swapaxes(2,3)``)::

            <pq||rs>^X = <pq|rs>^X - <pq|sr>^X,   <pq|rs>^X = (pr|qs)^X  (chemist)

        .. math::

            \langle pq\Vert rs\rangle^{X} = \langle pq|rs\rangle^{X} - \langle pq|sr\rangle^{X},
            \qquad \langle pq|rs\rangle^{X} = (pr|qs)^{X}

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
        r"""Spin-orbital second overlap derivatives ``S^{XY}`` for the ``(atom1, atom2)``
        pair: 9 arrays. Block-diagonal in spin (OVERLAP via :meth:`so_oei2`)::

            S^XY_pq = delta(spin_p, spin_q) C_mu,pbar (d2 S_mu,nu / dX dY) C_nu,qbar

        .. math::

            S^{XY}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial^2 S_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{\bar q}
        """
        return self.so_oei2(atom1, atom2, "OVERLAP", b1, b2)

    def so_oei2(self, atom1: int, atom2: int, kind: str, b1: str = 'all',
                b2: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital one-electron *second* derivative (block-diagonal in spin) for the
        ``(atom1, atom2)`` pair: 9 arrays indexed ``cart1*3 + cart2``::

            A^XY_pq = delta(spin_p, spin_q) C_mu,pbar (d2 A_mu,nu / dX dY) C_nu,qbar

        .. math::

            A^{XY}_{pq} = \delta_{\sigma_p \sigma_q}\,
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
        r"""Spin-orbital second core (kinetic + potential) derivatives ``h^{XY}`` for the
        ``(atom1, atom2)`` pair: 9 arrays. Block-diagonal in spin (KINETIC + POTENTIAL via
        :meth:`so_oei2`)::

            h^XY_pq = delta(spin_p, spin_q) C_mu,pbar (d2(T+V)_mu,nu / dX dY) C_nu,qbar

        .. math::

            h^{XY}_{pq} = \delta_{\sigma_p \sigma_q}\,
                C^{\mu}_{\bar p}\,\frac{\partial^2 (T+V)_{\mu\nu}}{\partial X\,\partial Y}\,C^{\nu}_{\bar q}
        """
        T = self.so_oei2(atom1, atom2, "KINETIC", b1, b2)
        V = self.so_oei2(atom1, atom2, "POTENTIAL", b1, b2)
        return [t + v for t, v in zip(T, V)]

    def so_eri2(self, atom1: int, atom2: int, b1: str = 'all', b2: str = 'all',
                b3: str = 'all', b4: str = 'all') -> List[np.ndarray]:
        r"""Spin-orbital antisymmetrized two-electron *second* derivatives ``<pq||rs>^{XY}``
        for the ``(atom1, atom2)`` pair: 9 arrays indexed ``cart1*3 + cart2``::

            <pq||rs>^XY = <pq|rs>^XY - <pq|sr>^XY,   <pq|rs>^XY = (pr|qs)^XY  (chemist)

        .. math::

            \langle pq\Vert rs\rangle^{XY} = \langle pq|rs\rangle^{XY} - \langle pq|sr\rangle^{XY},
            \qquad \langle pq|rs\rangle^{XY} = (pr|qs)^{XY}

        Psi4's ``mo_tei_deriv2(A, B)`` does not satisfy the integral's electron-exchange
        symmetry ``(pq|rs) = (rs|pq)`` term by term -- a single (A, B) call is one ordering
        of ``d^2/dXA dXB`` -- so the chemist integral is symmetrized over the bra<->ket
        swap here (``0.5 (ch + ch.transpose(2,3,0,1))``), which (the geometric derivative of
        a symmetric integral being symmetric) also restores the atom-pair-swap symmetry the
        molecular Hessian needs. The symmetrization assumes matching bra/ket block pairs
        (``b1,b2`` == ``b3,b4``), as in the occupied-block Hessian use; all four spin
        combinations are built independently."""
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
            ch = 0.5 * (ch + ch.transpose(2, 3, 0, 1))   # enforce (pq|rs) = (rs|pq)
            phys = ch.swapaxes(1, 2)
            out.append(phys - phys.swapaxes(2, 3))
        return out

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
        read-only (callers already build new arrays via swapaxes/arithmetic)."""
        if atom != self._d1_atom:
            self._d1_atom = atom
            self._d1_cache = {}
        if key not in self._d1_cache:
            self._d1_cache[key] = compute()
        return self._d1_cache[key]
