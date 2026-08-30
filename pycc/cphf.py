r"""
cphf.py: coupled-perturbed Hartree-Fock (CPHF) orbital response.

Solves the RHF first-order orbital-response (CPHF / CPSCF) equations in the MO
basis. For a perturbation ``x``, the occupied-virtual orbital rotations
``U^x_ia`` satisfy a single perturbation-independent linear operator (the
singlet orbital Hessian ``G``) applied to a perturbation-specific right-hand
side ``B^x`` (repeated indices summed)::

    G_iajb U^x_jb = B^x_ia

.. math::

    G_{ia,jb}\,U^{x}_{jb} = B^{x}_{ia}

The occupied-occupied and virtual-virtual blocks of the response are fixed
separately by the overlap derivative (orthonormality), so only the ov block is
solved here.

The two-electron part of ``G`` is written with the spin-adapted integrals
``L = 2<pq|rs> - <pq|sr>`` (physicist's/Dirac notation, already stored as
``H.L``)::

    electric (real perturbations -- nuclear, electric field):
        G_iajb = (e_a - e_i) d_ij d_ab + L[a,j,i,b] + L[a,b,i,j]
    magnetic (imaginary perturbations -- e.g. AATs):
        G_iajb = (e_a - e_i) d_ij d_ab + L[a,j,i,b] - L[a,b,i,j]

.. math::

    G_{ia,jb} = (\epsilon_a - \epsilon_i)\,\delta_{ij}\delta_{ab} + L_{ajib} \pm L_{abij}

(``+`` electric, ``-`` magnetic; the antisymmetrized ``<pq||rs>`` replaces ``L``
on the spin-orbital path.)

The simplest perturbation is the electric field: it does not move the basis
functions, so its right-hand side is just the (negated) MO dipole integrals
(``B = -mu``, since the field enters as ``-mu . E``) with no overlap/Pulay terms --
it exercises the solver in isolation.

Scope: this class computes only quantities intrinsic to the CPHF equations -- the
orbital Hessian, the perturbation right-hand sides (field / magnetic / nuclear),
and the response coefficients ``U`` (and the cached nuclear by-products the Hessian
reuses). The HF *property tensors* built from ``U`` -- polarizability, dipole
derivatives (APTs), molecular Hessian, atomic axial tensors -- live on :class:`HFwfn`,
which asks this solver for the relevant ``U`` and contracts it with the property
integrals.

Lives on the Wavefunction base (lazy ``self.cphf``); it depends only on base state
(``o``/``v``, the Fock diagonal, the orbital-Hessian integrals) plus the dipole
integrals, and is shared by HF, MP2, and CC orbital response.

Reference code: tensor contractions route through the device-aware ``self.contract``
(``ContractionBackend``, shared from the wavefunction), while the explicit orbital
Hessian is solved directly with ``numpy.linalg`` (a small dense solve, left on NumPy).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, List

import numpy as np

from .utils import diag
from .timing import timed


# A perturbation descriptor used to key the response and full-derivative caches.
# ``kind`` is 'field' (electric dipole, axis ``comp`` in 0/1/2), 'nuclear' (atomic
# displacement, ``comp`` = (atom, cart)), or 'magnetic' (axis ``comp``). It is the
# shared identity under which CPHF memoizes ``U^x`` and the full (CPHF-folded) first
# derivatives ``d_x f`` and ``d_x <pq||rs>`` so multi-property workflows (e.g. IR + VCD,
# or an MP2 gradient + polarizability) never recompute them. Only 'field' is wired so
# far; 'nuclear'/'magnetic' slot into the same machinery later.
Perturbation = namedtuple('Perturbation', ['kind', 'comp'])


def perturbation_label(pert, mol=None):
    """A short human label for a :data:`Perturbation`, for solver output -- e.g. ``'O1y'``
    (nuclear: atom 1, Cartesian y), ``'mu_x'`` (electric field), ``'m_z'`` (magnetic dipole),
    ``'p_x'`` (vector potential / linear momentum).  ``mol`` supplies the element symbols for a
    nuclear perturbation; without it the atom index is used."""
    xyz = 'xyz'
    kind, comp = pert.kind, pert.comp
    if kind == 'nuclear':
        atom, cart = comp
        sym = mol.symbol(atom) if mol is not None else 'atom%d' % atom
        return "%s%d%s" % (sym, atom + 1, xyz[cart])
    if kind == 'field':
        return "mu_%s" % xyz[comp]
    if kind == 'magnetic':
        return "m_%s" % xyz[comp]
    if kind == 'vecpot':
        return "p_%s" % xyz[comp]
    return "%s(%s)" % (kind, comp)


class CPHF(object):
    """RHF coupled-perturbed Hartree-Fock orbital-response solver (MO basis).

    Parameters
    ----------
    wfn : Wavefunction
        provides the active spaces (``o``/``v``, ``no``/``nv``), the Fock diagonal
        (orbital energies), the spin-adapted ERIs (``H.L``), and the MO dipole
        integrals (``H.mu``).

    Notes
    -----
    The orbital Hessian is perturbation-independent: it is built once per ``kind``
    ('electric' / 'magnetic') and cached, then reused across all perturbations.
    """

    def __init__(self, wfn: Any, full_occ: bool = False) -> None:
        """Bind the wavefunction and set up the CPHF working state: the active (or, with
        ``full_occ``, frozen-core + active) occupied/virtual spaces (o/v, no/nv), the orbital
        energies (Fock diagonal), the ROHF flag (orbital response unsupported -- see
        :meth:`solve`), and the empty per-kind / per-perturbation caches (orbital Hessian,
        field/magnetic/momentum/nuclear responses, and the skeleton and CPHF-folded
        derivatives)."""
        self.wfn = wfn
        self.contract = wfn.contract        # device-aware ContractionBackend (shared)
        # ``full_occ`` spans the full occupied space (frozen core + active) in the wfn's own
        # MO ordering, for frozen-core correlated derivatives whose orbital response (incl.
        # core<->active and core-virtual) the active-space CPHF can't supply. The core block
        # is the first ``o.stop - no`` orbitals (the frozen core leads ``o`` in both the
        # spatial and spin-orbital orderings). For ``nfzc=0`` it coincides with the default.
        if full_occ:
            self.o = slice(0, wfn.o.stop)
            self.no = wfn.o.stop
        else:
            self.o = wfn.o
            self.no = wfn.no
        self.v = wfn.v
        self.nv = wfn.nv
        # Orbital energies from the Fock diagonal (energy-ordered, all-electron --
        # HFwfn builds the full MO space).
        self.eps = np.asarray(diag(wfn.H.F))
        self._G: dict = {}  # kind -> reshaped (no*nv, no*nv) orbital Hessian
        # ROHF: restricted orbitals (same_a_b_orbs) but spin-polarized occupation
        # (not same_a_b_dens). UHF has same_a_b_orbs False; RHF has both True. The
        # CPHF orbital response is not supported for ROHF yet -- see solve().
        ref = wfn.ref
        self.is_rohf = bool(ref.same_a_b_orbs() and not ref.same_a_b_dens())
        # Persistent nuclear response, keyed by atom (geometry-bound -- valid for the
        # life of this CPHF object, which is tied to one wfn / structure). Built once
        # by solve_nuclear() and SHARED by every consumer of the nuclear response (the
        # molecular Hessian and the dipole derivatives / APTs), so an IR workflow
        # (Hessian -> normal modes, then APTs) never rebuilds the per-atom nmo**4
        # derivative ERIs or re-solves the CPHF equations.
        self._U_nuc: dict = {}  # atom -> list of 3 (no, nv) nuclear response U^X
        self._B_nuc: dict = {}  # atom -> list of 3 (no, nv) nuclear RHS B^X
        self._F_nuc: dict = {}  # atom -> list of 3 (no, no) skeleton deriv Fock F^(X)_ij
        self._S_nuc: dict = {}  # atom -> list of 3 (no, no) overlap deriv S^(X)_ij
        self._U_mag: dict = {}  # axis -> (no, nv) magnetic-field response U^B (real)
        self._U_mom: dict = {}  # axis -> (no, nv) linear-momentum response U^A (real)
        self._mag_int: dict = {}  # (axis, ncore, gauge) -> (U^H, dF^H, dERI^H) magnetic engine (MP2 AATs)
        self._mom_int: dict = {}  # (axis, ncore, gauge) -> (U^A, dF^A, dERI^A) momentum engine (MP2 VG APTs)
        # Full (CPHF-folded) first-derivative caches. These hold the response-dressed
        # derivatives d_x f and d_x <pq||rs>, persisting for the life of this CPHF object so
        # that multiple property calculations on one wavefunction share them.
        self._U_field: dict = {}  # axis -> (no, nv) electric-field response U^a
        self._skel: dict = {}     # Perturbation -> (fx, Sx) skeleton derivatives (nmo^2 each;
                                  # the nmo^4 gx is NOT cached -- see _skeleton_eri)
        self._dfock: dict = {}    # (pert, ncore, canonical) -> (nmo, nmo) full perturbed Fock deriv
        # the (nmo^4) full perturbed <pq||rs> deriv now lives in wfn.derivatives.store (persistent,
        # keyed on (pert, ncore, canonical)) rather than a per-CPHF RAM dict; see ``eri`` below.

    # ---- orbital (MO) Hessian ----
    def _mo_hessian(self, kind: str = "electric") -> np.ndarray:
        """Singlet orbital (MO) Hessian ``G`` as a ``(no*nv, no*nv)`` matrix, cached by ``kind``.

        The perturbation-independent left-hand operator of the CPHF equations (named to
        distinguish it from the molecular/nuclear Hessian property).  ``kind`` is 'electric'
        (real perturbations: nuclear displacements, electric field) or 'magnetic' (imaginary
        perturbations: magnetic field / AATs), which only flips the sign of the second L term.
        """
        if kind not in self._G:
            self._G[kind] = self._build_mo_hessian(kind)
        return self._G[kind]

    def _build_mo_hessian(self, kind: str) -> np.ndarray:
        r"""Build the singlet orbital (MO) Hessian ``G`` (``(no*nv, no*nv)``) for ``kind``
        ('electric' real / 'magnetic' imaginary). The two-electron weight ``W`` is the
        spin-adapted ``L`` (spatial) or antisymmetrized ``<pq||rs>`` (spin-orbital)::

            G_iajb = (e_a - e_i) d_ij d_ab + W[a,j,i,b] + sign * W[a,b,i,j]
            sign = +1 (electric),  -1 (magnetic)

        .. math::

            G_{ia,jb} = (\epsilon_a - \epsilon_i)\,\delta_{ij}\delta_{ab}
                + W_{ajib} \pm W_{abij}

        with :math:`+` for electric (real) and :math:`-` for magnetic (imaginary)
        perturbations.
        """
        o, v, no, nv = self.o, self.v, self.no, self.nv
        # Two-electron weight: the spin-adapted L for the spatial (closed-shell singlet)
        # path, the antisymmetrized <pq||rs> for the spin-orbital path. The index
        # structure is identical; only the integral differs.
        if self.wfn.orbital_basis == 'spinorbital':
            W = np.asarray(self.wfn.H.ERI)
        else:
            W = np.asarray(self.wfn.H.L)

        if kind == "electric":
            sign = 1.0
        elif kind == "magnetic":
            sign = -1.0
        else:
            raise ValueError("kind must be 'electric' or 'magnetic', got %r" % kind)

        # Two-electron part: G_iajb = W[a,j,i,b] + sign * W[a,b,i,j].
        G = (self.contract('ajib->iajb', W[v, o, o, v])
             + sign * self.contract('abij->iajb', W[v, v, o, o]))
        G = G.reshape(no * nv, no * nv)

        # Orbital-energy part: + (e_a - e_i) on the (ia)=(jb) diagonal.
        D = (self.eps[v][None, :] - self.eps[o][:, None]).reshape(-1)
        G[np.diag_indices(no * nv)] += D
        return G

    # ---- linear solve ----
    def solve(self, B: np.ndarray, kind: str = "electric") -> np.ndarray:
        r"""Solve the CPHF equations for the ov response. ``B`` is ``(no, nv)``; returns
        ``U`` ``(no, nv)``. ``kind`` selects the electric/magnetic orbital Hessian
        (repeated indices summed)::

            G_iajb U^x_jb = B^x_ia

        .. math::

            G_{ia,jb}\,U^{x}_{jb} = B^{x}_{ia}

        Not implemented for ROHF: the semicanonical spin-orbital response lets alpha and
        beta relax independently (UHF-like) and so does not reproduce the *restricted*
        ROHF response. Matching it requires adopting the reference's ROHF Brillouin /
        orbital-rotation conventions (docc-socc, socc-virt couplings), which are not
        uniquely defined. RHF and UHF are supported. The CPHF-free HF gradient is
        unaffected (it does not call this)."""
        if self.is_rohf:
            raise NotImplementedError(
                "CPHF orbital response is not implemented for ROHF references: the "
                "semicanonical spin-orbital response does not reproduce the restricted "
                "ROHF response, and the ROHF Brillouin/orbital-rotation conventions "
                "(not uniquely defined) must match the reference. RHF and UHF are "
                "supported; the CPHF-free HF gradient works for ROHF.")
        G = self._mo_hessian(kind)
        U = np.linalg.solve(G, np.asarray(B).reshape(-1))
        return U.reshape(self.no, self.nv)

    # ---- property integrals / perturbation right-hand sides ----
    def _dipole_ov(self, axis: int) -> np.ndarray:
        r"""ov block of the MO electric-dipole integral for ``axis`` (0/1/2)::

            mu_ia = <i| mu_axis |a>

        .. math::

            \mu_{ia} = \langle i|\,\mu_{\mathrm{axis}}\,|a\rangle
        """
        return np.asarray(self.wfn.H.mu[axis])[self.o, self.v]

    def _rhs_field(self, axis: int) -> np.ndarray:
        r"""Electric-field CPHF right-hand side for ``axis`` (0/1/2)::

            B^x_ia = +mu_ia

        .. math::

            B^{x}_{ia} = +\mu_{ia}

        The field enters as ``H' = -mu . E`` (``H.mu`` is the dipole operator ``-e r``), so the
        skeleton Fock derivative is ``f^(a) = -mu`` and the CPHF RHS is ``B = -f^(a) = +mu``
        (no overlap/Pulay term -- the field does not move the basis functions).
        """
        return self._dipole_ov(axis)

    def solve_field(self, axis: int) -> np.ndarray:
        """Electric-field CPHF response ``U^a`` for ``axis`` (0/1/2), ``(no, nv)``, solved
        once and cached (shared by the polarizability and the correlated field properties)."""
        if axis not in self._U_field:
            self._U_field[axis] = self.solve(self._rhs_field(axis), kind="electric")
        return self._U_field[axis]

    # ---- CPHF-folded full first derivatives of f and <pq||rs> ----
    # Fold the CPHF coefficients ``U^x`` directly into the full MO derivatives of the Fock
    # matrix and the two-electron integrals (derivints.pdf). These full first derivatives feed
    # the correlated perturbed-response machinery (the perturbed Lagrangian / perturbed relaxed
    # density in CorrelatedDerivs, which solves its own Z-vector on top of them). Each
    # perturbation differs only in its *skeleton* derivatives and its ov CPHF response; the
    # assembly below is shared, and basis-aware (spatial closed-shell default, spin-orbital via
    # the same ``orbital_basis`` switch as the rest of the code).

    def _skeleton_eri(self, pert: "Perturbation"):
        r"""Skeleton two-electron derivative for ``pert`` -- ``<pq|rs>^(x)`` (spatial) or
        ``<pq||rs>^(x)`` (spin-orbital), ``nmo^4`` -- fetched from the persistent
        :class:`~pycc.derivatives.DerivStore` on every call and **not** cached in RAM.
        ``0.0`` for an electric field, whose basis functions do not move.

        Deliberately uncached.  The store computes and writes the atom's 3-Cartesian block once
        and reads it back thereafter, so caching here would save no transform and no integral
        generation -- it would only pin the block.  A retained ``gx`` is a *view* into that
        3-Cartesian buffer, so one retained coordinate holds ``3*nmo^4`` and all ``3N`` of them
        held ``9N*nmo^4`` (measured 27x nmo^4 for water/6-31G, 3 atoms).  Four-index derivative
        tensors belong in the store, not in memory.
        """
        if pert.kind == 'field':
            return 0.0                                       # basis functions do not move
        if pert.kind != 'nuclear':
            raise NotImplementedError(
                "skeleton two-electron derivatives are wired for 'field' and 'nuclear' only.")
        atom, cart = pert.comp
        d = self.wfn.derivatives
        # Single-Cartesian read: ``eri(atom)[cart]`` would be a VIEW into the 3-Cartesian stack and
        # would keep 3*nmo^4 alive for as long as the caller holds it; the component accessors read
        # only the requested slab, so this holds nmo^4.
        if self.wfn.orbital_basis == 'spinorbital':
            return np.asarray(d.so_eri_component(atom, cart))    # <pq||rs>^(x)
        return np.asarray(d.eri_component(atom, cart))           # physicist <pq|rs>^(x)

    def _skeleton_derivatives(self, pert: "Perturbation"):
        r"""Skeleton (fixed-MO-coefficient) derivatives for ``pert``: the pair
        ``(fx, Sx)`` of the skeleton Fock derivative ``f^(x)_pq`` (``nmo x nmo``) and the
        overlap derivative ``S^(x)_pq`` (``nmo x nmo``), cached per ``pert``.

        Only these two ``nmo x nmo`` matrices are cached.  The ``nmo^4`` two-electron
        derivative that ``fx`` is built from is obtained from :meth:`_skeleton_eri` and
        released; consumers that need it (:meth:`perturbed_eri`) call that method themselves.

        - **field**: the basis functions do not move, so ``S^(x) = 0``; the skeleton Fock
          derivative is ``f^(a) = -mu`` (``H' = -mu.E``).
        - **nuclear** (``comp = (atom, cart)``): the skeleton derivative integrals come from
          the ``Derivatives`` provider; the skeleton Fock derivative is ``f^(x)_pq = h^(x)_pq
          + w[p,m,q,m]^(x)`` (``m`` over occupied) with ``w`` the spin-adapted ``L`` (spatial)
          or antisymmetrized ``<pq||rs>`` (spin-orbital).

        .. math::

            f^{(x)}_{pq} = h^{(x)}_{pq} + w^{(x)}_{pmqm}
        """
        if pert in self._skel:
            return self._skel[pert]
        nmo, o = self.wfn.nmo, self.o
        so = self.wfn.orbital_basis == 'spinorbital'
        if pert.kind == 'field':
            fx = -np.asarray(self.wfn.H.mu[pert.comp])       # f^(a) = -mu  (H' = -mu.E)
            Sx = np.zeros((nmo, nmo))
        elif pert.kind == 'nuclear':
            atom, cart = pert.comp
            d = self.wfn.derivatives
            if so:
                hx = np.asarray(d.so_core(atom)[cart])
                Sx = np.asarray(d.so_overlap(atom)[cart])
                gx = self._skeleton_eri(pert)                # <pq||rs>^(x)
                w = gx                                        # Fock 2e weight = <pq||rs>^(x)
            else:
                hx = np.asarray(d.core(atom)[cart])
                Sx = np.asarray(d.overlap(atom)[cart])
                gx = self._skeleton_eri(pert)                      # physicist <pq|rs>^(x)
                w = 2.0 * gx - gx.swapaxes(2, 3)              # spin-adapted L^(x)
            fx = hx + self.contract('pmqm->pq', w[:, o, :, o])   # skeleton Fock derivative
            # Deliberate, not tidying: gx (and w, a fresh nmo^4 on the spatial path) would
            # otherwise stay alive past the _skel assignment below.  Keeping the cache to nmo^2
            # is the whole point of splitting _skeleton_eri out -- see its docstring.
            del gx, w
        else:
            raise NotImplementedError(
                "explicit perturbed derivatives are wired for 'field' and 'nuclear' only; "
                "'magnetic' uses the same machinery but is not yet built.")
        self._skel[pert] = (fx, Sx)
        return self._skel[pert]

    def _ov_response(self, pert: "Perturbation") -> np.ndarray:
        """The independent ov CPHF response ``U_ai`` (``(no, nv)``, indexed ``[i, a]``) for
        ``pert`` -- ``solve_field`` for the electric field, the cached ``solve_nuclear`` for
        a nuclear displacement."""
        if pert.kind == 'field':
            return self.solve_field(pert.comp)
        if pert.kind == 'nuclear':
            atom, cart = pert.comp
            return self.solve_nuclear(atom)[cart]
        raise NotImplementedError("ov response wired for 'field'/'nuclear' only.")

    @timed("orbital response (CPHF)")
    def full_U(self, pert: "Perturbation", ncore: int = 0, canonical: bool = False) -> np.ndarray:
        r"""The full ``nmo x nmo`` orbital-rotation matrix ``U^x_pq`` for ``pert``.

        The matrix element ``U^x_qp = <phi_q|d phi_p/dx>`` (the coefficient of orbital ``q`` in
        the first-order change of orbital ``p``, as it enters the integral derivatives). The
        non-canonical perturbed-orbital conditions fix the diagonal blocks from the overlap
        derivative, ``U^x_ij = -1/2 S^(x)_ij`` and ``U^x_ab = -1/2 S^(x)_ab``; the CPHF solve gives the
        occupied response ``Uia[i,a] = <phi_a|d phi_i> = U^x_ai`` (so ``U[v,o] = Uia.T``), and
        orthonormality ``U^x_pq + U^x_qp = -S^(x)_pq`` fixes ``U[o,v] = -S^(x)[o,v] - Uia``. For an
        electric field ``S^(x) = 0``, so the oo/vv blocks vanish and ``U[o,v] = -Uia``::

            U^x_ij = -1/2 S^(x)_ij,   U^x_ab = -1/2 S^(x)_ab
            U^x_ai = <phi_a | d phi_i / dx>,   U^x_ia = -S^(x)_ia - U^x_ai   (U^x_pq + U^x_qp = -S^(x)_pq)

        .. math::

            \begin{aligned}
            U^{x}_{ij} &= -\tfrac{1}{2} S^{(x)}_{ij}, \quad U^{x}_{ab} = -\tfrac{1}{2} S^{(x)}_{ab} \\
            U^{x}_{ai} &= \langle \phi_a | \partial_x \phi_i \rangle, \quad
                U^{x}_{ia} = -S^{(x)}_{ia} - U^{x}_{ai} \quad (U^{x}_{pq} + U^{x}_{qp} = -S^{(x)}_{pq})
            \end{aligned}

        ``ncore > 0`` (frozen-core correlated derivatives): the lowest ``ncore`` occupied
        orbitals are the frozen core. Core<->active-occupied rotations are non-redundant (they
        move the frozen/active partition), so that block is *not* left at the orthonormality
        value but determined by the canonical Brillouin condition ``d_x f_ij = 0`` -- a direct
        divide by ``(eps_i - eps_j)``, using the already-solved ov response. The redundant
        core-core, active-active, and vir-vir blocks stay at ``-1/2 S^(x)``.

        ``canonical=True`` (CCSD(T) derivatives): the (T) energy is *not* invariant to
        active-occupied or virtual rotations, so the canonical perturbed-orbital condition
        ``d_x f_pq = 0`` must also fix the active-oo and vv off-diagonal blocks -- the same
        ``(eps_p - eps_q)`` divide, generalized from the frozen-core core<->active block to the
        full active-oo/vv space, so that ``d_x f`` is diagonal within oo and vv. Diagonal and
        (near-)degenerate pairs keep the orthonormality value ``-1/2 S^(x)`` (the divide is
        ill-conditioned there -- the standard degeneracy caveat)."""
        o, v, nmo = self.o, self.v, self.wfn.nmo
        fx, Sx = self._skeleton_derivatives(pert)
        Uia = self._ov_response(pert)                       # (no, nv): U_ai = <phi_a|d phi_i>
        U = np.zeros((nmo, nmo))
        U[o, o] = -0.5 * Sx[o, o]
        U[v, v] = -0.5 * Sx[v, v]
        U[v, o] = Uia.T
        U[o, v] = -Sx[o, v] - Uia
        if ncore:
            # Core <-> active-occupied block from d_x f_ij = 0 (i core, j active), with
            # U_ji = -S^(x)_ij - U_ij eliminated:
            #   U_ij = -[ f^(x)_ij - S^(x)_ij eps_j - 1/2 sum_nm S^(x)_nm A_injm
            #             + sum_cm U_cm A_icjm ] / (eps_i - eps_j),
            # A_pqrs = w[p,q,r,s] + w[p,s,r,q] with w the orbital-Hessian weight.
            W = (np.asarray(self.wfn.H.ERI) if self.wfn.orbital_basis == 'spinorbital'
                 else np.asarray(self.wfn.H.L))
            eps = self.eps
            co = slice(o.start, o.start + ncore)            # frozen core
            ao = slice(o.start + ncore, o.stop)             # active occupied
            Soo = Sx[o, o]
            Uvo = U[v, o]
            Sterm = 0.5 * (self.contract('nm,injm->ij', Soo, W[co, o, ao, o])
                           + self.contract('nm,imjn->ij', Soo, W[co, o, ao, o]))
            Uterm = (self.contract('cm,icjm->ij', Uvo, W[co, v, ao, o])
                     + self.contract('cm,imjc->ij', Uvo, W[co, o, ao, v]))
            num = fx[co, ao] - Sx[co, ao] * eps[ao][None, :] - Sterm + Uterm
            Uca = -num / (eps[co][:, None] - eps[ao][None, :])
            U[co, ao] = Uca
            U[ao, co] = -(Sx[co, ao] + Uca).T
        if canonical:
            # Canonical Brillouin divide d_x f_pq = 0 for the active-oo and vv off-diagonal
            # blocks (generalizes the frozen-core core<->active block above), so d_x f is
            # diagonal within oo and vv -- required by the non-invariant (T) kernels.
            W = (np.asarray(self.wfn.H.ERI) if self.wfn.orbital_basis == 'spinorbital'
                 else np.asarray(self.wfn.H.L))
            eps = self.eps
            ao = slice(o.start + ncore, o.stop)             # active occupied
            Soo = Sx[o, o]
            Uvo = U[v, o]
            for blk in (ao, v):
                Sblk = Sx[blk, blk]
                Sterm = 0.5 * (self.contract('nm,pnqm->pq', Soo, W[blk, o, blk, o])
                               + self.contract('nm,pmqn->pq', Soo, W[blk, o, blk, o]))
                Uterm = (self.contract('cm,pcqm->pq', Uvo, W[blk, v, blk, o])
                         + self.contract('cm,pmqc->pq', Uvo, W[blk, o, blk, v]))
                num = fx[blk, blk] - Sblk * eps[blk][None, :] - Sterm + Uterm
                gap = eps[blk][:, None] - eps[blk][None, :]
                offdiag = np.abs(gap) > 1e-8
                U[blk, blk] = np.where(offdiag, -num / np.where(offdiag, gap, 1.0), -0.5 * Sblk)
        return U

    def perturbed_fock(self, pert: "Perturbation", ncore: int = 0, canonical: bool = False) -> np.ndarray:
        r"""Full first derivative of the MO Fock matrix ``d_x f_pq`` (``nmo x nmo``) for
        ``pert``, with the CPHF response folded in (derivints.pdf)::

            d_x f_pq = f^(x)_pq + U^x_pq f_pp + U^x_qp f_qq
                       - 1/2 S^(x)_nm A_pnqm + U^x_cm A_pcqm

        .. math::

            \partial_x f_{pq} = f^{(x)}_{pq} + U^{x}_{pq}\,\epsilon_p + U^{x}_{qp}\,\epsilon_q
                - \tfrac{1}{2} S^{(x)}_{nm} A_{pnqm} + U^{x}_{cm} A_{pcqm}

        with ``A_pqrs = w[p,q,r,s] + w[p,s,r,q]`` the orbital-Hessian two-electron weight
        (``w`` = the spin-adapted ``L`` on the spatial path, the antisymmetrized ``<pq||rs>``
        on the spin-orbital path), ``n,m`` over occupied and ``c`` over virtual. The skeleton
        ``f^(x)``/``S^(x)`` come from :meth:`_skeleton_derivatives`; for an electric field ``S^(x) = 0`` so the
        ``S^(x)`` term drops. ``ncore`` selects the frozen-core core<->active response in ``U``
        (see :meth:`full_U`). ``canonical`` selects the canonical active-oo/vv perturbed
        orbitals (CCSD(T); makes ``d_x f`` diagonal within oo and vv). Cached per
        ``(pert, ncore, canonical)``."""
        key = (pert, ncore, canonical)
        if key in self._dfock:
            return self._dfock[key]
        o, v = self.o, self.v
        eps = self.eps
        # Orbital-Hessian two-electron weight: spin-adapted L (spatial) / <pq||rs> (SO).
        W = (np.asarray(self.wfn.H.ERI) if self.wfn.orbital_basis == 'spinorbital'
             else np.asarray(self.wfn.H.L))
        fx, Sx = self._skeleton_derivatives(pert)
        U = self.full_U(pert, ncore, canonical)
        df = fx + U * eps[:, None] + U.T * eps[None, :]     # f^(x) + U_pq f_pp + U_qp f_qq
        # - 1/2 sum_nm(occ) S^(x)_nm ( w[p,n,q,m] + w[p,m,q,n] )
        Soo = Sx[o, o]
        df = df - 0.5 * (self.contract('nm,pnqm->pq', Soo, W[:, o, :, o])
                         + self.contract('nm,pmqn->pq', Soo, W[:, o, :, o]))
        # + sum_cm U_cm ( w[p,c,q,m] + w[p,m,q,c] ),  c in vir, m in occ
        Uvo = U[v, o]
        df = df + (self.contract('cm,pcqm->pq', Uvo, W[:, v, :, o])
                   + self.contract('cm,pmqc->pq', Uvo, W[:, o, :, v]))
        self._dfock[key] = df
        return df

    def perturbed_eri(self, pert: "Perturbation", ncore: int = 0, blocks=None, canonical: bool = False) -> np.ndarray:
        r"""Full first derivative of the MO two-electron integrals ``d_x <pq|rs>`` for ``pert``
        (derivints.pdf)::

            d_x <pq|rs> = <pq|rs>^(x)
                          + U^x_tp <tq|rs> + U^x_tq <pt|rs>
                          + U^x_tr <pq|ts> + U^x_ts <pq|rt>

        .. math::

            \partial_x \langle pq|rs\rangle = \langle pq|rs\rangle^{(x)}
                + U^{x}_{tp}\langle tq|rs\rangle + U^{x}_{tq}\langle pt|rs\rangle
                + U^{x}_{tr}\langle pq|ts\rangle + U^{x}_{ts}\langle pq|rt\rangle

        in the basis's integral convention: the plain physicist ``<pq|rs>`` (= ``H.ERI``) on
        the spatial path, the antisymmetrized ``<pq||rs>`` (= ``H.ERI``) on the spin-orbital
        path -- so the same rotation of ``H.ERI`` serves both. The skeleton ``<pq|rs>^(x)``
        comes from :meth:`_skeleton_derivatives` (zero for an electric field). ``ncore`` selects the
        frozen-core core<->active response in ``U`` (see :meth:`full_U`). The full ``nmo^4``
        tensor is built and cached per ``(pert, ncore, canonical)`` (geometry-bound, shared across
        properties). ``blocks`` is an optional 4-tuple of block labels ('o'/'v'/'all'), e.g.
        ``('o','o','v','v')`` for the MP2 2PDM; the **default returns the full tensor** (CC
        consumers need the other blocks). ``canonical`` selects the canonical active-oo/vv
        perturbed orbitals (CCSD(T))."""
        def _build():
            ERI = np.asarray(self.wfn.H.ERI)
            gx = self._skeleton_eri(pert)
            U = self.full_U(pert, ncore, canonical)
            return gx + self._rotate_eri(U, ERI)
        # persistent, perturbation-keyed store (disk when enabled, RAM when not) -- shared across
        # property calls on this wavefunction, replacing the old per-CPHF ``_deri`` RAM dict.
        full = self.wfn.derivatives.store.get_or_compute('deri', pert, _build, ctx=(ncore, canonical))
        if blocks is None:
            return full
        sl = tuple({'o': self.o, 'v': self.v}.get(b, slice(None)) for b in blocks)
        return full[sl]

    def _rotate_eri(self, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        r"""Orbital-rotation of a 4-index integral tensor by ``U`` (each index rotated)::

            U_tp T_tqrs + U_tq T_ptrs + U_tr T_pqts + U_ts T_pqrt

        .. math::

            U_{tp} T_{tqrs} + U_{tq} T_{ptrs} + U_{tr} T_{pqts} + U_{ts} T_{pqrt}

        Shared by the first-order integral derivative (rotation of the unperturbed ERIs) and
        the second-order one (rotation of ``U^{ab}`` with the ERIs, and of ``U^a`` with the
        first perturbed ERIs)."""
        return (self.contract('tp,tqrs->pqrs', U, T)
                + self.contract('tq,ptrs->pqrs', U, T)
                + self.contract('tr,pqts->pqrs', U, T)
                + self.contract('ts,pqrt->pqrs', U, T))

    # ---- magnetic-field perturbation (imaginary; for AATs) ----
    def _magnetic_dipole_ov(self, axis: int) -> np.ndarray:
        r"""ov block of the (real) MO magnetic-dipole integral for ``axis`` (0/1/2)::

            m_ia = Re[ -i (H.m)_ia ]

        .. math::

            m_{ia} = \mathrm{Re}\,[-i\,(H.m)_{ia}]

        ``H.m`` carries the ``-1/2`` and an imaginary unit (it is the pure-imaginary
        operator ``-i/2 L``, with the ``i`` convention shared by the CC response code);
        the magnetic CPHF can be solved as a real problem, so this strips the ``i`` by
        multiplying by ``-i`` -- the final VCD rotatory strength takes the imaginary
        component of the APT*AAT product, not of the AAT itself."""
        return np.asarray(-1.0j * self.wfn.H.m[axis])[self.o, self.v].real

    def _rhs_magnetic(self, axis: int) -> np.ndarray:
        r"""Magnetic-field CPHF RHS for ``axis`` (0/1/2), real::

            B^x_ia = -m_ia

        .. math::

            B^{x}_{ia} = -m_{ia}

        The magnetic field enters as ``H' = -m . B`` (analogous to ``-mu . E`` for the
        electric field), and -- like the electric field -- it does not move the basis
        functions, so there is no overlap/Pulay term: the RHS is just the (negated)
        real magnetic-dipole ov integral. The magnetic perturbation is imaginary, so
        this response uses the antisymmetric (``kind='magnetic'``) orbital Hessian."""
        return -self._magnetic_dipole_ov(axis)

    def solve_magnetic(self, axis: int) -> np.ndarray:
        """Magnetic-field CPHF response ``U^B`` for ``axis`` (0/1/2), ``(no, nv)``,
        solved once and cached. Real (the magnetic-dipole RHS is stripped of its i)."""
        if axis not in self._U_mag:
            self._U_mag[axis] = self.solve(self._rhs_magnetic(axis), kind="magnetic")
        return self._U_mag[axis]

    def _momentum_ov(self, axis: int) -> np.ndarray:
        r"""ov block of the (real) MO linear-momentum integral for ``axis`` (0/1/2)::

            pi_ia = Re[ -i (H.p)_ia ]

        .. math::

            \pi_{ia} = \mathrm{Re}\,[-i\,(H.p)_{ia}]

        ``H.p`` is ``i * <mu|Del|nu>`` in the MO basis (the pure-imaginary linear-momentum
        operator, carrying the same ``i`` convention as ``H.m``); this strips the ``i`` by
        multiplying by ``-i`` so the momentum CPHF is a real problem, exactly as
        :meth:`_magnetic_dipole_ov` does for the magnetic dipole."""
        return np.asarray(-1.0j * self.wfn.H.p[axis])[self.o, self.v].real

    def _rhs_momentum(self, axis: int) -> np.ndarray:
        r"""Linear-momentum (magnetic vector-potential) CPHF RHS for ``axis`` (0/1/2), real::

            B^x_ia = +pi_ia

        .. math::

            B^{x}_{ia} = +\pi_{ia}

        The vector potential enters the Hamiltonian as ``H'(A) = A . pi`` (Amos, Jalkanen &
        Stephens, JPC 92, 5571 (1988), Eq. 10), so ``dH'/dA = +pi`` -- the (positive) real
        momentum ov integral (contrast the magnetic ``B = -m``). Like the field/magnetic
        perturbations, ``A`` does not move the basis functions, so there is no overlap/Pulay
        term. The momentum perturbation is imaginary, so this uses the antisymmetric
        (``kind='magnetic'``) orbital Hessian -- the same Hessian as the magnetic response."""
        return self._momentum_ov(axis)

    def solve_momentum(self, axis: int) -> np.ndarray:
        """Linear-momentum CPHF response ``U^A`` for ``axis`` (0/1/2), ``(no, nv)``, solved
        once and cached. Real. This is the ket derivative ``dPsi/dA`` in the velocity-gauge
        APT (:meth:`HFwfn.apt`)."""
        if axis not in self._U_mom:
            self._U_mom[axis] = self.solve(self._rhs_momentum(axis), kind="magnetic")
        return self._U_mom[axis]

    def magnetic_ints(self, axis: int, ncore: int = 0, gauge: str = 'non-canonical'):
        """Magnetic-field-perturbed MO integrals for ``axis`` (0/1/2): returns the tuple
        ``(U^H, dF^H, dERI^H)`` used by the MP2 atomic axial tensors
        (:meth:`~pycc.mpderiv.MPderiv.aat`). Cached per ``(axis, ncore, gauge)``.

        The magnetic perturbation ``H' = -m.H`` (``m = -1/2 L``, the real antisymmetric
        magnetic-dipole matrix, stripped of the ``i`` carried in ``H.m``) is imaginary, so the
        orbital response is antisymmetric and uses the ``kind='magnetic'`` orbital Hessian.

        ``U^H`` is the **full** ``nmo x nmo`` response: the ov block from the magnetic CPHF solve,
        the vo block as its (symmetric) transpose, and the oo/vv blocks set by ``gauge``:

        * ``'non-canonical'`` (default): the redundant within-space rotations (core-core,
          active-active, virtual-virtual) are left at zero -- the ``-1/2 S^H = 0`` common-origin
          choice.  Only the **non-redundant** core<->active-occupied block (present when
          ``ncore > 0``, frozen core) is filled from the canonical condition.  This avoids the
          near-degenerate divides of the fully canonical choice (e.g. close-lying core orbitals)
          and is numerically preferred.  The AAT total is invariant to this choice.
        * ``'canonical'``: every oo/vv block from ``d_H f_pq = 0`` (divide by the orbital-energy
          difference), degeneracy-guarded.

        ``dF^H``/``dERI^H`` are the perturbed Fock / two-electron integrals with the antisymmetric
        (ket ``+``, bra ``-``) rotation.  ``W`` is the antisymmetrized ``<pq||rs>`` (spin-orbital)
        or the spin-adapted ``L`` (spatial), matching the rest of the CPHF engine."""
        key = (axis, ncore, gauge)
        if key not in self._mag_int:
            hmag = np.asarray(-1.0j * self.wfn.H.m[axis]).real
            self._mag_int[key] = self._antisym_field_ints(hmag, ncore, gauge)
        return self._mag_int[key]

    def momentum_ints(self, axis: int, ncore: int = 0, gauge: str = 'non-canonical'):
        """Linear-momentum-perturbed MO integrals for ``axis`` (0/1/2): returns the tuple
        ``(U^A, dF^A, dERI^A)`` used by the MP2 velocity-gauge atomic polar tensors
        (:meth:`~pycc.mpderiv.MPderiv._correlation_velocity_dipole_derivatives`). Cached per
        ``(axis, ncore, gauge)``.

        This is the magnetic engine (:meth:`magnetic_ints`) with the magnetic-dipole operator
        replaced by the linear-momentum operator ``p = -i nabla`` (stripped of the ``i`` carried
        in ``H.p``).  Momentum is imaginary/anti-Hermitian just like the magnetic dipole, so the
        orbital response is antisymmetric and shares the ``kind='magnetic'`` orbital Hessian and
        the identical gauge handling of the redundant oo/vv blocks."""
        key = (axis, ncore, gauge)
        if key not in self._mom_int:
            hmom = np.asarray(-1.0j * self.wfn.H.p[axis]).real
            self._mom_int[key] = self._antisym_field_ints(hmom, ncore, gauge)
        return self._mom_int[key]

    def _antisym_field_ints(self, hmag, ncore: int, gauge: str):
        r"""Shared engine for the imaginary (anti-Hermitian) one-electron perturbations -- the
        magnetic dipole (:meth:`magnetic_ints`) and the linear momentum
        (:meth:`momentum_ints`).  ``hmag`` is the stripped real antisymmetric operator matrix
        in the MO basis; returns ``(U, dF, dERI)`` with the antisymmetric orbital response and
        the gauge treatment of the redundant oo/vv blocks documented on :meth:`magnetic_ints`.
        The ov response comes from the shared ``kind='magnetic'`` orbital Hessian via
        :meth:`solve` (``G^m`` = :meth:`_build_mo_hessian`, the same operator
        :meth:`solve_magnetic` / :meth:`solve_momentum` use)::

            U^x_ai = (G^m)^-1 h_ai,   U^x_ia = U^x_ai
            d<pq|rs> = U^x_tr <pq|ts> + U^x_ts <pq|rt> - U^x_tp <tq|rs> - U^x_tq <pt|rs>

        .. math::

            \begin{aligned}
            U^{x}_{ai} &= (G^{m})^{-1} h_{ai}, \qquad U^{x}_{ia} = U^{x}_{ai} \\
            \partial\langle pq|rs\rangle &= U^{x}_{tr}\langle pq|ts\rangle + U^{x}_{ts}\langle pq|rt\rangle
                - U^{x}_{tp}\langle tq|rs\rangle - U^{x}_{tq}\langle pt|rs\rangle
            \end{aligned}
        """
        o, v, nmo = self.o, self.v, self.wfn.nmo
        no = self.no
        t = slice(0, nmo)
        eps = self.eps
        c = self.contract
        ERI = np.asarray(self.wfn.H.ERI)
        W = ERI if self.wfn.orbital_basis == 'spinorbital' else np.asarray(self.wfn.H.L)
        core = -W + W.swapaxes(1, 3)                # antisymmetric mean-field weight (oo/vv blocks below)

        def _safe_div(num, e):
            # canonical oo/vv rotation num_pq / (eps_q - eps_p), degeneracy-guarded: where two
            # orbitals are (near-)degenerate the spin-diagonal magnetic numerator also vanishes,
            # so set 0/0 (and the diagonal) to zero rather than divide.
            gap = e[None, :] - e[:, None]
            out = np.zeros_like(num)
            m = np.abs(gap) > 1e-7
            out[m] = num[m] / gap[m]
            return out

        U = np.zeros((nmo, nmo))
        # ov response from the shared kind='magnetic' orbital Hessian and solve (one Hessian
        # build, one ov solve -- the same operator solve_magnetic/solve_momentum use).  solve()
        # works in the (no, nv) ov layout, so feed the vo RHS transposed and transpose the
        # (no, nv) result back into the (nv, no) U[v, o] block.
        U[v, o] = self.solve(hmag[v, o].T, kind="magnetic").T
        U[o, v] = U[v, o].T
        if gauge == 'canonical':
            U[o, o] = _safe_div(-hmag[o, o] + c('em,iejm->ij', U[v, o], core[o, v, o, o]), eps[o])
            U[v, v] = _safe_div(-hmag[v, v] + c('em,aebm->ab', U[v, o], core[v, v, v, o]), eps[v])
        elif ncore:
            # non-canonical: only the non-redundant core<->active-occupied block is canonical
            Uoo = _safe_div(-hmag[o, o] + c('em,iejm->ij', U[v, o], core[o, v, o, o]), eps[o])
            keep = np.zeros((no, no), dtype=bool)
            keep[:ncore, ncore:] = True
            keep[ncore:, :ncore] = True
            U[o, o] = Uoo * keep
        np.fill_diagonal(U, 0.0)
        dF = np.zeros((nmo, nmo))
        dF[o, o] = (-hmag[o, o] + (U[o, o] * eps[o].reshape(-1, 1) - U[o, o].T * eps[o])
                    + c('em,iejm->ij', U[v, o], core[o, v, o, o]))
        dF[v, v] = (-hmag[v, v] + (U[v, v] * eps[v].reshape(-1, 1) - U[v, v].T * eps[v])
                    + c('em,aebm->ab', U[v, o], core[v, v, v, o]))
        dERI = (c('tr,pqts->pqrs', U[:, t], ERI[t, t, :, t]) + c('ts,pqrt->pqrs', U[:, t], ERI[t, t, t, :])
                - c('tp,tqrs->pqrs', U[:, t], ERI[:, t, t, t]) - c('tq,ptrs->pqrs', U[:, t], ERI[t, :, t, t]))
        return (U, dF, dERI)

    def _build_rhs_nuclear(self, atom: int):
        r"""Nuclear CPHF RHS for ``atom`` -- one heavy pass over the derivative integrals;
        returns three lists of 3 (x,y,z) arrays: the CPHF RHS ``B^X_ia`` ``(no, nv)``, the
        skeleton derivative Fock oo block ``F^(X)_ij`` ``(no, no)``, and the overlap
        derivative oo block ``S^(X)_ij`` ``(no, no)`` (the latter two are free by-products the
        molecular Hessian needs). Unlike the field RHS, a nuclear displacement moves the
        basis functions, so the RHS folds in the skeleton Fock ``F^(X)``, the overlap term
        ``-eps_i S^(X)``, and the Pulay coupling of the overlap-determined occupied-occupied
        response ``U^X_kl = -1/2 S^(X)_kl`` back into the Fock matrix; cast with the
        spin-adapted ``L = 2<pq|rs> - <pq|sr>`` (physicist's notation), as in the orbital
        Hessian::

            F^(X)_pq = h^(X)_pq + L[p,k,q,k]^(X)          (skeleton Fock derivative)
            B^X_ia = -[ F^(X)_ia - eps_i S^(X)_ia
                        - 1/2 S^(X)_kl ( L[a,k,i,l] + L[a,l,i,k] ) ]

        .. math::

            \begin{aligned}
            F^{(X)}_{pq} &= h^{(X)}_{pq} + L^{(X)}_{pkqk} \\
            B^{X}_{ia} &= -\big[\, F^{(X)}_{ia} - \epsilon_i S^{(X)}_{ia}
                - \tfrac{1}{2} S^{(X)}_{kl}\,(L_{akil} + L_{alik}) \,\big]
            \end{aligned}

        Returns ``(B, Foo, Soo)``. The Pulay coupling uses the ``L[a,k,i,l]``/``L[a,l,i,k]``
        pair (not ``L[i,k,a,l]`` -- their Coulomb parts coincide but the exchange parts
        differ); validated to ~1e-8 via the APT-transpose check against finite difference of
        the SCF dipole.

        The skeleton derivative integrals (including the dominant ``nmo^4`` ERI derivative)
        come from the unified :meth:`_skeleton_derivatives` cache, computed once per atom/cart
        and reused by the explicit perturbed-integral engine (:meth:`perturbed_fock` /
        :meth:`perturbed_eri`) and by any second-derivative consumer. The spin-orbital path
        dispatches to :meth:`_so_build_rhs_nuclear`.
        """
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_build_rhs_nuclear(atom)
        o, v = self.o, self.v
        eps_o = self.eps[o]
        Lvooo = np.asarray(self.wfn.H.L)[v, o, o, o]   # spin-adapted, unperturbed
        B, Foo, Soo = [], [], []
        for c in range(3):
            Fx, Sx = self._skeleton_derivatives(Perturbation('nuclear', (atom, c)))
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^(X)_kl into the
            # ov block:  -1/2 S^(X)_kl ( L[a,k,i,l] + L[a,l,i,k] ).
            coupling = (self.contract('akil,kl->ia', Lvooo, Sx[o, o])
                        + self.contract('alik,kl->ia', Lvooo, Sx[o, o]))
            # The CPHF RHS is minus the first-order off-diagonal ("perturbation")
            # Fock that the response drives to zero -- B = -Q (nuclear analog of the
            # field's B = -mu).
            Q = Fx[o, v] - eps_o[:, None] * Sx[o, v] - 0.5 * coupling
            B.append(-Q)
            Foo.append(Fx[o, o])
            Soo.append(Sx[o, o])
        return B, Foo, Soo

    def _so_build_rhs_nuclear(self, atom: int):
        r"""Spin-orbital nuclear CPHF RHS for ``atom`` -- the spin-orbital analogue of
        :meth:`_build_rhs_nuclear`. Same structure with the antisymmetrized ``<pq||rs>`` in
        place of the spin-adapted ``L``, the spin-orbital skeleton derivative integrals
        from ``Derivatives.so_*``, and singly occupied spin orbitals::

            F^(X)_pq = h^(X)_pq + <pk||qk>^(X)             (skeleton Fock derivative)
            B^X_ia = -[ F^(X)_ia - eps_i S^(X)_ia
                        - 1/2 S^(X)_kl ( <ak||il> + <al||ik> ) ]

        .. math::

            \begin{aligned}
            F^{(X)}_{pq} &= h^{(X)}_{pq} + \langle pk\Vert qk\rangle^{(X)} \\
            B^{X}_{ia} &= -\big[\, F^{(X)}_{ia} - \epsilon_i S^{(X)}_{ia}
                - \tfrac{1}{2} S^{(X)}_{kl}\,(\langle ak\Vert il\rangle + \langle al\Vert ik\rangle) \,\big]
            \end{aligned}

        Returns ``(B, Foo, Soo)`` as for the spatial path.

        The skeleton derivative integrals (including the dominant ``nmo^4`` ``so_eri``
        derivative) come from the unified :meth:`_skeleton_derivatives` cache, so they are computed once
        per atom/cart and reused by the explicit perturbed-integral engine
        (:meth:`perturbed_fock` / :meth:`perturbed_eri`) and by any second-derivative
        consumer; ``Foo``/``Soo`` are then just the ``oo`` slices of the cached
        ``f^(x)``/``S^(x)``."""
        o, v = self.o, self.v
        eps_o = self.eps[o]
        ERI = np.asarray(self.wfn.H.ERI)         # antisymmetrized, unperturbed
        Evooo = ERI[v, o, o, o]
        B, Foo, Soo = [], [], []
        for c in range(3):
            Fx, Sx = self._skeleton_derivatives(Perturbation('nuclear', (atom, c)))
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^(X)_kl into the ov
            # block: -1/2 sum_kl S^(X)_kl ( <ak||il> + <al||ik> ).
            coupling = (self.contract('akil,kl->ia', Evooo, Sx[o, o])
                        + self.contract('alik,kl->ia', Evooo, Sx[o, o]))
            Q = Fx[o, v] - eps_o[:, None] * Sx[o, v] - 0.5 * coupling
            B.append(-Q)
            Foo.append(Fx[o, o])
            Soo.append(Sx[o, o])
        return B, Foo, Soo

    # ---- cached nuclear response (shared by the Hessian and the APTs) ----
    def solve_nuclear(self, atom: int) -> List[np.ndarray]:
        """Nuclear CPHF response for ``atom`` -- list of 3 (x, y, z) ``(no, nv)``
        ``U^X`` arrays -- solved once and cached.

        Building the RHS (:meth:`_build_rhs_nuclear`) regenerates the full-MO derivative
        ERIs (nmo**4 per atom), the dominant cost; both the molecular Hessian and the
        dipole derivatives / APTs need the same 3*natom ``U^X``, so this memoizes the
        response (and the RHS, in ``self._B_nuc``) per atom and they share it. The
        cache is geometry-bound: it lives for the life of this CPHF object.
        """
        if atom not in self._U_nuc:
            B, Foo, Soo = self._build_rhs_nuclear(atom)
            self._B_nuc[atom] = B
            self._F_nuc[atom] = Foo
            self._S_nuc[atom] = Soo
            self._U_nuc[atom] = [self.solve(B[c], kind="electric") for c in range(3)]
        return self._U_nuc[atom]

    def rhs_nuclear(self, atom: int) -> List[np.ndarray]:
        """Nuclear RHS ``B^X`` for ``atom`` from the shared cache (populated by
        :meth:`solve_nuclear`); the molecular Hessian's response term needs the RHS
        as well as the response."""
        if atom not in self._B_nuc:
            self.solve_nuclear(atom)
        return self._B_nuc[atom]

    def nuclear_skeleton_fock(self, atom: int) -> List[np.ndarray]:
        """Skeleton derivative Fock oo block ``F^(X)_ij`` for ``atom`` from the shared
        cache (a by-product of :meth:`solve_nuclear`); used by the molecular Hessian's
        first-derivative cross terms."""
        if atom not in self._F_nuc:
            self.solve_nuclear(atom)
        return self._F_nuc[atom]

    def nuclear_skeleton_overlap(self, atom: int) -> List[np.ndarray]:
        """Overlap derivative oo block ``S^(X)_ij`` for ``atom`` from the shared cache
        (a by-product of :meth:`solve_nuclear`); used by the molecular Hessian's
        first-derivative cross terms."""
        if atom not in self._S_nuc:
            self.solve_nuclear(atom)
        return self._S_nuc[atom]

    def nuclear_hessian_response(self) -> np.ndarray:
        r"""Ao-INDEPENDENT first-order CPHF-response part of the SCF electronic Hessian,
        shape ``(3*natom, 3*natom)``, built entirely from this object's cached nuclear
        response (``solve_nuclear`` and its by-products) and its wavefunction's Fock
        diagonal and ``oooo`` integral block.  With ``x = (A,a)``, ``y = (B,b)``;
        ``i,j,n,m`` occupied, ``a`` virtual; ``k = 2``, ``W = L`` (spatial closed shell)
        or ``k = 1``, ``W = <pq||rs>`` (spin-orbital)::

            H^resp_xy = -k [ 2 U^x_ia B^y_ia + S^x_ij F^y_ij + S^y_ij F^x_ij
                             - 2 eps_i S^x_ij S^y_ij - S^x_ij S^y_nm W_imjn ]

        .. math::

            H^{\mathrm{resp}}_{xy} = -k\Big[\,2\sum_{ia} U^x_{ia} B^y_{ia}
                + \sum_{ij} S^{x}_{ij} F^{y}_{ij} + \sum_{ij} S^{y}_{ij} F^{x}_{ij}
                - 2\sum_{ij} \epsilon_i S^{x}_{ij} S^{y}_{ij}
                - \sum_{ijnm} S^{x}_{ij} S^{y}_{nm} W_{imjn} \,\Big]

        Carries no second-derivative TEIs.  Every quantity is taken in **this** CPHF
        object's occupied/virtual space, so the expression is valid for any CPHF whose
        ``o``/``v`` span the all-electron SCF space: :attr:`HFwfn.cphf` (all-electron by
        construction) and :meth:`~pycc.correlatedderivs.CorrelatedDerivs._full_occ_cphf`
        (``full_occ=True``, so ``o`` covers frozen core + active).  The two differ in the
        *ordering* of the occupied spin orbitals under frozen core, which the expression is
        invariant to because every occupied index is summed.  Owning the expression here --
        rather than on ``HFwfn``, which would read ``eps``/``W`` in its own ordering -- is what
        makes it safe to evaluate on the correlated driver's CPHF, whose nuclear caches the
        correlated Hessian has already filled (plan doc s.12).
        """
        o = self.o
        so = self.wfn.orbital_basis == 'spinorbital'
        k = 1.0 if so else 2.0                                   # closed-shell spin factor
        eps_o = self.eps[o]
        W = np.asarray(self.wfn.H.ERI if so else self.wfn.H.L)[o, o, o, o]   # i,m,j,n
        natom = self.wfn.ref.molecule().natom()
        U = [self.solve_nuclear(A) for A in range(natom)]             # U[A][a] -> (no,nv)
        B = [self.rhs_nuclear(A) for A in range(natom)]               # B[A][a] -> (no,nv)
        Foo = [self.nuclear_skeleton_fock(A) for A in range(natom)]   # F^X_ij -> (no,no)
        Soo = [self.nuclear_skeleton_overlap(A) for A in range(natom)]  # S^X_ij -> (no,no)
        c = self.contract
        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(A, natom):
                for a in range(3):
                    for b in range(3):
                        Ux, By = U[A][a], B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        resp = k * (-2.0 * c('ia,ia->', Ux, By)
                                    - c('ij,ij->', Sx, Fy)
                                    - c('ij,ij->', Sy, Fx)
                                    + 2.0 * c('i,ij,ij->', eps_o, Sx, Sy)
                                    + c('ij,nm,imjn->', Sx, Sy, W))
                        H[A * 3 + a, Bat * 3 + b] = resp
                        if A != Bat:
                            H[Bat * 3 + b, A * 3 + a] = resp
        return H
