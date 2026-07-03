"""
cphf.py: coupled-perturbed Hartree-Fock (CPHF) orbital response.

Solves the RHF first-order orbital-response (CPHF / CPSCF) equations in the MO
basis. For a perturbation lambda, the occupied-virtual orbital rotations
``U^lambda_ia`` satisfy a single perturbation-independent linear operator (the
singlet orbital Hessian ``G``) applied to a perturbation-specific right-hand
side ``B``::

    sum_jb  G_iajb  U_jb  =  B_ia

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


# A perturbation descriptor used to key the response and full-derivative caches.
# ``kind`` is 'field' (electric dipole, axis ``comp`` in 0/1/2), 'nuclear' (atomic
# displacement, ``comp`` = (atom, cart)), or 'magnetic' (axis ``comp``). It is the
# shared identity under which CPHF memoizes ``U^x`` and the full (CPHF-folded) first
# derivatives ``d_x f`` and ``d_x <pq||rs>`` so multi-property workflows (e.g. IR + VCD,
# or an MP2 gradient + polarizability) never recompute them. Only 'field' is wired so
# far; 'nuclear'/'magnetic' slot into the same machinery later.
Perturbation = namedtuple('Perturbation', ['kind', 'comp'])


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
        self._F_nuc: dict = {}  # atom -> list of 3 (no, no) skeleton deriv Fock F^X_ij
        self._S_nuc: dict = {}  # atom -> list of 3 (no, no) overlap deriv S^X_ij
        self._U_mag: dict = {}  # axis -> (no, nv) magnetic-field response U^B (real)
        self._U_mom: dict = {}  # axis -> (no, nv) linear-momentum response U^A (real)
        self._mag_int: dict = {}  # axis -> (U^H, dF^H, dERI^H) magnetic engine (for MP2 AATs)
        self._mom_int: dict = {}  # axis -> (U^A, dF^A, dERI^A) momentum engine (for MP2 VG APTs)
        # Full (CPHF-folded) first-derivative caches, keyed by Perturbation. These hold
        # the response-dressed derivatives d_x f and d_x <pq||rs> (notes: the "simple but
        # inefficient" explicit form), persisting for the life of this CPHF object so that
        # multiple property calculations on one wavefunction share them.
        self._U_field: dict = {}  # axis -> (no, nv) electric-field response U^a
        self._skel: dict = {}     # Perturbation -> (fx, Sx, gx) skeleton derivatives
        self._dfock: dict = {}    # Perturbation -> (nmo, nmo) full perturbed Fock deriv
        self._deri: dict = {}     # Perturbation -> (nmo^4) full perturbed <pq||rs> deriv
        # Second-order (field) caches, keyed by (perta, pertb, ncore).
        self._U2cache: dict = {}  # -> (nmo, nmo) second-order response U^{ab}
        self._dfock2: dict = {}   # -> (nmo, nmo) full second perturbed Fock deriv
        self._deri2: dict = {}    # -> (nmo^4) full second perturbed <pq||rs> deriv
        # Nuclear-nuclear skeleton second-derivative integrals, keyed by atom pair (atom1,
        # atom2): the expensive mo_*_deriv2 calls are shared across the 3x3 Cartesian blocks
        # of a pair, so memoize per pair rather than recompute per coordinate pair.
        self._d2int: dict = {}    # (a1, a2) -> {'eri','core','overlap'}: 9-block lists

    # ---- orbital Hessian ----
    def hessian(self, kind: str = "electric") -> np.ndarray:
        """Singlet orbital Hessian as a ``(no*nv, no*nv)`` matrix, cached by ``kind``.

        ``kind`` is 'electric' (real perturbations: nuclear displacements, electric
        field) or 'magnetic' (imaginary perturbations: magnetic field / AATs), which
        only flips the sign of the second L term.
        """
        if kind not in self._G:
            self._G[kind] = self._build_hessian(kind)
        return self._G[kind]

    def _build_hessian(self, kind: str) -> np.ndarray:
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
        """Solve ``G U = B`` for the ov response. ``B`` is ``(no, nv)``; returns ``(no, nv)``.

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
        G = self.hessian(kind)
        U = np.linalg.solve(G, np.asarray(B).reshape(-1))
        return U.reshape(self.no, self.nv)

    # ---- property integrals / perturbation right-hand sides ----
    def _mu_ov(self, axis: int) -> np.ndarray:
        """ov block of the MO electric-dipole integral for ``axis`` (0/1/2)."""
        return np.asarray(self.wfn.H.mu[axis])[self.o, self.v]

    def rhs_field(self, axis: int) -> np.ndarray:
        """Electric-field CPHF right-hand side for ``axis`` (0/1/2): ``B = +mu``.

        The field enters as ``H' = -mu . E`` (``H.mu`` is the dipole operator ``-e r``), so the
        skeleton Fock derivative is ``f^(a) = -mu`` and the CPHF RHS is ``B = -f^(a) = +mu``
        (no overlap/Pulay term -- the field does not move the basis functions).
        """
        return self._mu_ov(axis)

    def solve_field(self, axis: int) -> np.ndarray:
        """Electric-field CPHF response ``U^a`` for ``axis`` (0/1/2), ``(no, nv)``, solved
        once and cached (shared by the polarizability and the correlated field properties)."""
        if axis not in self._U_field:
            self._U_field[axis] = self.solve(self.rhs_field(axis), kind="electric")
        return self._U_field[axis]

    # ---- explicit (CPHF-folded) full first derivatives of f and <pq||rs> ----
    # The "simple but inefficient" form (derivints.pdf): rather than separate the orbital
    # response into a Z-vector + relaxed density, fold the CPHF coefficients ``U^x``
    # directly into the full derivatives of the Fock matrix and the antisymmetrized
    # two-electron integrals, then contract with the (unrelaxed) densities. These are the
    # building blocks of the correlated properties -- the field dipole/polarizability and
    # the nuclear gradient. Each perturbation differs only in its *skeleton* derivatives and
    # its ov CPHF response; the assembly below is shared, and basis-aware (spatial closed-shell
    # default, spin-orbital via the same ``orbital_basis`` switch as the rest of the code).

    def _skeleton(self, pert: "Perturbation"):
        """Skeleton (fixed-MO-coefficient) derivatives for ``pert``: a triple
        ``(fx, Sx, gx)`` of the skeleton Fock derivative ``f^(x)_pq`` (``nmo x nmo``), the
        overlap derivative ``S^x_pq`` (``nmo x nmo``), and the two-electron derivative
        ``gx`` (``nmo^4``, in the basis's integral convention -- spin-adapted ``<pq|rs>^(x)``
        on the spatial path, antisymmetrized ``<pq||rs>^(x)`` on the spin-orbital path),
        cached per ``pert``.

        - **field**: the basis functions do not move, so ``S^x = 0`` and ``gx = 0``; the
          skeleton Fock derivative is ``f^(a) = -mu`` (``H' = -mu.E``).
        - **nuclear** (``comp = (atom, cart)``): the skeleton derivative integrals come from
          the ``Derivatives`` provider; the skeleton Fock derivative is ``f^(x)_pq = h^(x)_pq
          + sum_m(occ) w[p,m,q,m]^(x)`` with ``w`` the spin-adapted ``L`` (spatial) or
          antisymmetrized ``<pq||rs>`` (spin-orbital)."""
        if pert in self._skel:
            return self._skel[pert]
        nmo, o = self.wfn.nmo, self.o
        so = self.wfn.orbital_basis == 'spinorbital'
        if pert.kind == 'field':
            fx = -np.asarray(self.wfn.H.mu[pert.comp])       # f^(a) = -mu  (H' = -mu.E)
            Sx = np.zeros((nmo, nmo))
            gx = 0.0                                         # no skeleton 2e deriv (field)
        elif pert.kind == 'nuclear':
            atom, cart = pert.comp
            d = self.wfn.derivatives
            if so:
                hx = np.asarray(d.so_core(atom)[cart])
                Sx = np.asarray(d.so_overlap(atom)[cart])
                gx = np.asarray(d.so_eri(atom)[cart])        # <pq||rs>^(x)
                w = gx                                        # Fock 2e weight = <pq||rs>^(x)
            else:
                hx = np.asarray(d.core(atom)[cart])
                Sx = np.asarray(d.overlap(atom)[cart])
                gx = np.asarray(d.eri(atom)[cart]).swapaxes(1, 2)  # chemist -> physicist <pq|rs>^(x)
                w = 2.0 * gx - gx.swapaxes(2, 3)              # spin-adapted L^(x)
            fx = hx + self.contract('pmqm->pq', w[:, o, :, o])   # skeleton Fock derivative
        else:
            raise NotImplementedError(
                "explicit perturbed derivatives are wired for 'field' and 'nuclear' only; "
                "'magnetic' uses the same machinery but is not yet built.")
        self._skel[pert] = (fx, Sx, gx)
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

    def _full_U(self, pert: "Perturbation", ncore: int = 0) -> np.ndarray:
        """The full ``nmo x nmo`` orbital-rotation matrix ``U^x_pq`` for ``pert``.

        The matrix element ``U_qp = <phi_q|d phi_p/dx>`` (the coefficient of orbital ``q`` in
        the first-order change of orbital ``p``, as it enters the integral derivatives). The
        non-canonical perturbed-orbital conditions fix the diagonal blocks from the overlap
        derivative, ``U_ij = -1/2 S^x_ij`` and ``U_ab = -1/2 S^x_ab``; the CPHF solve gives the
        occupied response ``Uia[i,a] = <phi_a|d phi_i> = U_ai`` (so ``U[v,o] = Uia.T``), and
        orthonormality ``U_pq + U_qp = -S^x_pq`` fixes ``U[o,v] = -S^x[o,v] - Uia``. For an
        electric field ``S^x = 0``, so the oo/vv blocks vanish and ``U[o,v] = -Uia``.

        ``ncore > 0`` (frozen-core correlated derivatives): the lowest ``ncore`` occupied
        orbitals are the frozen core. Core<->active-occupied rotations are non-redundant (they
        move the frozen/active partition), so that block is *not* left at the orthonormality
        value but determined by the canonical Brillouin condition ``d_x f_ij = 0`` -- a direct
        divide by ``(eps_i - eps_j)``, using the already-solved ov response. The redundant
        core-core, active-active, and vir-vir blocks stay at ``-1/2 S^x``."""
        o, v, nmo = self.o, self.v, self.wfn.nmo
        fx, Sx, _ = self._skeleton(pert)
        Uia = self._ov_response(pert)                       # (no, nv): U_ai = <phi_a|d phi_i>
        U = np.zeros((nmo, nmo))
        U[o, o] = -0.5 * Sx[o, o]
        U[v, v] = -0.5 * Sx[v, v]
        U[v, o] = Uia.T
        U[o, v] = -Sx[o, v] - Uia
        if ncore:
            # Core <-> active-occupied block from d_x f_ij = 0 (i core, j active), with
            # U_ji = -S^x_ij - U_ij eliminated:
            #   U_ij = -[ f^x_ij - S^x_ij eps_j - 1/2 sum_nm S^x_nm A_injm
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
        return U

    def perturbed_fock(self, pert: "Perturbation", ncore: int = 0) -> np.ndarray:
        """Full first derivative of the MO Fock matrix ``d_x f_pq`` (``nmo x nmo``) for
        ``pert``, with the CPHF response folded in (derivints.pdf)::

            d_x f_pq = f^(x)_pq + U^x_pq f_pp + U^x_qp f_qq
                       - 1/2 sum_nm S^x_nm A_pnqm + sum_cm U^x_cm A_pcqm

        with ``A_pqrs = w[p,q,r,s] + w[p,s,r,q]`` the orbital-Hessian two-electron weight
        (``w`` = the spin-adapted ``L`` on the spatial path, the antisymmetrized ``<pq||rs>``
        on the spin-orbital path), ``n,m`` over occupied and ``c`` over virtual. The skeleton
        ``f^(x)``/``S^x`` come from :meth:`_skeleton`; for an electric field ``S^x = 0`` so the
        ``S^x`` term drops. ``ncore`` selects the frozen-core core<->active response in ``U``
        (see :meth:`_full_U`). Cached per ``(pert, ncore)``."""
        key = (pert, ncore)
        if key in self._dfock:
            return self._dfock[key]
        o, v = self.o, self.v
        eps = self.eps
        # Orbital-Hessian two-electron weight: spin-adapted L (spatial) / <pq||rs> (SO).
        W = (np.asarray(self.wfn.H.ERI) if self.wfn.orbital_basis == 'spinorbital'
             else np.asarray(self.wfn.H.L))
        fx, Sx, _ = self._skeleton(pert)
        U = self._full_U(pert, ncore)
        df = fx + U * eps[:, None] + U.T * eps[None, :]     # f^(x) + U_pq f_pp + U_qp f_qq
        # - 1/2 sum_nm(occ) S^x_nm ( w[p,n,q,m] + w[p,m,q,n] )
        Soo = Sx[o, o]
        df = df - 0.5 * (self.contract('nm,pnqm->pq', Soo, W[:, o, :, o])
                         + self.contract('nm,pmqn->pq', Soo, W[:, o, :, o]))
        # + sum_cm U_cm ( w[p,c,q,m] + w[p,m,q,c] ),  c in vir, m in occ
        Uvo = U[v, o]
        df = df + (self.contract('cm,pcqm->pq', Uvo, W[:, v, :, o])
                   + self.contract('cm,pmqc->pq', Uvo, W[:, o, :, v]))
        self._dfock[key] = df
        return df

    def perturbed_eri(self, pert: "Perturbation", ncore: int = 0, blocks=None) -> np.ndarray:
        """Full first derivative of the MO two-electron integrals ``d_x <pq|rs>`` for ``pert``
        (derivints.pdf)::

            d_x <pq|rs> = <pq|rs>^(x)
                          + sum_t ( U^x_tp <tq|rs> + U^x_tq <pt|rs>
                                    + U^x_tr <pq|ts> + U^x_ts <pq|rt> )

        in the basis's integral convention: the plain physicist ``<pq|rs>`` (= ``H.ERI``) on
        the spatial path, the antisymmetrized ``<pq||rs>`` (= ``H.ERI``) on the spin-orbital
        path -- so the same rotation of ``H.ERI`` serves both. The skeleton ``<pq|rs>^(x)``
        comes from :meth:`_skeleton` (zero for an electric field). ``ncore`` selects the
        frozen-core core<->active response in ``U`` (see :meth:`_full_U`). The full ``nmo^4``
        tensor is built and cached per ``(pert, ncore)`` (geometry-bound, shared across
        properties). ``blocks`` is an optional 4-tuple of block labels ('o'/'v'/'all'), e.g.
        ``('o','o','v','v')`` for the MP2 2PDM; the **default returns the full tensor** (CC
        consumers need the other blocks)."""
        key = (pert, ncore)
        full = self._deri.get(key)
        if full is None:
            ERI = np.asarray(self.wfn.H.ERI)
            _, _, gx = self._skeleton(pert)
            U = self._full_U(pert, ncore)
            full = gx + self._rotate_eri(U, ERI)
            self._deri[key] = full
        if blocks is None:
            return full
        sl = tuple({'o': self.o, 'v': self.v}.get(b, slice(None)) for b in blocks)
        return full[sl]

    def _rotate_eri(self, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Orbital-rotation of a 4-index integral tensor by ``U`` (each index rotated)::

            sum_t ( U_tp T_tqrs + U_tq T_ptrs + U_tr T_pqts + U_ts T_pqrt ).

        Shared by the first-order integral derivative (rotation of the unperturbed ERIs) and
        the second-order one (rotation of ``U^{ab}`` with the ERIs, and of ``U^a`` with the
        first perturbed ERIs)."""
        return (self.contract('tp,tqrs->pqrs', U, T)
                + self.contract('tq,ptrs->pqrs', U, T)
                + self.contract('tr,pqts->pqrs', U, T)
                + self.contract('ts,pqrt->pqrs', U, T))

    # ---- second-order (two-field) perturbed derivatives ----
    # For two electric fields the second-order skeleton integrals vanish (the field is linear
    # in F), so d_ab f and d_ab <pq||rs> are built purely from U^a, U^b, the second-order
    # response U^{ab}, and the second-order orthonormality term xi^{ab} (Eqs. 17/20). The MO
    # integrals are multilinear in the orbital rotation, so the mixed second derivative is the
    # U^{ab} rotation of each index plus U^a/U^b products on *distinct* indices -- never the
    # same index twice (that index's mixed second derivative is U^{ab}, not U^a U^b).

    def _xi(self, perta: "Perturbation", pertb: "Perturbation", ncore: int = 0) -> np.ndarray:
        """Second-order orthonormality term ``xi^{ab}`` (Eq. 18), ``nmo x nmo``::

            xi^{ab}_pq = S^{ab}_pq + sum_r (U^a_qr U^b_pr + U^a_pr U^b_qr
                                            - S^a_qr S^b_pr - S^a_pr S^b_qr)
                       = S^{ab} + (U^b U^a.T + U^a U^b.T) - (S^b S^a + S^a S^b),

        the U-products (Eq. 18 has no U*S cross terms -- the S info rides in the ``-1/2 S^x``
        oo/vv blocks of the U's) plus the mixed overlap ``S^{ab}`` minus the ``S^a S^b``
        products. For an electric field ``S^x = 0``, and for the mixed field-nuclear case
        ``S^F = S^{FX} = 0``, so all three skeleton-overlap terms vanish and only the
        U-products survive (polarizability / APT). Both become active only for nuclear-nuclear
        (the molecular Hessian). ``S^a`` is the first-order MO overlap derivative
        (:meth:`_skeleton`); ``S^{ab}`` is :meth:`_overlap2_skeleton`."""
        Ua = self._full_U(perta, ncore)
        Ub = self._full_U(pertb, ncore)
        xi = Ub @ Ua.T + Ua @ Ub.T
        _, Sa, _ = self._skeleton(perta)
        _, Sb, _ = self._skeleton(pertb)
        Sab = self._overlap2_skeleton(perta, pertb)
        return xi + Sab - (Sb @ Sa + Sa @ Sb)

    def _cross_eri(self, Ua: np.ndarray, Ub: np.ndarray, T: np.ndarray) -> np.ndarray:
        """The cross-index (different-slot) product of two single rotations on ``T`` (Eq. 20):
        ``U^a`` on one integral index, ``U^b`` on another, over the 6 slot pairs. (No
        same-index products -- the integral is multilinear, so a single index's mixed second
        derivative is ``U^{ab}``, not ``U^a U^b``.)"""
        c = self.contract
        return (c('tp,uq,turs->pqrs', Ua, Ub, T) + c('tp,ur,tqus->pqrs', Ua, Ub, T)
                + c('tp,us,tqru->pqrs', Ua, Ub, T) + c('tq,ur,ptus->pqrs', Ua, Ub, T)
                + c('tq,us,ptru->pqrs', Ua, Ub, T) + c('tr,us,pqtu->pqrs', Ua, Ub, T))

    def _oei_skeleton(self, pert: "Perturbation") -> np.ndarray:
        """First-order one-electron (core-Hamiltonian) skeleton derivative ``h^(x)_pq``
        (``nmo x nmo``, MO basis, fixed MO coefficients) -- the piece of the skeleton Fock
        derivative *without* the occupied mean field. Field: ``h^(F) = -mu`` (``H' = -mu.E``).
        Nuclear: the core-Hamiltonian derivative from the ``Derivatives`` provider."""
        so = self.wfn.orbital_basis == 'spinorbital'
        if pert.kind == 'field':
            return -np.asarray(self.wfn.H.mu[pert.comp])
        if pert.kind == 'nuclear':
            atom, cart = pert.comp
            d = self.wfn.derivatives
            return np.asarray((d.so_core(atom) if so else d.core(atom))[cart])
        raise NotImplementedError("one-electron skeleton wired for 'field'/'nuclear' only.")

    def _d2int_blocks(self, a1: int, a2: int) -> dict:
        """Cached nuclear-nuclear skeleton second-derivative integrals for the atom pair
        ``(a1, a2)``: the 9 ``(cart1, cart2)`` blocks of the core Hamiltonian ``h^{XY}``, the
        overlap ``S^{XY}``, and the two-electron ``<pq||rs>^{XY}`` (in the basis's ERI
        convention). The ``mo_*_deriv2`` calls -- ``mo_tei_deriv2`` especially -- are shared
        across a pair's 3x3 Cartesian blocks, so compute once per atom pair (not per coordinate
        pair). Returns ``{'core','overlap','eri'}`` -> lists of 9 arrays (indexed ``c1*3+c2``)."""
        key = (a1, a2)
        if key not in self._d2int:
            so = self.wfn.orbital_basis == 'spinorbital'
            d = self.wfn.derivatives
            if so:
                core = [np.asarray(m) for m in d.so_core2(a1, a2)]
                overlap = [np.asarray(m) for m in d.so_overlap2(a1, a2)]
                eri = [np.asarray(m) for m in d.so_eri2(a1, a2)]     # <pq||rs>^{XY} (antisym)
            else:
                core = [np.asarray(m) for m in d.core2(a1, a2)]
                overlap = [np.asarray(m) for m in d.overlap2(a1, a2)]
                eri = []
                for ch in d.eri2(a1, a2):                            # chemist (pq|rs)^{XY}
                    ch = np.asarray(ch)
                    ch = 0.5 * (ch + ch.transpose(2, 3, 0, 1))       # enforce (pq|rs)=(rs|pq)
                    eri.append(ch.swapaxes(1, 2))                    # -> physicist <pq|rs>^{XY}
            self._d2int[key] = {'core': core, 'overlap': overlap, 'eri': eri}
        return self._d2int[key]

    def _oei2_skeleton(self, perta: "Perturbation", pertb: "Perturbation"):
        """Mixed one-electron skeleton second derivative ``h^(ab)_pq`` (fixed MO coefficients).

        - field-field: ``0`` (the field is linear in F).
        - field-nuclear: ``-d(mu_alpha)/dX_beta`` (dipole-derivative integrals) -- because
          ``d_F h = -mu``, so ``d_{F X} h = -mu^X`` (``Derivatives.dipole`` / ``so_dipole``).
        - nuclear-nuclear: the core-Hamiltonian second derivative ``h^{XY}`` (``core2`` /
          ``so_core2``), indexed ``cart1*3 + cart2`` for the atom pair (molecular Hessian)."""
        kinds = {perta.kind, pertb.kind}
        if kinds == {'field'}:
            return 0.0
        so = self.wfn.orbital_basis == 'spinorbital'
        d = self.wfn.derivatives
        if kinds == {'field', 'nuclear'}:
            fld, nuc = (perta, pertb) if perta.kind == 'field' else (pertb, perta)
            alpha = fld.comp
            atom, beta = nuc.comp
            dip = (d.so_dipole(atom) if so else d.dipole(atom))[alpha * 3 + beta]
            return -np.asarray(dip)
        if kinds == {'nuclear'}:
            (a1, c1), (a2, c2) = perta.comp, pertb.comp
            return self._d2int_blocks(a1, a2)['core'][c1 * 3 + c2]
        raise NotImplementedError("mixed one-electron skeleton wired for 'field'/'nuclear'.")

    def _overlap2_skeleton(self, perta: "Perturbation", pertb: "Perturbation"):
        """Mixed second overlap skeleton ``S^{ab}_pq`` (fixed MO coefficients), for the
        ``xi^{ab}`` term (Eq. 18). Zero unless both perturbations are nuclear (``S^F = 0``);
        nuclear-nuclear reads ``overlap2`` / ``so_overlap2`` (indexed ``cart1*3 + cart2``)."""
        if not (perta.kind == 'nuclear' and pertb.kind == 'nuclear'):
            return 0.0
        (a1, c1), (a2, c2) = perta.comp, pertb.comp
        return self._d2int_blocks(a1, a2)['overlap'][c1 * 3 + c2]

    def _d2eri_skeleton(self, perta: "Perturbation", pertb: "Perturbation"):
        """Mixed second two-electron skeleton ``<pq||rs>^{ab}`` (fixed MO coefficients), in the
        basis's ERI convention (SO: antisymmetrized ``<pq||rs>``; spatial: physicist ``<pq|rs>``).
        Zero unless both perturbations are nuclear (the field never enters the 2-e integrals) --
        the ``d_{F X} <> = 0`` fact that made the APT need no such term. Nuclear-nuclear reads
        ``eri2`` / ``so_eri2`` (indexed ``cart1*3 + cart2``), mirroring the first-order skeleton
        convention in :meth:`_skeleton` (chemist -> physicist, with the bra<->ket
        symmetrization ``so_eri2`` already applies and ``eri2`` gets here)."""
        if not (perta.kind == 'nuclear' and pertb.kind == 'nuclear'):
            return 0.0
        (a1, c1), (a2, c2) = perta.comp, pertb.comp
        return self._d2int_blocks(a1, a2)['eri'][c1 * 3 + c2]

    def _d2eri(self, perta, pertb, U2: np.ndarray, ncore: int = 0) -> np.ndarray:
        """Second perturbed two-electron derivative ``d_ab <pq||rs>`` (``nmo^4``) given the
        second-order rotation ``U2`` = ``U^{ab}`` (Eq. 20)::

            d_ab <pq||rs> = rotate(U^{ab}, <pq||rs>)
                            + cross(U^a, U^b, <pq||rs>) + cross(U^b, U^a, <pq||rs>)
                            + rotate(U^a, <pq||rs>^(b)) + rotate(U^b, <pq||rs>^(a))
                            + <pq||rs>^(ab)

        where ``<pq||rs>^(x)`` is the first-order two-electron skeleton (fixed MO coefficients,
        from :meth:`_skeleton`) and ``<pq||rs>^(ab)`` the mixed second-order skeleton
        (:meth:`_d2eri_skeleton`). For the electric field ``<pq||rs>^(x) = 0`` (the field never
        enters the two-electron integrals), so only the first three (rotation) terms survive --
        the polarizability case. Field-nuclear adds the ``rotate(U^F, <>^X)`` cross-skeleton term
        (``<>^(FX) = 0``, the APT case); nuclear-nuclear adds the second cross-skeleton and the
        direct ``<pq||rs>^(XY)`` (the molecular Hessian).

        Takes ``U2`` as an argument so the second-order CPHF solve can call it with the ov
        response zeroed (no recursion through :meth:`_full_U2`)."""
        ERI = np.asarray(self.wfn.H.ERI)
        Ua = self._full_U(perta, ncore)
        Ub = self._full_U(pertb, ncore)
        _, _, gxa = self._skeleton(perta)
        _, _, gxb = self._skeleton(pertb)
        d2e = (self._rotate_eri(U2, ERI)
               + self._cross_eri(Ua, Ub, ERI) + self._cross_eri(Ub, Ua, ERI))
        if np.ndim(gxb):                                    # gxb is an array (nuclear), not 0.0 (field)
            d2e = d2e + self._rotate_eri(Ua, np.asarray(gxb))
        if np.ndim(gxa):
            d2e = d2e + self._rotate_eri(Ub, np.asarray(gxa))
        gxab = self._d2eri_skeleton(perta, pertb)          # mixed 2e skeleton (nuclear-nuclear)
        if np.ndim(gxab):
            d2e = d2e + gxab
        return d2e

    def _d2fock(self, perta, pertb, U2: np.ndarray, ncore: int = 0) -> np.ndarray:
        """Second perturbed Fock derivative ``d_ab f_pq`` (``nmo x nmo``) given ``U2`` = ``U^{ab}``.

        The Fock is the one-electron ``h`` plus the occupied mean field ``G_pq = sum_m
        w[p,m,q,m]`` (``w`` = ``<pq||rs>`` (SO) / ``L`` (spatial)). Differentiating each piece
        with the correct multilinear rule (rotation + cross-index products, *no* same-index
        terms)::

            d_ab h = rotate2(U^{ab}, h) + cross2(U^a, U^b, h) + cross2(U^b, U^a, h)
                     + rotate2(U^a, h^(b)) + rotate2(U^b, h^(a))   (first-order oei skeletons)
                     + h^(ab)                                      (mixed oei skeleton)
            d_ab G_pq = sum_m d_ab w[p,m,q,m]                      (from d_ab <pq||rs>)

        with ``rotate2(U, M) = U.T M + M U`` and ``cross2(U^a,U^b,M) = U^a.T M U^b``. The
        one-electron skeletons come from :meth:`_oei_skeleton` (field ``h^(F) = -mu``, nuclear
        the core-Hamiltonian derivative) and :meth:`_oei2_skeleton` (field-field ``0``,
        field-nuclear ``-mu^X``). ``d_ab G`` inherits the (now perturbation-general) second
        two-electron derivative from :meth:`_d2eri`. Takes ``U2`` as an argument (no recursion
        through :meth:`_full_U2`).

        Handles field-field (polarizability) and field-nuclear (APT). NUCLEAR-NUCLEAR (the
        molecular Hessian) additionally needs ``h^(XY)`` (``core2``) and ``<pq||rs>^(XY)``
        (guarded in :meth:`_d2eri`), and the ``S^{XY}`` term in :meth:`_xi`."""
        o = self.o
        c = self.contract
        f = np.asarray(self.wfn.H.F)
        so = self.wfn.orbital_basis == 'spinorbital'
        W = np.asarray(self.wfn.H.ERI) if so else np.asarray(self.wfn.H.L)
        h = f - c('pmqm->pq', W[:, o, :, o])                 # bare one-electron MO Fock
        Ua = self._full_U(perta, ncore)
        Ub = self._full_U(pertb, ncore)
        hxa = self._oei_skeleton(perta)                      # first-order oei skeletons
        hxb = self._oei_skeleton(pertb)
        hxab = self._oei2_skeleton(perta, pertb)             # mixed oei skeleton (0 for field-field)
        d2h = (c('rp,rq->pq', U2, h) + c('pr,rq->pq', h, U2)             # rotate2(U^{ab}, h)
               + c('rp,rs,sq->pq', Ua, h, Ub) + c('rp,rs,sq->pq', Ub, h, Ua)  # cross2
               + c('rp,rq->pq', Ua, hxb) + c('pr,rq->pq', hxb, Ua)      # rotate2(U^a, h^(b))
               + c('rp,rq->pq', Ub, hxa) + c('pr,rq->pq', hxa, Ub)      # rotate2(U^b, h^(a))
               + hxab)                                                   # mixed oei skeleton
        d2e = self._d2eri(perta, pertb, U2, ncore)
        d2W = d2e if so else (2.0 * d2e - d2e.swapaxes(2, 3))
        d2G = c('pmqm->pq', d2W[:, o, :, o])                 # sum_m d_ab w[p,m,q,m]
        return d2h + d2G

    def _full_U2(self, perta, pertb, ncore: int = 0) -> np.ndarray:
        """Full second-order orbital-rotation matrix ``U^{ab}`` for two fields (cached). The
        symmetric (oo/vv) blocks come from the second-order orthonormality (Eq. 19,
        ``U^{ab}_pq + U^{ab}_qp + xi^{ab}_pq = 0`` -> ``-1/2 xi``); the ov block from the
        second-order CPHF ``G U^{ab}_ov = B^{ab}`` (canonical Brillouin ``d_ab f_ai = 0``),
        with ``B^{ab}`` the ov block of ``d_ab f`` evaluated with the ov response zeroed --
        the same ``G`` as the first-order solve.

        ``ncore > 0`` (frozen-core correlated second derivatives): the core<->active-occupied
        block is non-redundant (it moves the frozen/active partition), so it is fixed by the
        second-order canonical condition ``d_ab f_ij = 0`` (``i`` core, ``j`` active) rather than
        left at ``-1/2 xi``. As at first order, this decouples: the antisymmetric core<->active
        part enters ``d_ab f`` only through the eps-diagonal ``eps_i U^{ab}_ij + eps_j U^{ab}_ji``
        (the occupied mean field sees only the *symmetric* part ``-1/2 xi``, unchanged between the
        placeholder and the true value, and the ov solve is likewise blind to it). Evaluating
        ``d_ab f`` with the ``-1/2 xi`` placeholder in place (``F0``) and imposing Eq. 19
        (``U^{ab}_ji = -xi_ij - U^{ab}_ij``) gives the explicit correction
        ``U^{ab}_ij = -F0_ij / (eps_i - eps_j) - 1/2 xi_ij``. The redundant core-core,
        active-active, and vir-vir blocks stay at ``-1/2 xi``.

        The frozen-core oo block also makes the ov orthonormality term ``xi_ia`` nonzero, so
        the ov block is not purely antisymmetric; its ``-xi_ia`` part is seeded before the
        RHS so the second-order CPHF solve still enforces Brillouin ``d_ab f_ai = 0``.

        ASSUMES A PERTURBATION-INDEPENDENT AO BASIS (field only), inherited from :meth:`_xi`
        (no ``S^{ab}``) and :meth:`_d2fock` (no skeleton second derivatives). Not valid as-is
        for nuclear displacements / the molecular Hessian."""
        key = (perta, pertb, ncore)
        if key in self._U2cache:
            return self._U2cache[key]
        o, v, nmo = self.o, self.v, self.wfn.nmo
        xi = self._xi(perta, pertb, ncore)
        U2 = np.zeros((nmo, nmo))
        U2[o, o] = -0.5 * xi[o, o]
        U2[v, v] = -0.5 * xi[v, v]
        # Seed the CPHF-response-independent ov orthonormality part U^{ab}_ia = -xi_ia
        # (Eq. 19 with the ov response zeroed) *before* forming the RHS, so B captures its
        # contribution to d_ab f_ov. This is zero for the electric field with no frozen core
        # (xi_ov vanishes when U has no oo block), but the frozen-core core<->active oo block
        # makes xi_ov nonzero; omitting it leaves the second-order Brillouin condition
        # d_ab f_ai = 0 unsatisfied (the Hessian G only maps the antisymmetric ov rotation).
        U2[o, v] = -xi[o, v]
        B = -self._d2fock(perta, pertb, U2, ncore)[o, v]     # ov CPHF response still zeroed
        Uai = self.solve(B, kind="electric")
        U2[v, o] = Uai.T
        U2[o, v] = -xi[o, v] - Uai
        if ncore:
            eps = self.eps
            co = slice(o.start, o.start + ncore)             # frozen core
            ao = slice(o.start + ncore, o.stop)              # active occupied
            F0 = self._d2fock(perta, pertb, U2, ncore)       # placeholder core<->active in U2
            gap = eps[co][:, None] - eps[ao][None, :]
            Uca = -F0[co, ao] / gap - 0.5 * xi[co, ao]
            U2[co, ao] = Uca
            U2[ao, co] = -(xi[co, ao] + Uca).T               # Eq. 19: U^{ab}_ji = -xi_ij - U^{ab}_ij
        self._U2cache[key] = U2
        return U2

    def perturbed_fock2(self, perta, pertb, ncore: int = 0) -> np.ndarray:
        """Full second perturbed Fock derivative ``d_ab f_pq`` (``nmo x nmo``) for two fields,
        cached per ``(perta, pertb, ncore)``. Field only -- assumes a perturbation-independent
        AO basis (see :meth:`_d2fock`); not valid for nuclear displacements."""
        key = (perta, pertb, ncore)
        if key not in self._dfock2:
            self._dfock2[key] = self._d2fock(perta, pertb, self._full_U2(perta, pertb, ncore), ncore)
        return self._dfock2[key]

    def perturbed_eri2(self, perta, pertb, ncore: int = 0, blocks=None) -> np.ndarray:
        """Full second perturbed two-electron derivative ``d_ab <pq||rs>`` for two fields
        (Eq. 20), cached per ``(perta, pertb, ncore)``. ``blocks`` slices as in
        :meth:`perturbed_eri`. Field only -- assumes a perturbation-independent AO basis (see
        :meth:`_d2eri`); not valid for nuclear displacements."""
        key = (perta, pertb, ncore)
        full = self._deri2.get(key)
        if full is None:
            full = self._d2eri(perta, pertb, self._full_U2(perta, pertb, ncore), ncore)
            self._deri2[key] = full
        if blocks is None:
            return full
        sl = tuple({'o': self.o, 'v': self.v}.get(b, slice(None)) for b in blocks)
        return full[sl]

    # ---- magnetic-field perturbation (imaginary; for AATs) ----
    def _m_ov(self, axis: int) -> np.ndarray:
        """ov block of the (real) MO magnetic-dipole integral for ``axis`` (0/1/2).

        ``H.m`` carries the ``-1/2`` and an imaginary unit (it is the pure-imaginary
        operator ``-i/2 L``, with the ``i`` convention shared by the CC response code);
        the magnetic CPHF can be solved as a real problem, so this strips the ``i`` by
        multiplying by ``-i`` -- the final VCD rotatory strength takes the imaginary
        component of the APT*AAT product, not of the AAT itself."""
        return np.asarray(-1.0j * self.wfn.H.m[axis])[self.o, self.v].real

    def rhs_magnetic(self, axis: int) -> np.ndarray:
        """Magnetic-field CPHF RHS for ``axis`` (0/1/2), i.e. ``B = -m`` (real).

        The magnetic field enters as ``H' = -m . B`` (analogous to ``-mu . E`` for the
        electric field), and -- like the electric field -- it does not move the basis
        functions, so there is no overlap/Pulay term: the RHS is just the (negated)
        real magnetic-dipole ov integral. The magnetic perturbation is imaginary, so
        this response uses the antisymmetric (``kind='magnetic'``) orbital Hessian."""
        return -self._m_ov(axis)

    def solve_magnetic(self, axis: int) -> np.ndarray:
        """Magnetic-field CPHF response ``U^B`` for ``axis`` (0/1/2), ``(no, nv)``,
        solved once and cached. Real (the magnetic-dipole RHS is stripped of its i)."""
        if axis not in self._U_mag:
            self._U_mag[axis] = self.solve(self.rhs_magnetic(axis), kind="magnetic")
        return self._U_mag[axis]

    def _p_ov(self, axis: int) -> np.ndarray:
        """ov block of the (real) MO linear-momentum integral for ``axis`` (0/1/2).

        ``H.p`` is ``i * <mu|Del|nu>`` in the MO basis (the pure-imaginary linear-momentum
        operator, carrying the same ``i`` convention as ``H.m``); this strips the ``i`` by
        multiplying by ``-i`` so the momentum CPHF is a real problem, exactly as
        :meth:`_m_ov` does for the magnetic dipole."""
        return np.asarray(-1.0j * self.wfn.H.p[axis])[self.o, self.v].real

    def rhs_momentum(self, axis: int) -> np.ndarray:
        """Linear-momentum (magnetic vector-potential) CPHF RHS for ``axis`` (0/1/2), real.

        The vector potential enters the Hamiltonian as ``H'(A) = A . pi`` (Amos, Jalkanen &
        Stephens, JPC 92, 5571 (1988), Eq. 10), so ``dH'/dA = +pi`` -- the (positive) real
        momentum ov integral (contrast the magnetic ``B = -m``). Like the field/magnetic
        perturbations, ``A`` does not move the basis functions, so there is no overlap/Pulay
        term. The momentum perturbation is imaginary, so this uses the antisymmetric
        (``kind='magnetic'``) orbital Hessian -- the same Hessian as the magnetic response."""
        return self._p_ov(axis)

    def solve_momentum(self, axis: int) -> np.ndarray:
        """Linear-momentum CPHF response ``U^A`` for ``axis`` (0/1/2), ``(no, nv)``, solved
        once and cached. Real. This is the ket derivative ``dPsi/dA`` in the velocity-gauge
        APT (:meth:`HFwfn.velocity_dipole_derivatives`)."""
        if axis not in self._U_mom:
            self._U_mom[axis] = self.solve(self.rhs_momentum(axis), kind="magnetic")
        return self._U_mom[axis]

    def magnetic_ints(self, axis: int, ncore: int = 0, gauge: str = 'non-canonical'):
        """Magnetic-field-perturbed MO integrals for ``axis`` (0/1/2): returns the tuple
        ``(U^H, dF^H, dERI^H)`` used by the MP2 atomic axial tensors
        (:meth:`MPwfn.atomic_axial_tensors`). Cached per ``(axis, ncore, gauge)``.

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
        (:meth:`MPwfn.velocity_dipole_derivatives`). Cached per ``(axis, ncore, gauge)``.

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
        """Shared engine for the imaginary (anti-Hermitian) one-electron perturbations -- the
        magnetic dipole (:meth:`magnetic_ints`) and the linear momentum
        (:meth:`momentum_ints`).  ``hmag`` is the stripped real antisymmetric operator matrix
        in the MO basis; returns ``(U, dF, dERI)`` with the antisymmetric orbital response and
        the gauge treatment of the redundant oo/vv blocks documented on :meth:`magnetic_ints`."""
        o, v, nmo = self.o, self.v, self.wfn.nmo
        no, nv = self.no, self.nv
        t = slice(0, nmo)
        eps = self.eps
        c = self.contract
        ERI = np.asarray(self.wfn.H.ERI)
        W = ERI if self.wfn.orbital_basis == 'spinorbital' else np.asarray(self.wfn.H.L)
        core = -W + W.swapaxes(1, 3)                # antisymmetric mean-field weight
        Gm = (np.einsum('ab,ij->aibj', np.eye(nv), np.eye(no))
              * (eps[v].reshape(nv, 1, 1, 1) - eps[o].reshape(1, no, 1, 1))
              + core.swapaxes(1, 2)[v, o, v, o]).reshape(nv * no, nv * no)

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
        U[v, o] = np.linalg.solve(Gm, hmag[v, o].reshape(nv * no)).reshape(nv, no)
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

    def _build_nuclear(self, atom: int):
        """One heavy pass over the derivative integrals for ``atom``; returns three
        lists of 3 (x,y,z) arrays: the CPHF RHS ``B^X_ia`` ``(no, nv)``, the skeleton
        derivative Fock oo block ``F^X_ij`` ``(no, no)``, and the overlap derivative oo
        block ``S^X_ij`` ``(no, no)``. The latter two are free by-products of the RHS
        build that the molecular Hessian's response terms need, so all are produced and
        cached together (the full-MO ERI derivative is the dominant nmo**4 cost).

        Unlike the electric-field RHS, a nuclear displacement moves the basis
        functions, so the right-hand side folds in (a) the skeleton derivative Fock
        ``F^X`` (built from the first-derivative one- and two-electron integrals at
        fixed MO coefficients), (b) the overlap-derivative term ``-eps_i S^X_ia``, and
        (c) the coupling of the overlap-determined occupied-occupied response
        ``U^X_kl = -1/2 S^X_kl`` back into the Fock matrix (the Pulay term). The CPHF
        RHS is minus that first-order off-diagonal ("perturbation") Fock, ``B = -Q``
        (the nuclear analog of the field's ``B = -mu``)::

            B^X_ia = -[ F^X_ia - eps_i S^X_ia
                        - 1/2 sum_kl S^X_kl ( L[a,k,i,l] + L[a,l,i,k] ) ]

        with the skeleton derivative Fock  F^X_pq = h^X_pq + sum_k(occ) L[p,k,q,k]^X.

        Everything is cast with the spin-adapted L = 2<pq|rs> - <pq|sr> (physicist's
        notation), as in the orbital Hessian. The Pulay coupling is the closed-shell
        G-operator for the symmetric occupied-occupied perturbation S^X_kl,
        ``sum_kl (-1/2 S^X_kl)(L[a,k,i,l] + L[a,l,i,k])`` (the L[a,k,i,l]/L[a,l,i,k]
        pair, not L[i,k,a,l] -- their Coulomb parts coincide but the exchange parts
        differ). The first-derivative integrals come from the (full-MO) Derivatives
        provider; the unperturbed L is H.L. Validated to ~1e-8 via the dipole-
        derivative / APT-transpose check against finite difference of the SCF dipole.

        The skeleton derivatives (including the dominant ``nmo^4`` ERI derivative) come from
        the unified :meth:`_skeleton` cache, computed once per atom/cart and reused by the
        explicit perturbed-integral engine. The spin-orbital path dispatches to
        :meth:`_so_build_nuclear`.
        """
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_build_nuclear(atom)
        o, v = self.o, self.v
        eps_o = self.eps[o]
        Lvooo = np.asarray(self.wfn.H.L)[v, o, o, o]   # spin-adapted, unperturbed
        B, Foo, Soo = [], [], []
        for c in range(3):
            Fx, Sx, _ = self._skeleton(Perturbation('nuclear', (atom, c)))
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^X_kl into the
            # ov block:  -1/2 S^X_kl ( L[a,k,i,l] + L[a,l,i,k] ).
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

    def _so_build_nuclear(self, atom: int):
        """Spin-orbital nuclear CPHF RHS for ``atom`` -- the spin-orbital analogue of
        :meth:`_build_nuclear`. Same structure with the antisymmetrized ``<pq||rs>`` in
        place of the spin-adapted ``L``, the spin-orbital skeleton derivative integrals
        from ``Derivatives.so_*``, and singly occupied spin orbitals::

            F^X_pq = h^X_pq + sum_k(occ) <pk||qk>^X        (skeleton Fock derivative)
            B^X_ia = -[ F^X_ia - eps_i S^X_ia
                        - 1/2 sum_kl S^X_kl ( <ak||il> + <al||ik> ) ]

        Returns ``(B, Foo, Soo)`` as for the spatial path.

        The skeleton derivative integrals (including the dominant ``nmo^4`` ``so_eri``
        derivative) come from the unified :meth:`_skeleton` cache, so they are computed once
        per atom/cart and reused by the explicit perturbed-integral engine
        (:meth:`perturbed_fock` / :meth:`perturbed_eri`) and by any second-derivative
        consumer; ``Foo``/``Soo`` are then just the ``oo`` slices of the cached
        ``f^(x)``/``S^x``."""
        o, v = self.o, self.v
        eps_o = self.eps[o]
        ERI = np.asarray(self.wfn.H.ERI)         # antisymmetrized, unperturbed
        Evooo = ERI[v, o, o, o]
        B, Foo, Soo = [], [], []
        for c in range(3):
            Fx, Sx, _ = self._skeleton(Perturbation('nuclear', (atom, c)))
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^X_kl into the ov
            # block: -1/2 sum_kl S^X_kl ( <ak||il> + <al||ik> ).
            coupling = (self.contract('akil,kl->ia', Evooo, Sx[o, o])
                        + self.contract('alik,kl->ia', Evooo, Sx[o, o]))
            Q = Fx[o, v] - eps_o[:, None] * Sx[o, v] - 0.5 * coupling
            B.append(-Q)
            Foo.append(Fx[o, o])
            Soo.append(Sx[o, o])
        return B, Foo, Soo

    def rhs_nuclear(self, atom: int) -> List[np.ndarray]:
        """Nuclear-perturbation CPHF RHS for ``atom``: list of 3 (x,y,z) ``(no, nv)``
        arrays ``B^X_ia``. Thin accessor over the cached :meth:`_build_nuclear`."""
        return self.rhs_nuclear_cached(atom)

    # ---- cached nuclear response (shared by the Hessian and the APTs) ----
    def solve_nuclear(self, atom: int) -> List[np.ndarray]:
        """Nuclear CPHF response for ``atom`` -- list of 3 (x, y, z) ``(no, nv)``
        ``U^X`` arrays -- solved once and cached.

        Building the RHS (:meth:`rhs_nuclear`) regenerates the full-MO derivative
        ERIs (nmo**4 per atom), the dominant cost; both the molecular Hessian and the
        dipole derivatives / APTs need the same 3*natom ``U^X``, so this memoizes the
        response (and the RHS, in ``self._B_nuc``) per atom and they share it. The
        cache is geometry-bound: it lives for the life of this CPHF object.
        """
        if atom not in self._U_nuc:
            B, Foo, Soo = self._build_nuclear(atom)
            self._B_nuc[atom] = B
            self._F_nuc[atom] = Foo
            self._S_nuc[atom] = Soo
            self._U_nuc[atom] = [self.solve(B[c], kind="electric") for c in range(3)]
        return self._U_nuc[atom]

    def rhs_nuclear_cached(self, atom: int) -> List[np.ndarray]:
        """Nuclear RHS ``B^X`` for ``atom`` from the shared cache (populated by
        :meth:`solve_nuclear`); the molecular Hessian's response term needs the RHS
        as well as the response."""
        if atom not in self._B_nuc:
            self.solve_nuclear(atom)
        return self._B_nuc[atom]

    def fock_nuclear_cached(self, atom: int) -> List[np.ndarray]:
        """Skeleton derivative Fock oo block ``F^X_ij`` for ``atom`` from the shared
        cache (a by-product of :meth:`solve_nuclear`); used by the molecular Hessian's
        first-derivative cross terms."""
        if atom not in self._F_nuc:
            self.solve_nuclear(atom)
        return self._F_nuc[atom]

    def overlap_nuclear_cached(self, atom: int) -> List[np.ndarray]:
        """Overlap derivative oo block ``S^X_ij`` for ``atom`` from the shared cache
        (a by-product of :meth:`solve_nuclear`); used by the molecular Hessian's
        first-derivative cross terms."""
        if atom not in self._S_nuc:
            self.solve_nuclear(atom)
        return self._S_nuc[atom]
