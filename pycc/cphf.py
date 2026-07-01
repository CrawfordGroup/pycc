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
    # The "simple but inefficient" form (notes.pdf): rather than separate the orbital
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
        ``pert``, with the CPHF response folded in (notes.pdf)::

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
        (notes.pdf)::

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
        """Second-order orthonormality term ``xi^{ab}`` (Eq. 18), ``nmo x nmo``. For the field
        (``S^x = S^{ab} = 0``) it reduces to ``xi^{ab}_pq = sum_r (U^a_qr U^b_pr + U^a_pr U^b_qr)
        = U^b U^a.T + U^a U^b.T`` (symmetric).

        ASSUMES A PERTURBATION-INDEPENDENT AO BASIS (electric/magnetic field). For a nuclear
        displacement the AOs move, so ``S^a`` and ``S^{ab}`` are nonzero and Eq. 18 gains the
        skeleton overlap terms (``-S^{ab}`` and ``S^a``-``U`` cross terms) that are dropped here.
        Extending this route to the molecular Hessian requires restoring them."""
        Ua = self._full_U(perta, ncore)
        Ub = self._full_U(pertb, ncore)
        return Ub @ Ua.T + Ua @ Ub.T

    def _cross_eri(self, Ua: np.ndarray, Ub: np.ndarray, T: np.ndarray) -> np.ndarray:
        """The cross-index (different-slot) product of two single rotations on ``T`` (Eq. 20):
        ``U^a`` on one integral index, ``U^b`` on another, over the 6 slot pairs. (No
        same-index products -- the integral is multilinear, so a single index's mixed second
        derivative is ``U^{ab}``, not ``U^a U^b``.)"""
        c = self.contract
        return (c('tp,uq,turs->pqrs', Ua, Ub, T) + c('tp,ur,tqus->pqrs', Ua, Ub, T)
                + c('tp,us,tqru->pqrs', Ua, Ub, T) + c('tq,ur,ptus->pqrs', Ua, Ub, T)
                + c('tq,us,ptru->pqrs', Ua, Ub, T) + c('tr,us,pqtu->pqrs', Ua, Ub, T))

    def _d2eri(self, perta, pertb, U2: np.ndarray, ncore: int = 0) -> np.ndarray:
        """Second perturbed two-electron derivative ``d_ab <pq||rs>`` (``nmo^4``) given the
        second-order rotation ``U2`` = ``U^{ab}`` (Eq. 20; field skeleton zero)::

            d_ab <pq||rs> = rotate(U^{ab}, <pq||rs>)
                            + cross(U^a, U^b, <pq||rs>) + cross(U^b, U^a, <pq||rs>).

        Takes ``U2`` as an argument so the second-order CPHF solve can call it with the ov
        response zeroed (no recursion through :meth:`_full_U2`).

        ASSUMES A PERTURBATION-INDEPENDENT AO BASIS (field only): there is no skeleton
        (direct AO) second-derivative term ``<pq||rs>^{ab}``, which vanishes for a field but
        not for a nuclear displacement. The molecular Hessian would add it here."""
        ERI = np.asarray(self.wfn.H.ERI)
        Ua = self._full_U(perta, ncore)
        Ub = self._full_U(pertb, ncore)
        return (self._rotate_eri(U2, ERI)
                + self._cross_eri(Ua, Ub, ERI) + self._cross_eri(Ub, Ua, ERI))

    def _d2fock(self, perta, pertb, U2: np.ndarray, ncore: int = 0) -> np.ndarray:
        """Second perturbed Fock derivative ``d_ab f_pq`` (``nmo x nmo``) given ``U2`` = ``U^{ab}``.

        The Fock is the one-electron ``h`` plus the occupied mean field ``G_pq = sum_m
        w[p,m,q,m]`` (``w`` = ``<pq||rs>`` (SO) / ``L`` (spatial)). Differentiating each piece
        with the correct multilinear rule (rotation + cross-index products, *no* same-index
        terms)::

            d_ab h = rotate2(U^{ab}, h) + cross2(U^a, U^b, h)          [+ cross2(U^b,U^a,h)]
                     + rotate2(U^a, -mu_b) + rotate2(U^b, -mu_a)       (field skeleton f^(x)=-mu)
            d_ab G_pq = sum_m d_ab w[p,m,q,m]                          (from d_ab <pq||rs>)

        with ``rotate2(U, M) = U.T M + M U`` and ``cross2(U^a,U^b,M) = U^a.T M U^b``. Takes
        ``U2`` as an argument (no recursion through :meth:`_full_U2`).

        ASSUMES A PERTURBATION-INDEPENDENT AO BASIS (field only). The one-electron skeleton is
        just ``f^(x) = -mu`` with no ``h^{ab}``, and ``d_ab G`` inherits the field-only
        ``d_ab <pq||rs>`` from :meth:`_d2eri`. A nuclear displacement would add the direct AO
        second derivatives ``h^{ab}`` and ``<pq||rs>^{ab}`` (and the ``S``-dependent terms via
        ``U2``); this routine does not."""
        o = self.o
        c = self.contract
        f = np.asarray(self.wfn.H.F)
        so = self.wfn.orbital_basis == 'spinorbital'
        W = np.asarray(self.wfn.H.ERI) if so else np.asarray(self.wfn.H.L)
        h = f - c('pmqm->pq', W[:, o, :, o])                 # bare one-electron MO Fock
        mua = np.asarray(self.wfn.H.mu[perta.comp])
        mub = np.asarray(self.wfn.H.mu[pertb.comp])
        Ua = self._full_U(perta, ncore)
        Ub = self._full_U(pertb, ncore)
        d2h = (c('rp,rq->pq', U2, h) + c('pr,rq->pq', h, U2)             # rotate2(U^{ab}, h)
               + c('rp,rs,sq->pq', Ua, h, Ub) + c('rp,rs,sq->pq', Ub, h, Ua)  # cross2
               - c('rp,rq->pq', Ua, mub) - c('pr,rq->pq', mub, Ua)      # rotate2(U^a, -mu_b)
               - c('rp,rq->pq', Ub, mua) - c('pr,rq->pq', mua, Ub))     # rotate2(U^b, -mu_a)
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
