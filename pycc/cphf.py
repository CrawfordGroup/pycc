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

    def __init__(self, wfn: Any) -> None:
        self.wfn = wfn
        self.contract = wfn.contract        # device-aware ContractionBackend (shared)
        self.o = wfn.o
        self.v = wfn.v
        self.no = wfn.no
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
        self._dfock: dict = {}    # Perturbation -> (nmo, nmo) full perturbed Fock deriv
        self._deri: dict = {}     # Perturbation -> (nmo^4) full perturbed <pq||rs> deriv

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
        """Electric-field CPHF RHS for ``axis`` (0/1/2), i.e. ``B = -mu``.

        The field enters the Hamiltonian as ``H' = -mu . E`` -- the field-interaction
        sign is distinct from the electron-charge sign already carried by the dipole
        integrals (``H.mu`` = -e r) -- so ``dH'/dE = -mu`` and the response RHS is the
        negated dipole. (For the polarizability the two sign conventions cancel, but
        the response ``U`` is reused by other properties where they do not.)
        """
        return -self._mu_ov(axis)

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
    # building blocks of the correlated field properties (dipole now; polarizability next)
    # and, later, of the nuclear gradient/Hessian. Spin-orbital path first.

    def _full_U(self, pert: "Perturbation") -> np.ndarray:
        """The full ``nmo x nmo`` orbital-rotation matrix ``U^x_pq`` for ``pert``.

        The matrix element ``U_qp = <phi_q|d phi_p/dx>`` (the coefficient of orbital ``q``
        in the first-order change of orbital ``p``, as it enters the integral derivatives).
        The CPHF solve returns the occupied response ``Uia[i,a] = <phi_a|d phi_i> = U_ai``,
        so the virtual-occupied block is ``U[v,o] = Uia.T``. For an electric field the basis
        functions do not move (``S^x = 0``): the occupied-occupied and virtual-virtual blocks
        vanish (``U_ij = U_ab = -1/2 S^x = 0``) and the ov/vo blocks are antisymmetric
        (``U_ia = -U_ai``), so ``U[o,v] = -Uia``."""
        if pert.kind != 'field':
            raise NotImplementedError(
                "full perturbed derivatives are wired for 'field' only so far; "
                "'nuclear'/'magnetic' use the same machinery but are not yet built.")
        nmo = self.wfn.nmo
        Uia = self.solve_field(pert.comp)                  # (no, nv): U_ai = <phi_a|d phi_i>
        U = np.zeros((nmo, nmo))
        U[self.o, self.v] = -Uia
        U[self.v, self.o] = Uia.T
        return U

    def perturbed_fock(self, pert: "Perturbation") -> np.ndarray:
        """Full first derivative of the MO Fock matrix ``d_x f_pq`` (``nmo x nmo``) for
        ``pert``, with the CPHF response folded in (notes.pdf)::

            d_x f_pq = f^(x)_pq + U^x_pq f_pp + U^x_qp f_qq
                       - 1/2 sum_nm S^x_nm A_pnqm + sum_cm U^x_cm A_pcqm

        with ``A_pqrs = <pq||rs> + <ps||qr>`` (the orbital-Hessian two-electron weight).
        For an electric field ``S^x = 0`` and the skeleton Fock derivative ``f^(x) = mu``
        (the MO dipole integral, sign consistent with ``rhs_field``'s ``B = -mu``), so the
        ``S^x`` term drops. Cached per ``pert``. Spin-orbital path only for now."""
        if self.wfn.orbital_basis != 'spinorbital':
            raise NotImplementedError(
                "perturbed_fock is implemented for the spin-orbital path only so far.")
        if pert in self._dfock:
            return self._dfock[pert]
        o, v = self.o, self.v
        eps = self.eps
        ERI = np.asarray(self.wfn.H.ERI)
        U = self._full_U(pert)
        fx = np.asarray(self.wfn.H.mu[pert.comp])          # skeleton field Fock deriv = mu
        df = fx + U * eps[:, None] + U.T * eps[None, :]     # f^(x) + U_pq f_pp + U_qp f_qq
        # + sum_cm U_cm ( <pc||qm> + <pm||qc> ),  c in vir, m in occ
        Uvo = U[v, o]
        df = df + (self.contract('cm,pcqm->pq', Uvo, ERI[:, v, :, o])
                   + self.contract('cm,pmqc->pq', Uvo, ERI[:, o, :, v]))
        self._dfock[pert] = df
        return df

    def perturbed_eri(self, pert: "Perturbation", blocks=None) -> np.ndarray:
        """Full first derivative of the antisymmetrized two-electron integrals
        ``d_x <pq||rs>`` for ``pert`` (notes.pdf)::

            d_x <pq||rs> = <pq||rs>^(x)
                           + sum_t ( U^x_tp <tq||rs> + U^x_tq <pt||rs>
                                     + U^x_tr <pq||ts> + U^x_ts <pq||rt> )

        For an electric field the skeleton derivative ``<pq||rs>^(x) = 0``, leaving the
        ``U``-rotation. The full ``nmo^4`` tensor is built and cached per ``pert`` (memory
        is geometry-bound, shared across properties). ``blocks`` is an optional 4-tuple of
        block labels ('o'/'v'/'all'), e.g. ``('o','o','v','v')`` for the MP2 2PDM; the
        **default returns the full tensor** (CC consumers need the other blocks). Spin-orbital
        path only for now."""
        if self.wfn.orbital_basis != 'spinorbital':
            raise NotImplementedError(
                "perturbed_eri is implemented for the spin-orbital path only so far.")
        full = self._deri.get(pert)
        if full is None:
            ERI = np.asarray(self.wfn.H.ERI)
            U = self._full_U(pert)                          # skeleton <pq||rs>^(x) = 0 (field)
            full = (self.contract('tp,tqrs->pqrs', U, ERI)
                    + self.contract('tq,ptrs->pqrs', U, ERI)
                    + self.contract('tr,pqts->pqrs', U, ERI)
                    + self.contract('ts,pqrt->pqrs', U, ERI))
            self._deri[pert] = full
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

        The spin-orbital path dispatches to :meth:`_so_build_nuclear`.
        """
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_build_nuclear(atom)
        o, v, no, nv = self.o, self.v, self.no, self.nv
        eps_o = self.eps[o]
        L = np.asarray(self.wfn.H.L)  # spin-adapted, physicist, unperturbed

        d = self.wfn.derivatives
        Sx = d.overlap(atom)                                   # 3 x (nmo,nmo)
        hx = d.core(atom)                                      # 3 x (nmo,nmo)
        gx = [g.swapaxes(1, 2) for g in                        # chemist -> physicist
              d.eri(atom)]                                     # 3 x (nmo^4) <pq|rs>^X

        Lvooo = L[v, o, o, o]
        B, Foo, Soo = [], [], []
        for c in range(3):
            Lx = 2.0 * gx[c] - gx[c].swapaxes(2, 3)            # derivative spin-adapted L
            # skeleton derivative Fock: F^X_pq = h^X + sum_k L[p,k,q,k]^X
            Fx = hx[c] + self.contract('pkqk->pq', Lx[:, o, :, o])
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^X_kl into the
            # ov block:  -1/2 S^X_kl ( L[a,k,i,l] + L[a,l,i,k] ).
            coupling = (self.contract('akil,kl->ia', Lvooo, Sx[c][o, o])
                        + self.contract('alik,kl->ia', Lvooo, Sx[c][o, o]))
            # The CPHF RHS is minus the first-order off-diagonal ("perturbation")
            # Fock that the response drives to zero -- B = -Q (nuclear analog of the
            # field's B = -mu).
            Q = Fx[o, v] - eps_o[:, None] * Sx[c][o, v] - 0.5 * coupling
            B.append(-Q)
            Foo.append(Fx[o, o])
            Soo.append(Sx[c][o, o])
        return B, Foo, Soo

    def _so_build_nuclear(self, atom: int):
        """Spin-orbital nuclear CPHF RHS for ``atom`` -- the spin-orbital analogue of
        :meth:`_build_nuclear`. Same structure with the antisymmetrized ``<pq||rs>`` in
        place of the spin-adapted ``L``, the spin-orbital skeleton derivative integrals
        from ``Derivatives.so_*``, and singly occupied spin orbitals::

            F^X_pq = h^X_pq + sum_k(occ) <pk||qk>^X        (skeleton Fock derivative)
            B^X_ia = -[ F^X_ia - eps_i S^X_ia
                        - 1/2 sum_kl S^X_kl ( <ak||il> + <al||ik> ) ]

        Returns ``(B, Foo, Soo)`` as for the spatial path."""
        o, v = self.o, self.v
        eps_o = self.eps[o]
        ERI = np.asarray(self.wfn.H.ERI)         # antisymmetrized, unperturbed
        d = self.wfn.derivatives
        hx = d.so_core(atom)                     # 3 x (nmo,nmo)
        Sxa = d.so_overlap(atom)                 # 3 x (nmo,nmo)
        gx = d.so_eri(atom)                      # 3 x (nmo^4), antisymmetrized <pq||rs>^X

        Evooo = ERI[v, o, o, o]
        B, Foo, Soo = [], [], []
        for c in range(3):
            # skeleton derivative Fock F^X_pq = h^X + sum_k(occ) <pk||qk>^X
            Fx = hx[c] + self.contract('pkqk->pq', gx[c][:, o, :, o])
            Sx = Sxa[c]
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
