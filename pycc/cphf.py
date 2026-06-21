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

The first validation target is the static dipole polarizability (electric-field
response): the field does not move the basis functions, so its right-hand side
is just the (negated) MO dipole integrals (``B = -mu``, since the field enters as
``-mu . E``) with no overlap/Pulay terms -- it exercises the solver in isolation.

Created and owned by HFwfn for now; it depends only on base state (``o``/``v``,
the Fock diagonal, ``H.L``) plus the property integrals (``H.mu``), so it can be
promoted to the Wavefunction base when MP2/CI/CC orbital response is needed.

This is reference, CPU/NumPy code, consistent with the rest of the HF derivative
engine (the explicit orbital Hessian is solved directly with ``numpy.linalg``).
"""

from __future__ import annotations

from typing import Any, List

import psi4
import numpy as np

from .utils import diag


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
        self.o = wfn.o
        self.v = wfn.v
        self.no = wfn.no
        self.nv = wfn.nv
        # Orbital energies from the Fock diagonal (energy-ordered, all-electron --
        # HFwfn builds the full MO space).
        self.eps = np.asarray(diag(wfn.H.F))
        self._G: dict = {}  # kind -> reshaped (no*nv, no*nv) orbital Hessian
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
        G = (np.einsum('ajib->iajb', W[v, o, o, v])
             + sign * np.einsum('abij->iajb', W[v, v, o, o]))
        G = G.reshape(no * nv, no * nv)

        # Orbital-energy part: + (e_a - e_i) on the (ia)=(jb) diagonal.
        D = (self.eps[v][None, :] - self.eps[o][:, None]).reshape(-1)
        G[np.diag_indices(no * nv)] += D
        return G

    # ---- linear solve ----
    def solve(self, B: np.ndarray, kind: str = "electric") -> np.ndarray:
        """Solve ``G U = B`` for the ov response. ``B`` is ``(no, nv)``; returns ``(no, nv)``."""
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
        """
        o, v, no, nv = self.o, self.v, self.no, self.nv
        eps_o = self.eps[o]
        L = np.asarray(self.wfn.H.L)  # spin-adapted, physicist, unperturbed

        C = np.asarray(self.wfn.C)
        Call = psi4.core.Matrix.from_array(C)
        d = self.wfn.derivatives
        Sx = d.overlap(atom, Call, Call)                       # 3 x (nmo,nmo)
        hx = d.core(atom, Call, Call)                          # 3 x (nmo,nmo)
        gx = [g.swapaxes(1, 2) for g in                        # chemist -> physicist
              d.eri(atom, Call, Call, Call, Call)]             # 3 x (nmo^4) <pq|rs>^X

        Lvooo = L[v, o, o, o]
        B, Foo, Soo = [], [], []
        for c in range(3):
            Lx = 2.0 * gx[c] - gx[c].swapaxes(2, 3)            # derivative spin-adapted L
            # skeleton derivative Fock: F^X_pq = h^X + sum_k L[p,k,q,k]^X
            Fx = hx[c] + np.einsum('pkqk->pq', Lx[:, o, :, o])
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^X_kl into the
            # ov block:  -1/2 S^X_kl ( L[a,k,i,l] + L[a,l,i,k] ).
            coupling = (np.einsum('akil,kl->ia', Lvooo, Sx[c][o, o])
                        + np.einsum('alik,kl->ia', Lvooo, Sx[c][o, o]))
            # The CPHF RHS is minus the first-order off-diagonal ("perturbation")
            # Fock that the response drives to zero -- B = -Q (nuclear analog of the
            # field's B = -mu).
            Q = Fx[o, v] - eps_o[:, None] * Sx[c][o, v] - 0.5 * coupling
            B.append(-Q)
            Foo.append(Fx[o, o])
            Soo.append(Sx[c][o, o])
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

    def dipole_derivatives(self) -> np.ndarray:
        """Analytic nuclear dipole derivatives ``d(mu_alpha)/d(X_A,beta)`` (a.u.),
        shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` -- the atomic polar
        tensors (APTs), transposed.

        Built from the nuclear CPHF response, so the ov term probes ``U^X`` directly
        (unlike the energy gradient, which is variationally insensitive to it). The
        assembly is nuclear + explicit electronic + overlap (Pulay) + CPHF response::

            d mu_a / d X_Ab = Z_A delta_ab                         (nuclear)
                            + 2 sum_i (d mu_a / d X_Ab)_ii         (explicit electronic)
                            - 2 sum_ik S^X_ki (mu_a)_ik            (oo / Pulay response)
                            + 4 sum_ia U^X_ia (mu_a)_ia            (ov / CPHF response)
        """
        o, v = self.o, self.v
        mu = [np.asarray(self.wfn.H.mu[a]) for a in range(3)]
        d = self.wfn.derivatives
        C = np.asarray(self.wfn.C)
        Cocc = psi4.core.Matrix.from_array(C[:, o])
        mol = self.wfn.ref.molecule()
        natom = mol.natom()

        dmu = np.zeros((natom, 3, 3))
        for A in range(natom):
            Ux = self.solve_nuclear(A)           # cached; shared with the Hessian
            Sx = d.overlap(A, Cocc, Cocc)        # 3 x (no,no)
            dip = d.dipole(A, Cocc, Cocc)        # 9 x (no,no): index alpha*3 + beta
            for beta in range(3):
                for alpha in range(3):
                    val = mol.Z(A) if alpha == beta else 0.0
                    val += 2.0 * np.trace(dip[alpha * 3 + beta])
                    val -= 2.0 * np.einsum('ki,ki->', Sx[beta], mu[alpha][o, o])
                    val += 4.0 * np.einsum('ia,ia->', Ux[beta], mu[alpha][o, v])
                    dmu[A, beta, alpha] = val
        return dmu

    def molecular_hessian(self) -> np.ndarray:
        """RHF nuclear (molecular) Hessian ``d^2 E / dX_Aa dX_Bb`` (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3 + a, B*3 + b)`` -- the force-constant
        matrix, matching ``psi4.hessian('scf')`` layout (gate 4).

        Built from (i) the second-derivative ("skeleton") integral terms -- the
        gradient's integrals differentiated a second time -- and (ii) the first-order
        CPHF response, which is where the nuclear ``U^X`` (and its RHS ``B^X``) enter.
        The response is taken from the shared :meth:`solve_nuclear` cache, so the
        3*natom solves done here are reused for free by :meth:`dipole_derivatives`
        (the APTs) in an IR workflow.

        The assembly has two parts. (i) The skeleton second-derivative terms mirror the
        validated CPHF-free gradient with the integrals differentiated twice::

            2 h^{ab}_ii + (2(ii|jj) - (ij|ij))^{ab} - 2 eps_i S^{ab}_ii + V_NN^{ab}

        (ii) The first-order CPHF response and the first-derivative *product* cross
        terms (x = (A,a), y = (B,b); i,j,n,m occupied; spin-adapted ``L`` = H.L)::

            -4 U^x_ai B^y_ai - 2 S^x_ij F^y_ij - 2 S^y_ij F^x_ij
            + 4 eps_i S^x_ij S^y_ij + 2 S^x_ij S^y_nm L_imjn

        where ``U^x``/``B^x`` are the cached nuclear response/RHS and ``F^x_ij``/``S^x_ij``
        are the skeleton derivative Fock and overlap oo blocks (cached by-products of
        :meth:`solve_nuclear`). Validated against ``psi4.hessian('scf')`` (gate 4).
        """
        o, v = self.o, self.v
        eps_o = self.eps[o]
        d = self.wfn.derivatives
        C = np.asarray(self.wfn.C)
        Cocc = psi4.core.Matrix.from_array(C[:, o])
        mol = self.wfn.ref.molecule()
        natom = mol.natom()

        Loooo = np.asarray(self.wfn.H.L)[o, o, o, o]       # i,m,j,n (spin-adapted)
        Vnn2 = d.nuclear_repulsion2()                      # (3*natom, 3*natom)
        # Pre-solve & cache the nuclear response for every atom (shared with the APTs).
        # All four come from a single heavy per-atom pass (solve_nuclear).
        U = [self.solve_nuclear(A) for A in range(natom)]          # U[A][a] -> (no,nv)
        B = [self.rhs_nuclear_cached(A) for A in range(natom)]     # B[A][a] -> (no,nv)
        Foo = [self.fock_nuclear_cached(A) for A in range(natom)]  # F^X_ij -> (no,no)
        Soo = [self.overlap_nuclear_cached(A) for A in range(natom)]  # S^X_ij -> (no,no)

        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(natom):
                S2 = d.overlap2(A, Bat, Cocc, Cocc)               # 9 x (no,no)
                h2 = d.core2(A, Bat, Cocc, Cocc)                  # 9 x (no,no)
                g2 = d.eri2(A, Bat, Cocc, Cocc, Cocc, Cocc)       # 9 x (no,no,no,no) chemist
                for a in range(3):
                    for b in range(3):
                        c = a * 3 + b
                        Ux, Uy = U[A][a], U[Bat][b]
                        Bx, By = B[A][a], B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        # --- skeleton (second-derivative integrals), as in the gradient ---
                        skel = (2.0 * np.trace(h2[c])
                                + 2.0 * np.einsum('iijj->', g2[c])
                                - np.einsum('ijij->', g2[c])
                                - 2.0 * np.einsum('i,ii->', eps_o, S2[c])
                                + Vnn2[A * 3 + a, Bat * 3 + b])
                        # --- first-order CPHF response + first-derivative cross terms:
                        #   -4 U^x_ai B^y_ai - 2 S^x_ij F^y_ij - 2 S^y_ij F^x_ij
                        #   + 4 eps_i S^x_ij S^y_ij + 2 S^x_ij S^y_nm L_imjn
                        resp = (-4.0 * np.einsum('ia,ia->', Ux, By)
                                - 2.0 * np.einsum('ij,ij->', Sx, Fy)
                                - 2.0 * np.einsum('ij,ij->', Sy, Fx)
                                + 4.0 * np.einsum('i,ij,ij->', eps_o, Sx, Sy)
                                + 2.0 * np.einsum('ij,nm,imjn->', Sx, Sy, Loooo))
                        H[A * 3 + a, Bat * 3 + b] = skel + resp
        return H

    def atomic_axial_tensors(self) -> np.ndarray:
        """RHF atomic axial tensors (AATs) ``I^lambda_{alpha,beta}`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[lambda, alpha, beta]`` -- the overlap of the
        nuclear (``R_{lambda,alpha}``) and magnetic-field (``B_beta``) wavefunction
        derivatives, the electronic part of the magnetic-dipole vibrational transition
        moment (common gauge origin).

        Implements Eq. (16) of the AAT note (CPHF coefficients over dependent pairs
        already cancelled analytically)::

            I^lambda_{alpha,beta} = 2 sum_{ia} [ U^{R}_{ai} U^{B}_{ai}
                                                 + U^{B}_{ai} <phi^{R}_i | phi_a> ]

        with ``U^{R}`` the nuclear CPHF response (shared cache, :meth:`solve_nuclear`),
        ``U^{B}`` the magnetic-field response (:meth:`solve_magnetic`, antisymmetric
        Hessian), and ``<phi^R_i|phi_a>`` the nuclear half-derivative overlap (occupied
        derivative against the unperturbed virtual; ``Derivatives.overlap_half`` 'LEFT').

        This is the electronic part only (the magnetic-dipole vibrational transition
        moment); the full VCD AAT adds the nuclear term
        ``(Z_lambda / 4) eps_{alpha,beta,gamma} R_{lambda,gamma}``. The magnetic-dipole
        integrals are stripped of their ``i`` (:meth:`_m_ov`) so the magnetic response
        and the AAT are real -- the VCD rotatory strength takes the imaginary component
        of the APT*AAT product, not of the AAT. Validated against DALTON's STO-3G H2O2
        electronic AATs (via the psi4numpy SCF-VCD reference) to ~5e-9.
        """
        o, v = self.o, self.v
        d = self.wfn.derivatives
        C = np.asarray(self.wfn.C)
        Cocc = psi4.core.Matrix.from_array(C[:, o])
        Cvir = psi4.core.Matrix.from_array(C[:, v])
        natom = self.wfn.ref.molecule().natom()

        Ub = [self.solve_magnetic(beta) for beta in range(3)]      # 3 x (no,nv), real
        aat = np.zeros((natom, 3, 3))
        for lam in range(natom):
            Ur = self.solve_nuclear(lam)                           # 3 x (no,nv), cached
            Shalf = d.overlap_half(lam, Cocc, Cvir, side="LEFT")   # 3 x (no,nv)
            for alpha in range(3):
                for beta in range(3):
                    aat[lam, alpha, beta] = 2.0 * (
                        np.einsum('ia,ia->', Ur[alpha], Ub[beta])
                        + np.einsum('ia,ia->', Ub[beta], Shalf[alpha]))
        return aat

    # ---- property drivers ----
    def polarizability(self) -> np.ndarray:
        """Static electric-dipole polarizability tensor (a.u.), shape ``(3, 3)``.

        Solves the electric-field CPHF response for each Cartesian axis and contracts
        with the dipole integrals: ``alpha_ab = -4 sum_ia mu^a_ia U^b_ia`` (closed
        shell), where ``U^b`` solves ``G U^b = -mu^b`` in the ov block. (The two minus
        signs -- the ``-mu`` RHS and the ``-4`` contraction -- make alpha positive
        definite, and cancel against each other so the tensor is unchanged.)
        """
        mu = [self._mu_ov(c) for c in range(3)]
        U = [self.solve(self.rhs_field(b), kind="electric") for b in range(3)]
        alpha = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                alpha[a, b] = -4.0 * np.einsum('ia,ia->', mu[a], U[b])
        return alpha
