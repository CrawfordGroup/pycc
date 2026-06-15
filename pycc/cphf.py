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

from typing import Any

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
        L = np.asarray(self.wfn.H.L)

        if kind == "electric":
            sign = 1.0
        elif kind == "magnetic":
            sign = -1.0
        else:
            raise ValueError("kind must be 'electric' or 'magnetic', got %r" % kind)

        # Two-electron part: G_iajb = L[a,j,i,b] + sign * L[a,b,i,j].
        G = (np.einsum('ajib->iajb', L[v, o, o, v])
             + sign * np.einsum('abij->iajb', L[v, v, o, o]))
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
