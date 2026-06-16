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

    def rhs_nuclear(self, atom: int) -> List[np.ndarray]:
        """Nuclear-perturbation CPHF RHS for ``atom``: list of 3 (x,y,z) ``(no, nv)``
        arrays ``B^X_ia``.

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

        B = []
        for c in range(3):
            Lx = 2.0 * gx[c] - gx[c].swapaxes(2, 3)            # derivative spin-adapted L
            # skeleton derivative Fock: F^X_pq = h^X + sum_k L[p,k,q,k]^X
            Fx = hx[c] + np.einsum('pkqk->pq', Lx[:, o, :, o])
            # Pulay coupling of the overlap-determined U^X_kl = -1/2 S^X_kl into the
            # ov block:  -1/2 S^X_kl ( L[a,k,i,l] + L[a,l,i,k] ).
            Lvooo = L[v, o, o, o]
            coupling = (np.einsum('akil,kl->ia', Lvooo, Sx[c][o, o])
                        + np.einsum('alik,kl->ia', Lvooo, Sx[c][o, o]))
            # The CPHF RHS is minus the first-order off-diagonal ("perturbation")
            # Fock that the response drives to zero -- B = -Q (nuclear analog of the
            # field's B = -mu).
            Q = Fx[o, v] - eps_o[:, None] * Sx[c][o, v] - 0.5 * coupling
            B.append(-Q)
        return B

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
            Bx = self.rhs_nuclear(A)
            Ux = [self.solve(Bx[c], kind="electric") for c in range(3)]
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
