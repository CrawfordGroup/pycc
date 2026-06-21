"""
hfwfn.py: Hartree-Fock wavefunction and MO-basis analytic derivative properties.
"""

from __future__ import annotations

from typing import Any

import psi4
import numpy as np

from .wavefunction import Wavefunction
from .utils import diag


class HFwfn(Wavefunction):
    """An RHF wavefunction on the shared :class:`Wavefunction` base, and the home
    for MO-basis HF analytic derivative properties (gradient now; Hessian, APTs,
    AATs, and a CPHF solver to follow).

    The SCF energy is the base's reference energy (``self.eref``); this class adds
    the response/derivative engine, including a :class:`Derivatives` provider.

    Attributes
    ----------
    derivatives : Derivatives
        MO-basis derivative-integral provider, inherited from the :class:`Wavefunction`
        base (lazy ``self.derivatives``)
    cphf : CPHF
        coupled-perturbed HF orbital-response solver (promotable to the base later,
        once a correlated-gradient consumer fixes the solver/assembly seam)
    grad : numpy.ndarray
        the most recently computed nuclear gradient, shape (natom, 3)
    """

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
        # HF properties are all-electron: always use the full MO space, regardless
        # of any frozen core the reference was run with.
        kwargs.pop('frozen_core', None)
        super().__init__(scf_wfn, frozen_core=False, **kwargs)
        # The derivative-integral provider and the CPHF orbital-response solver now live
        # on the base, built lazily (``self.derivatives`` / ``self.cphf``).

    def gradient(self) -> np.ndarray:
        """RHF analytic energy gradient (a.u.), shape (natom, 3).

        Closed-shell, MO-basis, CPHF-free RHF gradient (i, j over occupied; the
        ``^x`` skeleton derivatives come from :class:`Derivatives`, ``eps_i`` are the
        occupied orbital energies, and the ``-2 eps_i S^x_ii`` term is the
        energy-weighted-density / orbital-response contribution that makes the HF
        gradient CPHF-free)::

            dE/dX = sum_i 2 h^x_ii + sum_ij (2 (ii|jj)^x - (ij|ij)^x)
                    - sum_i 2 eps_i S^x_ii + dV_NN/dX

        The derivative integrals are transformed with the base's symmetry-handled
        ``self.C`` (single irrep block, global energy order), so this works with
        molecular symmetry left on. HFwfn always uses the full (all-electron) MO
        space, so a frozen core on the reference does not affect the gradient.
        """
        o = self.o
        no = self.no
        # Occupied block of the symmetry-handled MO coefficients, and the matching
        # (energy-ordered) occupied orbital energies from the Fock diagonal.
        Cocc = psi4.core.Matrix.from_array(np.asarray(self.C)[:, :no])
        eps = np.asarray(diag(self.H.F))[o]

        d = self.derivatives
        grad = np.zeros((d.natom, 3))
        Vnn = d.nuclear_repulsion()
        for atom in range(d.natom):
            Sx = d.overlap(atom, Cocc, Cocc)
            hx = d.core(atom, Cocc, Cocc)
            erix = d.eri(atom, Cocc, Cocc, Cocc, Cocc)
            for c in range(3):
                grad[atom, c] = (2.0 * np.trace(hx[c])
                                 + 2.0 * np.einsum('iijj->', erix[c])
                                 - np.einsum('ijij->', erix[c])
                                 - 2.0 * np.einsum('i,ii->', eps, Sx[c])
                                 + Vnn[atom, c])
        self.grad = grad
        return grad

    def polarizability(self) -> np.ndarray:
        """Static electric-dipole polarizability tensor (a.u.), shape ``(3, 3)``.

        The field perturbation does not move the basis functions, so the response has
        no overlap/Pulay contribution: solve the electric-field CPHF response for each
        Cartesian axis (:class:`CPHF`) and contract with the MO dipole integrals,
        ``alpha_ab = -4 sum_ia mu^a_ia U^b_ia`` (closed shell), where ``U^b`` solves
        ``G U^b = -mu^b`` in the ov block. (The two minus signs -- the ``-mu`` RHS and
        the ``-4`` contraction -- make alpha positive definite and cancel.)
        """
        o, v = self.o, self.v
        mu = [np.asarray(self.H.mu[c])[o, v] for c in range(3)]
        U = [self.cphf.solve(self.cphf.rhs_field(b), kind="electric") for b in range(3)]
        alpha = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                alpha[a, b] = -4.0 * np.einsum('ia,ia->', mu[a], U[b])
        self.alpha = alpha
        return self.alpha

    def dipole_derivatives(self) -> np.ndarray:
        """Analytic nuclear dipole derivatives ``d(mu_alpha)/d(X_A,beta)`` (a.u.),
        shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` -- the atomic polar
        tensors (APTs), transposed.

        Built from the nuclear CPHF response (:meth:`CPHF.solve_nuclear`), so the ov
        term probes ``U^X`` directly (unlike the energy gradient, which is variationally
        insensitive to it). The assembly is nuclear + explicit electronic + overlap
        (Pulay) + CPHF response::

            d mu_a / d X_Ab = Z_A delta_ab                         (nuclear)
                            + 2 sum_i (d mu_a / d X_Ab)_ii         (explicit electronic)
                            - 2 sum_ik S^X_ki (mu_a)_ik            (oo / Pulay response)
                            + 4 sum_ia U^X_ia (mu_a)_ia            (ov / CPHF response)
        """
        o, v = self.o, self.v
        mu = [np.asarray(self.H.mu[a]) for a in range(3)]
        d = self.derivatives
        C = np.asarray(self.C)
        Cocc = psi4.core.Matrix.from_array(C[:, o])
        mol = self.ref.molecule()
        natom = mol.natom()

        dmu = np.zeros((natom, 3, 3))
        for A in range(natom):
            Ux = self.cphf.solve_nuclear(A)      # cached; shared with the Hessian
            Sx = d.overlap(A, Cocc, Cocc)        # 3 x (no,no)
            dip = d.dipole(A, Cocc, Cocc)        # 9 x (no,no): index alpha*3 + beta
            for beta in range(3):
                for alpha in range(3):
                    val = mol.Z(A) if alpha == beta else 0.0
                    val += 2.0 * np.trace(dip[alpha * 3 + beta])
                    val -= 2.0 * np.einsum('ki,ki->', Sx[beta], mu[alpha][o, o])
                    val += 4.0 * np.einsum('ia,ia->', Ux[beta], mu[alpha][o, v])
                    dmu[A, beta, alpha] = val
        self.dipder = dmu
        return self.dipder

    def hessian(self) -> np.ndarray:
        """RHF nuclear (molecular) Hessian ``d^2 E / dX_Aa dX_Bb`` (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3 + a, B*3 + b)`` -- the force-constant
        matrix, matching ``psi4.hessian('scf')`` layout.

        Built from (i) the second-derivative ("skeleton") integral terms -- the
        gradient's integrals differentiated a second time -- and (ii) the first-order
        nuclear CPHF response (:meth:`CPHF.solve_nuclear`), taken from the shared cache
        so the 3*natom solves are reused for free by :meth:`dipole_derivatives` in an IR
        workflow. The skeleton terms mirror the CPHF-free gradient with the integrals
        differentiated twice::

            2 h^{ab}_ii + (2(ii|jj) - (ij|ij))^{ab} - 2 eps_i S^{ab}_ii + V_NN^{ab}

        and the response + first-derivative product cross terms (x = (A,a), y = (B,b);
        i,j,n,m occupied; spin-adapted ``L`` = H.L)::

            -4 U^x_ai B^y_ai - 2 S^x_ij F^y_ij - 2 S^y_ij F^x_ij
            + 4 eps_i S^x_ij S^y_ij + 2 S^x_ij S^y_nm L_imjn

        where ``U^x``/``B^x`` are the cached nuclear response/RHS and ``F^x_ij``/``S^x_ij``
        the skeleton derivative Fock/overlap oo blocks (cached by :meth:`CPHF.solve_nuclear`).
        """
        o, v = self.o, self.v
        eps_o = np.asarray(diag(self.H.F))[o]
        d = self.derivatives
        C = np.asarray(self.C)
        Cocc = psi4.core.Matrix.from_array(C[:, o])
        mol = self.ref.molecule()
        natom = mol.natom()

        Loooo = np.asarray(self.H.L)[o, o, o, o]       # i,m,j,n (spin-adapted)
        Vnn2 = d.nuclear_repulsion2()                  # (3*natom, 3*natom)
        # Pre-solve & cache the nuclear response for every atom (shared with the APTs);
        # all four come from a single heavy per-atom pass (CPHF.solve_nuclear).
        U = [self.cphf.solve_nuclear(A) for A in range(natom)]            # U[A][a]->(no,nv)
        B = [self.cphf.rhs_nuclear_cached(A) for A in range(natom)]       # B[A][a]->(no,nv)
        Foo = [self.cphf.fock_nuclear_cached(A) for A in range(natom)]    # F^X_ij->(no,no)
        Soo = [self.cphf.overlap_nuclear_cached(A) for A in range(natom)]  # S^X_ij->(no,no)

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
                        By = B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        # --- skeleton (second-derivative integrals), as in the gradient ---
                        skel = (2.0 * np.trace(h2[c])
                                + 2.0 * np.einsum('iijj->', g2[c])
                                - np.einsum('ijij->', g2[c])
                                - 2.0 * np.einsum('i,ii->', eps_o, S2[c])
                                + Vnn2[A * 3 + a, Bat * 3 + b])
                        # --- first-order CPHF response + first-derivative cross terms ---
                        resp = (-4.0 * np.einsum('ia,ia->', Ux, By)
                                - 2.0 * np.einsum('ij,ij->', Sx, Fy)
                                - 2.0 * np.einsum('ij,ij->', Sy, Fx)
                                + 4.0 * np.einsum('i,ij,ij->', eps_o, Sx, Sy)
                                + 2.0 * np.einsum('ij,nm,imjn->', Sx, Sy, Loooo))
                        H[A * 3 + a, Bat * 3 + b] = skel + resp
        self.hess = H
        return self.hess

    def atomic_axial_tensors(self) -> np.ndarray:
        """RHF atomic axial tensors (AATs) ``I^lambda_{alpha,beta}`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[lambda, alpha, beta]`` -- the electronic part of
        the magnetic-dipole vibrational transition moment (common gauge origin), for VCD.

        Eq. (16) of the AAT note (CPHF coefficients over dependent pairs already
        cancelled analytically)::

            I^lambda_{alpha,beta} = 2 sum_ia [ U^R_ai U^B_ai + U^B_ai <phi^R_i | phi_a> ]

        with ``U^R`` the nuclear CPHF response (:meth:`CPHF.solve_nuclear`, shared cache),
        ``U^B`` the magnetic-field response (:meth:`CPHF.solve_magnetic`, antisymmetric
        Hessian), and ``<phi^R_i|phi_a>`` the nuclear half-derivative overlap
        (``Derivatives.overlap_half`` 'LEFT'). The magnetic integrals are stripped of
        their ``i`` (so the response and AAT are real); the full VCD AAT adds the nuclear
        term ``(Z_lambda/4) eps_{alpha,beta,gamma} R_{lambda,gamma}``.
        """
        o, v = self.o, self.v
        d = self.derivatives
        C = np.asarray(self.C)
        Cocc = psi4.core.Matrix.from_array(C[:, o])
        Cvir = psi4.core.Matrix.from_array(C[:, v])
        natom = self.ref.molecule().natom()

        Ub = [self.cphf.solve_magnetic(beta) for beta in range(3)]   # 3 x (no,nv), real
        aat = np.zeros((natom, 3, 3))
        for lam in range(natom):
            Ur = self.cphf.solve_nuclear(lam)                        # 3 x (no,nv), cached
            Shalf = d.overlap_half(lam, Cocc, Cvir, side="LEFT")     # 3 x (no,nv)
            for alpha in range(3):
                for beta in range(3):
                    aat[lam, alpha, beta] = 2.0 * (
                        np.einsum('ia,ia->', Ur[alpha], Ub[beta])
                        + np.einsum('ia,ia->', Ub[beta], Shalf[alpha]))
        self.aat = aat
        return self.aat
