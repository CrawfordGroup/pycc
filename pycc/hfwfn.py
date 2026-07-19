"""
hfwfn.py: Hartree-Fock wavefunction and MO-basis analytic derivative properties.
"""

from __future__ import annotations

from typing import Any

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

    # HF properties are all-electron: force nfzc=0 regardless of the reference's
    # frozen-core designation (see Wavefunction._all_electron).
    _all_electron = True

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
        super().__init__(scf_wfn, **kwargs)
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

        The spin-orbital path (open-shell UHF/ROHF, or a closed shell forced to spin
        orbitals) is handled by :meth:`_so_gradient_electronic`.  The electronic and
        nuclear-repulsion pieces are computed separately (:meth:`_gradient_electronic` /
        :meth:`Derivatives.nuclear_repulsion`) and summed here; the :func:`pycc.gradient`
        facade exposes them as :class:`pycc.PropertyComponents`.
        """
        grad = self._gradient_electronic() + self.derivatives.nuclear_repulsion()
        self.grad = grad
        return grad

    def _gradient_electronic(self) -> np.ndarray:
        """Electronic (CPHF-free) part of the RHF gradient, shape (natom, 3) -- the public
        :meth:`gradient` adds the nuclear-repulsion derivative ``dV_NN/dX``.  Dispatches the
        spin-orbital path to :meth:`_so_gradient_electronic`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_gradient_electronic()
        o = self.o
        eps = np.asarray(diag(self.H.F))[o]
        d = self.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            Sx = d.overlap(atom, 'o', 'o')
            hx = d.core(atom, 'o', 'o')
            erix = d.eri(atom, 'o', 'o', 'o', 'o')
            for c in range(3):
                grad[atom, c] = (2.0 * np.trace(hx[c])
                                 + 2.0 * self.contract('iijj->', erix[c])
                                 - self.contract('ijij->', erix[c])
                                 - 2.0 * self.contract('i,ii->', eps, Sx[c]))
        return grad

    def _so_gradient_electronic(self) -> np.ndarray:
        """Electronic part of the spin-orbital HF gradient (i, j over occupied spin orbitals;
        ``<ij||ij>`` antisymmetrized), valid for UHF/ROHF and a closed shell forced to spin
        orbitals.  For a closed shell this equals the spatial electronic gradient term-for-term;
        :meth:`gradient` adds ``dV_NN/dX``."""
        eps = np.asarray(diag(self.H.F))[self.o]
        d = self.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.so_core(atom, 'o', 'o')
            Sx = d.so_overlap(atom, 'o', 'o')
            erix = d.so_eri(atom, 'o', 'o', 'o', 'o')
            for c in range(3):
                grad[atom, c] = (self.contract('ii->', hx[c])
                                 + 0.5 * self.contract('ijij->', erix[c])
                                 - self.contract('i,ii->', eps, Sx[c]))
        return grad

    def dipole(self) -> np.ndarray:
        """Total SCF electric-dipole moment (a.u.), shape ``(3,)``: the nuclear term plus
        the electronic term, in the molecule's frame (gauge-independent for a neutral system).

        ``mu_a = sum_A Z_A R_Aa  +  k sum_i (mu_a)_ii``, the electronic part being the occupied
        trace of the MO dipole integrals (``H.mu`` = ``-e r``; same ``Tr(D mu)`` convention as
        :meth:`MPwfn.relaxed_dipole`, with the SCF density's occupied block). The prefactor is
        the orbital occupancy: ``k = 2`` on the spatial closed-shell path, ``k = 1`` for singly
        occupied spin orbitals. Kept separate from the correlation dipole so the total MP2
        dipole is ``HFwfn.dipole() + MPwfn.relaxed_dipole()`` (mirroring the gradient split).
        Basis-aware. The :func:`pycc.dipole` facade exposes the nuclear/reference/correlation
        pieces as :class:`pycc.PropertyComponents`."""
        mol = self.ref.molecule()
        geom = np.asarray(mol.geometry())                      # (natom, 3), bohr
        Z = np.array([mol.Z(A) for A in range(mol.natom())])
        nuc = np.einsum('a,ax->x', Z, geom)
        self.mu_scf = nuc + self._dipole_electronic()
        return self.mu_scf

    def _dipole_electronic(self) -> np.ndarray:
        """Electronic part of the SCF dipole (a.u.), shape (3,): ``k sum_i (mu_a)_ii`` (the
        occupied trace of the MO dipole integrals; ``k = 2`` spatial, ``1`` spin-orbital).
        :meth:`dipole` adds the nuclear term ``sum_A Z_A R_A``."""
        o = self.o
        k = 1.0 if self.orbital_basis == 'spinorbital' else 2.0
        return np.array([k * np.trace(np.asarray(self.H.mu[a])[o, o]) for a in range(3)])

    def polarizability(self) -> np.ndarray:
        r"""Static electric-dipole polarizability tensor (a.u.), shape ``(3, 3)``.

        The field does not move the basis functions, so there is no overlap/Pulay term: solve the
        electric-field CPHF response per Cartesian axis (:class:`CPHF`) and contract with the MO
        dipole integrals::

            alpha_ab = k sum_ia mu^a_ia U^b_ia

        .. math::

            \begin{aligned}
            \alpha_{ab} = k \sum_{ia} \mu^{a}_{ia}\, U^{b}_{ia}
            \end{aligned}

        where ``U^b`` solves the ov-block field response ``G U^b = mu^b``.  The prefactor ``k``
        counts the ov + vo response (factor 2) and the orbital occupancy: ``k = 4`` on the spatial
        closed-shell path (double occupancy), ``k = 2`` for singly occupied spin orbitals.
        Basis-aware.
        """
        o, v = self.o, self.v
        k = 2.0 if self.orbital_basis == 'spinorbital' else 4.0
        mu = [np.asarray(self.H.mu[c])[o, v] for c in range(3)]
        U = [self.cphf.solve_field(b) for b in range(3)]
        alpha = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                alpha[a, b] = k * self.contract('ia,ia->', mu[a], U[b])
        self.alpha = alpha
        return self.alpha

    def dipole_derivatives(self) -> np.ndarray:
        r"""Analytic nuclear dipole derivatives ``d(mu_alpha)/d(X_A,beta)`` (a.u.),
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

        .. math::

            \begin{aligned}
            \frac{d\mu_\alpha}{dX_{A\beta}}
            &= Z_A\,\delta_{\alpha\beta} && \text{(nuclear)} \\
            &\quad + 2\sum_i \Big(\tfrac{\partial\mu_\alpha}{\partial X_{A\beta}}\Big)_{ii} && \text{(explicit electronic)} \\
            &\quad - 2\sum_{ik} S^X_{ki}\,(\mu_\alpha)_{ik} && \text{(oo / Pulay response)} \\
            &\quad + 4\sum_{ia} U^X_{ia}\,(\mu_\alpha)_{ia} && \text{(ov / CPHF response)}
            \end{aligned}

        The spin-orbital path is handled by :meth:`_so_dipole_derivatives_electronic`; the
        nuclear ``Z_A delta`` term is added here and the electronic part comes from
        :meth:`_dipole_derivatives_electronic`.  The :func:`pycc.apt` facade exposes the pieces
        as :class:`pycc.PropertyComponents`.
        """
        dmu = self._dipole_derivatives_electronic()
        mol = self.ref.molecule()
        for A in range(mol.natom()):
            for a in range(3):
                dmu[A, a, a] += mol.Z(A)             # nuclear Z_A delta_{alpha,beta}
        self.dipder = dmu
        return self.dipder

    def _dipole_derivatives_electronic(self) -> np.ndarray:
        """Electronic part of the APT (a.u.), shape ``(natom, 3, 3)`` -- the explicit + Pulay +
        CPHF-response terms without the nuclear ``Z_A delta`` (added by
        :meth:`dipole_derivatives`).  Dispatches to :meth:`_so_dipole_derivatives_electronic`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_dipole_derivatives_electronic()
        o, v = self.o, self.v
        mu = [np.asarray(self.H.mu[a]) for a in range(3)]
        d = self.derivatives
        natom = self.ref.molecule().natom()
        dmu = np.zeros((natom, 3, 3))
        for A in range(natom):
            Ux = self.cphf.solve_nuclear(A)      # cached; shared with the Hessian
            Sx = d.overlap(A, 'o', 'o')          # 3 x (no,no)
            dip = d.dipole(A, 'o', 'o')          # 9 x (no,no): index alpha*3 + beta
            for beta in range(3):
                for alpha in range(3):
                    val = 2.0 * np.trace(dip[alpha * 3 + beta])
                    val -= 2.0 * self.contract('ki,ki->', Sx[beta], mu[alpha][o, o])
                    val += 4.0 * self.contract('ia,ia->', Ux[beta], mu[alpha][o, v])
                    dmu[A, beta, alpha] = val
        return dmu

    def _so_dipole_derivatives_electronic(self) -> np.ndarray:
        """Electronic part of the spin-orbital APT (singly occupied spin orbitals: one-electron
        traces carry factor 1 and the ov response factor 2, vs 2 and 4 closed-shell), with the
        spin-orbital nuclear CPHF response and dipole derivatives.  :meth:`dipole_derivatives`
        adds ``Z_A delta``.  Valid for UHF and a closed shell forced to spin orbitals; ROHF
        raises (via :meth:`CPHF.solve`)."""
        o, v = self.o, self.v
        mu = [np.asarray(self.H.mu[a]) for a in range(3)]
        d = self.derivatives
        natom = self.ref.molecule().natom()
        dmu = np.zeros((natom, 3, 3))
        for A in range(natom):
            Ux = self.cphf.solve_nuclear(A)      # 3 x (no,nv), spin-orbital response
            Sx = d.so_overlap(A, 'o', 'o')       # 3 x (no,no)
            dip = d.so_dipole(A, 'o', 'o')       # 9 x (no,no): index alpha*3 + beta
            for beta in range(3):
                for alpha in range(3):
                    val = self.contract('ii->', dip[alpha * 3 + beta])
                    val -= self.contract('ki,ki->', Sx[beta], mu[alpha][o, o])
                    val += 2.0 * self.contract('ia,ia->', Ux[beta], mu[alpha][o, v])
                    dmu[A, beta, alpha] = val
        return dmu

    def hessian(self) -> np.ndarray:
        r"""RHF nuclear (molecular) Hessian ``d^2 E / dX_Aa dX_Bb`` (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3 + a, B*3 + b)`` -- the force-constant
        matrix, matching ``psi4.hessian('scf')`` layout.

        Built from (i) the second-derivative ("skeleton") integral terms -- the
        gradient's integrals differentiated a second time -- and (ii) the first-order
        nuclear CPHF response (:meth:`CPHF.solve_nuclear`), taken from the shared cache
        so the 3*natom solves are reused for free by :meth:`dipole_derivatives` in an IR
        workflow. The skeleton terms mirror the CPHF-free gradient with the integrals
        differentiated twice::

            2 h^{ab}_ii + (2(ii|jj) - (ij|ij))^{ab} - 2 eps_i S^{ab}_ii + V_NN^{ab}

        .. math::

            \begin{aligned}
            2\,h^{ab}_{ii} + \big(2(ii|jj) - (ij|ij)\big)^{ab} - 2\,\varepsilon_i S^{ab}_{ii} + V_{NN}^{ab}
            \end{aligned}

        and the response + first-derivative product cross terms (x = (A,a), y = (B,b);
        i,j,n,m occupied; spin-adapted ``L`` = H.L)::

            -4 U^x_ai B^y_ai - 2 S^x_ij F^y_ij - 2 S^y_ij F^x_ij
            + 4 eps_i S^x_ij S^y_ij + 2 S^x_ij S^y_nm L_imjn

        .. math::

            \begin{aligned}
            -4\,U^x_{ai} B^y_{ai} - 2\,S^x_{ij} F^y_{ij} - 2\,S^y_{ij} F^x_{ij} + 4\,\varepsilon_i S^x_{ij} S^y_{ij} + 2\,S^x_{ij} S^y_{nm} L_{imjn}
            \end{aligned}

        where ``U^x``/``B^x`` are the cached nuclear response/RHS and ``F^x_ij``/``S^x_ij``
        the skeleton derivative Fock/overlap oo blocks (cached by :meth:`CPHF.solve_nuclear`).

        The spin-orbital path is handled by :meth:`_so_hessian_electronic`; the electronic terms
        (:meth:`_hessian_electronic`) and the nuclear-repulsion second derivative are computed
        separately and summed.  The :func:`pycc.hessian` facade exposes the pieces as
        :class:`pycc.PropertyComponents`.
        """
        H = self._hessian_electronic() + self.derivatives.nuclear_repulsion2()
        self.hess = H
        return self.hess

    def _hessian_electronic(self) -> np.ndarray:
        """Electronic part of the RHF molecular Hessian, shape ``(3*natom, 3*natom)`` -- the
        skeleton (second-derivative integral) and CPHF-response terms, without the nuclear-
        repulsion second derivative ``V_NN^{ab}`` (added by :meth:`hessian`).  Dispatches to
        :meth:`_so_hessian_electronic`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_hessian_electronic()
        o, v = self.o, self.v
        eps_o = np.asarray(diag(self.H.F))[o]
        d = self.derivatives
        natom = self.ref.molecule().natom()

        Loooo = np.asarray(self.H.L)[o, o, o, o]       # i,m,j,n (spin-adapted)
        # Pre-solve & cache the nuclear response for every atom (shared with the APTs);
        # all four come from a single heavy per-atom pass (CPHF.solve_nuclear).
        U = [self.cphf.solve_nuclear(A) for A in range(natom)]            # U[A][a]->(no,nv)
        B = [self.cphf.rhs_nuclear(A) for A in range(natom)]       # B[A][a]->(no,nv)
        Foo = [self.cphf.nuclear_skeleton_fock(A) for A in range(natom)]    # F^X_ij->(no,no)
        Soo = [self.cphf.nuclear_skeleton_overlap(A) for A in range(natom)]  # S^X_ij->(no,no)

        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(natom):
                S2 = d.overlap2(A, Bat, 'o', 'o')                 # 9 x (no,no)
                h2 = d.core2(A, Bat, 'o', 'o')                    # 9 x (no,no)
                g2 = d.eri2(A, Bat, 'o', 'o', 'o', 'o')           # 9 x (no,no,no,no) chemist
                for a in range(3):
                    for b in range(3):
                        c = a * 3 + b
                        Ux, Uy = U[A][a], U[Bat][b]
                        By = B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        # --- skeleton (second-derivative integrals), as in the gradient ---
                        skel = (2.0 * np.trace(h2[c])
                                + 2.0 * self.contract('iijj->', g2[c])
                                - self.contract('ijij->', g2[c])
                                - 2.0 * self.contract('i,ii->', eps_o, S2[c]))
                        # --- first-order CPHF response + first-derivative cross terms ---
                        resp = (-4.0 * self.contract('ia,ia->', Ux, By)
                                - 2.0 * self.contract('ij,ij->', Sx, Fy)
                                - 2.0 * self.contract('ij,ij->', Sy, Fx)
                                + 4.0 * self.contract('i,ij,ij->', eps_o, Sx, Sy)
                                + 2.0 * self.contract('ij,nm,imjn->', Sx, Sy, Loooo))
                        H[A * 3 + a, Bat * 3 + b] = skel + resp
        return H

    def _so_hessian_electronic(self) -> np.ndarray:
        """Electronic part of the spin-orbital molecular Hessian, shape ``(3*natom, 3*natom)``.
        The spin-orbital form of :meth:`_hessian_electronic`: singly occupied spin orbitals (the
        closed-shell prefactors halve) with the antisymmetrized ``<ij||kl>`` and the spin-orbital
        second-derivative integrals (:meth:`Derivatives.so_core2` / ``so_eri2`` / ``so_overlap2``,
        occupied block); the nuclear-repulsion second derivative ``V_NN^{ab}`` is added by
        :meth:`hessian`::

            skeleton:  h^{ab}_ii + 1/2 <ij||ij>^{ab} - eps_i S^{ab}_ii
            response:  -2 U^x_ai B^y_ai - S^x_ij F^y_ij - S^y_ij F^x_ij
                       + 2 eps_i S^x_ij S^y_ij + S^x_ij S^y_nm <im||jn>

        Valid for UHF as well as a closed-shell RHF reference forced to spin orbitals;
        ROHF raises (the nuclear response goes through :meth:`CPHF.solve`)."""
        o = self.o
        eps_o = np.asarray(diag(self.H.F))[o]
        ERIoooo = np.asarray(self.H.ERI)[o, o, o, o]   # i,m,j,n (antisymmetrized)
        d = self.derivatives
        natom = self.ref.molecule().natom()

        U = [self.cphf.solve_nuclear(A) for A in range(natom)]            # U[A][a]->(no,nv)
        B = [self.cphf.rhs_nuclear(A) for A in range(natom)]       # B[A][a]->(no,nv)
        Foo = [self.cphf.nuclear_skeleton_fock(A) for A in range(natom)]    # F^X_ij->(no,no)
        Soo = [self.cphf.nuclear_skeleton_overlap(A) for A in range(natom)]  # S^X_ij->(no,no)

        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(natom):
                S2 = d.so_overlap2(A, Bat, 'o', 'o')      # 9 x (no,no)
                h2 = d.so_core2(A, Bat, 'o', 'o')         # 9 x (no,no)
                g2 = d.so_eri2(A, Bat, 'o', 'o', 'o', 'o')  # 9 x (no^4) <ij||kl>^{ab}
                for a in range(3):
                    for b in range(3):
                        c = a * 3 + b
                        Ux, Uy = U[A][a], U[Bat][b]
                        By = B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        # --- skeleton (second-derivative integrals), as in the gradient ---
                        skel = (self.contract('ii->', h2[c])
                                + 0.5 * self.contract('ijij->', g2[c])
                                - self.contract('i,ii->', eps_o, S2[c]))
                        # --- first-order CPHF response + first-derivative cross terms ---
                        resp = (-2.0 * self.contract('ia,ia->', Ux, By)
                                - self.contract('ij,ij->', Sx, Fy)
                                - self.contract('ij,ij->', Sy, Fx)
                                + 2.0 * self.contract('i,ij,ij->', eps_o, Sx, Sy)
                                + self.contract('ij,nm,imjn->', Sx, Sy, ERIoooo))
                        H[A * 3 + a, Bat * 3 + b] = skel + resp
        return H

    def atomic_axial_tensors(self) -> np.ndarray:
        r"""RHF atomic axial tensors (AATs) ``I^lambda_{alpha,beta}`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[lambda, alpha, beta]`` -- the electronic part of
        the magnetic-dipole vibrational transition moment (common gauge origin), for VCD.

        Eq. (16) of the AAT note (CPHF coefficients over dependent pairs already
        cancelled analytically)::

            I^lambda_{alpha,beta} = 2 sum_ia [ U^R_ai U^B_ai + U^B_ai <phi^R_i | phi_a> ]

        .. math::

            \begin{aligned}
            I^\lambda_{\alpha\beta} = 2 \sum_{ia} \big[\, U^R_{ai} U^B_{ai} + U^B_{ai} \langle \phi^R_i | \phi_a \rangle \,\big]
            \end{aligned}

        with ``U^R`` the nuclear CPHF response (:meth:`CPHF.solve_nuclear`, shared cache),
        ``U^B`` the magnetic-field response (:meth:`CPHF.solve_magnetic`, antisymmetric
        Hessian), and ``<phi^R_i|phi_a>`` the nuclear half-derivative overlap
        (``Derivatives.overlap_half`` 'LEFT'). The magnetic integrals are stripped of
        their ``i`` (so the response and AAT are real); the full VCD AAT adds the nuclear
        term ``(Z_lambda/4) eps_{alpha,beta,gamma} R_{lambda,gamma}``.

        The spin-orbital path dispatches to :meth:`_so_atomic_axial_tensors`.
        """
        if self.orbital_basis == 'spinorbital':
            return self._so_atomic_axial_tensors()
        d = self.derivatives
        natom = self.ref.molecule().natom()

        Ub = [self.cphf.solve_magnetic(beta) for beta in range(3)]   # 3 x (no,nv), real
        aat = np.zeros((natom, 3, 3))
        for lam in range(natom):
            Ur = self.cphf.solve_nuclear(lam)                        # 3 x (no,nv), cached
            Shalf = d.overlap_half(lam, 'o', 'v', side="LEFT")       # 3 x (no,nv)
            for alpha in range(3):
                for beta in range(3):
                    aat[lam, alpha, beta] = 2.0 * (
                        self.contract('ia,ia->', Ur[alpha], Ub[beta])
                        + self.contract('ia,ia->', Ub[beta], Shalf[alpha]))
        self.aat = aat
        return self.aat

    def _so_atomic_axial_tensors(self) -> np.ndarray:
        r"""Spin-orbital RHF/UHF atomic axial tensors (AATs), shape ``(natom, 3, 3)``. The
        spin-orbital form of :meth:`atomic_axial_tensors`: singly occupied spin orbitals
        (the closed-shell prefactor 2 -> 1), with the spin-orbital nuclear response
        (:meth:`CPHF.solve_nuclear`), magnetic response (:meth:`CPHF.solve_magnetic`), and
        nuclear half-derivative overlaps (:meth:`Derivatives.so_overlap_half`)::

            I^lam_{a,b} = sum_ia [ U^R_ai U^B_ai + U^B_ai <phi^R_i | phi_a> ]

        .. math::

            \begin{aligned}
            I^\lambda_{ab} = \sum_{ia} \big[\, U^R_{ai} U^B_{ai} + U^B_{ai} \langle \phi^R_i | \phi_a \rangle \,\big]
            \end{aligned}

        Valid for UHF as well as a closed-shell RHF reference forced to spin orbitals
        (ROHF raises, via :meth:`CPHF.solve`). Note: there is no prior open-shell UHF AAT
        implementation to validate against; the closed-shell keystone (SO-RHF == the
        DALTON-validated spatial AAT) validates the spin-orbital machinery, and the UHF
        result runs through the same code path."""
        d = self.derivatives
        natom = self.ref.molecule().natom()

        Ub = [self.cphf.solve_magnetic(beta) for beta in range(3)]   # 3 x (no,nv), real
        aat = np.zeros((natom, 3, 3))
        for lam in range(natom):
            Ur = self.cphf.solve_nuclear(lam)                        # 3 x (no,nv), cached
            Shalf = d.so_overlap_half(lam, 'o', 'v', side="LEFT")    # 3 x (no,nv)
            for alpha in range(3):
                for beta in range(3):
                    aat[lam, alpha, beta] = (
                        self.contract('ia,ia->', Ur[alpha], Ub[beta])
                        + self.contract('ia,ia->', Ub[beta], Shalf[alpha]))
        self.aat = aat
        return self.aat

    def velocity_dipole_derivatives(self) -> np.ndarray:
        r"""Velocity-gauge (VG) atomic polar tensors ``[P^A_{beta,alpha}]^VG`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` = ``d(mu_alpha)/d(X_A,beta)`` -- the
        momentum-form APT, an alternative to the length-gauge :meth:`dipole_derivatives`.

        Formulated (Amos, Jalkanen & Stephens, JPC 92, 5571 (1988), Eq. 14; Shumberger et al.,
        LG(OI) VCD, Eq. 15) as an overlap of wave-function derivatives -- the nuclear derivative
        of the bra with the magnetic-vector-potential derivative of the ket::

            [P^A_{beta,alpha}]^VG = -4 sum_ia (U^R_{ia,beta} + <phi^R_i|phi_a>) U^A_{ia,alpha}
                                    + Z_A delta_{alpha,beta}

        .. math::

            \begin{aligned}
            [P^A_{\beta\alpha}]^{\mathrm{VG}} = -4 \sum_{ia}
            \big( U^R_{ia,\beta} + \langle \phi^R_i | \phi_a \rangle \big) U^A_{ia,\alpha} +
            Z_A\,\delta_{\alpha\beta}
            \end{aligned}

        the same overlap structure as the AAT (:meth:`atomic_axial_tensors`) with the linear-
        momentum response ``U^A`` (:meth:`CPHF.solve_momentum`, ``dPsi/dA``) in place of the
        magnetic response ``U^B``, and the length-gauge ``Z_A delta`` nuclear term in place of
        the AAT's Levi-Civita term. ``U^R`` is the nuclear CPHF response; ``<phi^R_i|phi_a>`` the
        nuclear half-derivative overlap (the vector potential does not move the basis, so this
        rides on the R side only). The ``-4`` prefactor is the closed-shell value (real-wf
        doubling x double occupancy; the two imaginary units of ``-2i`` x ``H.p`` fix the sign)
        -- pinned to the Amos et al. NH3 ``P(pi)`` values.

        Unlike the length-gauge APT, the VG APT differs from it in a finite basis and converges
        to it only toward the basis-set limit; both are origin-independent. The spin-orbital path
        is handled by :meth:`_so_velocity_dipole_derivatives_electronic`; the nuclear ``Z_A delta``
        term is added here.  The :func:`pycc.apt` (``gauge='velocity'``) facade exposes the pieces
        as :class:`pycc.PropertyComponents`."""
        P = self._velocity_dipole_derivatives_electronic()
        mol = self.ref.molecule()
        for A in range(mol.natom()):
            for a in range(3):
                P[A, a, a] += mol.Z(A)              # nuclear Z_A delta_{alpha,beta}
        self.vgapt = P
        return self.vgapt

    def _velocity_dipole_derivatives_electronic(self) -> np.ndarray:
        """Electronic part of the VG APT (a.u.), shape ``(natom, 3, 3)`` -- the wave-function-
        overlap term ``-4 sum_ia (U^R + <phi^R|phi>) U^A`` without the nuclear ``Z_A delta``
        (added by :meth:`velocity_dipole_derivatives`).  Dispatches to
        :meth:`_so_velocity_dipole_derivatives_electronic`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_velocity_dipole_derivatives_electronic()
        d = self.derivatives
        natom = self.ref.molecule().natom()
        Ua = [self.cphf.solve_momentum(alpha) for alpha in range(3)]   # 3 x (no,nv)
        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            Ur = self.cphf.solve_nuclear(A)                            # 3 x (no,nv)
            Sh = d.overlap_half(A, 'o', 'v', side="LEFT")              # 3 x (no,nv)
            for beta in range(3):
                for alpha in range(3):
                    P[A, beta, alpha] = -4.0 * self.contract('ia,ia->', Ur[beta] + Sh[beta], Ua[alpha])
        return P

    def _so_velocity_dipole_derivatives_electronic(self) -> np.ndarray:
        """Electronic part of the spin-orbital VG APT: singly occupied spin orbitals halve the
        closed-shell prefactor (``-4 -> -2``), with the spin-orbital nuclear/momentum responses
        (:meth:`CPHF.solve_nuclear`/:meth:`CPHF.solve_momentum`) and half-derivative overlaps
        (:meth:`Derivatives.so_overlap_half`); :meth:`velocity_dipole_derivatives` adds
        ``Z_A delta``.  Valid for UHF and a closed shell forced to spin orbitals; ROHF raises
        (via :meth:`CPHF.solve`)."""
        d = self.derivatives
        natom = self.ref.molecule().natom()
        Ua = [self.cphf.solve_momentum(alpha) for alpha in range(3)]
        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            Ur = self.cphf.solve_nuclear(A)
            Sh = d.so_overlap_half(A, 'o', 'v', side="LEFT")
            for beta in range(3):
                for alpha in range(3):
                    P[A, beta, alpha] = -2.0 * self.contract('ia,ia->', Ur[beta] + Sh[beta], Ua[alpha])
        return P
