"""
hfwfn.py: Hartree-Fock wavefunction and MO-basis analytic derivative properties.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .wavefunction import Wavefunction
from .utils import diag


class HFwfn(Wavefunction):
    """An RHF wavefunction on the shared :class:`Wavefunction` base, and the home for
    the MO-basis HF analytic derivative properties: gradient, (relaxed) dipole,
    polarizability, dipole derivatives (APTs), Hessian, and atomic axial tensors (AATs).

    The SCF energy is the base's reference energy (``self.eref``).  The
    response/derivative *engine* these build on -- the :class:`~pycc.cphf.CPHF`
    orbital-response solver and the :class:`~pycc.derivatives.Derivatives` integral
    provider -- lives on the :class:`Wavefunction` base, not on this class; it is reached
    through ``self.cphf`` / ``self.derivatives``.

    Attributes
    ----------
    derivatives : Derivatives
        MO-basis derivative-integral provider, inherited from the :class:`Wavefunction`
        base (lazy ``self.derivatives``)
    cphf : CPHF
        coupled-perturbed HF orbital-response solver, inherited from the
        :class:`Wavefunction` base (lazy ``self.cphf``)
    grad : numpy.ndarray
        the most recently computed nuclear gradient, shape (natom, 3)
    """

    # HF properties are all-electron: force nfzc=0 regardless of the reference's
    # frozen-core designation (see Wavefunction._all_electron).
    _all_electron = True

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
        self.method = 'HF'                  # preamble label (see Wavefunction._report_preamble)
        super().__init__(scf_wfn, **kwargs)
        # The derivative-integral provider and the CPHF orbital-response solver now live
        # on the base, built lazily (``self.derivatives`` / ``self.cphf``).

    def gradient(self) -> "PropertyComponents":
        r"""RHF analytic energy gradient (a.u.), shape (natom, 3).

        Closed-shell, MO-basis, CPHF-free RHF gradient (i, j over occupied; the
        ``^x`` skeleton derivatives come from :class:`Derivatives`, ``eps_i`` are the
        occupied orbital energies, and the ``-2 eps_i S^x_ii`` term is the
        energy-weighted-density / orbital-response contribution that makes the HF
        gradient CPHF-free)::

            dE/dX = 2 sum_i h^x_ii + 2 sum_ij (ii|jj)^x - sum_ij (ij|ij)^x
                    - 2 sum_i eps_i S^x_ii + dV_NN/dX

        .. math::

            \begin{aligned}
            \frac{dE}{dX} = 2\sum_i h^x_{ii} + 2\sum_{ij} (ii|jj)^x - \sum_{ij} (ij|ij)^x - 2\sum_i \varepsilon_i S^x_{ii} + \frac{dV_{NN}}{dX}
            \end{aligned}

        The derivative integrals are transformed with the base's symmetry-handled
        ``self.C`` (single irrep block, global energy order), so this works with
        molecular symmetry left on. HFwfn always uses the full (all-electron) MO
        space, so a frozen core on the reference does not affect the gradient.

        The spin-orbital path (open-shell UHF/ROHF, or a closed shell forced to spin
        orbitals) is handled by :meth:`_so_gradient_electronic`.  Returns the
        :class:`pycc.PropertyComponents` decomposition (``nuclear + reference + correlation``, the
        last zero for an SCF wavefunction); ``.total`` is the value above.  The electronic block is
        :meth:`_gradient_electronic`; the nuclear block is :meth:`Derivatives.nuclear_repulsion`.
        """
        from . import properties
        return properties.gradient(self)

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
            erix = d.eri(atom, 'o', 'o', 'o', 'o')   # physicist <ij|kl>^X (2J - K)
            for c in range(3):
                grad[atom, c] = (2.0 * np.trace(hx[c])
                                 + 2.0 * self.contract('ijij->', erix[c])
                                 - self.contract('iijj->', erix[c])
                                 - 2.0 * self.contract('i,ii->', eps, Sx[c]))
        return grad

    def _so_gradient_electronic(self) -> np.ndarray:
        r"""Electronic part of the spin-orbital HF gradient (i, j over occupied spin orbitals;
        ``<ij||ij>`` antisymmetrized), valid for UHF/ROHF and a closed shell forced to spin
        orbitals.  :meth:`gradient` adds ``dV_NN/dX``::

            (dE/dX)_elec = sum_i h^x_ii + 1/2 sum_ij <ij||ij>^x - sum_i eps_i S^x_ii

        .. math::

            \begin{aligned}
            \Big(\frac{dE}{dX}\Big)_{\mathrm{elec}} = \sum_i h^x_{ii} + \tfrac{1}{2}\sum_{ij} \langle ij||ij \rangle^x - \sum_i \varepsilon_i S^x_{ii}
            \end{aligned}

        For a closed shell this equals the spatial electronic gradient term-for-term."""
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

    def dipole(self) -> "PropertyComponents":
        r"""Total SCF electric-dipole moment (a.u.), shape ``(3,)``: the nuclear term plus
        the electronic term, in the molecule's frame (gauge-independent for a neutral system)::

            mu_a = sum_A Z_A R_Aa + k sum_i (mu_a)_ii

        .. math::

            \begin{aligned}
            \mu_a = \sum_A Z_A R_{Aa} + k \sum_i (\mu_a)_{ii}
            \end{aligned}

        The electronic part is the occupied trace of the MO dipole integrals (``H.mu`` = ``-e r``;
        same ``Tr(D mu)`` convention as :meth:`~pycc.correlatedderivs.CorrelatedDerivs.dipole`, with the SCF density's occupied
        block). The prefactor is the orbital occupancy: ``k = 2`` on the spatial closed-shell path,
        ``k = 1`` for spin orbitals. Kept separate from the correlation dipole so the total MP2
        dipole is ``HFwfn.dipole() + MPderiv.dipole()`` (mirroring the gradient split).
        Basis-aware. Returns the :class:`pycc.PropertyComponents` decomposition (``.total`` is the
        value above; the correlation block is zero for an SCF wavefunction)."""
        from . import properties
        return properties.dipole(self)

    def _dipole_electronic(self) -> np.ndarray:
        """Electronic part of the SCF dipole (a.u.), shape (3,): ``k sum_i (mu_a)_ii`` (the
        occupied trace of the MO dipole integrals; ``k = 2`` spatial, ``1`` spin-orbital).
        :meth:`dipole` adds the nuclear term ``sum_A Z_A R_A``."""
        o = self.o
        k = 1.0 if self.orbital_basis == 'spinorbital' else 2.0
        return np.array([k * np.trace(np.asarray(self.H.mu[a])[o, o]) for a in range(3)])

    def polarizability(self, omega: float = 0.0, relaxed: bool = None,
                       units: str = 'Eh') -> "PropertyComponents":
        """Static electric-dipole polarizability as a :class:`pycc.PropertyComponents` (``.total``
        is the SCF value; correlation block zero).  Delegates to the :func:`pycc.polarizability`
        assembler; the SCF electronic tensor is :meth:`_polarizability_electronic`.  ``omega`` must
        be ``0`` for an SCF wavefunction (there is no dynamic HF response)."""
        from . import properties
        return properties.polarizability(self, omega=omega, relaxed=relaxed, units=units)

    def _polarizability_electronic(self) -> np.ndarray:
        r"""Static electric-dipole polarizability tensor (a.u.), shape ``(3, 3)`` -- the SCF
        (reference) block of :meth:`polarizability`.

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
        closed-shell path (double occupancy), ``k = 2`` for spin orbitals.
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

    def apt(self, gauge: str = 'length', route: str = '2n+1-field',
            orbital_gauge: str = 'non-canonical') -> "PropertyComponents":
        r"""Atomic polar tensors ``d(mu_alpha)/d(X_A,beta)`` (a.u.), shape ``(natom, 3, 3)`` indexed
        ``[A, beta, alpha]``, as a :class:`pycc.PropertyComponents` (``.total`` is the value below).
        ``gauge='length'`` (default) is the length-gauge APT; ``gauge='velocity'`` the velocity
        gauge.  The SCF electronic blocks are :meth:`_dipole_derivatives_electronic` /
        :meth:`_velocity_dipole_derivatives_electronic`; the nuclear block is ``Z_A delta``.  The
        length-gauge value is::

        Built from the nuclear CPHF response (:meth:`CPHF.solve_nuclear`), so the ov
        term probes ``U^X`` directly (unlike the energy gradient, which is variationally
        insensitive to it). The assembly is nuclear + explicit electronic + overlap
        (Pulay) + CPHF response::

            d mu_a / d X_Ab = Z_A delta_ab                (nuclear)
                            + 2 sum_i (mu_a)^X_ii         (explicit electronic)
                            - 2 sum_ik S^X_ki (mu_a)_ik   (oo / Pulay response)
                            + 4 sum_ia U^X_ia (mu_a)_ia   (ov / CPHF response)

        .. math::

            \begin{aligned}
            \frac{d\mu_\alpha}{dX_{A\beta}}
            &= Z_A\,\delta_{\alpha\beta} && \text{(nuclear)} \\
            &\quad + 2\sum_i (\mu_\alpha)^X_{ii} && \text{(explicit electronic)} \\
            &\quad - 2\sum_{ik} S^X_{ki}\,(\mu_\alpha)_{ik} && \text{(oo / Pulay response)} \\
            &\quad + 4\sum_{ia} U^X_{ia}\,(\mu_\alpha)_{ia} && \text{(ov / CPHF response)}
            \end{aligned}

        The spin-orbital path is handled by :meth:`_so_dipole_derivatives_electronic`.
        """
        from . import properties
        return properties.apt(self, gauge=gauge, route=route, orbital_gauge=orbital_gauge)

    def _dipole_derivatives_electronic(self) -> np.ndarray:
        """Electronic part of the APT (a.u.), shape ``(natom, 3, 3)`` -- the explicit + Pulay +
        CPHF-response terms without the nuclear ``Z_A delta`` (added by
        :meth:`apt`).  Dispatches to :meth:`_so_dipole_derivatives_electronic`."""
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
        r"""Electronic part of the spin-orbital APT (spin orbitals: one-electron traces carry
        factor 1 and the ov response factor 2, vs 2 and 4 closed-shell), with the spin-orbital
        nuclear CPHF response and dipole derivatives.  :meth:`apt` adds the nuclear
        ``Z_A delta_ab``::

            (d mu_a / d X_Ab)_elec = sum_i (mu_a)^X_ii - sum_ik S^X_ki (mu_a)_ik + 2 sum_ia U^X_ia (mu_a)_ia

        .. math::

            \begin{aligned}
            \Big(\frac{d\mu_\alpha}{dX_{A\beta}}\Big)_{\mathrm{elec}} = \sum_i (\mu_\alpha)^X_{ii} - \sum_{ik} S^X_{ki}\,(\mu_\alpha)_{ik} + 2 \sum_{ia} U^X_{ia}\,(\mu_\alpha)_{ia}
            \end{aligned}

        Valid for UHF and a closed shell forced to spin orbitals; ROHF raises (via
        :meth:`CPHF.solve`)."""
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

    def hessian(self) -> "PropertyComponents":
        r"""RHF nuclear (molecular) Hessian ``d^2 E / dX_Aa dX_Bb`` (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3 + a, B*3 + b)`` -- the force-constant
        matrix, matching ``psi4.hessian('scf')`` layout.

        Built from the second-derivative ("skeleton") integrals and the first-order nuclear CPHF
        response (:meth:`CPHF.solve_nuclear`, taken from the shared cache so the ``3*natom`` solves
        are reused for free by :meth:`apt` in an IR workflow).  With x = (A,a),
        y = (B,b); i,j,n,m occupied; spin-adapted ``L`` = H.L::

            d^2E/dx dy = 2 sum_i h^{xy}_ii + 2 sum_ij (ii|jj)^{xy} - sum_ij (ij|ij)^{xy}
                         - 2 sum_i eps_i S^{xy}_ii + V_NN^{xy}
                         - 4 sum_ia U^x_ai B^y_ai - 2 sum_ij S^x_ij F^y_ij - 2 sum_ij S^y_ij F^x_ij
                         + 4 sum_ij eps_i S^x_ij S^y_ij + 2 sum_ijnm S^x_ij S^y_nm L_imjn

        .. math::

            \begin{aligned}
            \frac{d^2 E}{dx\,dy} &= 2\sum_i h^{xy}_{ii} + 2\sum_{ij} (ii|jj)^{xy} - \sum_{ij} (ij|ij)^{xy} - 2\sum_i \varepsilon_i S^{xy}_{ii} + V_{NN}^{xy} \\
            &\quad - 4\sum_{ia} U^x_{ai} B^y_{ai} - 2\sum_{ij} S^x_{ij} F^y_{ij} - 2\sum_{ij} S^y_{ij} F^x_{ij} + 4\sum_{ij} \varepsilon_i S^x_{ij} S^y_{ij} + 2\sum_{ijnm} S^x_{ij} S^y_{nm} L_{imjn}
            \end{aligned}

        where ``U^x``/``B^x`` are the cached nuclear response/RHS and ``F^x_ij``/``S^x_ij`` the
        skeleton derivative Fock/overlap oo blocks (cached by :meth:`CPHF.solve_nuclear`).

        The spin-orbital path is handled by :meth:`_so_hessian_electronic`.  Returns the
        :class:`pycc.PropertyComponents` decomposition (``.total`` is the value above; correlation
        block zero); the electronic block is :meth:`_hessian_electronic`, the nuclear block the
        nuclear-repulsion second derivative.
        """
        from . import properties
        return properties.hessian(self)

    def _hessian_electronic(self) -> np.ndarray:
        """Electronic part of the RHF molecular Hessian, shape ``(3*natom, 3*natom)`` -- the
        skeleton (second-derivative integral) plus CPHF-response terms, without the nuclear-
        repulsion second derivative ``V_NN^{ab}`` (added by :meth:`hessian`).  The two pieces are
        computed separately (:meth:`_hessian_skeleton` + :meth:`_hessian_response`) so the correlated
        Hessian, which shares the ``ao_tei_deriv2`` skeleton in its own assembly, can add the
        ao-independent reference response with a single :meth:`_hessian_response` call.  Dispatches
        (via the two helpers) to the spin-orbital path."""
        return self._hessian_skeleton() + self._hessian_response()

    def _hessian_skeleton(self) -> np.ndarray:
        """Ao-DEPENDENT skeleton (second-derivative integral) part of the RHF electronic Hessian,
        shape ``(3*natom, 3*natom)``: the ``h^{XY}``/``S^{XY}`` occupied traces and the 2-electron
        ``2 D D - D D`` contraction of ``ao_tei_deriv2`` with the occupied AO density.  The
        correlated Hessian duplicates this contraction inline (against its own shared ``ao`` block)
        rather than calling here, to avoid recomputing ``ao_tei_deriv2``.  Dispatches to
        :meth:`_so_hessian_skeleton`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_hessian_skeleton()
        o = self.o
        eps_o = np.asarray(diag(self.H.F))[o]
        d = self.derivatives
        natom = self.ref.molecule().natom()
        c = self.contract
        Co = np.asarray(self.C)[:, o]                   # occupied MO coefficients (AO x no)
        D = Co @ Co.T                                   # occupied AO density  D_mu,nu = sum_i C_mu,i C_nu,i

        H = np.zeros((3 * natom, 3 * natom))
        # Unique atom pairs (A <= B): the electronic Hessian is symmetric under (A,a)<->(B,b), so
        # fill the transpose instead of recomputing -- halving the ao_tei_deriv2 calls.  The
        # 2-electron skeleton is contracted with the occupied AO density D directly (Coulomb
        # 2 D_mn D_ls (mn|ls) minus exchange D_ml D_ns (mn|ls)), so the 2nd-deriv TEIs are never
        # transformed to the MO basis; ao_tei_deriv2 (9*nao^4) alone bounds the peak.  The raw
        # (bra<->ket-doubled) integral is exact here without _complete_deriv2 because D (x) D is a
        # bra<->ket-symmetric density.
        for A in range(natom):
            for Bat in range(A, natom):
                S2 = d.overlap2(A, Bat, 'o', 'o')                 # 9 x (no,no), cheap OEI
                h2 = d.core2(A, Bat, 'o', 'o')                    # 9 x (no,no), cheap OEI
                ao = d.ao_eri2(A, Bat)                            # 9 x (nao^4) raw chemist (mu nu|la si)
                for a in range(3):
                    for b in range(3):
                        cc = a * 3 + b
                        aoc = ao[cc]
                        two_e = 2.0 * c('mn,ls,mnls->', D, D, aoc) - c('ml,ns,mnls->', D, D, aoc)
                        skel = 2.0 * np.trace(h2[cc]) + two_e - 2.0 * c('i,ii->', eps_o, S2[cc])
                        H[A * 3 + a, Bat * 3 + b] = skel
                        if A != Bat:
                            H[Bat * 3 + b, A * 3 + a] = skel        # H symmetric in (A,a)<->(B,b)
                del ao
        return H

    def _hessian_response(self) -> np.ndarray:
        r"""Ao-INDEPENDENT first-order CPHF-response part of the RHF electronic Hessian, shape
        ``(3*natom, 3*natom)``::

            -4 sum_ia U^x_ai B^y_ai - 2 sum_ij S^x_ij F^y_ij - 2 sum_ij S^y_ij F^x_ij
            + 4 sum_ij eps_i S^x_ij S^y_ij + 2 sum_ijnm S^x_ij S^y_nm L_imjn

        Built from the pre-solved nuclear response (:meth:`CPHF.solve_nuclear`, shared with the
        APTs) and the skeleton derivative Fock/overlap oo blocks.  It carries no second-derivative
        TEIs, so the correlated Hessian delegates its reference response here (one call, no
        ao_tei_deriv2 recompute).  Dispatches to :meth:`_so_hessian_response`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_hessian_response()
        o = self.o
        eps_o = np.asarray(diag(self.H.F))[o]
        Loooo = np.asarray(self.H.L)[o, o, o, o]       # i,m,j,n (spin-adapted)
        natom = self.ref.molecule().natom()
        U = [self.cphf.solve_nuclear(A) for A in range(natom)]            # U[A][a]->(no,nv)
        B = [self.cphf.rhs_nuclear(A) for A in range(natom)]             # B[A][a]->(no,nv)
        Foo = [self.cphf.nuclear_skeleton_fock(A) for A in range(natom)]  # F^X_ij->(no,no)
        Soo = [self.cphf.nuclear_skeleton_overlap(A) for A in range(natom)]  # S^X_ij->(no,no)
        c = self.contract
        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(A, natom):
                for a in range(3):
                    for b in range(3):
                        Ux, By = U[A][a], B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        resp = (-4.0 * c('ia,ia->', Ux, By)
                                - 2.0 * c('ij,ij->', Sx, Fy)
                                - 2.0 * c('ij,ij->', Sy, Fx)
                                + 4.0 * c('i,ij,ij->', eps_o, Sx, Sy)
                                + 2.0 * c('ij,nm,imjn->', Sx, Sy, Loooo))
                        H[A * 3 + a, Bat * 3 + b] = resp
                        if A != Bat:
                            H[Bat * 3 + b, A * 3 + a] = resp
        return H

    def _so_hessian_electronic(self) -> np.ndarray:
        r"""Electronic part of the spin-orbital molecular Hessian, shape ``(3*natom, 3*natom)``.
        The spin-orbital form of :meth:`_hessian_electronic`: spin orbitals (the closed-shell
        prefactors halve) with the antisymmetrized ``<ij||kl>`` and the spin-orbital second-
        derivative integrals (:meth:`Derivatives.so_core2` / ``so_eri2`` / ``so_overlap2``, occupied
        block); the nuclear-repulsion second derivative ``V_NN^{xy}`` is added by :meth:`hessian`.
        With x = (A,a), y = (B,b); i,j,n,m occupied::

            d^2E/dx dy = sum_i h^{xy}_ii + 1/2 sum_ij <ij||ij>^{xy} - sum_i eps_i S^{xy}_ii
                         - 2 sum_ia U^x_ai B^y_ai - sum_ij S^x_ij F^y_ij - sum_ij S^y_ij F^x_ij
                         + 2 sum_ij eps_i S^x_ij S^y_ij + sum_ijnm S^x_ij S^y_nm <im||jn>

        .. math::

            \begin{aligned}
            \frac{d^2 E}{dx\,dy} &= \sum_i h^{xy}_{ii} + \tfrac{1}{2}\sum_{ij} \langle ij||ij \rangle^{xy} - \sum_i \varepsilon_i S^{xy}_{ii} \\
            &\quad - 2\sum_{ia} U^x_{ai} B^y_{ai} - \sum_{ij} S^x_{ij} F^y_{ij} - \sum_{ij} S^y_{ij} F^x_{ij} + 2\sum_{ij} \varepsilon_i S^x_{ij} S^y_{ij} + \sum_{ijnm} S^x_{ij} S^y_{nm} \langle im||jn \rangle
            \end{aligned}

        Valid for UHF as well as a closed-shell RHF reference forced to spin orbitals;
        ROHF raises (the nuclear response goes through :meth:`CPHF.solve`).  Skeleton and CPHF
        response are computed apart (:meth:`_so_hessian_skeleton` + :meth:`_so_hessian_response`),
        matching the spatial split so the correlated Hessian can delegate the reference response."""
        return self._so_hessian_skeleton() + self._so_hessian_response()

    def _so_hessian_skeleton(self) -> np.ndarray:
        """Ao-DEPENDENT skeleton of the spin-orbital electronic Hessian (spin-orbital form of
        :meth:`_hessian_skeleton`).  The 2-electron ``1/2 <ij||ij>^(XY)`` is contracted with the
        spin AO densities directly -- Coulomb of the total density minus same-spin exchange -- so
        the spatial ao_tei_deriv2 (9*nao^4) is contracted without the four per-spin-combination
        mo_tei_deriv2 transforms so_eri2 does.  The raw integral needs no _complete_deriv2: each
        effective density (D_t (x) D_t, D_a (x) D_a, D_b (x) D_b) is bra<->ket-symmetric."""
        o = self.o
        eps_o = np.asarray(diag(self.H.F))[o]
        d = self.derivatives
        natom = self.ref.molecule().natom()
        c = self.contract
        Ca = np.asarray(self.H.Ca); Cb = np.asarray(self.H.Cb)   # pycc C1-flattened alpha/beta MOs
        na, nb = self.ref.nalpha(), self.ref.nbeta()             # (self.ref.Da() is symmetry-blocked)
        Da = Ca[:, :na] @ Ca[:, :na].T                  # alpha occupied AO density (= beta for RHF)
        Db = Cb[:, :nb] @ Cb[:, :nb].T                  # beta occupied AO density
        Dt = Da + Db                                    # total AO density

        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(A, natom):
                S2 = d.so_overlap2(A, Bat, 'o', 'o')      # 9 x (no,no), cheap SO OEI
                h2 = d.so_core2(A, Bat, 'o', 'o')         # 9 x (no,no), cheap SO OEI
                ao = d.ao_eri2(A, Bat)                    # 9 x (nao^4) spatial raw chemist
                for a in range(3):
                    for b in range(3):
                        cc = a * 3 + b
                        aoc = ao[cc]
                        two_e = 0.5 * (c('mn,ls,mnls->', Dt, Dt, aoc)          # Coulomb (total)
                                       - c('ms,nl,mnls->', Da, Da, aoc)        # exchange (alpha)
                                       - c('ms,nl,mnls->', Db, Db, aoc))       # exchange (beta)
                        skel = c('ii->', h2[cc]) + two_e - c('i,ii->', eps_o, S2[cc])
                        H[A * 3 + a, Bat * 3 + b] = skel
                        if A != Bat:
                            H[Bat * 3 + b, A * 3 + a] = skel
                del ao
        return H

    def _so_hessian_response(self) -> np.ndarray:
        """Ao-INDEPENDENT CPHF-response part of the spin-orbital electronic Hessian (spin-orbital
        form of :meth:`_hessian_response`): the ``-2 U.B``/``-S.F``/``+eps S S``/``+S S <im||jn>``
        terms, with the spin-orbital (halved) closed-shell prefactors.  Carries no second-derivative
        TEIs, so the correlated Hessian delegates its reference response here."""
        o = self.o
        eps_o = np.asarray(diag(self.H.F))[o]
        ERIoooo = np.asarray(self.H.ERI)[o, o, o, o]   # i,m,j,n (antisymmetrized)
        natom = self.ref.molecule().natom()
        U = [self.cphf.solve_nuclear(A) for A in range(natom)]            # U[A][a]->(no,nv)
        B = [self.cphf.rhs_nuclear(A) for A in range(natom)]             # B[A][a]->(no,nv)
        Foo = [self.cphf.nuclear_skeleton_fock(A) for A in range(natom)]  # F^X_ij->(no,no)
        Soo = [self.cphf.nuclear_skeleton_overlap(A) for A in range(natom)]  # S^X_ij->(no,no)
        c = self.contract
        H = np.zeros((3 * natom, 3 * natom))
        for A in range(natom):
            for Bat in range(A, natom):
                for a in range(3):
                    for b in range(3):
                        Ux, By = U[A][a], B[Bat][b]
                        Sx, Sy = Soo[A][a], Soo[Bat][b]
                        Fx, Fy = Foo[A][a], Foo[Bat][b]
                        resp = (-2.0 * c('ia,ia->', Ux, By)
                                - c('ij,ij->', Sx, Fy)
                                - c('ij,ij->', Sy, Fx)
                                + 2.0 * c('i,ij,ij->', eps_o, Sx, Sy)
                                + c('ij,nm,imjn->', Sx, Sy, ERIoooo))
                        H[A * 3 + a, Bat * 3 + b] = resp
                        if A != Bat:
                            H[Bat * 3 + b, A * 3 + a] = resp
        return H

    def aat(self, origin=None, orbital_gauge: str = 'non-canonical') -> "PropertyComponents":
        """Atomic axial tensors (AATs, for VCD) as a :class:`pycc.PropertyComponents`
        (``nuclear + reference + correlation``; correlation zero for an SCF wavefunction).  Delegates
        to the :func:`pycc.aat` assembler; the SCF electronic block is :meth:`_aat_electronic`.
        ``origin`` sets the nuclear-term gauge origin (see :func:`pycc.aat`)."""
        from . import properties
        return properties.aat(self, origin=origin, orbital_gauge=orbital_gauge)

    def _aat_electronic(self) -> np.ndarray:
        r"""RHF atomic axial tensors (AATs) ``I^lambda_{alpha,beta}`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[lambda, alpha, beta]`` -- the electronic (reference) block of
        :meth:`aat`: the magnetic-dipole vibrational transition moment (common gauge origin), for VCD.

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
        return aat

    def _so_atomic_axial_tensors(self) -> np.ndarray:
        r"""Spin-orbital RHF/UHF atomic axial tensors (AATs), shape ``(natom, 3, 3)``. The
        spin-orbital form of :meth:`aat`: spin orbitals
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
        return aat

    def _velocity_dipole_derivatives_electronic(self) -> np.ndarray:
        r"""Velocity-gauge (VG) atomic polar tensors ``[P^A_{beta,alpha}]^VG`` electronic block
        (a.u.), shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` -- the momentum-form APT
        (reference block of :meth:`apt` with ``gauge='velocity'``), the wave-function-overlap term
        without the nuclear ``Z_A delta`` (added by the :func:`pycc.apt` assembler).

        Formulated (Amos, Jalkanen & Stephens, JPC 92, 5571 (1988), Eq. 14; Shumberger et al.,
        LG(OI) VCD, Eq. 15) as an overlap of wave-function derivatives -- the nuclear derivative
        of the bra with the magnetic-vector-potential derivative of the ket::

            [P^A_{beta,alpha}]^VG = -4 sum_ia (U^R_{ia,beta} + <phi^R_i|phi_a>) U^A_{ia,alpha}

        .. math::

            \begin{aligned}
            [P^A_{\beta\alpha}]^{\mathrm{VG}} = -4 \sum_{ia}
            \big( U^R_{ia,\beta} + \langle \phi^R_i | \phi_a \rangle \big) U^A_{ia,\alpha}
            \end{aligned}

        the same overlap structure as the AAT (:meth:`_aat_electronic`) with the linear-
        momentum response ``U^A`` (:meth:`CPHF.solve_momentum`, ``dPsi/dA``) in place of the
        magnetic response ``U^B``. ``U^R`` is the nuclear CPHF response; ``<phi^R_i|phi_a>`` the
        nuclear half-derivative overlap (the vector potential does not move the basis, so this
        rides on the R side only). The ``-4`` prefactor is the closed-shell value (real-wf
        doubling x double occupancy; the two imaginary units of ``-2i`` x ``H.p`` fix the sign)
        -- pinned to the Amos et al. NH3 ``P(pi)`` values.  The spin-orbital path is handled by
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
        r"""Electronic part of the spin-orbital VG APT: spin orbitals halve the closed-shell
        prefactor (``-4 -> -2``), with the spin-orbital nuclear/momentum responses
        (:meth:`CPHF.solve_nuclear`/:meth:`CPHF.solve_momentum`) and half-derivative overlaps
        (:meth:`Derivatives.so_overlap_half`); :meth:`apt` adds the nuclear
        ``Z_A delta_{alpha,beta}``.  With x = (A,beta)::

            [P^A_{beta,alpha}]^VG_elec = -2 sum_ia (U^R_{ia,beta} + <phi^R_i|phi_a>) U^A_{ia,alpha}

        .. math::

            \begin{aligned}
            [P^A_{\beta\alpha}]^{\mathrm{VG}}_{\mathrm{elec}} = -2 \sum_{ia} \big( U^R_{ia,\beta} + \langle \phi^R_i | \phi_a \rangle \big) U^A_{ia,\alpha}
            \end{aligned}

        Valid for UHF and a closed shell forced to spin orbitals; ROHF raises
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
