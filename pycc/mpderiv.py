"""MP2 analytic-derivative property driver.

`MPderiv` is the MP2 leaf of the :class:`~pycc.correlatedderivs.CorrelatedDerivs` hierarchy (see
docs/DERIVATIVES_PLAN_2026-06.md section 9): it supplies MP2's two density hooks -- the unrelaxed
reduced densities (:meth:`_unrelaxed_densities`) and their first-order response
(:meth:`_perturbed_unrelaxed_densities`) -- and adds the MP2-specific atomic axial tensors (AAT) and
velocity-gauge APT, which are overlap formulations the base does not provide.  Everything else is
inherited: the base owns the method-agnostic orbital-response (Z-vector) solve and the 2n+1 assembly,
so the relaxed dipole, gradient, polarizability, length-gauge APT, and Hessian come from it unchanged,
driven by the two hooks above.

State is read from the MP2 wavefunction ``self.mp`` (an :class:`~pycc.mpwfn.MPwfn`).  The unrelaxed
correlation-density seeds (``_*_opdm`` / ``_*_tpdm``) live here on the derivative driver,
mirroring the CC layout where the reduced densities live in :mod:`pycc.ccdensity` (reached from
:class:`~pycc.ccderiv.CCderiv`) rather than on :class:`~pycc.ccwfn.ccwfn`.  The intermediate
normalization (:meth:`MPwfn._mp2_normalization` / ``_so_``) is a wavefunction-level quantity and
stays on ``MPwfn``; the AAT/VG-APT overlaps read it via ``self.mp``.
"""

from __future__ import annotations

import numpy as np

from .correlatedderivs import CorrelatedDerivs


class MPderiv(CorrelatedDerivs):
    """MP2 correlation derivative-property driver.

    Constructed from a converged :class:`~pycc.mpwfn.MPwfn`.  Supplies the MP2 density hooks
    (:meth:`_unrelaxed_densities`, :meth:`_perturbed_unrelaxed_densities`) and the MP2-specific atomic
    axial tensors (:meth:`atomic_axial_tensors`) and velocity-gauge APT
    (:meth:`velocity_dipole_derivatives`); the orbital-response (Z-vector) solve and the 2n+1 assembly
    of the relaxed dipole, gradient, polarizability, length-gauge APT, and Hessian are inherited from
    :class:`~pycc.correlatedderivs.CorrelatedDerivs`.  Both the spin-adapted (closed-shell RHF) and
    spin-orbital (``_so_``) paths are frozen-core aware.
    """

    def __init__(self, wfn, e_conv=None, r_conv=None, maxiter=None,
                 max_diis=8, start_diis=1) -> None:
        """Bind the converged MPwfn (aliased as ``mp``; the base stores it as ``wfn``); the
        reduced-density seeds and their derivatives are built on demand from ``self.mp.t2``.

        The convergence / DIIS keyword arguments are accepted for signature parity with
        :class:`~pycc.ccderiv.CCderiv` but have no effect -- MP2 amplitudes are closed-form
        (non-iterative)."""
        super().__init__(wfn)
        self.mp = wfn                       # alias: the MP2 wavefunction whose densities we differentiate

    # ---- unrelaxed correlation-density seeds ----
    # The unrelaxed one- and two-particle correlation densities (pure functions of the MP2
    # amplitudes ``self.mp.t2``) seed the relaxed densities and orbital response.  They live here on
    # the derivative driver (the MP2 analogue of pycc.ccdensity, reached from CCderiv), since nothing
    # outside the derivative code consumes them.  The spin-orbital (_so_) and spin-adapted
    # (closed-shell, unlabeled) paths are paired; the spatial spin sum rides in
    # l2 = 2(2 t2 - t2.swap).

    def _mp2_opdm(self):
        r"""Spin-adapted unrelaxed MP2 one-particle correlation density blocks ``(Doo, Dvv)``, with
        the spin-adapted lambda ``l2 = 2(2 t2 - t2.swap)`` (the factor-2 carries the closed-shell
        spin sum)::

            Doo_ij  = -t_imef l2_jmef
            Dvv_ab  =  t_mnbe l2_mnae
            l2_ijab = 2(2 t2_ijab - t2_ijba)

        .. math::

            \begin{aligned}
            D^{oo}_{ij} &= -t_{imef} \lambda_{jmef} \\
            D^{vv}_{ab} &=  t_{mnbe} \lambda_{mnae} \\
            \lambda_{ijab} &= 2(2 t^{ab}_{ij} - t^{ba}_{ij})
            \end{aligned}
        """
        c = self.contract
        t2 = self.mp.t2
        l2 = 2.0 * (2.0 * t2 - t2.swapaxes(2, 3))
        Doo = -c('imef,jmef->ij', t2, l2)
        Dvv = c('mnbe,mnae->ab', t2, l2)
        return Doo, Dvv

    def _so_mp2_opdm(self):
        r"""Spin-orbital unrelaxed MP2 one-particle correlation density blocks ``(Doo, Dvv)``.
        The ``1/2`` is the normalization that makes the densities close the energy
        (``Tr(F Doo) + Tr(F Dvv) = -E_MP2``)::

            Doo_ij = -1/2 t_imef t_jmef
            Dvv_ab =  1/2 t_mnbe t_mnae

        .. math::

            \begin{aligned}
            D^{oo}_{ij} &= -\tfrac{1}{2} t_{imef} t_{jmef} \\
            D^{vv}_{ab} &=  \tfrac{1}{2} t_{mnbe} t_{mnae}
            \end{aligned}
        """
        c = self.contract
        t2 = self.mp.t2
        Doo = -0.5 * c('imef,jmef->ij', t2, t2)
        Dvv = 0.5 * c('mnbe,mnae->ab', t2, t2)
        return Doo, Dvv

    def _mp2_tpdm(self) -> np.ndarray:
        r"""Spin-adapted MP2 cumulant 2-PDM in the ``oovv``/``vvoo`` blocks. Built over the full MO
        space (``self.mp.nmo``); the active ``o``/``v`` slices place the amplitudes, leaving any
        frozen-core rows/columns zero::

            Gamma_ijab = 2 t2_ijab - t2_ijba

        .. math::

            \begin{aligned}
            \Gamma_{ijab} = 2 t^{ab}_{ij} - t^{ba}_{ij}
            \end{aligned}
        """
        o, v = self.mp.o, self.mp.v
        nmo = self.mp.nmo
        t2 = np.asarray(self.mp.t2)
        u = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
        Gam = np.zeros((nmo, nmo, nmo, nmo))
        Gam[o, o, v, v] = u
        Gam[v, v, o, o] = u.transpose(2, 3, 0, 1)
        return Gam

    def _so_mp2_tpdm(self) -> np.ndarray:
        r"""Spin-orbital MP2 cumulant 2-PDM in the ``oovv``/``vvoo`` blocks -- the only blocks that
        contribute (determined from the MP2 energy Lagrangian, in which Lambda and T2 enter
        linearly). Built over the full MO space (``self.mp.nmo``); the active ``o``/``v`` slices place
        the amplitudes::

            Gamma_ijab = 1/4 t2_ijab

        .. math::

            \begin{aligned}
            \Gamma_{ijab} = \tfrac{1}{4} t^{ab}_{ij}
            \end{aligned}
        """
        o, v = self.mp.o, self.mp.v
        nmo = self.mp.nmo
        t2 = np.asarray(self.mp.t2)
        Gam = np.zeros((nmo, nmo, nmo, nmo))
        Gam[o, o, v, v] = 0.25 * t2
        Gam[v, v, o, o] = 0.25 * t2.transpose(2, 3, 0, 1)
        return Gam

    def _unrelaxed_densities(self):
        """MP2 unrelaxed reduced densities as full-MO arrays: the 1-PDM ``D`` (the ``Doo``/``Dvv``
        correlation blocks on the occupied/virtual diagonal) and the cumulant 2-PDM ``Gamma``, from
        the amplitude seeds (:meth:`_mp2_opdm` / :meth:`_mp2_tpdm` and the ``_so_`` siblings).
        Supplies the base Z-vector (:meth:`CorrelatedDerivs._orbital_response` /
        :meth:`_so_orbital_response`)."""
        nmo, o, v = self.mp.nmo, self.mp.o, self.mp.v
        if self.mp.orbital_basis == 'spinorbital':
            Doo, Dvv = self._so_mp2_opdm()
            Gam = self._so_mp2_tpdm()
        else:
            Doo, Dvv = self._mp2_opdm()
            Gam = self._mp2_tpdm()
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        return D, np.asarray(Gam)

    # ---- perturbed amplitudes / densities (for the second derivatives) ----

    def _perturbed_t2(self, pert) -> np.ndarray:
        r"""First-order response of the MP2 doubles amplitudes to ``pert`` -- closed form,
        since the MP2 amplitudes are non-iterative (repeated indices summed)::

            t^x_ijab = [ d_x<ij||ab> + d_x f_ac t_ijcb + d_x f_bc t_ijac
                         - d_x f_ik t_kjab - d_x f_jk t_ikab ] / D_ijab

        .. math::

            \partial_x t^{ab}_{ij} = \frac{1}{D^{ab}_{ij}}\big[ \partial_x\langle ij\Vert ab\rangle
                + \partial_x f_{ac}\, t^{cb}_{ij} + \partial_x f_{bc}\, t^{ac}_{ij}
                - \partial_x f_{ik}\, t^{ab}_{kj} - \partial_x f_{jk}\, t^{ab}_{ik} \big]

        from the active ``oovv`` block of :meth:`CPHF.perturbed_eri` and the active ``oo``/``vv``
        blocks of :meth:`CPHF.perturbed_fock` (the diagonal of ``d_x f`` recovers ``-t d_x D``;
        the off-diagonal ``oo``/``vv`` blocks are the non-canonical coupling). Basis-agnostic --
        the integral convention rides in ``H.ERI``. Built over the full occupied space
        (frozen-core aware) but indexed on the active amplitudes."""
        o, v = self.mp.o, self.mp.v
        ncore = o.stop - self.mp.no
        cphf = self._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dfoo, dfvv = df[o, o], df[v, v]
        t2 = np.asarray(self.mp.t2)
        c = self.contract
        num = (deri[o, o, v, v]
               + c('ac,ijcb->ijab', dfvv, t2) + c('bc,ijac->ijab', dfvv, t2)
               - c('ik,kjab->ijab', dfoo, t2) - c('jk,ikab->ijab', dfoo, t2))
        return num / np.asarray(self.mp.Dijab)

    def _perturbed_unrelaxed_densities(self, pert, df=None, deri=None, dL=None):
        r"""First-order response of the unrelaxed correlation densities to ``pert``: returns
        ``(d_x gamma, d_x Gamma)`` (full-MO arrays), from the perturbed amplitudes ``ta = d_x t2``
        (:meth:`_perturbed_t2`) by the product rule -- the unrelaxed-density expressions
        (:meth:`_mp2_opdm`/:meth:`_mp2_tpdm` and the ``_so_`` siblings) differentiated
        (repeated indices summed; spin-adapted shown, ``la = 2(2 ta - ta.swap)``)::

            d_x gamma_oo_ij = -(ta_imef l2_jmef + t_imef la_jmef)
            d_x gamma_vv_ab =  (ta_mnbe l2_mnae + t_mnbe la_mnae)
            d_x Gamma_ijab  =  2 ta_ijab - ta_ijba

        .. math::

            \begin{aligned}
            \partial_x \gamma^{oo}_{ij} &= -(\partial_x t_{imef}\,\lambda_{jmef} + t_{imef}\,\partial_x\lambda_{jmef}) \\
            \partial_x \gamma^{vv}_{ab} &=  (\partial_x t_{mnbe}\,\lambda_{mnae} + t_{mnbe}\,\partial_x\lambda_{mnae}) \\
            \partial_x \Gamma_{ijab} &= 2\,\partial_x t^{ab}_{ij} - \partial_x t^{ba}_{ij}
            \end{aligned}

        This is MP2's implementation of the base
        :meth:`CorrelatedDerivs._perturbed_unrelaxed_densities` hook.  The MP2 response is closed
        form, so it rebuilds its own perturbed integrals from ``pert`` and ignores the CPHF-folded
        ``df``/``deri``/``dL`` the base passes in (those exist for the CC iterative path); they
        default to ``None`` so the closed form can also be called directly with just ``pert``."""
        o, v, nmo = self.mp.o, self.mp.v, self.mp.nmo
        t2 = np.asarray(self.mp.t2)
        ta = self._perturbed_t2(pert)
        c = self.contract
        dgam = np.zeros((nmo, nmo))
        dGam = np.zeros((nmo, nmo, nmo, nmo))
        if self.mp.orbital_basis == 'spinorbital':
            dgam[o, o] = -0.5 * (c('imef,jmef->ij', ta, t2) + c('imef,jmef->ij', t2, ta))
            dgam[v, v] = 0.5 * (c('mnbe,mnae->ab', ta, t2) + c('mnbe,mnae->ab', t2, ta))
            u = 0.25 * ta
        else:
            l2 = 2.0 * (2.0 * t2 - t2.swapaxes(2, 3))
            la = 2.0 * (2.0 * ta - ta.swapaxes(2, 3))
            dgam[o, o] = -(c('imef,jmef->ij', ta, l2) + c('imef,jmef->ij', t2, la))
            dgam[v, v] = (c('mnbe,mnae->ab', ta, l2) + c('mnbe,mnae->ab', t2, la))
            u = 2.0 * ta - ta.swapaxes(2, 3)
        dGam[o, o, v, v] = u
        dGam[v, v, o, o] = u.transpose(2, 3, 0, 1)
        return dgam, dGam

    # ---- 2n+1 route: perturbed relaxed density (assembly hoisted to CorrelatedDerivs) ----
    # The relaxed-density gradient is already the 2n+1 first derivative; its second derivative
    # (polarizability/APT/Hessian) needs the *response* of the relaxed density, whose new piece
    # is the perturbed Z-vector z^x (same orbital Hessian as the gradient, perturbed RHS).  The
    # assembly (and the polarizability/APT/Hessian orchestration) lives in CorrelatedDerivs,
    # driven by the _perturbed_unrelaxed_densities hook above.  See mp2_2n1_perturbed.tex /
    # DERIVATIVES_PLAN.

    # ---- second derivatives: polarizability, APT (dipole derivatives), Hessian ----
    # Inherited unchanged from CorrelatedDerivs (polarizability / dipole_derivatives / hessian);
    # MP2 adds no method-specific handling, so there is no override here.

    # ---- atomic axial tensors (VCD, magnetic/nuclear mixed derivative) ----

    def atomic_axial_tensors(self, gauge: str = 'non-canonical') -> np.ndarray:
        r"""MP2 **correlation** atomic axial tensors ``I^A_{alpha,beta}`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[A, alpha, beta]`` -- the nuclear(``alpha``)/magnetic-field
        (``beta``) mixed derivative of the wave function, as an overlap of its perturbed
        derivatives.  This is the **correlation** contribution only; the SCF reference AAT
        (:meth:`HFwfn.atomic_axial_tensors`) and the nuclear (charge x position) term are kept
        separate and summed by the :func:`pycc.aat` facade.  The correlation is computed directly
        from the correlation 1-PDM/amplitude derivatives (the reference ``2 delta_ij`` density
        block never enters), so the pieces are separated in fact, not by subtraction.  Dropping
        the reference density leaves the result orbital-gauge invariant on its own: the reference
        block it removes is itself gauge invariant (the antisymmetric magnetic oo/vv response
        contracts to zero against the symmetric nuclear response).  The electron-density formulation follows the diagonal Born-Oppenheimer
        correction of Gauss, Tajti, Kallay, Stanton & Szalay, J. Chem. Phys. 125, 144111 (2006)
        [Eqs. (16), (18), (19)], generalized to the mixed nuclear/magnetic derivative (Krishnan,
        Shumberger & Crawford, in prep.)::

            I = dc^R_I dc^H_I                    (coefficient overlap,            Icc)
              + g^R_pq <phi_p|d_H phi_q>         (left derivative density x U^H,  Icphi)
              + g^H_pq <phi_p|d_R phi_q>         (right derivative density x U^R, Iphic)
              + gamma_pq <d_R phi_p|d_H phi_q>   (density x both MO derivatives,  Ipp)

        (R = nuclear (alpha), H = magnetic (beta); repeated indices summed.)

        .. math::

            \begin{aligned}
            I^{A}_{\alpha\beta} = &\;\partial_R c_I\,\partial_H c_I
                + g^{R}_{pq}\langle\phi_p|\partial_H\phi_q\rangle \\
            &+ g^{H}_{pq}\langle\phi_p|\partial_R\phi_q\rangle
                + \gamma_{pq}\langle\partial_R\phi_p|\partial_H\phi_q\rangle
            \end{aligned}

        ``g^R``/``g^H`` are derivatives of the unrelaxed correlation 1-PDM: ``g^R`` the
        **symmetric** derivative (real nuclear perturbation) and ``g^H`` the **antisymmetric** one
        (imaginary magnetic perturbation -- the anti-Hermitian derivative of a Hermitian density).
        With that symmetry the folded density expression is **orbital-gauge invariant**.  ``gamma``
        is the unrelaxed correlation 1-PDM; the orbital relaxation lives entirely in the
        MO-derivative overlaps, so the cumulant 2-PDM does not contribute (it cancels by the
        magnetic antisymmetry).

        ``gauge`` selects the redundant magnetic oo/vv orbital response:

        * ``'non-canonical'`` (default): the redundant blocks are zero (numerically preferred --
          it avoids near-degenerate canonical divides such as close-lying core orbitals); only the
          non-redundant core<->active block is canonical (frozen core).
        * ``'canonical'``: all oo/vv blocks canonical.

        The total is invariant to this choice (:meth:`CPHF.magnetic_ints`).  The nuclear MO
        responses use pycc's ``-1/2 S`` gauge (:meth:`_perturbed_t2` / :meth:`CPHF.full_U`).
        Magnetic quantities are stripped of their ``i`` (as in the HF AAT); the VCD rotatory
        strength takes ``Im`` of the APT*AAT product.

        Frozen-core aware (the correlation densities/amplitudes stay in the active space while the
        orbital responses and reference density span the full occupied space; no Z-vector is
        needed because the densities are unrelaxed).  Both spin paths (the spin-orbital form is
        :meth:`_so_atomic_axial_tensors`).  Validated all-electron and frozen-core, both spins,
        both gauges, against the independent apyib MP2-VCD implementation."""
        if self.mp.orbital_basis == 'spinorbital':
            return self._so_atomic_axial_tensors(gauge)
        from .cphf import Perturbation
        o, v, nmo = self.mp.o, self.mp.v, self.mp.nmo
        no = o.stop
        ncore = o.stop - self.mp.no
        c = self.contract
        cphf = self._full_occ_cphf()
        d = self.mp.derivatives
        natom = d.natom
        t2 = np.asarray(self.mp.t2)
        Dijab = np.asarray(self.mp.Dijab)
        tau = 2.0 * t2 - t2.swapaxes(2, 3)
        N = self.mp._mp2_normalization()
        c0, c2 = N, N * t2
        # correlation part of the unrelaxed, normalized 1-PDM (the 2 delta_ij reference block is
        # excluded: it contributes the SCF reference AAT via Ipp, kept separate -- see the method
        # docstring).  This makes the return the correlation contribution only.
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = -2.0 * N**2 * c('ikab,jkab->ij', tau, t2)
        gamma[v, v] = +2.0 * N**2 * c('ijac,ijbc->ab', tau, t2)

        def dt2_from(dF, dERI):
            # magnetic (imaginary) perturbed T2: dERI enters via the vvoo block (antisymmetric)
            return ((dERI.swapaxes(0, 2).swapaxes(1, 3)[o, o, v, v]
                     + c('ac,ijcb->ijab', dF[v, v], t2) + c('bc,ijac->ijab', dF[v, v], t2)
                     - c('ki,kjab->ijab', dF[o, o], t2) - c('kj,ikab->ijab', dF[o, o], t2)) / Dijab)

        # magnetic side (3): U^H, magnetic derivative density gamma^H.  gamma^H tilde's the
        # *perturbed* amplitude (imaginary perturbation -> antisymmetric density derivative),
        # mirroring gamma^R; this is what makes the folded form orbital-gauge invariant.
        UH, gH = [], []
        for b in range(3):
            U, dF, dERI = cphf.magnetic_ints(b, ncore, gauge)
            dc2H = c0 * dt2_from(dF, dERI)
            tauH = 2.0 * dc2H - dc2H.swapaxes(2, 3)
            UH.append(U)
            R = np.zeros((nmo, nmo))
            R[o, o] = -2.0 * c('ikab,jkab->ij', tauH, c2)
            R[v, v] = +2.0 * c('ijac,ijbc->ab', tauH, c2)
            gH.append((R, dc2H))

        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            hs = d.overlap_half(A)                             # 3 x (nmo, nmo), full
            for cart in range(3):
                pX = Perturbation('nuclear', (A, cart))
                dt2R = np.asarray(self._perturbed_t2(pX))      # -1/2 S gauge
                dc0R = -c0**3 * c('ijab,ijab->', tau, dt2R)
                dc2R = dc0R * t2 + c0 * dt2R
                tauR = 2.0 * dc2R - dc2R.swapaxes(2, 3)
                UReff = np.asarray(cphf.full_U(pX, ncore)) + np.asarray(hs[cart]).T
                gR = np.zeros((nmo, nmo))
                gR[o, o] = -2.0 * c('ikab,jkab->ij', tauR, c2)
                gR[v, v] = +2.0 * c('ijac,ijbc->ab', tauR, c2)
                gR[np.arange(no), np.arange(no)] += 2.0 * c0 * dc0R
                for b in range(3):
                    RH, dc2H = gH[b]
                    Icc = c('ijab,ijab->', tauR, dc2H)
                    Icphi = c('ij,ij->', gR[o, o], UH[b][o, o]) + c('ab,ab->', gR[v, v], UH[b][v, v])
                    Iphic = c('ij,ji->', RH[o, o], UReff[o, o]) + c('ab,ab->', RH[v, v], UReff[v, v])
                    Ipp = c('pq,pq->', gamma, UH[b].T @ UReff)
                    P[A, cart, b] = Icc + Icphi + Iphic + Ipp
        self.aat = P
        return P

    def _so_atomic_axial_tensors(self, gauge: str = 'non-canonical') -> np.ndarray:
        r"""Spin-orbital MP2 electronic AATs (``(natom, 3, 3)``) -- the spin-orbital form of
        :meth:`atomic_axial_tensors` (see there for the theory), in the bare (already-
        antisymmetrized) spin-orbital amplitudes.  Same four-term overlap (R = nuclear, H =
        magnetic; repeated indices summed)::

            I = dc^R_I dc^H_I + g^R_pq <phi_p|d_H phi_q>
              + g^H_pq <phi_p|d_R phi_q> + gamma_pq <d_R phi_p|d_H phi_q>

        .. math::

            I^{A}_{\alpha\beta} = \partial_R c_I\,\partial_H c_I
                + g^{R}_{pq}\langle\phi_p|\partial_H\phi_q\rangle
                + g^{H}_{pq}\langle\phi_p|\partial_R\phi_q\rangle
                + \gamma_{pq}\langle\partial_R\phi_p|\partial_H\phi_q\rangle

        The derivative densities carry the same symmetry as in the spin-adapted path: ``gamma^R``
        is the **symmetric** part of
        ``-1/2 <d c2, c2>`` (real perturbation) and ``gamma^H`` the **antisymmetric** part of
        ``+1/2 <d c2, c2>`` (imaginary perturbation).  With those, the folded form is
        orbital-gauge invariant.  Verified equal to the spin-adapted path to machine precision."""
        from .cphf import Perturbation
        o, v, nmo = self.mp.o, self.mp.v, self.mp.nmo
        no = o.stop
        ncore = o.stop - self.mp.no
        c = self.contract
        cphf = self._full_occ_cphf()
        d = self.mp.derivatives
        natom = d.natom
        t2 = np.asarray(self.mp.t2)
        Dijab = np.asarray(self.mp.Dijab)
        N = self.mp._so_mp2_normalization()
        c0, c2 = N, N * t2
        Doo, Dvv = self._so_mp2_opdm()
        gamma = np.zeros((nmo, nmo))                    # correlation part of the 1-PDM (no delta_ij
        gamma[o, o] = N**2 * np.asarray(Doo)            # reference: it rides in the SCF AAT, kept
        gamma[v, v] = N**2 * np.asarray(Dvv)            # separate -- return is correlation only)

        def dt2_from(dF, dERI):
            # magnetic (imaginary) perturbed T2: dERI enters via the vvoo block (antisymmetric)
            return ((dERI.swapaxes(0, 2).swapaxes(1, 3)[o, o, v, v]
                     + c('ac,ijcb->ijab', dF[v, v], t2) + c('bc,ijac->ijab', dF[v, v], t2)
                     - c('ki,kjab->ijab', dF[o, o], t2) - c('kj,ikab->ijab', dF[o, o], t2)) / Dijab)

        def gdens(dc2, imaginary):
            """Derivative density from ``A = <d c2, c2>``: the symmetric part (real perturbation,
            gamma^R) or the antisymmetric part (imaginary perturbation, gamma^H).  The oo and vv
            blocks carry opposite signs (the Doo/Dvv sign), and gamma^R vs gamma^H flip together."""
            Aoo = c('imef,jmef->ij', dc2, c2)
            Avv = c('mnbe,mnae->ab', dc2, c2)
            R = np.zeros((nmo, nmo))
            if imaginary:                                  # gamma^H: antisymmetric
                R[o, o] = +0.25 * (Aoo - Aoo.T)
                R[v, v] = -0.25 * (Avv - Avv.T)
            else:                                          # gamma^R: symmetric
                R[o, o] = -0.25 * (Aoo + Aoo.T)
                R[v, v] = +0.25 * (Avv + Avv.T)
            return R

        UH, gH, dc2Hs = [], [], []
        for b in range(3):
            U, dF, dERI = cphf.magnetic_ints(b, ncore, gauge)
            dc2H = c0 * dt2_from(dF, dERI)
            UH.append(U)
            dc2Hs.append(dc2H)
            gH.append(gdens(dc2H, imaginary=True))        # antisymmetric

        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            hs = d.so_overlap_half(A)
            for cart in range(3):
                pX = Perturbation('nuclear', (A, cart))
                dt2R = np.asarray(self._perturbed_t2(pX))
                dc0R = -0.25 * c0**3 * c('ijab,ijab->', t2, dt2R)
                dc2R = dc0R * t2 + c0 * dt2R
                UReff = np.asarray(cphf.full_U(pX, ncore)) + np.asarray(hs[cart]).T
                gR = gdens(dc2R, imaginary=False)          # symmetric
                for b in range(3):
                    Icc = 0.25 * c('ijab,ijab->', dc2R, dc2Hs[b])
                    Icphi = c('ij,ij->', gR[o, o], UH[b][o, o]) + c('ab,ab->', gR[v, v], UH[b][v, v])
                    Iphic = c('ij,ij->', gH[b][o, o], UReff[o, o]) + c('ab,ab->', gH[b][v, v], UReff[v, v])
                    Ipp = c('pq,pq->', gamma, UH[b].T @ UReff)
                    P[A, cart, b] = Icc + Icphi + Iphic + Ipp
        self.aat = P
        return P

    def velocity_dipole_derivatives(self, gauge: str = 'non-canonical') -> np.ndarray:
        r"""MP2 velocity-gauge (VG) atomic polar tensors ``[P^A_{beta,alpha}]^VG`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` = ``d(mu_alpha)/d(X_A,beta)`` -- the
        momentum-form APT.  This is the **correlation** contribution only; the SCF reference VG
        APT (:meth:`HFwfn.velocity_dipole_derivatives`) and the nuclear ``Z_A delta_{alpha,beta}``
        term are kept separate and summed by the :func:`pycc.apt` (``gauge='velocity'``) facade.
        Built on the atomic-axial-tensor machinery (:meth:`atomic_axial_tensors`) with the
        magnetic-dipole operator replaced by the linear momentum ``p = -i nabla``
        (:meth:`CPHF.momentum_ints`)::

            correlation = 2 <d_R Psi | d_A Psi>_correlation

        .. math::

            [\text{correlation}] = 2\,\langle \partial_R \Psi \,|\, \partial_A \Psi \rangle_\mathrm{corr}

        As for the AAT, the correlation is computed directly (the reference density block never
        enters) and is orbital-gauge invariant on its own (``gauge`` selects the redundant momentum
        oo/vv response, default ``'non-canonical'``); frozen-core aware; both spin paths
        (spin-orbital: :meth:`_so_velocity_dipole_derivatives`).  The ``+2`` prefactor is the
        closed-shell value.

        Unlike the length-gauge APT (:meth:`dipole_derivatives`) the VG APT differs from it in a
        finite basis, converging to it toward the basis-set limit; both are origin-independent."""
        if self.mp.orbital_basis == 'spinorbital':
            return self._so_velocity_dipole_derivatives(gauge)
        from .cphf import Perturbation
        o, v, nmo = self.mp.o, self.mp.v, self.mp.nmo
        no = o.stop
        ncore = o.stop - self.mp.no
        c = self.contract
        cphf = self._full_occ_cphf()
        d = self.mp.derivatives
        natom = d.natom
        t2 = np.asarray(self.mp.t2)
        Dijab = np.asarray(self.mp.Dijab)
        tau = 2.0 * t2 - t2.swapaxes(2, 3)
        N = self.mp._mp2_normalization()
        c0, c2 = N, N * t2
        gamma = np.zeros((nmo, nmo))                    # correlation 1-PDM only (no 2 delta ref)
        gamma[o, o] = -2.0 * N**2 * c('ikab,jkab->ij', tau, t2)
        gamma[v, v] = +2.0 * N**2 * c('ijac,ijbc->ab', tau, t2)

        def dt2_from(dF, dERI):
            # momentum (imaginary) perturbed T2: dERI enters via the vvoo block (antisymmetric)
            return ((dERI.swapaxes(0, 2).swapaxes(1, 3)[o, o, v, v]
                     + c('ac,ijcb->ijab', dF[v, v], t2) + c('bc,ijac->ijab', dF[v, v], t2)
                     - c('ki,kjab->ijab', dF[o, o], t2) - c('kj,ikab->ijab', dF[o, o], t2)) / Dijab)

        UA, gA = [], []
        for a in range(3):
            U, dF, dERI = cphf.momentum_ints(a, ncore, gauge)
            dc2A = c0 * dt2_from(dF, dERI)
            tauA = 2.0 * dc2A - dc2A.swapaxes(2, 3)
            UA.append(U)
            R = np.zeros((nmo, nmo))
            R[o, o] = -2.0 * c('ikab,jkab->ij', tauA, c2)
            R[v, v] = +2.0 * c('ijac,ijbc->ab', tauA, c2)
            gA.append((R, dc2A))

        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            hs = d.overlap_half(A)
            for beta in range(3):
                pX = Perturbation('nuclear', (A, beta))
                dt2R = np.asarray(self._perturbed_t2(pX))
                dc0R = -c0**3 * c('ijab,ijab->', tau, dt2R)
                dc2R = dc0R * t2 + c0 * dt2R
                tauR = 2.0 * dc2R - dc2R.swapaxes(2, 3)
                UReff = np.asarray(cphf.full_U(pX, ncore)) + np.asarray(hs[beta]).T
                gR = np.zeros((nmo, nmo))
                gR[o, o] = -2.0 * c('ikab,jkab->ij', tauR, c2)
                gR[v, v] = +2.0 * c('ijac,ijbc->ab', tauR, c2)
                gR[np.arange(no), np.arange(no)] += 2.0 * c0 * dc0R
                for alpha in range(3):
                    RA, dc2A = gA[alpha]
                    Icc = c('ijab,ijab->', tauR, dc2A)
                    Icphi = c('ij,ij->', gR[o, o], UA[alpha][o, o]) + c('ab,ab->', gR[v, v], UA[alpha][v, v])
                    Iphic = c('ij,ji->', RA[o, o], UReff[o, o]) + c('ab,ab->', RA[v, v], UReff[v, v])
                    Ipp = c('pq,pq->', gamma, UA[alpha].T @ UReff)
                    P[A, beta, alpha] = 2.0 * (Icc + Icphi + Iphic + Ipp)
        self.vgapt = P
        return P

    def _so_velocity_dipole_derivatives(self, gauge: str = 'non-canonical') -> np.ndarray:
        r"""Spin-orbital MP2 velocity-gauge APTs (``(natom, 3, 3)``) -- the correlation-only
        spin-orbital form of :meth:`velocity_dipole_derivatives` (see there for the theory),
        sharing the spin-orbital AAT densities (:meth:`_so_atomic_axial_tensors`) with the
        linear-momentum response (:meth:`CPHF.momentum_ints`)::

            correlation = 2 <d_R Psi | d_A Psi>_correlation

        .. math::

            [\text{correlation}] = 2\,\langle \partial_R \Psi \,|\, \partial_A \Psi \rangle_\mathrm{corr}

        The ``+2`` prefactor is the same as
        the spin-adapted path (the spin-orbital overlap equals the spin-adapted overlap by
        construction); verified equal to the spin-adapted path to machine precision."""
        from .cphf import Perturbation
        o, v, nmo = self.mp.o, self.mp.v, self.mp.nmo
        no = o.stop
        ncore = o.stop - self.mp.no
        c = self.contract
        cphf = self._full_occ_cphf()
        d = self.mp.derivatives
        natom = d.natom
        t2 = np.asarray(self.mp.t2)
        Dijab = np.asarray(self.mp.Dijab)
        N = self.mp._so_mp2_normalization()
        c0, c2 = N, N * t2
        Doo, Dvv = self._so_mp2_opdm()
        gamma = np.zeros((nmo, nmo))                    # correlation 1-PDM only (no delta ref)
        gamma[o, o] = N**2 * np.asarray(Doo)
        gamma[v, v] = N**2 * np.asarray(Dvv)

        def dt2_from(dF, dERI):
            return ((dERI.swapaxes(0, 2).swapaxes(1, 3)[o, o, v, v]
                     + c('ac,ijcb->ijab', dF[v, v], t2) + c('bc,ijac->ijab', dF[v, v], t2)
                     - c('ki,kjab->ijab', dF[o, o], t2) - c('kj,ikab->ijab', dF[o, o], t2)) / Dijab)

        def gdens(dc2, imaginary):
            Aoo = c('imef,jmef->ij', dc2, c2)
            Avv = c('mnbe,mnae->ab', dc2, c2)
            R = np.zeros((nmo, nmo))
            if imaginary:
                R[o, o] = +0.25 * (Aoo - Aoo.T)
                R[v, v] = -0.25 * (Avv - Avv.T)
            else:
                R[o, o] = -0.25 * (Aoo + Aoo.T)
                R[v, v] = +0.25 * (Avv + Avv.T)
            return R

        UA, gA, dc2As = [], [], []
        for a in range(3):
            U, dF, dERI = cphf.momentum_ints(a, ncore, gauge)
            dc2A = c0 * dt2_from(dF, dERI)
            UA.append(U)
            dc2As.append(dc2A)
            gA.append(gdens(dc2A, imaginary=True))

        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            hs = d.so_overlap_half(A)
            for beta in range(3):
                pX = Perturbation('nuclear', (A, beta))
                dt2R = np.asarray(self._perturbed_t2(pX))
                dc0R = -0.25 * c0**3 * c('ijab,ijab->', t2, dt2R)
                dc2R = dc0R * t2 + c0 * dt2R
                UReff = np.asarray(cphf.full_U(pX, ncore)) + np.asarray(hs[beta]).T
                gR = gdens(dc2R, imaginary=False)
                for alpha in range(3):
                    Icc = 0.25 * c('ijab,ijab->', dc2R, dc2As[alpha])
                    Icphi = c('ij,ij->', gR[o, o], UA[alpha][o, o]) + c('ab,ab->', gR[v, v], UA[alpha][v, v])
                    Iphic = c('ij,ij->', gA[alpha][o, o], UReff[o, o]) + c('ab,ab->', gA[alpha][v, v], UReff[v, v])
                    Ipp = c('pq,pq->', gamma, UA[alpha].T @ UReff)
                    P[A, beta, alpha] = 2.0 * (Icc + Icphi + Iphic + Ipp)
        self.vgapt = P
        return P
