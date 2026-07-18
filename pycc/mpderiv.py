"""MP2 analytic-derivative property driver.

`MPderiv` is the MP2 leaf of the :class:`~pycc.correlatedderivs.CorrelatedDerivs` hierarchy (see
docs/DERIVATIVES_PLAN_2026-06.md section 9): it supplies the MP2 reduced densities and their
first-order responses and carries the MP2 correlation-property methods (dipole, gradient,
polarizability, APT, Hessian, AAT, VG-APT).  The base holds the method-agnostic machinery
(reference ``HFwfn``, canonical dependent-pair rotations); later phases hoist more shared
orbital-response and 2n+1 assembly into it.

State is read from the MP2 wavefunction ``self.mp`` (an :class:`~pycc.mpwfn.MPwfn`); the unrelaxed
correlation-density seeds (``_*_corr_opdm`` / ``_*_tpdm``) and the intermediate normalization stay
on ``MPwfn`` (they are amplitude-derived quantities exercised directly by the energy/test surface).
"""

from __future__ import annotations

import numpy as np

from .correlatedderivs import CorrelatedDerivs


class MPderiv(CorrelatedDerivs):
    """MP2 correlation derivative-property driver.

    Constructed from a converged :class:`~pycc.mpwfn.MPwfn`; carries the orbital-response
    (Z-vector) machinery and the correlation contributions to the analytic first and second
    derivative properties.  Both the spin-adapted (closed-shell RHF) and spin-orbital (``_so_``)
    paths are frozen-core aware.
    """

    def __init__(self, wfn) -> None:
        super().__init__(wfn)
        self.mp = wfn                       # alias: the MP2 wavefunction whose densities we differentiate

    # ---- relaxed one-particle density accessor (the base owns the Z-vector build) ----

    def mp2_relaxed_opdm(self) -> np.ndarray:
        """Relaxed MP2 one-particle correlation density (``nmo x nmo``): the unrelaxed
        ``Doo``/``Dvv`` plus the orbital-relaxation blocks from the base Z-vector solve
        (:meth:`CorrelatedDerivs._relaxed_density`).  The MP2-named accessor the
        :class:`~pycc.mpwfn.MPwfn` facade shim delegates to."""
        return self._relaxed_density()[0]

    # ---- full-occupied CPHF (shared by the 2n+1 perturbed-response routes) ----

    def _full_occ_cphf(self):
        """A CPHF over the **full** occupied space (frozen core + active) in the wavefunction's
        own MO ordering (cached). The 2n+1 perturbed-response routes need the orbital response over
        the full occupied space (core<->active and core-virtual), which ``self.mp.cphf`` (active
        only) can't supply; building it here -- rather than borrowing an all-electron ``HFwfn``
        -- keeps the spin-orbital ordering consistent with the densities (the all-electron SO
        ``HFwfn`` orders the spins differently). For ``nfzc=0`` it coincides with ``self.mp.cphf``."""
        if getattr(self, '_focphf', None) is None:
            from .cphf import CPHF
            self._focphf = CPHF(self.mp, full_occ=True)
        return self._focphf

    # ---- perturbed amplitudes / densities (for the second derivatives) ----

    def _perturbed_t2(self, pert) -> np.ndarray:
        """First-order response of the MP2 doubles amplitudes to ``pert`` -- closed form,
        since the MP2 amplitudes are non-iterative::

            t^x_ijab = [ d_x<ij||ab> + sum_c (d_x f_ac t_ijcb + d_x f_bc t_ijac)
                         - sum_k (d_x f_ik t_kjab + d_x f_jk t_ikab) ] / D_ijab

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

    def _perturbed_densities(self, pert):
        """First-order response of the unrelaxed correlation densities to ``pert``: returns
        ``(d_a gamma, d_a Gamma)`` (full-MO arrays), from the perturbed amplitudes
        :meth:`_perturbed_t2` by the product rule -- the same density expressions as the
        unrelaxed densities (:meth:`MPwfn._so_mp2_corr_opdm`/`MPwfn._so_mp2_tpdm` and the
        spatial siblings), differentiated. Basis-aware."""
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

    # ---- 2n+1 route: perturbed relaxed density ----
    # The relaxed-density gradient is already the 2n+1 first derivative; its second derivative
    # (polarizability/APT/Hessian) needs the *response* of the relaxed density, whose new piece
    # is the perturbed Z-vector z^x (same orbital Hessian as the gradient, perturbed RHS). The
    # spin-orbital path builds G inline; the spatial analogue carries the spin-adapted L and
    # solves through the all-electron HFwfn CPHF. See mp2_2n1_perturbed.tex / DERIVATIVES_PLAN.

    def _so_perturbed_lagrangian(self, pert, D=None, dD=None) -> np.ndarray:
        """First-order response ``d_x I'`` of the GSB orbital Lagrangian (``nmo x nmo``),
        spin-orbital. Differentiates :meth:`_so_lagrangian` term by term with the perturbed
        integrals (:meth:`CPHF.perturbed_fock`/`perturbed_eri`) and the density response.

        With ``D``/``dD`` omitted this uses the **unrelaxed** 1-PDM and its response
        (:meth:`_perturbed_densities`) -- the Z-vector RHS derivative. Passing ``D`` = the
        relaxed 1-PDM and ``dD`` = its response (:meth:`_so_perturbed_relaxed_opdm`) gives
        instead ``d_x W`` (the perturbed energy-weighted density; :meth:`_so_gradient` uses
        ``W = I'(D_rel)`` with the nuclear ``S^X``). The 2-PDM response ``dGam`` is the same
        either way (:meth:`_perturbed_densities`). The ``termA`` derivative is the full Fock
        matrix product ``d_x f @ (D + D.T)`` (the diagonal ``d_x eps`` alone suffices for the
        unrelaxed ``D`` -- no ov block -- but the relaxed ``D`` has ov/core-active blocks that
        the off-diagonal ``d_x f`` couples)."""
        nmo, o, v = self.mp.nmo, self.mp.o, self.mp.v
        ofull = slice(0, o.stop)
        ncore = o.stop - self.mp.no
        ERI = np.asarray(self.mp.H.ERI)
        eps = np.diag(np.asarray(self.mp.H.F))
        c = self.contract
        cphf = self._full_occ_cphf()
        Gam = np.asarray(self.mp._so_mp2_tpdm())
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dDg, dGam = self._perturbed_densities(pert)
        dGam = np.asarray(dGam)
        if D is None:
            _, _, D, _, _, _ = self._so_zvector()      # unrelaxed 1-PDM
            dD = np.asarray(dDg)
        dA = df @ (D + D.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, ERI[:, :, :, ofull])
                        + c('rs,rpsq->pq', D, deri[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, ERI[:, ofull, :, :])
                        + c('rs,rqsp->pq', D, deri[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    def _so_perturbed_relaxed_opdm(self, pert) -> np.ndarray:
        """First-order response ``d_x D_rel`` of the relaxed MP2 1-PDM (``nmo x nmo``),
        spin-orbital, frozen-core aware: the unrelaxed oo/vv responses
        (:meth:`_perturbed_densities`), the perturbed core-active divide ``d_x P_co``, and the
        **perturbed Z-vector** ``z^x`` in the ov/vo blocks over the full occupied space. ``z^x``
        solves the same orbital Hessian ``G`` as the gradient's Z-vector with the perturbed RHS
        ``X^x - A^x z`` (``A^x z`` uses the full non-canonical ``d_x f`` vv/oo blocks).

        ``d_x P_co`` is the derivative of the divide ``(eps_c - eps_i) P_ci = I'_ci - I'_ic``,
        a Sylvester relation ``(eps_c - eps_i) d_x P_ci = d_x(I'_ci - I'_ic) - sum_d d_x f_cd
        P_di + sum_j P_cj d_x f_ji``: the diagonal ``d=c``/``j=i`` terms are the
        ``-P_ci(d_x eps_c - d_x eps_i)`` orbital-energy shift, and the off-diagonal ``j != i``
        active-active ``d_x f`` couples the block because a field leaves the active occupied
        space non-canonical. ``P_co`` also feeds the ov RHS coupling (perturbed here too)."""
        nmo, nfzc, nv = self.mp.nmo, self.mp.nfzc, self.mp.nv
        o, v, co = self.mp.o, self.mp.v, self.mp.co
        ofull = slice(0, o.stop)
        nof = o.stop
        ncore = o.stop - self.mp.no
        c = self.contract
        cphf = self._full_occ_cphf()
        _, _, _, zia, G, Pco = self._so_zvector()
        ERI = np.asarray(self.mp.H.ERI)
        eps = np.diag(np.asarray(self.mp.H.F))
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dIp = self._so_perturbed_lagrangian(pert)
        dDg, _ = self._perturbed_densities(pert)
        dDg = np.asarray(dDg)
        dD = np.zeros((nmo, nmo))
        dD[o, o] = dDg[o, o]
        dD[v, v] = dDg[v, v]
        dX = dIp[ofull, v] - dIp[v, ofull].T
        if nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            dD[co, o] = dPco
            dD[o, co] = dPco.T
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, ERI[v, o, ofull, co]) + c('jc,acij->ia', dzjc, ERI[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, deri[v, o, ofull, co]) + c('jc,acij->ia', zjc, deri[v, co, ofull, o]))
        Axz = (c('ajib,jb->ia', deri[v, ofull, ofull, v], zia) + c('abij,jb->ia', deri[v, v, ofull, ofull], zia)
               + c('ab,ib->ia', df[v, v], zia) - c('ij,ja->ia', df[ofull, ofull], zia))
        zx = np.linalg.solve(G, (dX - Axz).reshape(-1)).reshape(nof, nv)
        dD[v, ofull] = -zx.T
        dD[ofull, v] = -zx
        return dD

    def _unrelaxed_densities(self):
        """MP2 unrelaxed reduced densities as full-MO arrays: the 1-PDM ``D_u`` (the ``Doo``/``Dvv``
        correlation blocks on the occupied/virtual diagonal) and the cumulant 2-PDM ``Gamma``, from
        the amplitude seeds (:meth:`MPwfn._{so_}mp2_corr_opdm` / ``_{so_}mp2_tpdm``).  Supplies the
        base Z-vector (:meth:`CorrelatedDerivs._zvector` / :meth:`_so_zvector`)."""
        nmo, o, v = self.mp.nmo, self.mp.o, self.mp.v
        if self.mp.orbital_basis == 'spinorbital':
            Doo, Dvv = self.mp._so_mp2_corr_opdm()
            Gam = self.mp._so_mp2_tpdm()
        else:
            Doo, Dvv = self.mp._mp2_corr_opdm()
            Gam = self.mp._mp2_tpdm()
        Du = np.zeros((nmo, nmo))
        Du[o, o] = np.asarray(Doo)
        Du[v, v] = np.asarray(Dvv)
        return Du, np.asarray(Gam)

    def _perturbed_lagrangian(self, pert, D=None, dD=None) -> np.ndarray:
        """Spatial first-order response ``d_x I'`` of the GSB orbital Lagrangian (``nmo x nmo``),
        the closed-shell analogue of :meth:`_so_perturbed_lagrangian`: the spin-adapted ``L``
        (and its derivative ``dL = 2 d_x<pq|rs> - d_x<pq|sr>``) in the two-electron 1-PDM term,
        ``<pr|st>`` (H.ERI) with ``Gamma`` in the 2-PDM term. ``D``/``dD`` default to the
        unrelaxed 1-PDM and its response (Z-vector RHS); pass the relaxed 1-PDM and its response
        for ``d_x W``. See :meth:`_so_perturbed_lagrangian`."""
        nmo, o, v = self.mp.nmo, self.mp.o, self.mp.v
        ofull = slice(0, o.stop)
        ncore = o.stop - self.mp.no
        ERI = np.asarray(self.mp.H.ERI)
        L = np.asarray(self.mp.H.L)
        eps = np.diag(np.asarray(self.mp.H.F))
        c = self.contract
        cphf = self._full_occ_cphf()
        Gam = np.asarray(self.mp._mp2_tpdm())
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        dDg, dGam = self._perturbed_densities(pert)
        dGam = np.asarray(dGam)
        if D is None:
            _, _, D, _, _, _ = self._zvector()          # unrelaxed 1-PDM
            dD = np.asarray(dDg)
        dA = df @ (D + D.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, L[:, :, :, ofull])
                        + c('rs,rpsq->pq', D, dL[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, L[:, ofull, :, :])
                        + c('rs,rqsp->pq', D, dL[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    def _perturbed_relaxed_opdm(self, pert) -> np.ndarray:
        """Spatial first-order response ``d_x D_rel`` of the relaxed MP2 1-PDM (``nmo x nmo``),
        frozen-core aware: the unrelaxed oo/vv responses, the perturbed core-active divide
        ``d_x P_co`` (the Sylvester derivative -- see :meth:`_so_perturbed_relaxed_opdm`), and
        the perturbed Z-vector ``z^x`` in the ov/vo blocks over the full occupied space. ``z^x``
        reuses the all-electron HFwfn CPHF (same orbital Hessian as the gradient's Z-vector)
        with the perturbed RHS ``X^x - A^x z``; ``A^x z`` uses ``dL`` and the full non-canonical
        ``d_x f`` vv/oo blocks, and (for ``nfzc>0``) the perturbed ``P_co`` RHS coupling."""
        nmo, nfzc, no = self.mp.nmo, self.mp.nfzc, self.mp.no
        o, v = self.mp.o, self.mp.v
        co = slice(0, nfzc)
        ofull = slice(0, nfzc + no)
        ncore = o.stop - self.mp.no
        c = self.contract
        cphf = self._full_occ_cphf()
        _, _, _, zia, hf, Pco = self._zvector()
        L = np.asarray(self.mp.H.L)
        eps = np.diag(np.asarray(self.mp.H.F))
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        dIp = self._perturbed_lagrangian(pert)
        dDg, _ = self._perturbed_densities(pert)
        dDg = np.asarray(dDg)
        dD = np.zeros((nmo, nmo))
        dD[o, o] = dDg[o, o]
        dD[v, v] = dDg[v, v]
        dX = dIp[ofull, v] - dIp[v, ofull].T
        if nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            dD[co, o] = dPco
            dD[o, co] = dPco.T
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, L[v, o, ofull, co]) + c('jc,acij->ia', dzjc, L[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, dL[v, o, ofull, co]) + c('jc,acij->ia', zjc, dL[v, co, ofull, o]))
        Axz = (c('ajib,jb->ia', dL[v, ofull, ofull, v], zia) + c('abij,jb->ia', dL[v, v, ofull, ofull], zia)
               + c('ab,ib->ia', df[v, v], zia) - c('ij,ja->ia', df[ofull, ofull], zia))
        zx = hf.cphf.solve(dX - Axz)
        dD[v, ofull] = -zx.T
        dD[ofull, v] = -zx
        return dD

    # ---- second derivatives: polarizability, APT (dipole derivatives), Hessian ----

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        """MP2 **correlation** contribution to the static (omega=0) dipole polarizability
        (a.u.), shape ``(3, 3)``: ``alpha_corr_ab = -d^2 E_corr / dF_a dF_b``, via the 2n+1
        route (:meth:`_polarizability_2n1`): differentiate the relaxed-density gradient in a
        field, using only the first-order perturbed *relaxed* density
        (:meth:`_so_perturbed_relaxed_opdm` / :meth:`_perturbed_relaxed_opdm`, which carry the
        perturbed Z-vector). Frozen-core aware (both spin-orbital and spin-adapted paths).

        ``route`` accepts only ``'2n+1'`` (the sole route; the argument is retained for a uniform
        property signature). The reference part is kept separate (:meth:`HFwfn.polarizability`);
        the total is their sum. Electric field only."""
        if route != '2n+1':
            raise ValueError(f"unknown polarizability route {route!r} (only '2n+1')")
        return self._polarizability_2n1()

    def _polarizability_2n1(self) -> np.ndarray:
        """MP2 correlation polarizability via the 2n+1 route (frozen-core aware; spin-orbital
        and spin-adapted paths).

        Differentiating the relaxed-density gradient in a field: ``d_b E = -Tr(D_rel mu_b)``
        (field skeleton ``f^(b) = -mu_b``), so::

            alpha_ab = sum_pq d_a D_rel_pq (mu_b)_pq
                     + sum_pq D_rel_pq [ (U^a).T mu_b + mu_b U^a ]_pq

        The first term is the perturbed relaxed density (:meth:`_so_perturbed_relaxed_opdm`,
        carrying the perturbed Z-vector and, for frozen core, the perturbed core-active divide);
        the second is the MO dipole rotating under the field (``d_a f^(b) = rotate(U^a, -mu_b)``,
        with ``U^a`` over the full occupied space -- ``ncore`` canonical core-active block). No
        second-order CPHF ``U^{ab}`` -- only first-order responses. Reproduces the explicit
        route to ~machine precision."""
        from .cphf import Perturbation
        ncore = self.mp.o.stop - self.mp.no
        cphf = self._full_occ_cphf()
        c = self.contract
        if self.mp.orbital_basis == 'spinorbital':
            Drel = self._so_zvector()[0]
            perturbed_opdm = self._so_perturbed_relaxed_opdm
        else:
            Drel = self._zvector()[0]
            perturbed_opdm = self._perturbed_relaxed_opdm
        mu = [np.asarray(self.mp.H.mu[a]) for a in range(3)]
        field = [Perturbation('field', b) for b in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            dDrel = perturbed_opdm(field[b])
            Ub = np.asarray(cphf._full_U(field[b], ncore))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def dipole_derivatives(self, route: str = '2n+1-field') -> np.ndarray:
        """MP2 **correlation** contribution to the atomic polar tensors (nuclear dipole
        derivatives, a.u.), shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` =
        ``d(mu_alpha)/d(X_{A,beta}) = -d^2 E_corr / dF_alpha dX_{A,beta}`` -- the mixed
        field/nuclear analog of :meth:`polarizability`.

        ``route='2n+1-field'`` (default) / ``'2n+1-nuclear'`` -- the two 2n+1 routes
        (:meth:`_dipole_derivatives_2n1`), which use only first-order responses (no ``U^{FX}``).
        ``'2n+1-nuclear'`` differentiates the relaxed dipole w.r.t. the nuclei (``3N`` nuclear
        relaxed-density responses); ``'2n+1-field'`` differentiates the relaxed nuclear gradient
        w.r.t. the field (3 field responses -- ``d_F D_rel``, ``d_F Gamma``, ``d_F W`` -- contracted
        with the ``3N`` nuclear skeletons). Both give the same tensor; ``'2n+1-field'`` is cheaper
        and is what the IR/VCD path stores.

        Correlation only; the nuclear ``Z_A`` and SCF reference terms are separate, and the total
        (nuclear + reference + correlation) is assembled by :func:`pycc.apt`. Basis-aware.
        Validated against a finite nuclear displacement of the analytic dipole (``test_068``)."""
        if route not in ('2n+1-nuclear', '2n+1-field'):
            raise ValueError(f"unknown dipole-derivative route {route!r} "
                             "(use '2n+1-nuclear' or '2n+1-field')")
        return self._dipole_derivatives_2n1(route)

    def _dipole_derivatives_2n1(self, route: str) -> np.ndarray:
        """MP2 correlation atomic polar tensors via the 2n+1 route (both spin paths, frozen-core
        aware); ``route`` is ``'2n+1-nuclear'`` or ``'2n+1-field'``. See :meth:`dipole_derivatives`.

        Nuclear side -- differentiate the relaxed dipole ``Tr(D_rel mu_a)`` w.r.t. the nucleus
        (the field gradient has no ``S^X``/2e-skeleton term, so no energy-weighted density
        appears)::

            P[X,a] = Tr(d_X D_rel mu_a) + Tr(D_rel [mu_a^X + rotate(U^X, mu_a)]).

        Field side -- differentiate the relaxed nuclear gradient
        ``E^X = sum D_rel f^X + sum Gamma <pq||rs>^X + sum W S^X`` w.r.t. the field::

            P[X,a] = -[ sum d_a D_rel f^X + sum D_rel d_a f^X + sum d_a Gamma <>^X
                        + sum Gamma d_a <>^X + sum d_a W S^X + sum W d_a S^X ],

        with the 3 field responses ``d_a D_rel`` (:meth:`_so_perturbed_relaxed_opdm`),
        ``d_a Gamma`` (:meth:`_perturbed_densities`), and the perturbed energy-weighted density
        ``d_a W`` (:meth:`_so_perturbed_lagrangian` at the relaxed density). The field-derivatives
        of the nuclear skeletons carry the orbital rotation ``rotate(U^a, .)`` plus, for
        ``d_a f^X``, the occupied-sum response and the ``-mu_a^X`` mixed skeleton (the field
        enters ``h``)."""
        from .cphf import Perturbation
        c = self.contract
        so = self.mp.orbital_basis == 'spinorbital'
        o = self.mp.o
        ofull = slice(0, o.stop)
        ncore = o.stop - self.mp.no
        cphf = self._full_occ_cphf()
        d = self.mp.derivatives
        natom = d.natom
        Drel = (self._so_zvector() if so else self._zvector())[0]
        popdm = self._so_perturbed_relaxed_opdm if so else self._perturbed_relaxed_opdm
        mu = [np.asarray(self.mp.H.mu[a]) for a in range(3)]
        P = np.zeros((natom, 3, 3))

        if route == '2n+1-nuclear':
            for A in range(natom):
                dip = d.so_dipole(A) if so else d.dipole(A)          # [alpha*3 + beta]
                for beta in range(3):
                    pX = Perturbation('nuclear', (A, beta))
                    dDrel = popdm(pX)
                    UX = np.asarray(cphf._full_U(pX, ncore))
                    for alpha in range(3):
                        dmu = np.asarray(dip[alpha * 3 + beta])       # skeleton d(mu_a)/dX_beta
                        rot = UX.T @ mu[alpha] + mu[alpha] @ UX
                        P[A, beta, alpha] = (c('pq,pq->', dDrel, mu[alpha])
                                             + c('pq,pq->', Drel, dmu + rot))
            return P

        # route == '2n+1-field'
        lagr = self._so_perturbed_lagrangian if so else self._perturbed_lagrangian
        Gam = np.asarray(self.mp._so_mp2_tpdm() if so else self.mp._mp2_tpdm())
        W = self._lagrangian(Drel, Gam)
        field = [Perturbation('field', a) for a in range(3)]
        dDrel = [popdm(field[a]) for a in range(3)]
        dGamF = [np.asarray(self._perturbed_densities(field[a])[1]) for a in range(3)]
        dW = [lagr(field[a], Drel, dDrel[a]) for a in range(3)]      # perturbed energy-weighted density
        U = [np.asarray(cphf._full_U(field[a], ncore)) for a in range(3)]

        def rot1(Um, M):
            return Um.T @ M + M @ Um

        def rot4(Um, T):
            return (c('tp,tqrs->pqrs', Um, T) + c('tq,ptrs->pqrs', Um, T)
                    + c('tr,pqts->pqrs', Um, T) + c('ts,pqrt->pqrs', Um, T))

        for A in range(natom):
            hx = d.so_core(A) if so else d.core(A)
            Sx = d.so_overlap(A) if so else d.overlap(A)
            dip = d.so_dipole(A) if so else d.dipole(A)
            if so:
                eriF = [np.asarray(e) for e in d.so_eri(A)]          # <pq||rs>^X (Fock and Gamma)
                gamX = eriF
            else:
                phys = [np.asarray(ch).transpose(0, 2, 1, 3) for ch in d.eri(A)]  # <pq|rs>^X (Gamma)
                eriF = [2.0 * p - p.transpose(0, 1, 3, 2) for p in phys]          # L^X (Fock)
                gamX = phys
            for beta in range(3):
                fX = np.asarray(hx[beta]) + c('pmqm->pq', eriF[beta][:, ofull, :, ofull])
                SX = np.asarray(Sx[beta])
                gX, eX = gamX[beta], eriF[beta]
                for alpha in range(3):
                    Um = U[alpha]
                    muX = np.asarray(dip[alpha * 3 + beta])
                    occ = (c('rm,prqm->pq', Um[:, ofull], eX[:, :, :, ofull])
                           + c('rm,pmqr->pq', Um[:, ofull], eX[:, ofull, :, :]))
                    dfX = rot1(Um, fX) - muX + occ
                    P[A, beta, alpha] = -(c('pq,pq->', dDrel[alpha], fX) + c('pq,pq->', Drel, dfX)
                                          + c('pqrs,pqrs->', dGamF[alpha], gX) + c('pqrs,pqrs->', Gam, rot4(Um, gX))
                                          + c('pq,pq->', dW[alpha], SX) + c('pq,pq->', W, rot1(Um, SX)))
        return P

    def hessian(self, route: str = '2n+1') -> np.ndarray:
        """MP2 **correlation** contribution to the molecular (nuclear) Hessian (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3+a, B*3+b)`` = ``d^2 E_corr / dX_{Aa} dX_{Bb}`` --
        the nuclear-nuclear analog of :meth:`polarizability`/:meth:`dipole_derivatives`, via the
        2n+1 route (:meth:`_hessian_2n1`): differentiate the relaxed nuclear gradient w.r.t. a
        second nucleus, using only ``3N`` *first-order* solves (the perturbed relaxed density /
        energy-weighted density and ``U^Y``). The nuclear-nuclear analog of the ``'2n+1-field'``
        APT, with the full second skeletons ``f^{XY}``/``<pq||rs>^{XY}``/``S^{XY}``.

        ``route`` accepts only ``'2n+1'`` (retained for a uniform property signature). The
        reference and nuclear parts are separate; the total (nuclear + reference + correlation) is
        assembled by :func:`pycc.hessian`. Basis-aware. Validated against a finite difference of the
        analytic gradient (``test_069``)."""
        if route != '2n+1':
            raise ValueError(f"unknown hessian route {route!r} (only '2n+1')")
        return self._hessian_2n1()

    def _hessian_2n1(self) -> np.ndarray:
        """MP2 correlation molecular Hessian via the 2n+1 route (both spin paths, frozen-core
        aware). Differentiate the relaxed nuclear gradient
        ``E^X = sum D_rel f^X + sum Gamma <pq||rs>^X + sum W S^X`` w.r.t. a second nucleus ``Y``::

            H[X,Y] = sum d_Y D_rel f^X + sum D_rel d_Y f^X + sum d_Y Gamma <>^X
                     + sum Gamma d_Y <>^X + sum d_Y W S^X + sum W d_Y S^X,

        the nuclear-nuclear analog of the ``'2n+1-field'`` APT (:meth:`_dipole_derivatives_2n1`).
        Only ``3N`` first-order solves -- the perturbed relaxed density ``d_Y D_rel``
        (:meth:`_so_perturbed_relaxed_opdm`), the perturbed energy-weighted density ``d_Y W``
        (:meth:`_so_perturbed_lagrangian` at ``D_rel``), ``d_Y Gamma``
        (:meth:`_perturbed_densities`), and ``U^Y`` (:meth:`CPHF._full_U`) -- vs the explicit
        route's ``O(N^2)`` ``U^{XY}`` solves.

        The field-derivatives of the nuclear skeletons carry (i) the full second integral
        skeletons ``f^{XY}``/``<>^{XY}``/``S^{XY}`` (:meth:`CPHF._d2int_blocks`, cached per atom
        pair -- all nonzero here, unlike the field case where only ``-mu^X`` survived), and (ii)
        the ``U^Y`` orbital rotation of the ``X`` skeletons. The rotations are hoisted off the
        ``O(N^2)`` pair loop onto the (per-``Y``) densities via ``sum A rot(U,B) = sum rot(U^T,A) B``:
        ``Dtil = U D + D U^T``, ``Wtil`` likewise, ``Gtil = rotate4(U^T, Gamma)``, and the
        Fock skeleton's occupied-sum response as the per-``X`` intermediate ``J^X`` contracted
        with ``U^Y`` (so no ``O(N^2)`` four-index rotation)."""
        from .cphf import Perturbation
        c = self.contract
        so = self.mp.orbital_basis == 'spinorbital'
        ofull = slice(0, self.mp.o.stop)
        ncore = self.mp.o.stop - self.mp.no
        cphf = self._full_occ_cphf()
        d = self.mp.derivatives
        natom = d.natom
        nc = 3 * natom
        Drel = (self._so_zvector() if so else self._zvector())[0]
        Gam = np.asarray(self.mp._so_mp2_tpdm() if so else self.mp._mp2_tpdm())
        W = self._lagrangian(Drel, Gam)
        popdm = self._so_perturbed_relaxed_opdm if so else self._perturbed_relaxed_opdm
        lagr = self._so_perturbed_lagrangian if so else self._perturbed_lagrangian
        pert = [Perturbation('nuclear', (A, ct)) for A in range(natom) for ct in range(3)]

        def rot4(Um, T):
            return (c('tp,tqrs->pqrs', Um, T) + c('tq,ptrs->pqrs', Um, T)
                    + c('tr,pqts->pqrs', Um, T) + c('ts,pqrt->pqrs', Um, T))

        # first-order responses + hoisted per-Y rotated densities (sum A rot(U,B) = sum rot(U^T,A) B)
        dDrel = [popdm(p) for p in pert]
        dGamN = [np.asarray(self._perturbed_densities(p)[1]) for p in pert]
        dW = [lagr(pert[i], Drel, dDrel[i]) for i in range(nc)]
        U = [np.asarray(cphf._full_U(p, ncore)) for p in pert]
        Dtil = [U[i] @ Drel + Drel @ U[i].T for i in range(nc)]
        Wtil = [U[i] @ W + W @ U[i].T for i in range(nc)]
        Gtil = [rot4(U[i].T, Gam) for i in range(nc)]

        # per-X first skeletons; J^X carries the Fock skeleton's occupied-sum rotation response
        fX, gamX, SX, JX = [], [], [], []
        for p in pert:
            A, ct = p.comp
            hx = np.asarray((d.so_core(A) if so else d.core(A))[ct])
            Sx = np.asarray((d.so_overlap(A) if so else d.overlap(A))[ct])
            if so:
                eF = np.asarray(d.so_eri(A)[ct])
                gm = eF
            else:
                ph = np.asarray(d.eri(A)[ct]).transpose(0, 2, 1, 3)     # <pq|rs>^X (Gamma)
                eF = 2.0 * ph - ph.transpose(0, 1, 3, 2)                # L^X (Fock)
                gm = ph
            fX.append(hx + c('pmqm->pq', eF[:, ofull, :, ofull]))
            gamX.append(gm)
            SX.append(Sx)
            JX.append(c('pq,prqm->rm', Drel, eF[:, :, :, ofull])
                      + c('pq,pmqr->rm', Drel, eF[:, ofull, :, :]))

        H = np.zeros((nc, nc))
        for iy, py in enumerate(pert):
            Ay, cy = py.comp
            for ix, px in enumerate(pert):
                Ax, cx = px.comp
                blk = cphf._d2int_blocks(Ax, Ay)             # raw second skeletons (no U^{XY})
                core2 = blk['core'][cx * 3 + cy]
                ov2 = blk['overlap'][cx * 3 + cy]
                e2 = blk['eri'][cx * 3 + cy]
                L2 = e2 if so else 2.0 * e2 - e2.swapaxes(2, 3)
                f2 = core2 + c('pmqm->pq', L2[:, ofull, :, ofull])       # f^{XY}
                H[ix, iy] = (c('pq,pq->', dDrel[iy] + Dtil[iy], fX[ix]) + c('pq,pq->', Drel, f2)
                             + c('pqrs,pqrs->', dGamN[iy] + Gtil[iy], gamX[ix]) + c('pqrs,pqrs->', Gam, e2)
                             + c('pq,pq->', dW[iy] + Wtil[iy], SX[ix]) + c('pq,pq->', W, ov2)
                             + float(np.sum(U[iy][:, ofull] * JX[ix])))
        return H

    # ---- atomic axial tensors (VCD, magnetic/nuclear mixed derivative) ----

    def atomic_axial_tensors(self, gauge: str = 'non-canonical') -> np.ndarray:
        """MP2 **correlation** atomic axial tensors ``I^A_{alpha,beta}`` (a.u.), shape
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

            I = sum_I dc^R_I dc^H_I                    (coefficient overlap,            Icc)
              + sum_pq g^R_pq <phi_p|d_H phi_q>        (left derivative density x U^H,  Icphi)
              + sum_pq g^H_pq <phi_p|d_R phi_q>        (right derivative density x U^R, Iphic)
              + sum_pq gamma_pq <d_R phi_p|d_H phi_q>  (density x both MO derivatives,  Ipp)

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
        responses use pycc's ``-1/2 S`` gauge (:meth:`_perturbed_t2` / :meth:`CPHF._full_U`).
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
                UReff = np.asarray(cphf._full_U(pX, ncore)) + np.asarray(hs[cart]).T
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
        """Spin-orbital MP2 electronic AATs (``(natom, 3, 3)``) -- the spin-orbital form of
        :meth:`atomic_axial_tensors` (see there for the theory), in the bare (already-
        antisymmetrized) spin-orbital amplitudes.  The derivative densities carry the same
        symmetry as in the spin-adapted path: ``gamma^R`` is the **symmetric** part of
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
        Doo, Dvv = self.mp._so_mp2_corr_opdm()
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
                UReff = np.asarray(cphf._full_U(pX, ncore)) + np.asarray(hs[cart]).T
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
        """MP2 velocity-gauge (VG) atomic polar tensors ``[P^A_{beta,alpha}]^VG`` (a.u.), shape
        ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` = ``d(mu_alpha)/d(X_A,beta)`` -- the
        momentum-form APT.  This is the **correlation** contribution only; the SCF reference VG
        APT (:meth:`HFwfn.velocity_dipole_derivatives`) and the nuclear ``Z_A delta_{alpha,beta}``
        term are kept separate and summed by the :func:`pycc.apt` (``gauge='velocity'``) facade.
        Built on the atomic-axial-tensor machinery (:meth:`atomic_axial_tensors`) with the
        magnetic-dipole operator replaced by the linear momentum ``p = -i nabla``
        (:meth:`CPHF.momentum_ints`)::

            correlation = 2 <d_R Psi | d_A Psi>_correlation

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
                UReff = np.asarray(cphf._full_U(pX, ncore)) + np.asarray(hs[beta]).T
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
        """Spin-orbital MP2 velocity-gauge APTs (``(natom, 3, 3)``) -- the correlation-only
        spin-orbital form of :meth:`velocity_dipole_derivatives` (see there for the theory),
        sharing the spin-orbital AAT densities (:meth:`_so_atomic_axial_tensors`) with the
        linear-momentum response (:meth:`CPHF.momentum_ints`).  The ``+2`` prefactor is the same as
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
        Doo, Dvv = self.mp._so_mp2_corr_opdm()
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
                UReff = np.asarray(cphf._full_U(pX, ncore)) + np.asarray(hs[beta]).T
                gR = gdens(dc2R, imaginary=False)
                for alpha in range(3):
                    Icc = 0.25 * c('ijab,ijab->', dc2R, dc2As[alpha])
                    Icphi = c('ij,ij->', gR[o, o], UA[alpha][o, o]) + c('ab,ab->', gR[v, v], UA[alpha][v, v])
                    Iphic = c('ij,ij->', gA[alpha][o, o], UReff[o, o]) + c('ab,ab->', gA[alpha][v, v], UReff[v, v])
                    Ipp = c('pq,pq->', gamma, UA[alpha].T @ UReff)
                    P[A, beta, alpha] = 2.0 * (Icc + Icphi + Iphic + Ipp)
        self.vgapt = P
        return P
