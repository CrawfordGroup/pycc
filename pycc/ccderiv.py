"""Analytic derivative properties of a coupled-cluster wavefunction.

`CCderiv` is the downstream derivative driver for a converged :class:`~pycc.ccwfn.CCwfn`, sitting
at the end of the chain ``ccwfn -> cchbar -> cclambda -> ccdensity``: it lazily builds the Lambda
amplitudes and reduced densities it needs and assembles the analytic gradient (Hessian, APTs, etc.
to follow).  Keeping this out of `CCwfn` respects the layering -- `cclambda`/`ccdensity` are
downstream of `ccwfn`, so the wavefunction never reaches forward to build them.  The
:mod:`pycc.properties` facade routes ``pycc.gradient(ccwfn)`` here (see its registry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .correlatedderivs import CorrelatedDerivs

if TYPE_CHECKING:
    from .ccwfn import CCwfn


class CCderiv(CorrelatedDerivs):
    """Analytic derivative properties of a converged CCSD wavefunction.

    Parameters
    ----------
    ccwfn : CCwfn
        A converged coupled-cluster wavefunction (call :meth:`CCwfn.solve_cc` first).

    Notes
    -----
    Both the spatial (closed-shell RHF) and spin-orbital (UHF) paths are supported; ROHF is not
    (the semicanonical response does not reproduce the restricted ROHF response).  Lambda and the
    reduced densities are solved/built on first use and cached.  The analytic nuclear gradient is
    implemented (Hessian, APTs, etc. to follow).
    """

    def __init__(self, ccwfn: "CCwfn") -> None:
        super().__init__(ccwfn)
        self.ccwfn = ccwfn                              # alias: this class uses .ccwfn, the base .wfn
        self._dens = None

    def _density(self):
        """Converged Lambda amplitudes and the (Lambda-response) reduced densities, cached.
        Builds ``cchbar`` -> ``cclambda`` (solved) -> ``ccdensity`` on first use.  Lambda inherits
        the wavefunction's convergence criteria (``ccwfn.e_conv``/``r_conv``/``maxiter``, set by
        ``solve_cc``) rather than hardwiring its own."""
        if self._dens is None:
            from .cchbar import cchbar
            from .cclambda import cclambda
            from .ccdensity import ccdensity
            cc = self.ccwfn
            hbar = cchbar(cc)
            lam = cclambda(cc, hbar)
            lam.solve_lambda(e_conv=cc.e_conv, r_conv=cc.r_conv, maxiter=cc.maxiter)
            self._dens = ccdensity(cc, lam)
        return self._dens

    def _unrelaxed_densities(self):
        """CC unrelaxed reduced densities: the Lambda-response 1-PDM ``D`` and the (symmetrized)
        cumulant 2-PDM ``Gam`` (:meth:`ccdensity.gradient_densities`), full-MO arrays.  Supplies the
        base Z-vector (:meth:`CorrelatedDerivs._orbital_response` / :meth:`_so_orbital_response`)."""
        D, Gam = self._density().gradient_densities()
        return np.asarray(D), np.asarray(Gam)

    # ---- second derivatives: static dipole polarizability ----------------
    # The asymmetric (2n+1) route: differentiate the relaxed-density gradient a second time in a
    # field.  A static field leaves the AO basis fixed (S^F = <pq|rs>^F = 0), so
    #     alpha[a,b] = Tr(dD_rel(F_b) . mu_a)  +  Tr(D_rel . (U^bT mu_a + mu_a U^b)),
    # with D_rel the (unperturbed) relaxed density and dD_rel its field response.  dD_rel needs the
    # perturbed amplitudes/multipliers dt/dLambda (iterative, CPHF-folded RHS carrying the orbital
    # relaxation), the perturbed correlation densities, the perturbed Lagrangian (FULL-df Fock term
    # -- see the diagonal-eps gotcha below), and the perturbed Z-vector.  Only first-order responses;
    # no second-order CPHF.  See DERIVATIVES_PLAN_2026-06.md sec 8.  Spatial (closed-shell RHF) and
    # spin-orbital (UHF) paths, all-electron and frozen core; (T) to follow.  Each _so_* method
    # follows its spatial counterpart -- the SO path mirrors the spatial with antisymmetrized
    # <pq||rs> (no L, no Hovov; Hvvvo/Hovoo via a Zovov intermediate) and the inline orbital Hessian
    # G (as in _so_relaxed_density) rather than a borrowed all-electron CPHF.

    _HBAR_BLOCKS = ('Hov', 'Hvv', 'Hoo', 'Hoooo', 'Hvvvv', 'Hvovv', 'Hooov',
                    'Hovvo', 'Hovov', 'Hvvvo', 'Hovoo')

    _SO_HBAR_BLOCKS = ('Hov', 'Hvv', 'Hoo', 'Hoooo', 'Hvvvv', 'Hvovv', 'Hooov',
                       'Hovvo', 'Hvvvo', 'Hovoo')

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        """CCSD **correlation** contribution to the static (omega=0) dipole polarizability (a.u.),
        shape ``(3, 3)``: ``alpha_corr[a,b] = -d^2 E_corr/dF_a dF_b``, the CC analog of
        :meth:`MPwfn.polarizability`.

        Only the **asymmetric (2n+1) route** is available for CC -- differentiate the relaxed-density
        gradient a second time (a single (T)-capable formulation; see DERIVATIVES_PLAN sec 8).  The
        ``route`` argument (from the :func:`pycc.polarizability` facade) accepts only ``'2n+1'``.

        The reference (SCF) polarizability is kept separate (:meth:`HFwfn.polarizability`); the
        :func:`pycc.polarizability` facade sums nuclear (zero) + reference + this correlation part.

        Spatial closed-shell RHF and spin-orbital UHF, CCSD and CCSD(T), all-electron and frozen
        core.  CCSD is validated against a tight finite field of :meth:`relaxed_dipole` and the
        SO == spatial keystone; CCSD(T) against the energy second-derivative finite field and the
        SO == spatial keystone (SO CCSD(T) itself matched to CFOUR)."""
        if route != '2n+1':
            raise ValueError(f"CC polarizability supports only the asymmetric '2n+1' route, not {route!r}.")
        cc = self.ccwfn
        if cc.model.upper() not in ('CCSD', 'CCSD(T)'):
            raise NotImplementedError(f"CC polarizability: only CCSD and CCSD(T) are implemented (not {cc.model}).")
        is_t = cc.model.upper() == 'CCSD(T)'
        if is_t and not hasattr(cc, 'S1'):
            raise ValueError("CCSD(T) polarizability requires the (T) density intermediates; "
                             "build the wavefunction with make_t3_density=True.")
        if cc.orbital_basis == 'spinorbital':
            return self._so_polarizability()
        from .cchbar import cchbar
        from .cclambda import cclambda
        from .ccdensity import ccdensity
        from .cphf import Perturbation
        o, v = cc.o, cc.v
        co = slice(0, cc.nfzc)                            # frozen-core occupied
        ofull = slice(0, o.stop)                         # full occupied (core + active)
        c = self.contract
        ncore = o.stop - cc.no
        eps = np.diag(np.asarray(cc.H.F))
        L = np.asarray(cc.H.L)

        # Lambda inherits the wavefunction's convergence (set by solve_cc); the test/user tightens
        # ccwfn convergence to reach the second-derivative oracle rather than hardwiring it here.
        hbar = cchbar(cc)
        lam = cclambda(cc, hbar)
        lam.solve_lambda(e_conv=cc.e_conv, r_conv=cc.r_conv, maxiter=cc.maxiter)
        dens = ccdensity(cc, lam)
        D0, Gam0 = (np.asarray(x) for x in dens.gradient_densities())
        Ip0 = np.asarray(self._lagrangian(D0, Gam0))
        hf = self._reference_hf()
        Drel = D0.copy()
        Pco = None
        if cc.nfzc:                                      # core<->active-occupied divide (as in _relaxed_density)
            Pco = (Ip0[co, o] - Ip0[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            Drel[co, o] += Pco
            Drel[o, co] += Pco.T
        X0 = Ip0[ofull, v] - Ip0[v, ofull].T
        if cc.nfzc:                                      # couple P_co into the Z-vector RHS
            zjc = -Pco.T
            X0 = X0 - (c('jc,ajic->ia', zjc, L[v, o, ofull, co]) + c('jc,acij->ia', zjc, L[v, co, ofull, o]))
        Poo0 = Pvv0 = None
        if cc.model.upper() == 'CCSD(T)':                # (T) oo/vv dependent-pairs (as in _relaxed_density)
            Poo0 = self._dependent_pairs(Ip0[o, o], eps[o])
            Pvv0 = self._dependent_pairs(Ip0[v, v], eps[v])
            Drel[o, o] += Poo0
            Drel[v, v] += Pvv0
            X0 = X0 + (c('kl,akil->ia', Poo0, L[v, o, ofull, o]) + c('bc,ibac->ia', Pvv0, L[ofull, v, v, v]))
        z = np.asarray(hf.cphf.solve(X0))
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z

        cphf = cc.mp.deriv._full_occ_cphf()
        mu = [np.asarray(cc.H.mu[a]) for a in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            dDrel = self._perturbed_relaxed_density(pert, hbar, lam, D0, Gam0, z, Pco, Poo0, Pvv0)
            Ub = np.asarray(cphf._full_U(pert, ncore, canonical=is_t))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def _so_polarizability(self) -> np.ndarray:
        """Spin-orbital (UHF) CCSD correlation polarizability -- the SO analog of
        :meth:`polarizability` (all-electron or frozen core).  Raises for ROHF."""
        from .cchbar import cchbar
        from .cclambda import cclambda
        from .ccdensity import ccdensity
        from .cphf import Perturbation
        cc = self.ccwfn
        if cc.mp.cphf.is_rohf:
            raise NotImplementedError("CC polarizability: ROHF is not supported (RHF/UHF only).")
        o, v, nv = cc.o, cc.v, cc.nv
        co = slice(0, o.start)                            # frozen core (2*nfzc spin-orbitals)
        ofull = slice(0, o.stop)
        nof = o.stop
        c = self.contract
        ncore = o.stop - cc.no
        eps = np.diag(np.asarray(cc.H.F))
        ERI = np.asarray(cc.H.ERI)

        hbar = cchbar(cc)
        lam = cclambda(cc, hbar)
        lam.solve_lambda(e_conv=cc.e_conv, r_conv=cc.r_conv, maxiter=cc.maxiter)
        D0, Gam0 = (np.asarray(x) for x in ccdensity(cc, lam).gradient_densities())
        Ip0 = np.asarray(self._lagrangian(D0, Gam0))
        Drel = D0.copy()
        Pco = None
        if cc.nfzc:
            Pco = (Ip0[co, o] - Ip0[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            Drel[co, o] += Pco
            Drel[o, co] += Pco.T
        X0 = Ip0[ofull, v] - Ip0[v, ofull].T
        if cc.nfzc:
            zjc = -Pco.T
            X0 = X0 - (c('jc,ajic->ia', zjc, ERI[v, o, ofull, co]) + c('jc,acij->ia', zjc, ERI[v, co, ofull, o]))
        Poo0 = Pvv0 = None
        if cc.model.upper() == 'CCSD(T)':                 # (T) oo/vv dependent-pairs (as in _so_relaxed_density)
            Poo0 = self._dependent_pairs(Ip0[o, o], eps[o])
            Pvv0 = self._dependent_pairs(Ip0[v, v], eps[v])
            Drel[o, o] += Poo0
            Drel[v, v] += Pvv0
            X0 = X0 + (c('kl,akil->ia', Poo0, ERI[v, o, ofull, o]) + c('bc,ibac->ia', Pvv0, ERI[ofull, v, v, v]))
        G = (c('ajib->iajb', ERI[v, ofull, ofull, v]) + c('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof*nv, nof*nv)
        G[np.diag_indices(nof*nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
        z = np.linalg.solve(G, X0.reshape(-1)).reshape(nof, nv)
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z

        cphf = cc.mp.deriv._full_occ_cphf()
        mu = [np.asarray(cc.H.mu[a]) for a in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            dDrel = self._so_perturbed_relaxed_density(pert, hbar, lam, D0, Gam0, z, G, Pco, Poo0, Pvv0)
            Ub = np.asarray(cphf._full_U(pert, ncore, canonical=(cc.model.upper() == 'CCSD(T)')))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def _perturbed_relaxed_density(self, pert, hbar, lam, D0, Gam0, z, Pco=None,
                                   Poo0=None, Pvv0=None) -> np.ndarray:
        """Field response ``dD_rel`` of the CC relaxed 1-PDM (spatial; all-electron or frozen core).
        Mirrors :meth:`MPwfn._perturbed_relaxed_opdm` with the CC densities, but (i) keeps the CC
        *unrelaxed* ov/vo blocks (``D_ai != D_ia`` for CC; MP2 has none) and (ii) builds the perturbed
        Lagrangian with the FULL-df Fock term (see :meth:`CorrelatedDerivs._perturbed_lagrangian_matrix`).  For frozen core
        the perturbed core<->active Sylvester divide ``dP_co`` is added and coupled into the perturbed
        Z-vector RHS, exactly as in the MP2 template / the unperturbed :meth:`_relaxed_density`.

        CCSD(T): uses the **canonical** perturbed orbitals (``df``/``deri`` with ``canonical=True``, so
        ``df`` is diagonal in oo/vv -- the non-invariant (T) kernels require it), threads the perturbed
        (T) intermediates ``dt3`` into the density and the Lambda source, and adds the perturbed
        active-oo/vv dependent-pairs ``dPoo``/``dPvv`` (the field-derivative of the unperturbed
        ``Poo0``/``Pvv0``, including the ``d(eps_i-eps_j) = df_ii - df_jj`` denominator term) to
        ``dD_rel`` + the perturbed Z-vector RHS."""
        cc = self.ccwfn
        o, v = cc.o, cc.v
        co = slice(0, cc.nfzc)
        ofull = slice(0, o.stop)
        c = self.contract
        ncore = o.stop - cc.no
        L = np.asarray(cc.H.L)
        eps = np.diag(np.asarray(cc.H.F))
        is_t = cc.model.upper() == 'CCSD(T)'
        canon = is_t                                    # (T) needs canonical perturbed orbitals
        cphf = cc.mp.deriv._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore, canonical=canon))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore, canonical=canon))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        hf = self._reference_hf()

        dt1, dt2 = self._perturbed_amplitudes(df, deri, dL, hbar)
        dt3 = self._perturbed_t3_intermediates(df, deri, dL, dt1, dt2) if is_t else None
        dl1, dl2 = self._perturbed_lambda(df, deri, dL, dt1, dt2, hbar, lam,
                                          dS1=(dt3['S1'] if is_t else None),
                                          dS2=(dt3['S2'] if is_t else None))
        dDg, dGam = self._perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3=dt3)
        dIp = self._perturbed_lagrangian_matrix(df, deri, dL, D0, dDg, Gam0, dGam)
        dX = dIp[ofull, v] - dIp[v, ofull].T
        dPco = None
        if cc.nfzc:                                     # perturbed core<->active divide (Sylvester derivative)
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, L[v, o, ofull, co]) + c('jc,acij->ia', dzjc, L[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, dL[v, o, ofull, co]) + c('jc,acij->ia', zjc, dL[v, co, ofull, o]))
        dPoo = dPvv = None
        if is_t:                                        # perturbed (T) oo/vv dependent-pairs
            dfd = np.diag(df)                           # canonical df diagonal = the perturbed gaps
            dPoo = self._perturbed_dependent_pairs(dIp[o, o], Poo0, eps[o], dfd[o])
            dPvv = self._perturbed_dependent_pairs(dIp[v, v], Pvv0, eps[v], dfd[v])
            dX = dX + (c('kl,akil->ia', dPoo, L[v, o, ofull, o]) + c('kl,akil->ia', Poo0, dL[v, o, ofull, o])
                       + c('bc,ibac->ia', dPvv, L[ofull, v, v, v]) + c('bc,ibac->ia', Pvv0, dL[ofull, v, v, v]))
        # perturbed orbital-Hessian response (A^x z); reference-only, as in MP2
        Axz = (c('ajib,jb->ia', dL[v, ofull, ofull, v], z) + c('abij,jb->ia', dL[v, v, ofull, ofull], z)
               + c('ab,ib->ia', df[v, v], z) - c('ij,ja->ia', df[ofull, ofull], z))
        zx = np.asarray(hf.cphf.solve(dX - Axz))
        dDrel = dDg.copy()                              # keep unrelaxed dD_ov/dD_vo (CC-only)
        if cc.nfzc:
            dDrel[co, o] += dPco
            dDrel[o, co] += dPco.T
        if is_t:
            dDrel[o, o] += dPoo
            dDrel[v, v] += dPvv
        dDrel[v, ofull] += -zx.T
        dDrel[ofull, v] += -zx
        return dDrel

    def _so_perturbed_relaxed_density(self, pert, hbar, lam, D0, Gam0, z, G, Pco,
                                      Poo0=None, Pvv0=None):
        """Spin-orbital field response ``dD_rel``; the SO analog of
        :meth:`_perturbed_relaxed_density` (inline Hessian ``G``, antisymmetrized ERI).

        CCSD(T): uses the **canonical** perturbed orbitals (``df``/``deri`` with ``canonical=True``,
        so ``df`` is diagonal in oo/vv -- the non-invariant (T) kernels require it), threads the
        perturbed (T) intermediates ``dt3`` into the density and the Lambda source, and adds the
        perturbed active-oo/vv dependent-pairs ``dPoo``/``dPvv`` (the field derivative of the
        unperturbed ``Poo0``/``Pvv0``, including the ``d(eps_i-eps_j) = df_ii - df_jj`` denominator
        term) to ``dD_rel`` and the perturbed Z-vector RHS -- mirroring the spatial template."""
        cc = self.ccwfn
        o, v, nv = cc.o, cc.v, cc.nv
        co = slice(0, o.start)
        ofull = slice(0, o.stop)
        nof = o.stop
        c = self.contract
        ncore = o.stop - cc.no
        ERI = np.asarray(cc.H.ERI)
        eps = np.diag(np.asarray(cc.H.F))
        is_t = cc.model.upper() == 'CCSD(T)'
        canon = is_t                                    # (T) needs canonical perturbed orbitals
        cphf = cc.mp.deriv._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore, canonical=canon))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore, canonical=canon))

        dt1, dt2 = self._so_perturbed_amplitudes(df, deri, hbar)
        dt3 = self._so_perturbed_t3_intermediates(df, deri, dt1, dt2) if is_t else None
        dl1, dl2 = self._so_perturbed_lambda(df, deri, dt1, dt2, hbar, lam,
                                             dS1=(dt3['S1'] if is_t else None),
                                             dS2=(dt3['S2'] if is_t else None))
        dDg, dGam = self._perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3=dt3)
        dIp = self._so_perturbed_lagrangian_matrix(df, deri, D0, dDg, Gam0, dGam)
        dX = dIp[ofull, v] - dIp[v, ofull].T
        dPco = None
        if cc.nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, ERI[v, o, ofull, co]) + c('jc,acij->ia', dzjc, ERI[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, deri[v, o, ofull, co]) + c('jc,acij->ia', zjc, deri[v, co, ofull, o]))
        dPoo = dPvv = None
        if is_t:                                        # perturbed (T) oo/vv dependent-pairs
            dfd = np.diag(df)                           # canonical df diagonal = the perturbed gaps
            dPoo = self._perturbed_dependent_pairs(dIp[o, o], Poo0, eps[o], dfd[o])
            dPvv = self._perturbed_dependent_pairs(dIp[v, v], Pvv0, eps[v], dfd[v])
            dX = dX + (c('kl,akil->ia', dPoo, ERI[v, o, ofull, o]) + c('kl,akil->ia', Poo0, deri[v, o, ofull, o])
                       + c('bc,ibac->ia', dPvv, ERI[ofull, v, v, v]) + c('bc,ibac->ia', Pvv0, deri[ofull, v, v, v]))
        Axz = (c('ajib,jb->ia', deri[v, ofull, ofull, v], z) + c('abij,jb->ia', deri[v, v, ofull, ofull], z)
               + c('ab,ib->ia', df[v, v], z) - c('ij,ja->ia', df[ofull, ofull], z))
        zx = np.linalg.solve(G, (dX - Axz).reshape(-1)).reshape(nof, nv)
        dDrel = dDg.copy()
        if cc.nfzc:
            dDrel[co, o] += dPco
            dDrel[o, co] += dPco.T
        if is_t:
            dDrel[o, o] += dPoo
            dDrel[v, v] += dPvv
        dDrel[v, ofull] += -zx.T
        dDrel[ofull, v] += -zx
        return dDrel

    def _ccsd_jacobian(self, X1, X2, hbar):
        """The CCSD Jacobian applied to an amplitude pair, ``(HBAR . X)_singles``,
        ``(HBAR . X)_doubles`` (doubles un-symmetrized).  Method-agnostic (the same contraction
        pattern as ``ccresponse.r_X1/r_X2``, built from ``cchbar`` -- not a ccresponse dependency)."""
        c = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        t2 = self.ccwfn.t2
        L = self.ccwfn.H.L
        Hvvmvo = 2.0 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)
        Hooomv = 2.0 * hbar.Hooov - hbar.Hooov.swapaxes(0, 1)
        r1 = c('ie,ae->ia', X1, hbar.Hvv) - c('ma,mi->ia', X1, hbar.Hoo)
        r1 += 2.0 * c('me,maei->ia', X1, hbar.Hovvo) - c('me,maie->ia', X1, hbar.Hovov)
        r1 += c('me,miea->ia', hbar.Hov, (2.0 * X2 - X2.swapaxes(0, 1)))
        r1 += c('imef,amef->ia', X2, Hvvmvo)
        r1 -= c('mnae,mnie->ia', X2, Hooomv)
        Zvv = c('amef,mf->ae', Hvvmvo, X1) - c('mnef,mnaf->ae', L[o, o, v, v], X2)
        Zoo = -1.0 * c('mnie,ne->mi', Hooomv, X1) - c('mnef,inef->mi', L[o, o, v, v], X2)
        r2 = c('ie,abej->ijab', X1, hbar.Hvvvo) - c('ma,mbij->ijab', X1, hbar.Hovoo)
        r2 += c('mi,mjab->ijab', Zoo, t2) + c('ae,ijeb->ijab', Zvv, t2)
        r2 += c('ijeb,ae->ijab', X2, hbar.Hvv) - c('mjab,mi->ijab', X2, hbar.Hoo)
        r2 += 0.5 * c('mnab,mnij->ijab', X2, hbar.Hoooo) + 0.5 * c('ijef,abef->ijab', X2, hbar.Hvvvv)
        r2 -= c('imeb,maje->ijab', X2, hbar.Hovov) + c('imea,mbej->ijab', X2, hbar.Hovvo)
        r2 += 2.0 * c('miea,mbej->ijab', X2, hbar.Hovvo) - c('miea,mbje->ijab', X2, hbar.Hovov)
        return r1, r2

    def _so_ccsd_jacobian(self, X1, X2, hbar):
        """Spin-orbital CCSD Jacobian ``(HBAR . X)`` -- the SO ``ccresponse.r_X1/r_X2`` contraction
        pattern (antisymmetrized; the P(ij)P(ab) is built in, so no final symmetrization)."""
        c = self.contract
        o, v = self.ccwfn.o, self.ccwfn.v
        t2 = self.ccwfn.t2
        ERI = self.ccwfn.H.ERI
        r1 = c('ie,ae->ia', X1, hbar.Hvv) - c('ma,mi->ia', X1, hbar.Hoo)
        r1 += c('me,maei->ia', X1, hbar.Hovvo)
        r1 += c('me,imae->ia', hbar.Hov, X2)
        r1 += 0.5 * c('imef,amef->ia', X2, hbar.Hvovv)
        r1 -= 0.5 * c('mnae,mnie->ia', X2, hbar.Hooov)
        Zvv = c('amef,me->af', hbar.Hvovv, X1)
        Zoo = c('mnie,me->ni', hbar.Hooov, X1)
        Yoo = 0.5 * c('mnef,mjef->nj', ERI[o, o, v, v], X2)
        Yvv = 0.5 * c('mnef,mneb->fb', ERI[o, o, v, v], X2)
        r2 = c('ie,abej->ijab', X1, hbar.Hvvvo) - c('je,abei->ijab', X1, hbar.Hvvvo)
        r2 -= c('ma,mbij->ijab', X1, hbar.Hovoo) - c('mb,maij->ijab', X1, hbar.Hovoo)
        r2 += c('ni,njab->ijab', Zoo, t2) - c('nj,niab->ijab', Zoo, t2)
        r2 -= c('af,ijfb->ijab', Zvv, t2) - c('bf,ijfa->ijab', Zvv, t2)
        r2 -= c('nj,inab->ijab', Yoo, t2) - c('ni,jnab->ijab', Yoo, t2)
        r2 -= c('fb,ijaf->ijab', Yvv, t2) - c('fa,ijbf->ijab', Yvv, t2)
        r2 += c('ijae,be->ijab', X2, hbar.Hvv) - c('ijbe,ae->ijab', X2, hbar.Hvv)
        r2 -= c('imab,mj->ijab', X2, hbar.Hoo) - c('jmab,mi->ijab', X2, hbar.Hoo)
        r2 += 0.5 * c('mnab,mnij->ijab', X2, hbar.Hoooo)
        r2 += 0.5 * c('ijef,abef->ijab', X2, hbar.Hvvvv)
        tmp = c('imae,mbej->ijab', X2, hbar.Hovvo)
        r2 += tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3) + tmp.swapaxes(0, 1).swapaxes(2, 3)
        return r1, r2

    def _perturbed_amplitudes(self, df, deri, dL, hbar, maxiter=None, rconv=None):
        """Perturbed CCSD amplitudes ``dt/dF`` (iterative).  RHS = the field-derivative of the CC
        residual at fixed ``t`` -- since the residual is linear in H, that is just
        ``residuals(df, t1, t2)`` with the perturbed two-electron integrals swapped in (the
        CPHF-folded ``deri``/``dL`` carry the orbital relaxation).  LHS = the CCSD Jacobian;
        iterate ``dt += (B + HBAR.dt)/D`` with DIIS, like :meth:`ccresponse.solve_right`.
        ``maxiter``/``rconv`` default to the wavefunction's convergence (``ccwfn.maxiter``/``r_conv``)."""
        from .utils import helper_diis
        cc = self.ccwfn
        maxiter = cc.maxiter if maxiter is None else maxiter
        rconv = cc.r_conv if rconv is None else rconv
        Dia, Dijab = cc.Dia, cc.Dijab
        saveERI, saveL = cc.H.ERI, cc.H.L
        cc.H.ERI, cc.H.L = deri, dL
        try:
            B1, B2 = cc.residuals(df, cc.t1, cc.t2)
        finally:
            cc.H.ERI, cc.H.L = saveERI, saveL
        B1, B2 = np.asarray(B1), np.asarray(B2)
        X1, X2 = B1 / Dia, B2 / Dijab
        diis = helper_diis(X1, X2, 8)
        for _ in range(maxiter):
            j1, j2 = self._ccsd_jacobian(X1, X2, hbar)
            r1 = B1 + j1
            r2 = 0.5 * B2 + j2
            r2 = r2 + r2.swapaxes(0, 1).swapaxes(2, 3)
            X1 = X1 + r1 / Dia
            X2 = X2 + r2 / Dijab
            if np.sqrt(np.sum((r1 / Dia) ** 2) + np.sum((r2 / Dijab) ** 2)) < rconv:
                break
            diis.add_error_vector(X1, X2)
            X1, X2 = diis.extrapolate(X1, X2)
        return X1, X2

    def _so_perturbed_amplitudes(self, df, deri, hbar, maxiter=None, rconv=None):
        """Spin-orbital perturbed CCSD amplitudes ``dt/dF`` (SO Jacobian; SO residual has no 0.5 /
        no final symmetrization, matching ``ccresponse.solve_right``).  ``maxiter``/``rconv`` default
        to the wavefunction's convergence (``ccwfn.maxiter``/``r_conv``)."""
        from .utils import helper_diis
        cc = self.ccwfn
        maxiter = cc.maxiter if maxiter is None else maxiter
        rconv = cc.r_conv if rconv is None else rconv
        Dia, Dijab = cc.Dia, cc.Dijab
        saveERI = cc.H.ERI
        cc.H.ERI = deri
        try:
            B1, B2 = cc.residuals(df, cc.t1, cc.t2)
        finally:
            cc.H.ERI = saveERI
        B1, B2 = np.asarray(B1), np.asarray(B2)
        X1, X2 = B1 / Dia, B2 / Dijab
        diis = helper_diis(X1, X2, 8)
        for _ in range(maxiter):
            j1, j2 = self._so_ccsd_jacobian(X1, X2, hbar)
            r1, r2 = B1 + j1, B2 + j2
            X1 = X1 + r1 / Dia
            X2 = X2 + r2 / Dijab
            if np.sqrt(np.sum((r1 / Dia) ** 2) + np.sum((r2 / Dijab) ** 2)) < rconv:
                break
            diis.add_error_vector(X1, X2)
            X1, X2 = diis.extrapolate(X1, X2)
        return X1, X2

    def _perturbed_hbar(self, df, deri, dL, dt1, dt2, hbar):
        """Spatial (closed-shell RHF) perturbed HBAR ``dHBAR``: the term-by-term product-rule
        derivative of the :meth:`cchbar.build_H*` builders along
        ``(F+df, ERI+deri, L+dL, t1+dt1, t2+dt2)``.  Computed in dependency order (``Hov``,
        ``Hvvvv``, ``Hoooo`` feed ``Hvvvo``/``Hovoo``).  Needed for the perturbed-Lambda
        inhomogeneity; verified to ~1e-15 against a complex-step derivative of the HBAR builders."""
        cc = self.ccwfn
        o, v = cc.o, cc.v
        c = self.contract
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI); L0 = np.asarray(cc.H.L)
        t1 = np.asarray(cc.t1); t2 = np.asarray(cc.t2)
        def p2(sub, A, dA, B, dB): return c(sub, dA, B) + c(sub, A, dB)
        tau = t2 + c('ia,jb->ijab', t1, t1)
        dtau = dt2 + c('ia,jb->ijab', dt1, t1) + c('ia,jb->ijab', t1, dt1)
        Hov0 = np.asarray(hbar.build_Hov(o, v, F0, L0, t1))
        Hvvvv0 = np.asarray(hbar.build_Hvvvv(o, v, ERI0, t1, t2))
        Hoooo0 = np.asarray(hbar.build_Hoooo(o, v, ERI0, t1, t2))
        d = {}
        d['Hov'] = df[o, v] + p2('nf,mnef->me', t1, dt1, L0[o, o, v, v], dL[o, o, v, v])
        d['Hvv'] = (df[v, v] - p2('me,ma->ae', F0[o, v], df[o, v], t1, dt1)
                    + p2('mf,amef->ae', t1, dt1, L0[v, o, v, v], dL[v, o, v, v])
                    - p2('mnfa,mnfe->ae', tau, dtau, L0[o, o, v, v], dL[o, o, v, v]))
        d['Hoo'] = (df[o, o] + p2('ie,me->mi', t1, dt1, F0[o, v], df[o, v])
                    + p2('ne,mnie->mi', t1, dt1, L0[o, o, o, v], dL[o, o, o, v])
                    + p2('inef,mnef->mi', tau, dtau, L0[o, o, v, v], dL[o, o, v, v]))
        tmp = p2('je,mnie->mnij', t1, dt1, ERI0[o, o, o, v], deri[o, o, o, v])
        d['Hoooo'] = (deri[o, o, o, o] + (tmp + tmp.swapaxes(0, 1).swapaxes(2, 3))
                      + p2('ijef,mnef->mnij', tau, dtau, ERI0[o, o, v, v], deri[o, o, v, v]))
        tmp = p2('mb,amef->abef', t1, dt1, ERI0[v, o, v, v], deri[v, o, v, v])
        d['Hvvvv'] = (deri[v, v, v, v] - (tmp + tmp.swapaxes(0, 1).swapaxes(2, 3))
                      + p2('mnab,mnef->abef', tau, dtau, ERI0[o, o, v, v], deri[o, o, v, v]))
        d['Hvovv'] = deri[v, o, v, v] - p2('na,nmef->amef', t1, dt1, ERI0[o, o, v, v], deri[o, o, v, v])
        d['Hooov'] = deri[o, o, o, v] + p2('if,nmef->mnie', t1, dt1, ERI0[o, o, v, v], deri[o, o, v, v])
        d['Hovvo'] = (deri[o, v, v, o] + p2('jf,mbef->mbej', t1, dt1, ERI0[o, v, v, v], deri[o, v, v, v])
                      - p2('nb,mnej->mbej', t1, dt1, ERI0[o, o, v, o], deri[o, o, v, o])
                      - p2('jnfb,mnef->mbej', tau, dtau, ERI0[o, o, v, v], deri[o, o, v, v])
                      + p2('njfb,mnef->mbej', t2, dt2, L0[o, o, v, v], dL[o, o, v, v]))
        d['Hovov'] = (deri[o, v, o, v] + p2('jf,bmef->mbje', t1, dt1, ERI0[v, o, v, v], deri[v, o, v, v])
                      - p2('nb,mnje->mbje', t1, dt1, ERI0[o, o, o, v], deri[o, o, o, v])
                      - p2('jnfb,nmef->mbje', tau, dtau, ERI0[o, o, v, v], deri[o, o, v, v]))
        # Hvvvo (reuses Hov, Hvvvv and two tmp intermediates)
        w1 = ERI0[v, o, v, o] - c('infa,mnfe->amei', t2, ERI0[o, o, v, v])
        dw1 = deri[v, o, v, o] - p2('infa,mnfe->amei', t2, dt2, ERI0[o, o, v, v], deri[o, o, v, v])
        w2 = (ERI0[v, o, o, v] - c('infb,mnef->bmie', t2, ERI0[o, o, v, v])
              + c('nifb,mnef->bmie', t2, L0[o, o, v, v]))
        dw2 = (deri[v, o, o, v] - p2('infb,mnef->bmie', t2, dt2, ERI0[o, o, v, v], deri[o, o, v, v])
               + p2('nifb,mnef->bmie', t2, dt2, L0[o, o, v, v], dL[o, o, v, v]))
        d['Hvvvo'] = (deri[v, v, v, o]
            - p2('me,miab->abei', Hov0, d['Hov'], t2, dt2)
            + p2('if,abef->abei', t1, dt1, Hvvvv0, d['Hvvvv'])
            + p2('mnab,mnei->abei', tau, dtau, ERI0[o, o, v, o], deri[o, o, v, o])
            - p2('imfa,bmfe->abei', t2, dt2, ERI0[v, o, v, v], deri[v, o, v, v])
            - p2('imfb,amef->abei', t2, dt2, ERI0[v, o, v, v], deri[v, o, v, v])
            + p2('mifb,amef->abei', t2, dt2, L0[v, o, v, v], dL[v, o, v, v])
            - (c('mb,amei->abei', dt1, w1) + c('mb,amei->abei', t1, dw1))
            - (c('ma,bmie->abei', dt1, w2) + c('ma,bmie->abei', t1, dw2)))
        # Hovoo (reuses Hov, Hoooo and two tmp intermediates)
        u1 = ERI0[o, v, o, v] - c('infb,mnfe->mbie', t2, ERI0[o, o, v, v])
        du1 = deri[o, v, o, v] - p2('infb,mnfe->mbie', t2, dt2, ERI0[o, o, v, v], deri[o, o, v, v])
        u2 = (ERI0[v, o, o, v] - c('jnfb,mnef->bmje', t2, ERI0[o, o, v, v])
              + c('njfb,mnef->bmje', t2, L0[o, o, v, v]))
        du2 = (deri[v, o, o, v] - p2('jnfb,mnef->bmje', t2, dt2, ERI0[o, o, v, v], deri[o, o, v, v])
               + p2('njfb,mnef->bmje', t2, dt2, L0[o, o, v, v], dL[o, o, v, v]))
        d['Hovoo'] = (deri[o, v, o, o]
            + p2('me,ijeb->mbij', Hov0, d['Hov'], t2, dt2)
            - p2('nb,mnij->mbij', t1, dt1, Hoooo0, d['Hoooo'])
            + p2('ijef,mbef->mbij', tau, dtau, ERI0[o, v, v, v], deri[o, v, v, v])
            - p2('ineb,nmje->mbij', t2, dt2, ERI0[o, o, o, v], deri[o, o, o, v])
            - p2('jneb,mnie->mbij', t2, dt2, ERI0[o, o, o, v], deri[o, o, o, v])
            + p2('njeb,mnie->mbij', t2, dt2, L0[o, o, o, v], dL[o, o, o, v])
            + (c('je,mbie->mbij', dt1, u1) + c('je,mbie->mbij', t1, du1))
            + (c('ie,bmje->mbij', dt1, u2) + c('ie,bmje->mbij', t1, du2)))
        return d

    def _so_perturbed_hbar(self, df, deri, dt1, dt2, hbar):
        """Spin-orbital perturbed HBAR ``dHBAR``: the term-by-term product-rule derivative
        of the :meth:`cchbar._so_build_*` block builders along ``(F+df, ERI+deri, t1+dt1, t2+dt2)``.
        Blocks are computed in dependency order (``Hov`` first; ``Hvvvo``/``Hovoo`` last, reusing the
        unperturbed ``Hov``/``Hvvvv``/``Hoooo``/``Zovov`` intermediates).  Verified to ~1e-15 against
        a complex-step derivative of the ``_so_build_*`` builders."""
        cc = self.ccwfn
        o, v = cc.o, cc.v
        c = self.contract
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI)
        t1 = np.asarray(cc.t1); t2 = np.asarray(cc.t2)
        # tau (fact1*t2 + fact2*antisym(t1 t1)) and its response
        def so_tau(f1, f2):
            return f1*t2 + f2*(c('ia,jb->ijab', t1, t1) - c('ib,ja->ijab', t1, t1))
        def d_tau(f1, f2):
            return (f1*dt2 + f2*(c('ia,jb->ijab', dt1, t1) + c('ia,jb->ijab', t1, dt1)
                                 - c('ib,ja->ijab', dt1, t1) - c('ib,ja->ijab', t1, dt1)))
        tau, dtau = so_tau(1.0, 1.0), d_tau(1.0, 1.0)
        taut, dtaut = so_tau(1.0, 0.5), d_tau(1.0, 0.5)
        # unperturbed intermediates that appear as factors
        Hov0 = np.asarray(hbar._so_build_Hov(o, v, F0, ERI0, t1))
        Hvvvv0 = np.asarray(hbar._so_build_Hvvvv(o, v, ERI0, t1, t2))
        Hoooo0 = np.asarray(hbar._so_build_Hoooo(o, v, ERI0, t1, t2))
        Zovov0 = np.asarray(hbar._so_build_Zovov(o, v, ERI0, t2))
        d = {}
        d['Hov'] = (df[o, v] + c('nf,mnef->me', dt1, ERI0[o, o, v, v])
                    + c('nf,mnef->me', t1, deri[o, o, v, v]))
        d['Hvv'] = (df[v, v]
            - 0.5*(c('me,ma->ae', df[o, v], t1) + c('me,ma->ae', F0[o, v], dt1))
            - 0.5*(c('me,ma->ae', d['Hov'], t1) + c('me,ma->ae', Hov0, dt1))
            + c('mf,amef->ae', dt1, ERI0[v, o, v, v]) + c('mf,amef->ae', t1, deri[v, o, v, v])
            - 0.5*(c('mnaf,mnef->ae', dtaut, ERI0[o, o, v, v]) + c('mnaf,mnef->ae', taut, deri[o, o, v, v])))
        d['Hoo'] = (df[o, o]
            + 0.5*(c('ie,me->mi', dt1, F0[o, v]) + c('ie,me->mi', t1, df[o, v]))
            + 0.5*(c('ie,me->mi', dt1, Hov0) + c('ie,me->mi', t1, d['Hov']))
            + c('ne,mnie->mi', dt1, ERI0[o, o, o, v]) + c('ne,mnie->mi', t1, deri[o, o, o, v])
            + 0.5*(c('inef,mnef->mi', dtaut, ERI0[o, o, v, v]) + c('inef,mnef->mi', taut, deri[o, o, v, v])))
        d['Hoooo'] = (deri[o, o, o, o]
            + (c('je,mnie->mnij', dt1, ERI0[o, o, o, v]) + c('je,mnie->mnij', t1, deri[o, o, o, v])
               - c('ie,mnje->mnij', dt1, ERI0[o, o, o, v]) - c('ie,mnje->mnij', t1, deri[o, o, o, v]))
            + 0.5*(c('ijef,mnef->mnij', dtau, ERI0[o, o, v, v]) + c('ijef,mnef->mnij', tau, deri[o, o, v, v])))
        d['Hvvvv'] = (deri[v, v, v, v]
            - (c('mb,amef->abef', dt1, ERI0[v, o, v, v]) + c('mb,amef->abef', t1, deri[v, o, v, v])
               - c('ma,bmef->abef', dt1, ERI0[v, o, v, v]) - c('ma,bmef->abef', t1, deri[v, o, v, v]))
            + 0.5*(c('mnab,mnef->abef', dtau, ERI0[o, o, v, v]) + c('mnab,mnef->abef', tau, deri[o, o, v, v])))
        d['Hvovv'] = (deri[v, o, v, v]
            - (c('na,nmef->amef', dt1, ERI0[o, o, v, v]) + c('na,nmef->amef', t1, deri[o, o, v, v])))
        d['Hooov'] = (deri[o, o, o, v]
            + (c('if,mnfe->mnie', dt1, ERI0[o, o, v, v]) + c('if,mnfe->mnie', t1, deri[o, o, v, v])))
        taul = t2 + c('ia,jb->ijab', t1, t1)                # local (non-antisym) tau in Hovvo
        dtaul = dt2 + c('ia,jb->ijab', dt1, t1) + c('ia,jb->ijab', t1, dt1)
        d['Hovvo'] = (deri[o, v, v, o]
            + c('jf,mbef->mbej', dt1, ERI0[o, v, v, v]) + c('jf,mbef->mbej', t1, deri[o, v, v, v])
            - c('nb,mnej->mbej', dt1, ERI0[o, o, v, o]) - c('nb,mnej->mbej', t1, deri[o, o, v, o])
            - c('jnfb,mnef->mbej', dtaul, ERI0[o, o, v, v]) - c('jnfb,mnef->mbej', taul, deri[o, o, v, v]))
        dZovov = (deri[o, v, o, v]
            + c('nibf,mnef->mbie', dt2, ERI0[o, o, v, v]) + c('nibf,mnef->mbie', t2, deri[o, o, v, v]))
        d['Hvvvo'] = (deri[v, v, v, o]
            - (c('me,miab->abei', d['Hov'], t2) + c('me,miab->abei', Hov0, dt2))
            + (c('if,abef->abei', dt1, Hvvvv0) + c('if,abef->abei', t1, d['Hvvvv']))
            + 0.5*(c('mnab,mnei->abei', dtau, ERI0[o, o, v, o]) + c('mnab,mnei->abei', tau, deri[o, o, v, o]))
            - (c('miaf,mbef->abei', dt2, ERI0[o, v, v, v]) + c('miaf,mbef->abei', t2, deri[o, v, v, v])
               - c('mibf,maef->abei', dt2, ERI0[o, v, v, v]) - c('mibf,maef->abei', t2, deri[o, v, v, v]))
            + (c('ma,mbie->abei', dt1, Zovov0) + c('ma,mbie->abei', t1, dZovov)
               - c('mb,maie->abei', dt1, Zovov0) - c('mb,maie->abei', t1, dZovov)))
        d['Hovoo'] = (deri[o, v, o, o]
            - (c('me,ijbe->mbij', d['Hov'], t2) + c('me,ijbe->mbij', Hov0, dt2))
            - (c('nb,mnij->mbij', dt1, Hoooo0) + c('nb,mnij->mbij', t1, d['Hoooo']))
            + 0.5*(c('ijef,mbef->mbij', dtau, ERI0[o, v, v, v]) + c('ijef,mbef->mbij', tau, deri[o, v, v, v]))
            + (c('jnbe,mnie->mbij', dt2, ERI0[o, o, o, v]) + c('jnbe,mnie->mbij', t2, deri[o, o, o, v])
               - c('inbe,mnje->mbij', dt2, ERI0[o, o, o, v]) - c('inbe,mnje->mbij', t2, deri[o, o, o, v]))
            - (c('ie,mbje->mbij', dt1, Zovov0) + c('ie,mbje->mbij', t1, dZovov)
               - c('je,mbie->mbij', dt1, Zovov0) - c('je,mbie->mbij', t1, dZovov)))
        return d

    def _perturbed_t3_intermediates(self, df, deri, dL, dt1, dt2):
        """Field response of the (T) intermediates ``d{Doo,Dvv,Dov,Goovv,Gooov,Gvvvo,S1,S2}/dF``: the
        analytic product-rule derivative ``dt3 = (dN - t3 dD)/D`` (:func:`cctriples.dt3_density`) along
        the **canonical** field path ``(t1+dt1, t2+dt2, F+df, ERI+deri, L+dL)``.  ``df`` MUST be the
        canonical perturbed Fock (``perturbed_fock(..., canonical=True)``, diagonal in oo/vv) so the
        batched diagonal-denominator (T) formula stays valid."""
        from . import cctriples
        cc = self.ccwfn
        o, v, no, nv = cc.o, cc.v, cc.no, cc.nv
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI); L0 = np.asarray(cc.H.L)
        t01 = np.asarray(cc.t1); t02 = np.asarray(cc.t2)
        return cctriples.dt3_density(o, v, no, nv, t01, t02, dt1, dt2, F0, df, ERI0, deri, L0, dL, self.contract)

    def _so_perturbed_t3_intermediates(self, df, deri, dt1, dt2):
        """Spin-orbital field response of the (T) intermediates: the analytic product-rule derivative
        :func:`cctriples.so_dt3_density` along the canonical path (no ``L``); see
        :meth:`_perturbed_t3_intermediates`."""
        from . import cctriples
        cc = self.ccwfn
        o, v, no, nv = cc.o, cc.v, cc.no, cc.nv
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI)
        t01 = np.asarray(cc.t1); t02 = np.asarray(cc.t2)
        return cctriples.so_dt3_density(o, v, no, nv, t01, t02, dt1, dt2, F0, df, ERI0, deri, self.contract)

    def _perturbed_lambda(self, df, deri, dL, dt1, dt2, hbar, lam, dS1=None, dS2=None, maxiter=None, rconv=None):
        """Perturbed Lambda ``dLambda/dF`` (iterative, linear), staying in ``cclambda`` (no
        ``Y1/Y2``).  The inhomogeneity is ``r_L`` evaluated with the perturbed HBAR + ``dL``
        (unperturbed G) plus the explicit ``dG.H``/``dG.L`` product-rule halves; the Lambda-Jacobian
        action is ``r_L(dLambda) - r_L(0)`` (``r_L`` is affine in Lambda).  Iterate
        ``dLambda += (B + Jacobian)/D`` like :meth:`cclambda.solve_lambda`.  ``maxiter``/``rconv``
        default to the wavefunction's convergence (``ccwfn.maxiter``/``r_conv``).

        CCSD(T): the perturbed (T) sources ``dS1``/``dS2`` (from :meth:`_perturbed_t3_intermediates`)
        are passed straight into ``r_L1``/``r_L2`` via their ``s1``/``s2`` arguments, so they thread
        through the same residual construction and P_ij^ab symmetrization as the unperturbed
        ``cc.S1``/``cc.S2`` do in :meth:`cclambda.solve_lambda` -- no separate correction needed."""
        from .utils import helper_diis
        cc = self.ccwfn
        maxiter = cc.maxiter if maxiter is None else maxiter
        rconv = cc.r_conv if rconv is None else rconv
        o, v = cc.o, cc.v
        c = self.contract
        Dia, Dijab = cc.Dia, cc.Dijab
        l1, l2 = np.asarray(lam.l1), np.asarray(lam.l2)
        t2 = np.asarray(cc.t2)
        L0 = np.asarray(cc.H.L)

        def rL1(La, Lb, H, Gvv, Goo, s1=None):
            return np.asarray(lam.r_L1(o, v, La, Lb, H['Hov'], H['Hvv'], H['Hoo'], H['Hovvo'],
                                       H['Hovov'], H['Hvvvo'], H['Hovoo'], H['Hvovv'], H['Hooov'], Gvv, Goo, s1=s1))
        def rL2(La, Lb, Ld, H, Gvv, Goo, s2=None):
            return np.asarray(lam.r_L2(o, v, La, Lb, Ld, H['Hov'], H['Hvv'], H['Hoo'], H['Hoooo'],
                                       H['Hvvvv'], H['Hovvo'], H['Hovov'], H['Hvvvo'], H['Hovoo'],
                                       H['Hvovv'], H['Hooov'], Gvv, Goo, s2=s2))

        dH = self._perturbed_hbar(df, deri, dL, dt1, dt2, hbar)
        H0 = {b: np.asarray(getattr(hbar, b)) for b in self._HBAR_BLOCKS}
        Goo0 = np.asarray(lam.build_Goo(t2, l2)); Gvv0 = np.asarray(lam.build_Gvv(t2, l2))
        dGoo = np.asarray(lam.build_Goo(dt2, l2)); dGvv = np.asarray(lam.build_Gvv(dt2, l2))  # l2 fixed
        B1 = rL1(l1, l2, dH, Gvv0, Goo0, s1=dS1)        # (T): dS1/dS2 thread through r_L1/r_L2 exactly
        B2 = rL2(l1, l2, dL, dH, Gvv0, Goo0, s2=dS2)    # as cc.S1/cc.S2 do unperturbed (None for CCSD)
        Hvovv0, Hooov0 = H0['Hvovv'], H0['Hooov']
        c1 = -2.0*c('ef,eifa->ia', dGvv, Hvovv0) + c('ef,eiaf->ia', dGvv, Hvovv0)
        c1 += -2.0*c('mn,mina->ia', dGoo, Hooov0) + c('mn,imna->ia', dGoo, Hooov0)
        c2p = c('ae,ijeb->ijab', dGvv, L0[o, o, v, v]) - c('mi,mjab->ijab', dGoo, L0[o, o, v, v])
        B1 = B1 + c1
        B2 = B2 + (c2p + c2p.swapaxes(0, 1).swapaxes(2, 3))
        zl1 = np.zeros_like(l1); zl2 = np.zeros_like(l2)
        zGvv = np.zeros_like(Gvv0); zGoo = np.zeros_like(Goo0)
        rL1_0 = rL1(zl1, zl2, H0, zGvv, zGoo)           # = 2*Hov (the Lambda-independent constant)
        rL2_0 = rL2(zl1, zl2, L0, H0, zGvv, zGoo)       # = SYM[L]
        dl1, dl2 = B1 / Dia, B2 / Dijab
        diis = helper_diis(dl1, dl2, 8)
        for _ in range(maxiter):
            Gvv_d = np.asarray(lam.build_Gvv(t2, dl2)); Goo_d = np.asarray(lam.build_Goo(t2, dl2))
            j1 = rL1(dl1, dl2, H0, Gvv_d, Goo_d) - rL1_0
            j2 = rL2(dl1, dl2, L0, H0, Gvv_d, Goo_d) - rL2_0
            r1 = B1 + j1
            r2 = B2 + j2
            dl1 = dl1 + r1 / Dia
            dl2 = dl2 + r2 / Dijab
            if np.sqrt(np.sum((r1 / Dia) ** 2) + np.sum((r2 / Dijab) ** 2)) < rconv:
                break
            diis.add_error_vector(dl1, dl2)
            dl1, dl2 = diis.extrapolate(dl1, dl2)
        return dl1, dl2

    def _so_perturbed_lambda(self, df, deri, dt1, dt2, hbar, lam, dS1=None, dS2=None, maxiter=None, rconv=None):
        """Spin-orbital perturbed Lambda ``dLambda/dF`` (SO r_L; inhomogeneity = r_L with perturbed
        HBAR + perturbed ERI, unperturbed G, plus the dG.H / dG.<pq||rs> product-rule halves).

        CCSD(T): the perturbed (T) sources ``dS1``/``dS2`` are passed straight into the SO
        ``_so_r_L1``/``_so_r_L2`` via their ``s1``/``s2`` arguments (no 1/2 on S2, unlike the spatial
        path), so they thread through the same residual construction as ``cc.S1``/``cc.S2`` do
        unperturbed -- no separate correction needed.  ``maxiter``/``rconv`` default to the
        wavefunction's convergence (``ccwfn.maxiter``/``r_conv``)."""
        from .utils import helper_diis
        cc = self.ccwfn
        maxiter = cc.maxiter if maxiter is None else maxiter
        rconv = cc.r_conv if rconv is None else rconv
        o, v = cc.o, cc.v
        c = self.contract
        Dia, Dijab = cc.Dia, cc.Dijab
        l1, l2 = np.asarray(lam.l1), np.asarray(lam.l2)
        t2 = np.asarray(cc.t2)
        ERI0 = np.asarray(cc.H.ERI)

        def rL1(La, Lb, H, Gvv, Goo, s1=None):
            return np.asarray(lam._so_r_L1(o, v, La, Lb, H['Hov'], H['Hvv'], H['Hoo'], H['Hovvo'],
                                           H['Hvvvo'], H['Hovoo'], H['Hvovv'], H['Hooov'], Gvv, Goo, s1=s1))
        def rL2(La, Lb, E, H, Gvv, Goo, s2=None):
            return np.asarray(lam._so_r_L2(o, v, La, Lb, E, H['Hov'], H['Hvv'], H['Hoo'], H['Hoooo'],
                                           H['Hvvvv'], H['Hovvo'], H['Hvvvo'], H['Hovoo'], H['Hvovv'],
                                           H['Hooov'], Gvv, Goo, s2=s2))

        dH = self._so_perturbed_hbar(df, deri, dt1, dt2, hbar)
        H0 = {b: np.asarray(getattr(hbar, b)) for b in self._SO_HBAR_BLOCKS}
        Goo0 = np.asarray(lam.build_Goo(t2, l2)); Gvv0 = np.asarray(lam.build_Gvv(t2, l2))
        dGoo = np.asarray(lam.build_Goo(dt2, l2)); dGvv = np.asarray(lam.build_Gvv(dt2, l2))  # l2 fixed
        B1 = rL1(l1, l2, dH, Gvv0, Goo0, s1=dS1)        # (T): dS1/dS2 thread through _so_r_L1/_so_r_L2
        B2 = rL2(l1, l2, deri, dH, Gvv0, Goo0, s2=dS2)  # exactly as cc.S1/cc.S2 do (None for CCSD)
        Hvovv0, Hooov0 = H0['Hvovv'], H0['Hooov']
        B1 = B1 - c('ef,eifa->ia', dGvv, Hvovv0) - c('mn,mina->ia', dGoo, Hooov0)
        oovv = ERI0[o, o, v, v]
        B2 = B2 + (c('be,ijae->ijab', dGvv, oovv) - c('ae,ijbe->ijab', dGvv, oovv)) \
                - (c('mj,imab->ijab', dGoo, oovv) - c('mi,jmab->ijab', dGoo, oovv))
        zl1 = np.zeros_like(l1); zl2 = np.zeros_like(l2)
        zGvv = np.zeros_like(Gvv0); zGoo = np.zeros_like(Goo0)
        rL1_0 = rL1(zl1, zl2, H0, zGvv, zGoo)
        rL2_0 = rL2(zl1, zl2, ERI0, H0, zGvv, zGoo)
        dl1, dl2 = B1 / Dia, B2 / Dijab
        diis = helper_diis(dl1, dl2, 8)
        for _ in range(maxiter):
            Gvv_d = np.asarray(lam.build_Gvv(t2, dl2)); Goo_d = np.asarray(lam.build_Goo(t2, dl2))
            j1 = rL1(dl1, dl2, H0, Gvv_d, Goo_d) - rL1_0
            j2 = rL2(dl1, dl2, ERI0, H0, Gvv_d, Goo_d) - rL2_0
            r1, r2 = B1 + j1, B2 + j2
            dl1 = dl1 + r1 / Dia
            dl2 = dl2 + r2 / Dijab
            if np.sqrt(np.sum((r1 / Dia) ** 2) + np.sum((r2 / Dijab) ** 2)) < rconv:
                break
            diis.add_error_vector(dl1, dl2)
            dl1, dl2 = diis.extrapolate(dl1, dl2)
        return dl1, dl2

    def _perturbed_correlation_densities(self, dt1, dt2, dl1, dl2, lam, dt3=None):
        """Field response ``(dD, dGamma)`` of the unrelaxed CC correlation densities: the analytic
        product-rule derivative of the density builders, dispatched by orbital basis to
        :meth:`_so_perturbed_correlation_densities` or
        :meth:`_spatial_perturbed_correlation_densities`.  For CCSD(T), ``dt3`` (from
        :meth:`_perturbed_t3_intermediates`) supplies the perturbed (T) 1-/2-PDM increments, added
        with the same blocks and symmetrization the unperturbed builders use."""
        cc = self.ccwfn
        if cc.orbital_basis == 'spinorbital':
            return self._so_perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3)
        return self._spatial_perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3)

    def _so_perturbed_correlation_densities(self, dt1, dt2, dl1, dl2, lam, dt3=None):
        """Analytic field response ``(dD, dGamma)`` of the unrelaxed **spin-orbital CCSD**
        correlation densities: the product-rule derivative of the raw density builders
        (:meth:`ccdensity.build_Doo`/``Dvv``/``Dvo``/``Dov`` and :meth:`ccdensity.build_so_twopdm`),
        i.e. Appendix C of ``docs/cc_gradients_orbital_response.tex``.  Replaces the 5-point stencil
        for this path; verified to ~1e-15 against a complex-step derivative of those builders.

        The nine two-particle block derivatives are placed into the full ``nmo^4`` array and bra-ket
        symmetrized exactly as :meth:`ccdensity.gradient_densities` / :meth:`ccdensity._so_full_twopdm`
        do (overall ``1/4``, ``1/2 (Gamma + Gamma_rspq)``), so the result matches the stenciled
        convention block for block.  Frozen-core rows/columns stay zero (active ``o``/``v`` slices).

        For CCSD(T), ``dt3`` (from :meth:`_so_perturbed_t3_intermediates`) supplies the perturbed
        (T) increments, which enter the density builders additively; they are added to the matching
        derivative blocks before placement (``Goovv->ijab``, ``Gooov->ijka``, ``Gvovv->ciab``,
        ``Govoo->kaij``, ``Gvvvo->abci``, and ``Doo``/``Dvv``/``Dov`` on the 1-PDM)."""
        cc = self.ccwfn
        o, v, nmo = cc.o, cc.v, cc.nmo
        c = self.contract
        t1 = np.asarray(cc.t1); t2 = np.asarray(cc.t2)
        l1 = np.asarray(lam.l1); l2 = np.asarray(lam.l2)
        # recurring intermediates and their responses (product rule)
        tau = t2 + c('ia,jb->ijab', t1, t1) - c('ja,ib->ijab', t1, t1)
        dtau = (dt2 + c('ia,jb->ijab', dt1, t1) + c('ia,jb->ijab', t1, dt1)
                - c('ja,ib->ijab', dt1, t1) - c('ja,ib->ijab', t1, dt1))
        X = t2 + 2.0*c('ie,ma->miae', t1, t1)
        dX = dt2 + 2.0*(c('ie,ma->miae', dt1, t1) + c('ie,ma->miae', t1, dt1))
        # ---- one-particle derivative blocks ----
        dDoo = (-(c('ie,je->ij', dt1, l1) + c('ie,je->ij', t1, dl1))
                - 0.5*(c('imef,jmef->ij', dt2, l2) + c('imef,jmef->ij', t2, dl2)))
        dDvv = ((c('mb,ma->ab', dt1, l1) + c('mb,ma->ab', t1, dl1))
                + 0.5*(c('mnbe,mnae->ab', dt2, l2) + c('mnbe,mnae->ab', t2, dl2)))
        dDvo = dl1.T
        dDov = (dt1
                + c('me,imae->ia', dl1, t2) + c('me,imae->ia', l1, dt2)
                - c('me,ie,ma->ia', dl1, t1, t1) - c('me,ie,ma->ia', l1, dt1, t1) - c('me,ie,ma->ia', l1, t1, dt1)
                - 0.5*(c('mnef,ie,mnaf->ia', dl2, t1, t2) + c('mnef,ma,inef->ia', dl2, t1, t2))
                - 0.5*(c('mnef,ie,mnaf->ia', l2, dt1, t2) + c('mnef,ie,mnaf->ia', l2, t1, dt2)
                       + c('mnef,ma,inef->ia', l2, dt1, t2) + c('mnef,ma,inef->ia', l2, t1, dt2)))
        # ---- two-particle derivative blocks (product rule of build_so_twopdm) ----
        Pij = lambda Y: Y - Y.swapaxes(0, 1)
        Pab = lambda Y: Y - Y.swapaxes(2, 3)
        Pijab = lambda Y: Pij(Pab(Y))
        dG = {}
        dG['ijkl'] = 0.5*(c('ijef,klef->ijkl', dtau, l2) + c('ijef,klef->ijkl', tau, dl2))
        dG['abcd'] = 0.5*(c('mncd,mnab->abcd', dtau, l2) + c('mncd,mnab->abcd', tau, dl2))
        dG['ijka'] = (
            -(c('ke,ijea->ijka', dl1, tau) + c('ke,ijea->ijka', l1, dtau))
            + 0.5*(c('kmef,ijef,ma->ijka', dl2, tau, t1) + c('kmef,ijef,ma->ijka', l2, dtau, t1)
                   + c('kmef,ijef,ma->ijka', l2, tau, dt1))
            + Pij(c('mkef,imae,jf->ijka', dl2, t2, t1) + c('mkef,imae,jf->ijka', l2, dt2, t1)
                  + c('mkef,imae,jf->ijka', l2, t2, dt1))
            - 0.5*Pij(c('kmef,imef,ja->ijka', dl2, t2, t1) + c('kmef,imef,ja->ijka', l2, dt2, t1)
                      + c('kmef,imef,ja->ijka', l2, t2, dt1)))
        dG['ciab'] = (
            c('mc,miab->ciab', dl1, tau) + c('mc,miab->ciab', l1, dtau)
            - 0.5*(c('mnce,mnab,ie->ciab', dl2, tau, t1) + c('mnce,mnab,ie->ciab', l2, dtau, t1)
                   + c('mnce,mnab,ie->ciab', l2, tau, dt1))
            - Pab(c('mnce,inae,mb->ciab', dl2, t2, t1) + c('mnce,inae,mb->ciab', l2, dt2, t1)
                  + c('mnce,inae,mb->ciab', l2, t2, dt1))
            + 0.5*Pab(c('mnce,mnae,ib->ciab', dl2, t2, t1) + c('mnce,mnae,ib->ciab', l2, dt2, t1)
                      + c('mnce,mnae,ib->ciab', l2, t2, dt1)))
        dG['abci'] = c('miab,mc->abci', dl2, t1) + c('miab,mc->abci', l2, dt1)
        dG['kaij'] = -(c('ijea,ke->kaij', dl2, t1) + c('ijea,ke->kaij', l2, dt1))
        dG['ibaj'] = (
            c('ia,jb->ibaj', dt1, l1) + c('ia,jb->ibaj', t1, dl1)
            + c('jmbe,miea->ibaj', dl2, t2) + c('jmbe,miea->ibaj', l2, dt2)
            - c('jmbe,ma,ie->ibaj', dl2, t1, t1) - c('jmbe,ma,ie->ibaj', l2, dt1, t1)
            - c('jmbe,ma,ie->ibaj', l2, t1, dt1))
        dG['ijab'] = (
            dtau
            + 0.25*(c('mnef,ijef,mnab->ijab', dl2, tau, tau) + c('mnef,ijef,mnab->ijab', l2, dtau, tau)
                    + c('mnef,ijef,mnab->ijab', l2, tau, dtau))
            - 0.5*Pij(c('mnef,inef,mjab->ijab', dl2, t2, tau) + c('mnef,inef,mjab->ijab', l2, dt2, tau)
                      + c('mnef,inef,mjab->ijab', l2, t2, dtau))
            - Pij(c('me,mjab,ie->ijab', dl1, tau, t1) + c('me,mjab,ie->ijab', l1, dtau, t1)
                  + c('me,mjab,ie->ijab', l1, tau, dt1))
            - 0.5*Pab(c('mnef,mnaf,ijeb->ijab', dl2, t2, tau) + c('mnef,mnaf,ijeb->ijab', l2, dt2, tau)
                      + c('mnef,mnaf,ijeb->ijab', l2, t2, dtau))
            - Pab(c('me,ijeb,ma->ijab', dl1, tau, t1) + c('me,ijeb,ma->ijab', l1, dtau, t1)
                  + c('me,ijeb,ma->ijab', l1, tau, dt1))
            - 0.5*Pijab(c('miae,mnef,jnbf->ijab', dX, l2, t2) + c('miae,mnef,jnbf->ijab', X, dl2, t2)
                        + c('miae,mnef,jnbf->ijab', X, l2, dt2))
            - Pijab(c('miae,me,jb->ijab', dX, l1, t1) + c('miae,me,jb->ijab', X, dl1, t1)
                    + c('miae,me,jb->ijab', X, l1, dt1))
            + 3.0*Pijab(c('me,ia,je,mb->ijab', dl1, t1, t1, t1) + c('me,ia,je,mb->ijab', l1, dt1, t1, t1)
                        + c('me,ia,je,mb->ijab', l1, t1, dt1, t1) + c('me,ia,je,mb->ijab', l1, t1, t1, dt1)))
        dG['abij'] = dl2.transpose(2, 3, 0, 1)
        if dt3 is not None:                              # (T) increments (additive; see build_so_twopdm)
            dDoo = dDoo + dt3['Doo']; dDvv = dDvv + dt3['Dvv']; dDov = dDov + dt3['Dov']
            dG['ijab'] = dG['ijab'] + dt3['Goovv']
            dG['ijka'] = dG['ijka'] + dt3['Gooov']
            dG['ciab'] = dG['ciab'] + dt3['Gvovv']
            dG['kaij'] = dG['kaij'] + dt3['Govoo']
            dG['abci'] = dG['abci'] + dt3['Gvvvo']
        # ---- placement + symmetrization (mirrors ccdensity._so_full_twopdm + gradient_densities) ----
        dGam = np.zeros((nmo, nmo, nmo, nmo))
        dGam[o, o, o, o] = dG['ijkl']
        dGam[v, v, v, v] = dG['abcd']
        dGam[o, o, v, v] = dG['ijab']; dGam[v, v, o, o] = dG['abij']
        dGam[o, o, o, v] = dG['ijka']; dGam[o, o, v, o] = -dG['ijka'].transpose(0, 1, 3, 2)
        dGam[o, v, o, o] = dG['kaij']; dGam[v, o, o, o] = -dG['kaij'].transpose(1, 0, 2, 3)
        dGam[v, v, v, o] = dG['abci']; dGam[v, v, o, v] = -dG['abci'].transpose(0, 1, 3, 2)
        dGam[v, o, v, v] = dG['ciab']; dGam[o, v, v, v] = -dG['ciab'].transpose(1, 0, 2, 3)
        ib = dG['ibaj']
        dGam[o, v, v, o] = ib
        dGam[v, o, v, o] = -ib.transpose(1, 0, 2, 3)
        dGam[o, v, o, v] = -ib.transpose(0, 1, 3, 2)
        dGam[v, o, o, v] = ib.transpose(1, 0, 3, 2)
        dGam = 0.5 * (dGam + dGam.transpose(2, 3, 0, 1))
        dD = np.zeros((nmo, nmo))
        dD[o, o] = dDoo; dD[v, v] = dDvv; dD[o, v] = dDov; dD[v, o] = dDvo
        return dD, 0.25 * dGam

    def _spatial_perturbed_correlation_densities(self, dt1, dt2, dl1, dl2, lam, dt3=None):
        """Analytic field response ``(dD, dGamma)`` of the unrelaxed **spatial (closed-shell RHF)
        CCSD** correlation densities: the product-rule derivative of the ccdensity builders
        (:meth:`ccdensity.build_Doo`/``Dvv``/``Dvo``/``Dov`` and ``Doooo``/``Dvvvv``/``Dooov``/
        ``Dvvvo``/``Dovov``/``Doovv``), placed and symmetrized exactly as
        :meth:`ccdensity.gradient_densities` (blocks with the same per-block factors, then the
        four-fold ``1/4(G + G_qpsr + G_rspq + G_srqp)``).  Replaces the 5-point stencil for this
        path; verified to ~1e-15 against a complex-step derivative of the builders.  Frozen-core
        rows/columns stay zero (active ``o``/``v`` slices).

        For CCSD(T), ``dt3`` (from :meth:`_perturbed_t3_intermediates`) supplies the perturbed (T)
        increments, added to the matching derivative blocks before placement (``Gooov->Dooov``,
        ``Gvvvo->Dvvvo``, ``Goovv->Doovv``, and ``Doo``/``Dvv``/``Dov`` on the 1-PDM); the spatial
        route has no ``Gvovv``/``Govoo``."""
        cc = self.ccwfn
        o, v, nmo = cc.o, cc.v, cc.nmo
        c = self.contract
        t1 = np.asarray(cc.t1); t2 = np.asarray(cc.t2)
        l1 = np.asarray(lam.l1); l2 = np.asarray(lam.l2)
        def p2(sub, A, dA, B, dB):           # product rule for a two-factor contraction
            return c(sub, dA, B) + c(sub, A, dB)
        # shared intermediates and their responses
        tau = t2 + c('ia,jb->ijab', t1, t1)
        dtau = dt2 + c('ia,jb->ijab', dt1, t1) + c('ia,jb->ijab', t1, dt1)
        Goo, dGoo = c('mjab,ijab->mi', t2, l2), p2('mjab,ijab->mi', t2, dt2, l2, dl2)
        Gvv, dGvv = -c('ijeb,ijab->ae', t2, l2), -p2('ijeb,ijab->ae', t2, dt2, l2, dl2)
        tsp, dtsp = 2.0*tau - tau.swapaxes(2, 3), 2.0*dtau - dtau.swapaxes(2, 3)   # tau_spinad
        # ---- one-particle blocks ----
        dDoo = -(p2('ie,je->ij', t1, dt1, l1, dl1)) - (p2('imef,jmef->ij', t2, dt2, l2, dl2))
        dDvv = (p2('mb,ma->ab', t1, dt1, l1, dl1)) + (p2('mnbe,mnae->ab', t2, dt2, l2, dl2))
        dDvo = dl1.T
        G1, dG1 = c('mnef,inef->mi', l2, t2), p2('mnef,inef->mi', l2, dl2, t2, dt2)
        G2, dG2 = c('mnef,mnaf->ea', l2, t2), p2('mnef,mnaf->ea', l2, dl2, t2, dt2)
        dDov = (2.0*dt1
                + 2.0*p2('me,imae->ia', l1, dl1, t2, dt2)
                - p2('me,miae->ia', l1, dl1, tau, dtau)
                - p2('mi,ma->ia', G1, dG1, t1, dt1)
                - p2('ea,ie->ia', G2, dG2, t1, dt1))
        # ---- two-particle blocks ----
        dDoooo = p2('ijef,klef->ijkl', tau, dtau, l2, dl2)
        dDvvvv = p2('mnab,mncd->abcd', tau, dtau, l2, dl2)
        dDovov = (-p2('ia,jb->iajb', t1, dt1, l1, dl1)
                  - p2('mibe,jmea->iajb', tau, dtau, l2, dl2)
                  - p2('imbe,mjea->iajb', t2, dt2, l2, dl2))
        # Dooov
        tmpB, dtmpB = c('jmaf,kmef->jake', t2, l2), p2('jmaf,kmef->jake', t2, dt2, l2, dl2)
        tmpC, dtmpC = c('ijef,kmef->ijkm', t2, l2), p2('ijef,kmef->ijkm', t2, dt2, l2, dl2)
        tmpD, dtmpD = c('mjaf,kmef->jake', t2, l2), p2('mjaf,kmef->jake', t2, dt2, l2, dl2)
        tmpE, dtmpE = c('imea,kmef->iakf', t2, l2), p2('imea,kmef->iakf', t2, dt2, l2, dl2)
        F1, dF1 = c('kmef,jf->kmej', l2, t1), p2('kmef,jf->kmej', l2, dl2, t1, dt1)
        F2, dF2 = c('kmej,ie->kmij', F1, t1), p2('kmej,ie->kmij', F1, dF1, t1, dt1)
        dDooov = (-p2('ke,ijea->ijka', l1, dl1, tsp, dtsp)
                  - p2('ie,jkae->ijka', t1, dt1, l2, dl2)
                  - 2.0*p2('ik,ja->ijka', Goo, dGoo, t1, dt1)
                  + p2('jk,ia->ijka', Goo, dGoo, t1, dt1)
                  - 2.0*p2('jake,ie->ijka', tmpB, dtmpB, t1, dt1)
                  + p2('iake,je->ijka', tmpB, dtmpB, t1, dt1)
                  + p2('ijkm,ma->ijka', tmpC, dtmpC, t1, dt1)
                  + p2('jake,ie->ijka', tmpD, dtmpD, t1, dt1)
                  + p2('iakf,jf->ijka', tmpE, dtmpE, t1, dt1)
                  + p2('kmij,ma->ijka', F2, dF2, t1, dt1))
        # Dvvvo
        vB, dvB = c('imbe,nmce->ibnc', t2, l2), p2('imbe,nmce->ibnc', t2, dt2, l2, dl2)
        vC, dvC = c('nmab,nmce->abce', t2, l2), p2('nmab,nmce->abce', t2, dt2, l2, dl2)
        vD, dvD = c('niae,nmce->iamc', t2, l2), p2('niae,nmce->iamc', t2, dt2, l2, dl2)
        vE, dvE = c('mibe,nmce->ibnc', t2, l2), p2('mibe,nmce->ibnc', t2, dt2, l2, dl2)
        vF1, dvF1 = c('nmce,ie->nmci', l2, t1), p2('nmce,ie->nmci', l2, dl2, t1, dt1)
        vF2, dvF2 = c('nmci,na->amci', vF1, t1), p2('nmci,na->amci', vF1, dvF1, t1, dt1)
        dDvvvo = (p2('mc,miab->abci', l1, dl1, tsp, dtsp)
                  + p2('ma,imbc->abci', t1, dt1, l2, dl2)
                  - 2.0*p2('ca,ib->abci', Gvv, dGvv, t1, dt1)
                  + p2('cb,ia->abci', Gvv, dGvv, t1, dt1)
                  + 2.0*p2('ibnc,na->abci', vB, dvB, t1, dt1)
                  - p2('ianc,nb->abci', vB, dvB, t1, dt1)
                  - p2('abce,ie->abci', vC, dvC, t1, dt1)
                  - p2('iamc,mb->abci', vD, dvD, t1, dt1)
                  - p2('ibnc,na->abci', vE, dvE, t1, dt1)
                  - p2('amci,mb->abci', vF2, dvF2, t1, dt1))
        # Doovv
        T, dT = 2.0*t2 - t2.swapaxes(2, 3), 2.0*dt2 - dt2.swapaxes(2, 3)
        P, dP = 2.0*c('me,jmbe->jb', l1, T), 2.0*p2('me,jmbe->jb', l1, dl1, T, dT)
        Q, dQ = 2.0*c('ijeb,me->ijmb', T, l1), 2.0*p2('ijeb,me->ijmb', T, dT, l1, dl1)
        R, dR = 2.0*c('jmba,me->jeba', tsp, l1), 2.0*p2('jmba,me->jeba', tsp, dtsp, l1, dl1)
        Ooo, dOoo = c('ijef,mnef->ijmn', t2, l2), p2('ijef,mnef->ijmn', t2, dt2, l2, dl2)
        b1, db1 = c('njbf,mnef->jbme', t2, l2), p2('njbf,mnef->jbme', t2, dt2, l2, dl2)
        c1, dc1 = c('imfb,mnef->ibne', t2, l2), p2('imfb,mnef->ibne', t2, dt2, l2, dl2)
        d1, dd1 = c('inaf,mnef->iame', t2, l2), p2('inaf,mnef->iame', t2, dt2, l2, dl2)
        dDoovv = (4.0*p2('ia,jb->ijab', t1, dt1, l1, dl1)
                  + 2.0*dtsp + dl2
                  + 2.0*p2('jb,ia->ijab', P, dP, t1, dt1)
                  - p2('ja,ib->ijab', P, dP, t1, dt1)
                  - p2('ijmb,ma->ijab', Q, dQ, t1, dt1)
                  - p2('jeba,ie->ijab', R, dR, t1, dt1)
                  + 4.0*p2('imae,mjeb->ijab', t2, dt2, l2, dl2)
                  - 2.0*p2('mjbe,imae->ijab', tau, dtau, l2, dl2)
                  + p2('ijmn,mnab->ijab', Ooo, dOoo, t2, dt2)
                  + p2('jbme,miae->ijab', b1, db1, t2, dt2)
                  + p2('ibne,njae->ijab', c1, dc1, t2, dt2)
                  + 4.0*p2('eb,ijae->ijab', Gvv, dGvv, tau, dtau)
                  - 2.0*p2('ea,ijbe->ijab', Gvv, dGvv, tau, dtau)
                  - 4.0*p2('jm,imab->ijab', Goo, dGoo, tau, dtau)
                  + 2.0*p2('jm,imba->ijab', Goo, dGoo, tau, dtau)
                  - 4.0*p2('iame,mjbe->ijab', d1, dd1, tau, dtau)
                  + 2.0*p2('ibme,mjae->ijab', d1, dd1, tau, dtau)
                  + 4.0*p2('jbme,imae->ijab', d1, dd1, t2, dt2)
                  - 2.0*p2('jame,imbe->ijab', d1, dd1, t2, dt2))
        # six triple-nested t1 blocks + the final all-t1 block
        n1, dn1 = c('nb,ijmn->ijmb', t1, Ooo), p2('nb,ijmn->ijmb', t1, dt1, Ooo, dOoo)
        dDoovv = dDoovv + p2('ma,ijmb->ijab', t1, dt1, n1, dn1)
        m2a, dm2a = c('ie,mnef->mnif', t1, l2), p2('ie,mnef->mnif', t1, dt1, l2, dl2)
        m2b, dm2b = c('jf,mnif->mnij', t1, m2a), p2('jf,mnif->mnij', t1, dt1, m2a, dm2a)
        dDoovv = dDoovv + p2('mnij,mnab->ijab', m2b, dm2b, t2, dt2)
        m3a, dm3a = c('ie,mnef->mnif', t1, l2), p2('ie,mnef->mnif', t1, dt1, l2, dl2)
        m3b, dm3b = c('mnif,njbf->mijb', m3a, t2), p2('mnif,njbf->mijb', m3a, dm3a, t2, dt2)
        dDoovv = dDoovv + p2('ma,mijb->ijab', t1, dt1, m3b, dm3b)
        m4a, dm4a = c('jf,mnef->mnej', t1, l2), p2('jf,mnef->mnej', t1, dt1, l2, dl2)
        m4b, dm4b = c('mnej,miae->njia', m4a, t2), p2('mnej,miae->njia', m4a, dm4a, t2, dt2)
        dDoovv = dDoovv + p2('nb,njia->ijab', t1, dt1, m4b, dm4b)
        m5a, dm5a = c('je,mnef->mnjf', t1, l2), p2('je,mnef->mnjf', t1, dt1, l2, dl2)
        m5b, dm5b = c('mnjf,imfb->njib', m5a, t2), p2('mnjf,imfb->njib', m5a, dm5a, t2, dt2)
        dDoovv = dDoovv + p2('na,njib->ijab', t1, dt1, m5b, dm5b)
        m6a, dm6a = c('if,mnef->mnei', t1, l2), p2('if,mnef->mnei', t1, dt1, l2, dl2)
        m6b, dm6b = c('mnei,njae->mija', m6a, t2), p2('mnei,njae->mija', m6a, dm6a, t2, dt2)
        dDoovv = dDoovv + p2('mb,mija->ijab', t1, dt1, m6b, dm6b)
        g1, dg1 = c('jf,mnef->mnej', t1, l2), p2('jf,mnef->mnej', t1, dt1, l2, dl2)
        g2, dg2 = c('ie,mnej->mnij', t1, g1), p2('ie,mnej->mnij', t1, dt1, g1, dg1)
        g3, dg3 = c('nb,mnij->mbij', t1, g2), p2('nb,mnij->mbij', t1, dt1, g2, dg2)
        dDoovv = dDoovv + p2('ma,mbij->ijab', t1, dt1, g3, dg3)
        if dt3 is not None:                              # (T) increments (additive; see build_Dooov/Dvvvo/Doovv)
            dDoo = dDoo + dt3['Doo']; dDvv = dDvv + dt3['Dvv']; dDov = dDov + dt3['Dov']
            dDooov = dDooov + dt3['Gooov']
            dDvvvo = dDvvvo + dt3['Gvvvo']
            dDoovv = dDoovv + dt3['Goovv']
        # ---- placement + four-fold symmetrization (mirrors gradient_densities spatial path) ----
        dD = np.zeros((nmo, nmo))
        dD[o, o] = dDoo; dD[v, v] = dDvv; dD[o, v] = dDov; dD[v, o] = dDvo
        dG = np.zeros((nmo, nmo, nmo, nmo))
        dG[o, o, o, o] = 0.5 * dDoooo
        dG[v, v, v, v] = 0.5 * dDvvvv
        dG[o, v, o, v] = dDovov
        dG[o, o, v, v] = 0.25 * dDoovv
        dG[v, v, o, o] = 0.25 * dDoovv.transpose(2, 3, 0, 1)
        dG[o, o, o, v] = 0.5 * dDooov
        dG[o, v, o, o] = 0.5 * dDooov.transpose(2, 3, 0, 1)
        dG[v, v, v, o] = 0.5 * dDvvvo
        dG[v, o, v, v] = 0.5 * dDvvvo.transpose(2, 3, 0, 1)
        dG = 0.25 * (dG + dG.transpose(1, 0, 3, 2) + dG.transpose(2, 3, 0, 1) + dG.transpose(3, 2, 1, 0))
        return dD, dG
