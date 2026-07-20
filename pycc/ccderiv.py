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
    reduced densities are solved/built on first use and cached.  The analytic gradient, static
    polarizability, atomic polar tensors (length-gauge APTs), and molecular Hessian are implemented,
    for CCSD and CCSD(T), all-electron and frozen core.
    """

    def __init__(self, ccwfn: "CCwfn") -> None:
        """Bind the converged CCwfn (aliased as ``ccwfn``; the base stores it as ``wfn``) and
        initialize the lazy Lambda/reduced-density cache (see :meth:`_density`)."""
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
    # spin-orbital (UHF) paths, all-electron and frozen core, CCSD and CCSD(T).  Each _so_* method
    # follows its spatial counterpart -- the SO path mirrors the spatial with antisymmetrized
    # <pq||rs> (no L, no Hovov; Hvvvo/Hovoo via a Zovov intermediate) and the inline orbital Hessian
    # G (as in _so_relaxed_density) rather than a borrowed all-electron CPHF.

    _HBAR_BLOCKS = ('Hov', 'Hvv', 'Hoo', 'Hoooo', 'Hvvvv', 'Hvovv', 'Hooov',
                    'Hovvo', 'Hovov', 'Hvvvo', 'Hovoo')

    _SO_HBAR_BLOCKS = ('Hov', 'Hvv', 'Hoo', 'Hoooo', 'Hvvvv', 'Hvovv', 'Hooov',
                       'Hovvo', 'Hvvvo', 'Hovoo')

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        r"""CCSD **correlation** contribution to the static (omega=0) dipole polarizability (a.u.),
        shape ``(3, 3)``, the CC analog of :meth:`MPwfn.polarizability`::

            alpha_corr[a,b] = -d^2 E_corr / dF_a dF_b

        .. math::

            \alpha^\mathrm{corr}_{ab} = -\frac{\partial^2 E_\mathrm{corr}}{\partial F_a\,\partial F_b}

        Only the **asymmetric (2n+1) route** is available for CC -- differentiate the relaxed-density
        gradient a second time (a single (T)-capable formulation; see DERIVATIVES_PLAN sec 8).  The
        ``route`` argument (from the :func:`pycc.polarizability` facade) accepts only ``'2n+1'``.

        The reference (SCF) polarizability is kept separate (:meth:`HFwfn.polarizability`); the
        :func:`pycc.polarizability` facade sums nuclear (zero) + reference + this correlation part.

        Spatial closed-shell RHF and spin-orbital UHF, CCSD and CCSD(T), all-electron and frozen
        core.  CCSD is validated against a tight finite field of :meth:`relaxed_dipole` and the
        SO == spatial keystone; CCSD(T) against the energy second-derivative finite field and the
        SO == spatial keystone (SO CCSD(T) itself matched to CFOUR).

        Overrides :meth:`CorrelatedDerivs.polarizability` only to add the CC method-specific guards
        (supported model, (T) density intermediates); the shared 2n+1 assembly runs via ``super()``."""
        cc = self.ccwfn
        if cc.model.upper() not in ('CCSD', 'CCSD(T)'):
            raise NotImplementedError(f"CC polarizability: only CCSD and CCSD(T) are implemented (not {cc.model}).")
        if cc.model.upper() == 'CCSD(T)' and not hasattr(cc, 'S1'):
            raise ValueError("CCSD(T) polarizability requires the (T) density intermediates; "
                             "build the wavefunction with make_t3_density=True.")
        return super().polarizability(route)        # shared 2n+1 assembly (SO/spatial dispatch in the base)

    def dipole_derivatives(self, route: str = '2n+1-field') -> np.ndarray:
        r"""CCSD/CCSD(T) **correlation** contribution to the atomic polar tensors (nuclear dipole
        derivatives, a.u.), shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]``, the mixed
        field/nuclear analog of :meth:`polarizability`::

            d(mu_alpha)/d(X_{A,beta}) = -d^2 E_corr / dF_alpha dX_{A,beta}

        .. math::

            \frac{\partial \mu_\alpha}{\partial X_{A\beta}}
                = -\frac{\partial^2 E_\mathrm{corr}}{\partial F_\alpha\,\partial X_{A\beta}}

        ``route='2n+1-field'`` (default, 3 field solves) or ``'2n+1-nuclear'`` (``3N`` nuclear
        solves); both give the same tensor.  The nuclear ``Z_A`` and SCF reference terms are kept
        separate (:meth:`HFwfn.dipole_derivatives`) and summed with this correlation part by
        :func:`pycc.apt`.

        Spatial closed-shell RHF and spin-orbital UHF, CCSD and CCSD(T), all-electron and frozen
        core.  The (T) contribution enters entirely through the (T)-aware relaxed and perturbed
        densities the shared base already builds (no APT-specific (T) code); the spin-orbital CCSD(T)
        APT is validated against a CFOUR DIPDER oracle (``DIPDER(CCSD(T)) - DIPDER(SCF)``, see
        ``test_087_ccsdt_apt``).

        Overrides :meth:`CorrelatedDerivs.dipole_derivatives` only to add the CC method-specific
        guards (supported model, (T) density intermediates); the shared 2n+1 assembly runs via
        ``super()``."""
        cc = self.ccwfn
        if cc.model.upper() not in ('CCSD', 'CCSD(T)'):
            raise NotImplementedError(f"CC dipole derivatives: only CCSD and CCSD(T) are implemented (not {cc.model}).")
        if cc.model.upper() == 'CCSD(T)' and not hasattr(cc, 'S1'):
            raise ValueError("CCSD(T) dipole derivatives require the (T) density intermediates; "
                             "build the wavefunction with make_t3_density=True.")
        return super().dipole_derivatives(route)    # shared 2n+1 assembly (SO/spatial dispatch in the base)

    def hessian(self, route: str = '2n+1') -> np.ndarray:
        r"""CCSD/CCSD(T) **correlation** contribution to the molecular (nuclear) Hessian (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3+a, B*3+b)``, the nuclear-nuclear analog of
        :meth:`polarizability` / :meth:`dipole_derivatives`::

            H[Aa,Bb] = d^2 E_corr / dX_{Aa} dX_{Bb}

        .. math::

            H_{Aa,Bb} = \frac{\partial^2 E_\mathrm{corr}}{\partial X_{Aa}\,\partial X_{Bb}}

        ``route`` accepts only ``'2n+1'`` (``3N`` nuclear perturbed solves).  The nuclear-repulsion
        second derivative and the SCF reference Hessian are kept separate
        (:meth:`HFwfn.hessian` / :meth:`Derivatives.nuclear_repulsion2`) and summed with this
        correlation part by :func:`pycc.hessian`.

        Spatial closed-shell RHF and spin-orbital UHF, CCSD and CCSD(T), all-electron and frozen
        core.  As for the APT, the (T) contribution enters entirely through the (T)-aware relaxed and
        perturbed densities the shared base already builds (no Hessian-specific (T) code); validated
        against a CFOUR FCMFINAL oracle (``FCMFINAL(CCSD(T)) - FCMFINAL(SCF)``, see
        ``test_089_ccsdt_hessian``).

        Overrides :meth:`CorrelatedDerivs.hessian` only to add the CC method-specific guards
        (supported model, (T) density intermediates); the shared 2n+1 assembly runs via ``super()``."""
        cc = self.ccwfn
        if cc.model.upper() not in ('CCSD', 'CCSD(T)'):
            raise NotImplementedError(f"CC hessian: only CCSD and CCSD(T) are implemented (not {cc.model}).")
        if cc.model.upper() == 'CCSD(T)' and not hasattr(cc, 'S1'):
            raise ValueError("CCSD(T) hessian requires the (T) density intermediates; "
                             "build the wavefunction with make_t3_density=True.")
        return super().hessian(route)               # shared 2n+1 assembly (SO/spatial dispatch in the base)

    def _perturbed_unrelaxed_densities(self, pert, df, deri, dL):
        """CC perturbed unrelaxed densities ``(d_x gamma, d_x Gamma)`` -- the base
        :meth:`CorrelatedDerivs._perturbed_unrelaxed_densities` hook.  Runs the iterative perturbed
        amplitude solve ``dt = dt(df, deri, dL)`` and the perturbed Lambda solve ``dLambda``
        (reusing the converged ``hbar``/``lam`` from :meth:`_density`), threading the perturbed (T)
        intermediates ``dt3`` into both and into the density for CCSD(T), then builds the perturbed
        correlation densities (:meth:`_perturbed_correlation_densities`).  ``df``/``deri`` are already
        canonical per :attr:`perturbed_mo_gauge` (passed in by the base perturbed Z-vector solve)."""
        cc = self.ccwfn
        lam = self._density().cclambda
        hbar = lam.hbar
        is_t = cc.model.upper() == 'CCSD(T)'
        if cc.orbital_basis == 'spinorbital':
            dt1, dt2 = self._so_perturbed_amplitudes(df, deri, hbar)
            dt3 = self._so_perturbed_t3_intermediates(df, deri, dt1, dt2) if is_t else None
            dl1, dl2 = self._so_perturbed_lambda(df, deri, dt1, dt2, hbar, lam,
                                                 dS1=(dt3['S1'] if is_t else None),
                                                 dS2=(dt3['S2'] if is_t else None))
        else:
            dt1, dt2 = self._perturbed_amplitudes(df, deri, dL, hbar)
            dt3 = self._perturbed_t3_intermediates(df, deri, dL, dt1, dt2) if is_t else None
            dl1, dl2 = self._perturbed_lambda(df, deri, dL, dt1, dt2, hbar, lam,
                                              dS1=(dt3['S1'] if is_t else None),
                                              dS2=(dt3['S2'] if is_t else None))
        return self._perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3=dt3)

    def _ccsd_jacobian(self, X1, X2, hbar):
        r"""The CCSD Jacobian applied to an amplitude pair -- ``(HBAR . X)`` projected onto singles
        and doubles (doubles un-symmetrized), i.e. the singles/doubles action ``<mu|[HBAR, X]|0>``
        built from ``cchbar``::

            sigma1_ia   = (HBAR . X)_ia
            sigma2_ijab = (HBAR . X)_ijab

        .. math::

            \sigma^{a}_{i} = (\bar{H}\,X)^{a}_{i}, \qquad \sigma^{ab}_{ij} = (\bar{H}\,X)^{ab}_{ij}
        """
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
        r"""Spin-orbital CCSD Jacobian ``(HBAR . X)`` -- the antisymmetrized singles/doubles action
        ``<mu|[HBAR, X]|0>`` (the P(ij)P(ab) is built in, so no final symmetrization)::

            sigma1_ia   = (HBAR . X)_ia
            sigma2_ijab = (HBAR . X)_ijab

        .. math::

            \sigma^{a}_{i} = (\bar{H}\,X)^{a}_{i}, \qquad \sigma^{ab}_{ij} = (\bar{H}\,X)^{ab}_{ij}
        """
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
        r"""Perturbed CCSD amplitudes ``dt/dx`` (iterative).  Differentiating the CC amplitude
        equation ``R_mu = <mu|HBAR|0> = 0`` with respect to the perturbation ``x`` splits into the
        ``d_x t`` part (the CCSD Jacobian :meth:`_ccsd_jacobian`, ``<mu|[HBAR, d_x t]|0>``) and the
        fixed-``t`` part -- the perturbation-dependent inhomogeneity ``B^x``, the derivative of the
        **bare** Hamiltonian similarity-transformed at fixed ``t``::

            (HBAR . dt) = -B^x,   B^x_mu = <mu| e^-T (d_x H) e^T |0>

        .. math::

            \bar{H}\,\partial_x t = -B^{x}, \qquad
            B^{x}_\mu = \langle\mu|\, e^{-T}(\partial_x H)\, e^{T}\,|0\rangle

        (Note ``B^x`` differentiates only the integrals -- ``t`` is held fixed -- so it is NOT
        ``d_x HBAR``, whose ``[HBAR, d_x t]`` piece is the Jacobian LHS.)  ``B^x`` is computed by
        evaluating ``cc.residuals`` with the perturbed **bare** integrals (``df`` and the CPHF-folded
        ``deri``/``dL`` swapped into ``cc.H``, carrying the orbital relaxation), the residual formula
        supplying the ``e^-T ( ) e^T`` transform.  Iterate ``dt += (B + HBAR.dt)/D`` with DIIS.
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
        r"""Spin-orbital perturbed CCSD amplitudes ``dt/dx`` -- the spin-orbital analogue of
        :meth:`_perturbed_amplitudes` (SO Jacobian :meth:`_so_ccsd_jacobian`; the SO residual has
        no 0.5 and no final symmetrization)::

            (HBAR . dt) = -B^x,   B^x_mu = <mu| e^-T (d_x H) e^T |0>

        .. math::

            \bar{H}\,\partial_x t = -B^{x}, \qquad
            B^{x}_\mu = \langle\mu|\, e^{-T}(\partial_x H)\, e^{T}\,|0\rangle

        ``B^x`` (fixed-``t``) is ``cc.residuals`` evaluated with the perturbed **bare** integrals
        (``df``, ``deri`` swapped in).  ``maxiter``/``rconv`` default to the wavefunction's
        convergence (``ccwfn.maxiter``/``r_conv``)."""
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
        r"""Spatial (closed-shell RHF) perturbed HBAR ``dHBAR`` (``d`` = the full ``d/dx``
        derivative): the term-by-term product-rule derivative of each :meth:`cchbar.build_H*`
        builder along ``(F+df, ERI+deri, L+dL, t1+dt1, t2+dt2)``.  Each block is the product rule
        applied to its unperturbed builder (see :mod:`pycc.cchbar`); e.g. for ``Hov``::

            d Hov_me = df_me + dt1_nf L_mnef + t1_nf dL_mnef

        .. math::

            \partial_x \bar{H}_{me} = \partial_x f_{me} + \partial_x t^{f}_{n}\,L_{mnef}
                + t^{f}_{n}\,\partial_x L_{mnef}

        Computed in dependency order (``Hov``, ``Hvvvv``, ``Hoooo`` feed ``Hvvvo``/``Hovoo``).
        Needed for the perturbed-Lambda inhomogeneity; verified to ~1e-15 against a complex-step
        derivative of the HBAR builders."""
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
        r"""Spin-orbital perturbed HBAR ``dHBAR`` -- the spin-orbital analogue of
        :meth:`_perturbed_hbar`: the term-by-term product-rule derivative of each
        :meth:`cchbar._so_build_*` block builder along ``(F+df, ERI+deri, t1+dt1, t2+dt2)`` (the
        antisymmetrized ``<pq||rs>`` in place of ``L``); e.g. for ``Hov``::

            d Hov_me = df_me + dt1_nf <mn||ef> + t1_nf d<mn||ef>

        .. math::

            \partial_x \bar{H}_{me} = \partial_x f_{me} + \partial_x t^{f}_{n}\,\langle mn\Vert ef\rangle
                + t^{f}_{n}\,\partial_x \langle mn\Vert ef\rangle

        Blocks in dependency order (``Hov`` first; ``Hvvvo``/``Hovoo`` last, reusing the unperturbed
        ``Hov``/``Hvvvv``/``Hoooo``/``Zovov`` intermediates).  Verified to ~1e-15 against a
        complex-step derivative of the ``_so_build_*`` builders."""
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
        r"""First-order response of the (T) intermediates ``d{Doo,Dvv,Dov,Goovv,Gooov,Gvvvo,S1,S2}/dx``:
        the analytic product/quotient-rule derivative of the T3 amplitudes
        (:func:`cctriples.dt3_density`) along the **canonical** perturbation path
        ``(t1+dt1, t2+dt2, F+df, ERI+deri, L+dL)``::

            dt3 = (dN - t3 dD) / D

        .. math::

            \partial_x t_3 = \frac{\partial_x N - t_3\,\partial_x D}{D}

        with ``N``/``D`` the (T)-amplitude numerator / orbital-energy denominator.  ``df`` MUST be the
        canonical perturbed Fock (``perturbed_fock(..., canonical=True)``, diagonal in oo/vv) so the
        batched diagonal-denominator (T) formula stays valid."""
        from . import cctriples
        cc = self.ccwfn
        o, v, no, nv = cc.o, cc.v, cc.no, cc.nv
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI); L0 = np.asarray(cc.H.L)
        t01 = np.asarray(cc.t1); t02 = np.asarray(cc.t2)
        return cctriples.dt3_density(o, v, no, nv, t01, t02, dt1, dt2, F0, df, ERI0, deri, L0, dL, self.contract)

    def _so_perturbed_t3_intermediates(self, df, deri, dt1, dt2):
        """Spin-orbital first-order response of the (T) intermediates: the analytic product-rule
        derivative :func:`cctriples.so_dt3_density` along the canonical perturbation path (no
        ``L``); see :meth:`_perturbed_t3_intermediates`."""
        from . import cctriples
        cc = self.ccwfn
        o, v, no, nv = cc.o, cc.v, cc.no, cc.nv
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI)
        t01 = np.asarray(cc.t1); t02 = np.asarray(cc.t2)
        return cctriples.so_dt3_density(o, v, no, nv, t01, t02, dt1, dt2, F0, df, ERI0, deri, self.contract)

    def _perturbed_lambda(self, df, deri, dL, dt1, dt2, hbar, lam, dS1=None, dS2=None, maxiter=None, rconv=None):
        r"""Perturbed Lambda ``dLambda/dx`` (iterative, linear): a single inhomogeneous linear solve
        that reuses ``cclambda``'s ground-state Lambda residual ``r_L`` as the operator (no separate
        perturbed-multiplier amplitudes).  Differentiating the Lambda
        equation gives a linear equation with the same Lambda-Jacobian: its action is
        ``r_L(dLambda) - r_L(0)`` -- since ``r_L`` is affine in Lambda (linear plus a
        Lambda-independent constant), the subtraction cancels the constant and leaves the pure linear
        Jacobian action -- and an inhomogeneity ``B`` = ``r_L`` evaluated with the perturbed HBAR + ``dL`` (unperturbed
        ``G``) plus the explicit ``dG.H``/``dG.L`` product-rule halves; iterate
        ``dLambda += (B + Jacobian)/D`` like :meth:`cclambda.solve_lambda`::

            J_Lambda . dLambda = -B,   B = r_L(dHBAR, dL) + dG-halves

        .. math::

            J_\Lambda\,\partial_x \Lambda = -B

        ``maxiter``/``rconv`` default to the wavefunction's convergence (``ccwfn.maxiter``/``r_conv``).

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
        r"""Spin-orbital perturbed Lambda ``dLambda/dx`` -- the spin-orbital analogue of
        :meth:`_perturbed_lambda` (SO ``_so_r_L``; inhomogeneity = ``r_L`` with perturbed HBAR +
        perturbed ERI, unperturbed G, plus the ``dG.H`` / ``dG.<pq||rs>`` product-rule halves)::

            J_Lambda . dLambda = -B,   B = r_L(dHBAR, deri) + dG-halves

        .. math::

            J_\Lambda\,\partial_x \Lambda = -B

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
        r"""First-order response ``(dD, dGamma)`` of the unrelaxed CC correlation densities: the
        analytic product-rule (chain-rule) derivative of the density builders in the
        amplitudes/multipliers::

            dD     = (dD/dt) dt + (dD/dl) dl
            dGamma = (dGamma/dt) dt + (dGamma/dl) dl

        .. math::

            \partial_x D = \frac{\partial D}{\partial t}\,\partial_x t + \frac{\partial D}{\partial \lambda}\,\partial_x \lambda,
            \qquad \partial_x \Gamma = \frac{\partial \Gamma}{\partial t}\,\partial_x t + \frac{\partial \Gamma}{\partial \lambda}\,\partial_x \lambda

        Dispatched by orbital basis to :meth:`_so_perturbed_correlation_densities` or
        :meth:`_spatial_perturbed_correlation_densities`.  For CCSD(T), ``dt3`` (from
        :meth:`_perturbed_t3_intermediates`) supplies the perturbed (T) 1-/2-PDM increments, added
        with the same blocks and symmetrization the unperturbed builders use."""
        cc = self.ccwfn
        if cc.orbital_basis == 'spinorbital':
            return self._so_perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3)
        return self._spatial_perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam, dt3)

    def _so_perturbed_correlation_densities(self, dt1, dt2, dl1, dl2, lam, dt3=None):
        """Analytic first-order response ``(dD, dGamma)`` of the unrelaxed **spin-orbital CCSD**
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
        """Analytic first-order response ``(dD, dGamma)`` of the unrelaxed **spatial (closed-shell RHF)
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
