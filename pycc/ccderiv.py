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

if TYPE_CHECKING:
    from .ccwfn import CCwfn


class CCderiv:
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
        self.ccwfn = ccwfn
        self.contract = ccwfn.contract
        self._dens = None
        self._ref_hf = None

    def _density(self):
        """Converged Lambda amplitudes and the (Lambda-response) reduced densities, cached.
        Builds ``cchbar`` -> ``cclambda`` (solved) -> ``ccdensity`` on first use."""
        if self._dens is None:
            from .cchbar import cchbar
            from .cclambda import cclambda
            from .ccdensity import ccdensity
            hbar = cchbar(self.ccwfn)
            lam = cclambda(self.ccwfn, hbar)
            lam.solve_lambda(e_conv=1e-10, r_conv=1e-10)
            self._dens = ccdensity(self.ccwfn, lam)
        return self._dens

    def _reference_hf(self):
        """The all-electron :class:`~pycc.hfwfn.HFwfn` for the SCF reference (cached), supplying
        the reference gradient for the total CCSD property (the :func:`pycc.gradient` facade pairs
        it with this correlation gradient and the nuclear term)."""
        if self._ref_hf is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.ccwfn.ref, orbital_basis=self.ccwfn.orbital_basis)
        return self._ref_hf

    def _lagrangian(self, D, Gam):
        """The generalized-Fock (orbital-gradient) Lagrangian ``I'_pq`` from a full-MO 1-PDM ``D``
        and 2-PDM ``Gam`` -- reused verbatim from the MP2 gradient machinery (it is generic in the
        densities: ``-1/2 [ eps_p(D_pq+D_qp) + delta_{q,occ} sum_rs D_rs(<rp|sq>_L + <rq|sp>_L)
        + 4 sum_rst <pr|st> Gam_qrst ]``).  Its ov-antisymmetric part is the Z-vector RHS; evaluated
        at the relaxed density it is the energy-weighted density.  ``Gam`` must carry the proper
        2-PDM permutational symmetry (:meth:`ccdensity.gradient_densities` symmetrizes it), since
        the three-index ``termC`` is sensitive to it.  Dispatches on the orbital basis: the
        spin-orbital path uses the antisymmetrized ``_so_mp2_lagrangian`` (``<pq||rs>``), the spatial
        path the spin-adapted ``_mp2_lagrangian`` (``H.L``).  (When an MPderiv is split out this
        shared primitive will move with it.)"""
        if self.ccwfn.orbital_basis == 'spinorbital':
            return self.ccwfn.mp._so_mp2_lagrangian(D, Gam)
        return self.ccwfn.mp._mp2_lagrangian(D, Gam)

    @staticmethod
    def _dependent_pairs(Iblock, eps_block, thresh=1e-8):
        """Canonical dependent-pair rotation ``P_mn = (I'_mn - I'_nm)/(eps_m - eps_n)`` for a
        square occ-occ or virt-virt Lagrangian block ``Iblock`` and its orbital energies
        ``eps_block``.  Numerator-gated (``|Delta I'| < thresh`` -> 0), which skips both the
        diagonal (``m=n``) and near-degenerate pairs (Lee-Rendell's ``|Delta X| < 1e-8``
        threshold).  ``P`` is symmetric -- numerator and denominator are both antisymmetric.

        This is the frozen-core core<->active-occupied divide (used for the ``P_co`` block in
        :meth:`gradient`) generalized to an arbitrary square block; CCSD(T) needs it over the
        full oo and vv blocks because (T) breaks the occ-occ/virt-virt rotation invariance."""
        num = np.asarray(Iblock) - np.asarray(Iblock).T
        den = eps_block[:, None] - eps_block[None, :]
        P = np.zeros_like(num)
        m = np.abs(num) > thresh
        P[m] = num[m] / den[m]
        return P

    def _relaxed_density(self):
        """The relaxed correlation 1-PDM ``Drel`` and the symmetrized 2-PDM ``Gam`` (spatial
        closed-shell RHF).  ``Drel`` is the Lambda-response 1-PDM ``D`` plus the orbital-relaxation
        blocks: the frozen-core core<->active-occupied divide ``P_co``, the CCSD(T) oo/vv
        dependent-pair ``kappa_oo``/``kappa_vv`` (model-gated), and the ov Z-vector ``-z`` (one CPHF
        solve, RHS = the ov-antisymmetric generalized Fock ``I'_ia - I'_ai``).  This is the
        **perturbation-independent** relaxed density shared by :meth:`gradient` (contracted with the
        skeleton derivative integrals) and :meth:`relaxed_dipole` (contracted with the dipole
        integrals); the (T) response rides along in ``Drel`` for both."""
        cc = self.ccwfn
        o, v = cc.o, cc.v
        nfzc = cc.nfzc
        co = slice(0, nfzc)                              # frozen-core occupied
        ofull = slice(0, o.stop)                         # full occupied (core + active)
        c = self.contract
        eps = np.diag(np.asarray(cc.H.F))
        L = np.asarray(cc.H.L)
        D, Gam = self._density().gradient_densities()
        Ip = self._lagrangian(D, Gam)
        Drel = D.copy()
        if nfzc:                                         # core<->active-occupied: direct divide
            Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            Drel[co, o] += Pco
            Drel[o, co] += Pco.T
        X = Ip[ofull, v] - Ip[v, ofull].T               # ov-antisymmetric generalized Fock (full occ)
        if nfzc:                                         # couple P_co into the Z-vector RHS
            zjc = -Pco.T
            X = X - (c('jc,ajic->ia', zjc, L[v, o, ofull, co])
                     + c('jc,acij->ia', zjc, L[v, co, ofull, o]))
        if cc.model.upper() == 'CCSD(T)':
            # (T) breaks the occ-occ / virt-virt rotation invariance, so the canonical perturbed
            # orbitals acquire dependent-pair rotations kappa_oo/kappa_vv beyond the ov Z-vector:
            # (I'_ij-I'_ji)/(eps_i-eps_j) over the active oo pairs and the vv analog
            # (:meth:`_dependent_pairs`), added to the relaxed density and coupled into the ov
            # Z-vector RHS through the antisymmetrized ERI.  For CCSD both blocks vanish (oo/vv
            # invariance).  Frozen core: the core<->active-occupied (T) response is already carried
            # by P_co above (its I' is the (T)-inclusive Lagrangian); these blocks add the
            # active<->active oo and the vv pairs, and the ov-occupied index of the coupling runs
            # over the full occupied space (ofull), reducing to the active space when nfzc=0.
            Poo = self._dependent_pairs(Ip[o, o], eps[o])
            Pvv = self._dependent_pairs(Ip[v, v], eps[v])
            Drel[o, o] += Poo
            Drel[v, v] += Pvv
            X = X + (c('kl,akil->ia', Poo, L[v, o, ofull, o])
                     + c('bc,ibac->ia', Pvv, L[ofull, v, v, v]))
        z = self._reference_hf().cphf.solve(X)          # Z-vector: A z = X (full-occ SCF Hessian)
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z
        return Drel, Gam

    def _so_relaxed_density(self):
        """Spin-orbital (UHF) analog of :meth:`_relaxed_density`: the relaxed correlation 1-PDM
        ``Drel`` and 2-PDM ``Gam``, with the antisymmetrized-ERI Lagrangian and the inline orbital
        Hessian (``G_ia,jb = <aj||ib> + <ab||ij> + delta (eps_a - eps_i)``) rather than a borrowed
        all-electron CPHF.  **UHF only** -- raises for ROHF (the semicanonical response does not
        reproduce the restricted ROHF response)."""
        cc = self.ccwfn
        if cc.mp.cphf.is_rohf:
            raise NotImplementedError(
                "The spin-orbital CCSD/CCSD(T) gradient and relaxed dipole are not implemented for "
                "ROHF references (the semicanonical response does not reproduce the restricted ROHF "
                "response); RHF and UHF are supported.")
        o, v = cc.o, cc.v
        nfzc, nv = cc.nfzc, cc.nv
        co = slice(0, o.start)                            # frozen-core occupied (2*nfzc spin-orbitals)
        ofull = slice(0, o.stop)                          # full occupied (core + active)
        nof = o.stop
        c = self.contract
        eps = np.diag(np.asarray(cc.H.F))
        ERI = np.asarray(cc.H.ERI)                        # spin-orbital <pq||rs>
        D, Gam = self._density().gradient_densities()
        Ip = self._lagrangian(D, Gam)
        Drel = D.copy()
        if nfzc:                                          # core<->active-occupied: direct divide
            Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            Drel[co, o] += Pco
            Drel[o, co] += Pco.T
        X = Ip[ofull, v] - Ip[v, ofull].T                # ov-antisymmetric generalized Fock
        if nfzc:                                          # couple P_co into the Z-vector RHS
            zjc = -Pco.T
            X = X - (c('jc,ajic->ia', zjc, ERI[v, o, ofull, co])
                     + c('jc,acij->ia', zjc, ERI[v, co, ofull, o]))
        if cc.model.upper() == 'CCSD(T)':
            # (T) breaks the occ-occ / virt-virt rotation invariance: the canonical perturbed
            # orbitals acquire dependent-pair rotations kappa_oo/kappa_vv beyond the ov Z-vector --
            # the spin-orbital analog of the (T) branch in :meth:`_relaxed_density` (spatial L ->
            # <pq||rs>).  For CCSD both blocks vanish.
            Poo = self._dependent_pairs(Ip[o, o], eps[o])
            Pvv = self._dependent_pairs(Ip[v, v], eps[v])
            Drel[o, o] += Poo
            Drel[v, v] += Pvv
            X = X + (c('kl,akil->ia', Poo, ERI[v, o, ofull, o])
                     + c('bc,ibac->ia', Pvv, ERI[ofull, v, v, v]))
        G = (c('ajib->iajb', ERI[v, ofull, ofull, v])    # orbital Hessian, built inline
             + c('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof * nv, nof * nv)
        G[np.diag_indices(nof * nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
        z = np.linalg.solve(G, X.reshape(-1)).reshape(nof, nv)   # Z-vector A z = X
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z
        return Drel, Gam

    def gradient(self) -> np.ndarray:
        """CCSD **correlation** contribution to the analytic nuclear energy gradient (a.u.), shape
        ``(natom, 3)``, via the **Z-vector (relaxed-density) route** -- the efficient default
        (one CPHF solve, not ``3*natom``)::

            dE_corr/dX = sum_pq D~_pq f^X_pq + sum_pqrs Gamma_pqrs <pq|rs>^X + sum_pq W_pq S^X_pq

        with the relaxed 1-PDM ``D~`` (the Lambda-response ``D`` plus the orbital-relaxation
        Z-vector ``z``), the (symmetrized) 2-PDM ``Gamma``, and the energy-weighted density
        ``W = I'(D~)`` (:meth:`_lagrangian`).  The Z-vector solves ``A z = X`` once, with
        ``X = I'_ia - I'_ai`` the ov-antisymmetric generalized Fock, using the SCF orbital Hessian
        (:meth:`HFwfn.cphf`).  ``f^X``/``S^X``/``<pq|rs>^X`` are the skeleton derivative integrals
        from ``ccwfn.derivatives`` (no per-perturbation CPHF solve).

        The **reference (SCF) gradient is kept separate**: the total CCSD gradient is
        ``HFwfn(ref).gradient()`` plus this, assembled by the :func:`pycc.gradient` facade.

        Spatial (closed-shell RHF) path; all-electron and frozen-core.  Validated against
        ``psi4.gradient('ccsd')``, a finite difference of the CCSD energy, and the independent
        explicit-derivative route (:meth:`_gradient_explicit`).  The spin-orbital (UHF) path is
        dispatched to :meth:`_so_gradient`.

        Frozen core is handled as in the MP2 gradient: the correlation densities live in the active
        space while the orbital response spans the full occupied space.  The core<->active-occupied
        rotation is a direct divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`` (the SCF energy is
        invariant to occupied-occupied rotations), coupled into the Z-vector right-hand side."""
        cc = self.ccwfn
        if cc.orbital_basis == 'spinorbital':
            return self._so_gradient()
        o = cc.o
        ofull = slice(0, o.stop)                         # full occupied (core + active)
        c = self.contract
        Drel, Gam = self._relaxed_density()             # D + P_co + (T) kappa_oo/vv + ov Z-vector
        W = self._lagrangian(Drel, Gam)                 # energy-weighted density
        d = cc.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.core(atom); Sx = d.overlap(atom); ERIx = d.eri(atom)
            for cart in range(3):
                phys = ERIx[cart].transpose(0, 2, 1, 3)                 # chemist -> physicist <pq|rs>^X
                Lx = 2.0 * phys - phys.transpose(0, 1, 3, 2)
                fx = hx[cart] + c('pmqm->pq', Lx[:, ofull, :, ofull])   # skeleton Fock deriv (full occ)
                grad[atom, cart] = (c('pq,pq->', Drel, fx)
                                    + c('pqrs,pqrs->', Gam, phys)
                                    + c('pq,pq->', W, Sx[cart]))
        return grad

    def _so_gradient(self) -> np.ndarray:
        """Spin-orbital (UHF) CCSD / CCSD(T) **correlation** gradient via the Z-vector route -- the
        spin-orbital analog of :meth:`gradient`, mirroring :meth:`MPwfn._so_gradient` /
        :meth:`MPwfn._so_zvector`::

            dE_corr/dX = sum_pq D~_pq f^X_pq + sum_pqrs Gamma_pqrs <pq||rs>^X + sum_pq W_pq S^X_pq

        Differences from the spatial path: the generalized-Fock Lagrangian is the antisymmetrized
        ``_so_mp2_lagrangian`` (:meth:`_lagrangian` dispatches); the orbital Hessian is built
        **inline** (``G_ia,jb = <aj||ib> + <ab||ij> + delta (eps_a - eps_i)``) rather than borrowed
        from an all-electron ``HFwfn`` CPHF -- the all-electron spin-orbital ``HFwfn`` orders the
        spins differently from the densities, so there is no CPHF to reuse; and the skeleton
        derivative integrals are the spin-orbital ``derivatives.so_*`` (``f^X = h^X + sum_m
        <pm||qm>^X`` over the full occupied space).

        For ``model == 'CCSD(T)'`` the (T) density enters through ``gradient_densities`` and the
        (T) occ-occ/virt-virt dependent-pair rotations ``kappa_oo``/``kappa_vv`` are added exactly
        as in the spatial :meth:`gradient` (with the antisymmetrized ERI in place of ``H.L``).

        **UHF only** -- ROHF is not supported (the semicanonical response does not reproduce the
        restricted ROHF response).  Frozen-core aware (the core<->active-occupied ``P_co`` divide,
        coupled into the Z-vector RHS, exactly as the spatial path and :meth:`MPwfn._so_zvector`).
        Validated against ``psi4.gradient('ccsd')`` (UHF), the spatial closed-shell gradient (SO ==
        spatial, CCSD and CCSD(T)), a finite difference of pycc's own SO CCSD(T) energy (open-shell
        UHF), and the explicit-derivative route (:meth:`_gradient_explicit`, CCSD only -- the (T)
        dependent-pair is not yet carried there)."""
        cc = self.ccwfn
        o = cc.o
        ofull = slice(0, o.stop)                          # full occupied (core + active)
        c = self.contract
        Drel, Gam = self._so_relaxed_density()           # raises for ROHF; D + P_co + (T) kappa + z
        W = self._lagrangian(Drel, Gam)                  # energy-weighted density
        d = cc.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.so_core(atom); Sx = d.so_overlap(atom); ERIx = d.so_eri(atom)
            for cart in range(3):
                fx = hx[cart] + c('pmqm->pq', ERIx[cart][:, ofull, :, ofull])   # skeleton Fock deriv
                grad[atom, cart] = (c('pq,pq->', Drel, fx)
                                    + c('pqrs,pqrs->', Gam, ERIx[cart])
                                    + c('pq,pq->', W, Sx[cart]))
        return grad

    def relaxed_dipole(self) -> np.ndarray:
        """CCSD / CCSD(T) **correlation** contribution to the relaxed electronic dipole (a.u.),
        ``Tr(D_rel . mu)`` per Cartesian axis -- the CC analog of :meth:`MPwfn.relaxed_dipole`, and
        the correlation block that :func:`pycc.dipole` pairs with the reference (HF) and nuclear
        dipoles.

        A static electric field does not move the AO basis (``S^F = <pq|rs>^F = 0``), so the relaxed
        dipole is just the perturbation-independent relaxed density (:meth:`_relaxed_density` /
        :meth:`_so_relaxed_density`) contracted with the dipole integrals -- no energy-weighted
        density and no 2-PDM term (unlike the gradient).  The (T) density and its oo/vv
        dependent-pair orbital response ride along inside ``D_rel``, so the (T) contribution needs no
        separate handling.  Dispatches spatial vs spin-orbital on ``orbital_basis`` (ROHF raises via
        :meth:`_so_relaxed_density`)."""
        cc = self.ccwfn
        Drel, _ = (self._so_relaxed_density() if cc.orbital_basis == 'spinorbital'
                   else self._relaxed_density())
        c = self.contract
        return np.array([c('pq,pq->', Drel, np.asarray(cc.H.mu[a])) for a in range(3)])

    def _gradient_explicit(self) -> np.ndarray:
        """CCSD correlation gradient via the **explicit-derivative route** -- an independent
        cross-check of :meth:`gradient`::

            dE_corr/dX = sum_pq D_pq (d_X f)_pq + sum_pqrs Gamma_pqrs (d_X <pq|rs>)

        The Lambda-response densities are contracted with the CPHF-folded perturbed integrals
        (:meth:`CPHF.perturbed_fock` / :meth:`CPHF.perturbed_eri`), the orbital relaxation riding
        inside ``d_X f`` / ``d_X <pq|rs>`` -- one nuclear CPHF solve per perturbation (the "simple
        but inefficient" form, analog of :meth:`MPwfn._corr_gradient_explicit`).  Same result as
        :meth:`gradient` (which uses the single Z-vector solve).  Basis-agnostic: the perturbed
        integrals and densities dispatch on the orbital basis, so this cross-checks the spatial and
        spin-orbital gradients alike.

        .. note::
           **CCSD(T) caveat.** This route does not yet carry the (T) occ-occ/virt-virt
           dependent-pair orbital response (the ``kappa_oo``/``kappa_vv`` divides added to
           :meth:`gradient`), so for ``model == 'CCSD(T)'`` it is *not* equivalent to
           :meth:`gradient` and should not be used as a cross-check -- validate the (T) gradient
           against a finite difference of the CCSD(T) energy instead (``test_083``).  Extending
           the explicit route to (T) is a later phase."""
        from .cphf import Perturbation
        cc = self.ccwfn
        D, Gam = self._density().gradient_densities()
        cphf = cc.mp._full_occ_cphf()
        ncore = cc.o.stop - cc.no
        c = self.contract
        natom = cc.derivatives.natom
        grad = np.zeros((natom, 3))
        for atom in range(natom):
            for cart in range(3):
                pert = Perturbation('nuclear', (atom, cart))
                df = np.asarray(cphf.perturbed_fock(pert, ncore))
                deri = np.asarray(cphf.perturbed_eri(pert, ncore))
                grad[atom, cart] = (c('pq,pq->', D, df) + c('pqrs,pqrs->', Gam, deri))
        return grad

    # ---- second derivatives: static dipole polarizability ----------------
    # The asymmetric (2n+1) route: differentiate the relaxed-density gradient a second time in a
    # field.  A static field leaves the AO basis fixed (S^F = <pq|rs>^F = 0), so
    #     alpha[a,b] = Tr(dD_rel(F_b) . mu_a)  +  Tr(D_rel . (U^bT mu_a + mu_a U^b)),
    # with D_rel the (unperturbed) relaxed density and dD_rel its field response.  dD_rel needs the
    # perturbed amplitudes/multipliers dt/dLambda (iterative, CPHF-folded RHS carrying the orbital
    # relaxation), the perturbed correlation densities, the perturbed Lagrangian (FULL-df Fock term
    # -- see the diagonal-eps gotcha below), and the perturbed Z-vector.  Only first-order responses;
    # no second-order CPHF.  See DERIVATIVES_PLAN_2026-06.md sec 8.  Spatial (closed-shell RHF),
    # all-electron; frozen core / spin-orbital / (T) to follow.

    _HBAR_BLOCKS = ('Hov', 'Hvv', 'Hoo', 'Hoooo', 'Hvvvv', 'Hvovv', 'Hooov',
                    'Hovvo', 'Hovov', 'Hvvvo', 'Hovoo')

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        """CCSD **correlation** contribution to the static (omega=0) dipole polarizability (a.u.),
        shape ``(3, 3)``: ``alpha_corr[a,b] = -d^2 E_corr/dF_a dF_b``, the CC analog of
        :meth:`MPwfn.polarizability`.

        Only the **asymmetric (2n+1) route** is available for CC -- differentiate the relaxed-density
        gradient a second time (a single (T)-capable formulation; see DERIVATIVES_PLAN sec 8).  The
        ``route`` argument (from the :func:`pycc.polarizability` facade) accepts only ``'2n+1'``.

        The reference (SCF) polarizability is kept separate (:meth:`HFwfn.polarizability`); the
        :func:`pycc.polarizability` facade sums nuclear (zero) + reference + this correlation part.

        Spatial closed-shell RHF and spin-orbital UHF (both all-electron and frozen core); (T) to
        follow.  Validated against a tight finite field of :meth:`relaxed_dipole` and the SO ==
        spatial keystone."""
        if route != '2n+1':
            raise ValueError(f"CC polarizability supports only the asymmetric '2n+1' route, not {route!r}.")
        cc = self.ccwfn
        if cc.model.upper() != 'CCSD':
            raise NotImplementedError(f"CC polarizability: only CCSD is implemented (not {cc.model}).")
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

        # Tight Lambda (the second derivative wants Lambda well past the 1e-10 the gradient uses).
        hbar = cchbar(cc)
        lam = cclambda(cc, hbar)
        lam.solve_lambda(1e-13, 1e-13)
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
        z = np.asarray(hf.cphf.solve(X0))
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z

        cphf = cc.mp._full_occ_cphf()
        mu = [np.asarray(cc.H.mu[a]) for a in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            dDrel = self._perturbed_relaxed_density(pert, hbar, lam, D0, Gam0, z, Pco)
            Ub = np.asarray(cphf._full_U(pert, ncore))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def _perturbed_relaxed_density(self, pert, hbar, lam, D0, Gam0, z, Pco=None) -> np.ndarray:
        """Field response ``dD_rel`` of the CC relaxed 1-PDM (spatial; all-electron or frozen core).
        Mirrors :meth:`MPwfn._perturbed_relaxed_opdm` with the CC densities, but (i) keeps the CC
        *unrelaxed* ov/vo blocks (``D_ai != D_ia`` for CC; MP2 has none) and (ii) builds the perturbed
        Lagrangian with the FULL-df Fock term (see :meth:`_cc_perturbed_lagrangian`).  For frozen core
        the perturbed core<->active Sylvester divide ``dP_co`` is added and coupled into the perturbed
        Z-vector RHS, exactly as in the MP2 template / the unperturbed :meth:`_relaxed_density`."""
        cc = self.ccwfn
        o, v = cc.o, cc.v
        co = slice(0, cc.nfzc)
        ofull = slice(0, o.stop)
        c = self.contract
        ncore = o.stop - cc.no
        L = np.asarray(cc.H.L)
        eps = np.diag(np.asarray(cc.H.F))
        cphf = cc.mp._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        hf = self._reference_hf()

        dt1, dt2 = self._perturbed_amplitudes(df, deri, dL, hbar)
        dl1, dl2 = self._perturbed_lambda(df, deri, dL, dt1, dt2, hbar, lam)
        dDg, dGam = self._perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam)
        dIp = self._cc_perturbed_lagrangian(df, deri, dL, D0, dDg, Gam0, dGam)
        dX = dIp[ofull, v] - dIp[v, ofull].T
        dPco = None
        if cc.nfzc:                                     # perturbed core<->active divide (Sylvester derivative)
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, L[v, o, ofull, co]) + c('jc,acij->ia', dzjc, L[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, dL[v, o, ofull, co]) + c('jc,acij->ia', zjc, dL[v, co, ofull, o]))
        # perturbed orbital-Hessian response (A^x z); reference-only, as in MP2
        Axz = (c('ajib,jb->ia', dL[v, ofull, ofull, v], z) + c('abij,jb->ia', dL[v, v, ofull, ofull], z)
               + c('ab,ib->ia', df[v, v], z) - c('ij,ja->ia', df[ofull, ofull], z))
        zx = np.asarray(hf.cphf.solve(dX - Axz))
        dDrel = dDg.copy()                              # keep unrelaxed dD_ov/dD_vo (CC-only)
        if cc.nfzc:
            dDrel[co, o] += dPco
            dDrel[o, co] += dPco.T
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

    def _perturbed_amplitudes(self, df, deri, dL, hbar, maxiter=200, rconv=1e-13):
        """Perturbed CCSD amplitudes ``dt/dF`` (iterative).  RHS = the field-derivative of the CC
        residual at fixed ``t`` -- since the residual is linear in H, that is just
        ``residuals(df, t1, t2)`` with the perturbed two-electron integrals swapped in (the
        CPHF-folded ``deri``/``dL`` carry the orbital relaxation).  LHS = the CCSD Jacobian;
        iterate ``dt += (B + HBAR.dt)/D`` with DIIS, like :meth:`ccresponse.solve_right`."""
        from .utils import helper_diis
        cc = self.ccwfn
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

    def _hbar_blocks(self, hbar, F, ERI, L, t1, t2):
        """All (spatial CCSD) HBAR blocks built from explicit integrals/amplitudes -- the
        :meth:`cchbar._build` sequence with supplied arguments (no wavefunction-state mutation).
        Used by :meth:`_perturbed_hbar` for the exact stencil."""
        o, v = self.ccwfn.o, self.ccwfn.v
        Hov = hbar.build_Hov(o, v, F, L, t1)
        Hvv = hbar.build_Hvv(o, v, F, L, t1, t2)
        Hoo = hbar.build_Hoo(o, v, F, L, t1, t2)
        Hoooo = hbar.build_Hoooo(o, v, ERI, t1, t2)
        Hvvvv = hbar.build_Hvvvv(o, v, ERI, t1, t2)
        Hvovv = hbar.build_Hvovv(o, v, ERI, t1)
        Hooov = hbar.build_Hooov(o, v, ERI, t1)
        Hovvo = hbar.build_Hovvo(o, v, ERI, L, t1, t2)
        Hovov = hbar.build_Hovov(o, v, ERI, t1, t2)
        Hvvvo = hbar.build_Hvvvo(o, v, ERI, L, Hov, Hvvvv, t1, t2)
        Hovoo = hbar.build_Hovoo(o, v, ERI, L, Hov, Hoooo, t1, t2)
        return {'Hov': Hov, 'Hvv': Hvv, 'Hoo': Hoo, 'Hoooo': Hoooo, 'Hvvvv': Hvvvv,
                'Hvovv': Hvovv, 'Hooov': Hooov, 'Hovvo': Hovvo, 'Hovov': Hovov,
                'Hvvvo': Hvvvo, 'Hovoo': Hovoo}

    def _perturbed_hbar(self, df, deri, dL, dt1, dt2, hbar):
        """Perturbed HBAR ``dHBAR`` via a 5-point central stencil of the block builders in the
        step, evaluated at ``(F +- h df, ERI +- h deri, L +- h dL, t +- h dt)``.  Every spatial
        CCSD HBAR block is a polynomial of degree <= 4 in the step, so the stencil is algebraically
        exact (confirmed to roundoff).  Needed only for the perturbed-Lambda inhomogeneity."""
        cc = self.ccwfn
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI); L0 = np.asarray(cc.H.L)
        t01 = np.asarray(cc.t1); t02 = np.asarray(cc.t2)
        h = 1e-2
        def at(s):
            return self._hbar_blocks(hbar, F0 + s*h*df, ERI0 + s*h*deri, L0 + s*h*dL,
                                     t01 + s*h*dt1, t02 + s*h*dt2)
        m2, m1, p1, p2 = at(-2), at(-1), at(1), at(2)
        return {k: (np.asarray(m2[k]) - 8.0*np.asarray(m1[k]) + 8.0*np.asarray(p1[k])
                    - np.asarray(p2[k])) / (12.0*h) for k in m2}

    def _perturbed_lambda(self, df, deri, dL, dt1, dt2, hbar, lam, maxiter=200, rconv=1e-13):
        """Perturbed Lambda ``dLambda/dF`` (iterative, linear), staying in ``cclambda`` (no
        ``Y1/Y2``).  The inhomogeneity is ``r_L`` evaluated with the perturbed HBAR + ``dL``
        (unperturbed G) plus the explicit ``dG.H``/``dG.L`` product-rule halves; the Lambda-Jacobian
        action is ``r_L(dLambda) - r_L(0)`` (``r_L`` is affine in Lambda).  Iterate
        ``dLambda += (B + Jacobian)/D`` like :meth:`cclambda.solve_lambda`."""
        from .utils import helper_diis
        cc = self.ccwfn
        o, v = cc.o, cc.v
        c = self.contract
        Dia, Dijab = cc.Dia, cc.Dijab
        l1, l2 = np.asarray(lam.l1), np.asarray(lam.l2)
        t2 = np.asarray(cc.t2)
        L0 = np.asarray(cc.H.L)

        def rL1(La, Lb, H, Gvv, Goo):
            return np.asarray(lam.r_L1(o, v, La, Lb, H['Hov'], H['Hvv'], H['Hoo'], H['Hovvo'],
                                       H['Hovov'], H['Hvvvo'], H['Hovoo'], H['Hvovv'], H['Hooov'], Gvv, Goo))
        def rL2(La, Lb, Ld, H, Gvv, Goo):
            return np.asarray(lam.r_L2(o, v, La, Lb, Ld, H['Hov'], H['Hvv'], H['Hoo'], H['Hoooo'],
                                       H['Hvvvv'], H['Hovvo'], H['Hovov'], H['Hvvvo'], H['Hovoo'],
                                       H['Hvovv'], H['Hooov'], Gvv, Goo))

        dH = self._perturbed_hbar(df, deri, dL, dt1, dt2, hbar)
        H0 = {b: np.asarray(getattr(hbar, b)) for b in self._HBAR_BLOCKS}
        Goo0 = np.asarray(lam.build_Goo(t2, l2)); Gvv0 = np.asarray(lam.build_Gvv(t2, l2))
        dGoo = np.asarray(lam.build_Goo(dt2, l2)); dGvv = np.asarray(lam.build_Gvv(dt2, l2))  # l2 fixed
        B1 = rL1(l1, l2, dH, Gvv0, Goo0)
        B2 = rL2(l1, l2, dL, dH, Gvv0, Goo0)
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

    def _perturbed_correlation_densities(self, dt1, dt2, dl1, dl2, lam):
        """Field response ``(dD, dGamma)`` of the unrelaxed CC correlation densities via a 5-point
        stencil of :meth:`ccdensity.gradient_densities` in the amplitudes/multipliers (the densities
        are pure polynomials in ``t``/``lambda`` -- exact).  Temporarily sets ``cc.t1/t2`` and
        ``lam.l1/l2`` to the stepped values (restored)."""
        from .ccdensity import ccdensity
        cc = self.ccwfn
        t01, t02 = np.asarray(cc.t1).copy(), np.asarray(cc.t2).copy()
        l01, l02 = np.asarray(lam.l1).copy(), np.asarray(lam.l2).copy()
        h = 1e-2
        def at(s):
            cc.t1 = t01 + s*h*dt1; cc.t2 = t02 + s*h*dt2
            lam.l1 = l01 + s*h*dl1; lam.l2 = l02 + s*h*dl2
            D_s, Gam_s = ccdensity(cc, lam).gradient_densities()
            return np.asarray(D_s), np.asarray(Gam_s)
        try:
            m2, m1, p1, p2 = at(-2), at(-1), at(1), at(2)
        finally:
            cc.t1, cc.t2 = t01, t02
            lam.l1, lam.l2 = l01, l02
        dDg = (m2[0] - 8.0*m1[0] + 8.0*p1[0] - p2[0]) / (12.0*h)
        dGam = (m2[1] - 8.0*m1[1] + 8.0*p1[1] - p2[1]) / (12.0*h)
        return dDg, dGam

    def _cc_perturbed_lagrangian(self, df, deri, dL, D, dD, Gam, dGam):
        """Field response ``dI'`` of the generalized-Fock (GSB) Lagrangian, density-generic.  The
        Fock term is the **full matrix product** ``df @ (D + D.T)`` -- NOT the diagonal ``diag(df)``
        a stencil of :meth:`MPwfn._mp2_lagrangian` (which uses ``eps[:,None]*(D+D.T)``, valid only
        at the canonical F=0) would give.  The omitted ``df_offdiag @ (D+D.T)`` vanishes for MP2
        (unrelaxed ``D`` has no ov block) but is nonzero for CC (couples to ``Dov``/``Dvo``); see
        DERIVATIVES_PLAN sec 8.2.  (Same formula as :meth:`MPwfn._perturbed_lagrangian`; destined to
        merge with it under ``CorrelatedDerivs``.)"""
        cc = self.ccwfn
        o = cc.o
        ofull = slice(0, o.stop)
        c = self.contract
        ERI = np.asarray(cc.H.ERI)
        L = np.asarray(cc.H.L)
        eps = np.diag(np.asarray(cc.H.F))
        nmo = cc.nmo
        dA = df @ (D + D.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, L[:, :, :, ofull]) + c('rs,rpsq->pq', D, dL[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, L[:, ofull, :, :]) + c('rs,rqsp->pq', D, dL[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    # ---- spin-orbital (UHF) polarizability ------------------------------
    # Mirror of the spatial path with antisymmetrized <pq||rs> (no L), the spin-orbital HBAR (no
    # Hovov; Hvvvo/Hovoo use a Zovov intermediate), the spin-orbital Jacobian / r_L, and the inline
    # orbital Hessian G (as in _so_relaxed_density) rather than a borrowed all-electron CPHF.
    # Validated by the RHF-forced-to-SO == spatial keystone.

    _SO_HBAR_BLOCKS = ('Hov', 'Hvv', 'Hoo', 'Hoooo', 'Hvvvv', 'Hvovv', 'Hooov',
                       'Hovvo', 'Hvvvo', 'Hovoo')

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
        lam.solve_lambda(1e-13, 1e-13)
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
        G = (c('ajib->iajb', ERI[v, ofull, ofull, v]) + c('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof*nv, nof*nv)
        G[np.diag_indices(nof*nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
        z = np.linalg.solve(G, X0.reshape(-1)).reshape(nof, nv)
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z

        cphf = cc.mp._full_occ_cphf()
        mu = [np.asarray(cc.H.mu[a]) for a in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            dDrel = self._so_perturbed_relaxed_density(pert, hbar, lam, D0, Gam0, z, G, Pco)
            Ub = np.asarray(cphf._full_U(pert, ncore))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def _so_perturbed_relaxed_density(self, pert, hbar, lam, D0, Gam0, z, G, Pco):
        """Spin-orbital field response ``dD_rel``; the SO analog of
        :meth:`_perturbed_relaxed_density` (inline Hessian ``G``, antisymmetrized ERI)."""
        cc = self.ccwfn
        o, v, nv = cc.o, cc.v, cc.nv
        co = slice(0, o.start)
        ofull = slice(0, o.stop)
        nof = o.stop
        c = self.contract
        ncore = o.stop - cc.no
        ERI = np.asarray(cc.H.ERI)
        eps = np.diag(np.asarray(cc.H.F))
        cphf = cc.mp._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))

        dt1, dt2 = self._so_perturbed_amplitudes(df, deri, hbar)
        dl1, dl2 = self._so_perturbed_lambda(df, deri, dt1, dt2, hbar, lam)
        dDg, dGam = self._perturbed_correlation_densities(dt1, dt2, dl1, dl2, lam)
        dIp = self._so_cc_perturbed_lagrangian(df, deri, D0, dDg, Gam0, dGam)
        dX = dIp[ofull, v] - dIp[v, ofull].T
        dPco = None
        if cc.nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, ERI[v, o, ofull, co]) + c('jc,acij->ia', dzjc, ERI[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, deri[v, o, ofull, co]) + c('jc,acij->ia', zjc, deri[v, co, ofull, o]))
        Axz = (c('ajib,jb->ia', deri[v, ofull, ofull, v], z) + c('abij,jb->ia', deri[v, v, ofull, ofull], z)
               + c('ab,ib->ia', df[v, v], z) - c('ij,ja->ia', df[ofull, ofull], z))
        zx = np.linalg.solve(G, (dX - Axz).reshape(-1)).reshape(nof, nv)
        dDrel = dDg.copy()
        if cc.nfzc:
            dDrel[co, o] += dPco
            dDrel[o, co] += dPco.T
        dDrel[v, ofull] += -zx.T
        dDrel[ofull, v] += -zx
        return dDrel

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

    def _so_perturbed_amplitudes(self, df, deri, hbar, maxiter=200, rconv=1e-13):
        """Spin-orbital perturbed CCSD amplitudes ``dt/dF`` (SO Jacobian; SO residual has no 0.5 /
        no final symmetrization, matching ``ccresponse.solve_right``)."""
        from .utils import helper_diis
        cc = self.ccwfn
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

    def _so_hbar_blocks(self, hbar, F, ERI, t1, t2):
        """Spin-orbital HBAR blocks from explicit integrals/amplitudes (the ``cchbar._so_build``
        sequence -- no Hovov; Hvvvo/Hovoo via the Zovov intermediate)."""
        o, v = self.ccwfn.o, self.ccwfn.v
        Hov = hbar._so_build_Hov(o, v, F, ERI, t1)
        Hvv = hbar._so_build_Hvv(o, v, F, ERI, Hov, t1, t2)
        Hoo = hbar._so_build_Hoo(o, v, F, ERI, Hov, t1, t2)
        Hoooo = hbar._so_build_Hoooo(o, v, ERI, t1, t2)
        Hvvvv = hbar._so_build_Hvvvv(o, v, ERI, t1, t2)
        Hvovv = hbar._so_build_Hvovv(o, v, ERI, t1)
        Hooov = hbar._so_build_Hooov(o, v, ERI, t1)
        Hovvo = hbar._so_build_Hovvo(o, v, ERI, t1, t2)
        Zovov = hbar._so_build_Zovov(o, v, ERI, t2)
        Hvvvo = hbar._so_build_Hvvvo(o, v, ERI, Hov, Hvvvv, Zovov, t1, t2)
        Hovoo = hbar._so_build_Hovoo(o, v, ERI, Hov, Hoooo, Zovov, t1, t2)
        return {'Hov': Hov, 'Hvv': Hvv, 'Hoo': Hoo, 'Hoooo': Hoooo, 'Hvvvv': Hvvvv,
                'Hvovv': Hvovv, 'Hooov': Hooov, 'Hovvo': Hovvo, 'Hvvvo': Hvvvo, 'Hovoo': Hovoo}

    def _so_perturbed_hbar(self, df, deri, dt1, dt2, hbar):
        """Spin-orbital perturbed HBAR via the exact 5-point stencil (degree-<=4 blocks)."""
        cc = self.ccwfn
        F0 = np.asarray(cc.H.F); ERI0 = np.asarray(cc.H.ERI)
        t01 = np.asarray(cc.t1); t02 = np.asarray(cc.t2)
        h = 1e-2
        def at(s):
            return self._so_hbar_blocks(hbar, F0 + s*h*df, ERI0 + s*h*deri, t01 + s*h*dt1, t02 + s*h*dt2)
        m2, m1, p1, p2 = at(-2), at(-1), at(1), at(2)
        return {k: (np.asarray(m2[k]) - 8.0*np.asarray(m1[k]) + 8.0*np.asarray(p1[k])
                    - np.asarray(p2[k])) / (12.0*h) for k in m2}

    def _so_perturbed_lambda(self, df, deri, dt1, dt2, hbar, lam, maxiter=200, rconv=1e-13):
        """Spin-orbital perturbed Lambda ``dLambda/dF`` (SO r_L; inhomogeneity = r_L with perturbed
        HBAR + perturbed ERI, unperturbed G, plus the dG.H / dG.<pq||rs> product-rule halves)."""
        from .utils import helper_diis
        cc = self.ccwfn
        o, v = cc.o, cc.v
        c = self.contract
        Dia, Dijab = cc.Dia, cc.Dijab
        l1, l2 = np.asarray(lam.l1), np.asarray(lam.l2)
        t2 = np.asarray(cc.t2)
        ERI0 = np.asarray(cc.H.ERI)

        def rL1(La, Lb, H, Gvv, Goo):
            return np.asarray(lam._so_r_L1(o, v, La, Lb, H['Hov'], H['Hvv'], H['Hoo'], H['Hovvo'],
                                           H['Hvvvo'], H['Hovoo'], H['Hvovv'], H['Hooov'], Gvv, Goo))
        def rL2(La, Lb, E, H, Gvv, Goo):
            return np.asarray(lam._so_r_L2(o, v, La, Lb, E, H['Hov'], H['Hvv'], H['Hoo'], H['Hoooo'],
                                           H['Hvvvv'], H['Hovvo'], H['Hvvvo'], H['Hovoo'], H['Hvovv'],
                                           H['Hooov'], Gvv, Goo))

        dH = self._so_perturbed_hbar(df, deri, dt1, dt2, hbar)
        H0 = {b: np.asarray(getattr(hbar, b)) for b in self._SO_HBAR_BLOCKS}
        Goo0 = np.asarray(lam.build_Goo(t2, l2)); Gvv0 = np.asarray(lam.build_Gvv(t2, l2))
        dGoo = np.asarray(lam.build_Goo(dt2, l2)); dGvv = np.asarray(lam.build_Gvv(dt2, l2))  # l2 fixed
        B1 = rL1(l1, l2, dH, Gvv0, Goo0)
        B2 = rL2(l1, l2, deri, dH, Gvv0, Goo0)
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

    def _so_cc_perturbed_lagrangian(self, df, deri, D, dD, Gam, dGam):
        """Spin-orbital field response ``dI'`` of the GSB Lagrangian (full-df Fock term, as in
        :meth:`MPwfn._so_perturbed_lagrangian`), density-generic with the CC densities."""
        cc = self.ccwfn
        o = cc.o
        ofull = slice(0, o.stop)
        c = self.contract
        ERI = np.asarray(cc.H.ERI)
        eps = np.diag(np.asarray(cc.H.F))
        nmo = cc.nmo
        dA = df @ (D + D.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, ERI[:, :, :, ofull]) + c('rs,rpsq->pq', D, deri[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, ERI[:, ofull, :, :]) + c('rs,rqsp->pq', D, deri[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)
