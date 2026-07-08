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
        """Spin-orbital (UHF) CCSD **correlation** gradient via the Z-vector route -- the
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

        **UHF only** -- ROHF is not supported (the semicanonical response does not reproduce the
        restricted ROHF response).  Frozen-core aware (the core<->active-occupied ``P_co`` divide,
        coupled into the Z-vector RHS, exactly as the spatial path and :meth:`MPwfn._so_zvector`).
        Validated against ``psi4.gradient('ccsd')`` (UHF), the spatial closed-shell gradient (SO ==
        spatial), and the explicit-derivative route (:meth:`_gradient_explicit`)."""
        cc = self.ccwfn
        if cc.mp.cphf.is_rohf:
            raise NotImplementedError(
                "The spin-orbital CCSD gradient is not implemented for ROHF references (the "
                "semicanonical response does not reproduce the restricted ROHF response); RHF and "
                "UHF are supported.")
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
        G = (c('ajib->iajb', ERI[v, ofull, ofull, v])    # orbital Hessian, built inline
             + c('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof * nv, nof * nv)
        G[np.diag_indices(nof * nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
        z = np.linalg.solve(G, X.reshape(-1)).reshape(nof, nv)   # Z-vector A z = X
        Drel[v, ofull] += -z.T
        Drel[ofull, v] += -z
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
