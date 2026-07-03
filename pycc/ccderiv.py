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
    Spatial (closed-shell RHF) path only for now -- the spin-orbital two-particle density is not
    yet implemented.  Lambda and the reduced densities are solved/built on first use and cached.
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
        the three-index ``termC`` is sensitive to it.  (When an MPderiv is split out this shared
        primitive will move with it.)"""
        return self.ccwfn.mp._mp2_lagrangian(D, Gam)

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

        Spatial (closed-shell RHF) path only for now; all-electron and frozen-core.  Validated
        against ``psi4.gradient('ccsd')``, a finite difference of the CCSD energy, and the
        independent explicit-derivative route (:meth:`_gradient_explicit`).

        Frozen core is handled as in the MP2 gradient: the correlation densities live in the active
        space while the orbital response spans the full occupied space.  The core<->active-occupied
        rotation is a direct divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`` (the SCF energy is
        invariant to occupied-occupied rotations), coupled into the Z-vector right-hand side."""
        cc = self.ccwfn
        if cc.orbital_basis != 'spatial':
            raise NotImplementedError(
                "The CCSD analytic gradient currently requires the spatial (closed-shell RHF) "
                "path; the spin-orbital two-particle density is not yet implemented.")
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

    def _gradient_explicit(self) -> np.ndarray:
        """CCSD correlation gradient via the **explicit-derivative route** -- an independent
        cross-check of :meth:`gradient`::

            dE_corr/dX = sum_pq D_pq (d_X f)_pq + sum_pqrs Gamma_pqrs (d_X <pq|rs>)

        The Lambda-response densities are contracted with the CPHF-folded perturbed integrals
        (:meth:`CPHF.perturbed_fock` / :meth:`CPHF.perturbed_eri`), the orbital relaxation riding
        inside ``d_X f`` / ``d_X <pq|rs>`` -- one nuclear CPHF solve per perturbation (the "simple
        but inefficient" form, analog of :meth:`MPwfn._corr_gradient_explicit`).  Same result as
        :meth:`gradient` (which uses the single Z-vector solve)."""
        from .cphf import Perturbation
        cc = self.ccwfn
        if cc.orbital_basis != 'spatial':
            raise NotImplementedError("Explicit CCSD gradient: spatial (closed-shell) only.")
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
