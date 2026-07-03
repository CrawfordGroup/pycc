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

    def gradient(self) -> np.ndarray:
        """CCSD **correlation** contribution to the analytic nuclear energy gradient (a.u.), shape
        ``(natom, 3)``, via the **explicit-derivative route**::

            dE_corr/dX = sum_pq D_pq (d_X f)_pq + sum_pqrs Gamma_pqrs (d_X <pq|rs>)

        The CCSD Lambda-response 1- and 2-particle densities
        (:meth:`ccdensity.gradient_densities`, no-prefactor convention) are contracted with the
        CPHF-folded perturbed integrals (:meth:`CPHF.perturbed_fock` / :meth:`CPHF.perturbed_eri`)
        -- one nuclear CPHF solve per perturbation (``3*natom``), the orbital relaxation riding
        inside ``d_X f`` / ``d_X <pq|rs>`` rather than in a Z-vector.  The perturbed-integral
        engine is the one built for the MP2 gradient (:meth:`MPwfn._full_occ_cphf`, shared through
        ``ccwfn.mp``).

        The **reference (SCF) gradient is kept separate** (as for MP2): the total CCSD gradient is
        ``HFwfn(ref).gradient()`` plus this, assembled by the :func:`pycc.gradient` facade.

        Spatial (closed-shell RHF) path only for now; all-electron.  Validated against
        ``psi4.gradient('ccsd')`` and a finite difference of the CCSD energy.  This "simple but
        inefficient" route is the analog of :meth:`MPwfn._corr_gradient_explicit`; the Z-vector
        (relaxed-density) route is the efficient alternative (in development)."""
        from .cphf import Perturbation
        cc = self.ccwfn
        if cc.orbital_basis != 'spatial':
            raise NotImplementedError(
                "The CCSD analytic gradient currently requires the spatial (closed-shell RHF) "
                "path; the spin-orbital two-particle density is not yet implemented.")
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
