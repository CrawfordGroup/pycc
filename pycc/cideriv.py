"""
cideriv.py: CISD analytic-derivative property driver (SCAFFOLD / STUB).

This is the Phase-4 slot of the CorrelatedDerivs refactor: the place where CISD's analytic
derivative properties will live behind a driver, exactly as MP2 lives on :class:`~pycc.mpderiv.MPderiv`
and CC on :class:`~pycc.ccderiv.CCderiv`.  It is intentionally a STUB -- the two density hooks below
raise :class:`NotImplementedError`.  Filling them in is a programmer task.

WHY A STUB IS ENOUGH
--------------------
:class:`~pycc.correlatedderivs.CorrelatedDerivs` already owns the entire method-agnostic
orbital-response + property-assembly machinery: the unperturbed Z-vector
(:meth:`~pycc.correlatedderivs.CorrelatedDerivs._orbital_response`), the perturbed relaxed density
(:meth:`~pycc.correlatedderivs.CorrelatedDerivs._perturbed_relaxed_density`), and the public
properties ``gradient`` / ``relaxed_dipole`` / ``polarizability`` / ``dipole_derivatives`` /
``hessian``.  All of that is driven by just two method-specific hooks, which each leaf supplies:

  * :meth:`_unrelaxed_densities`            -- the unrelaxed reduced densities ``(D, Gam)``
  * :meth:`_perturbed_unrelaxed_densities`  -- their first-order response to a perturbation

Implement those two hooks for CISD and ``CIderiv`` immediately inherits gradients, the relaxed
dipole, the polarizability, the atomic polar tensors (LG-APT), and the molecular Hessian -- no
per-property CISD code required.  (Atomic axial tensors and the velocity-gauge APT are NOT provided
by the base -- they are method-specific overlap formulations -- so those stay as bespoke methods,
as they are for MP2 on ``MPderiv`` and currently for CISD on ``CIwfn``.)

MIGRATION ROADMAP (for the programmer)
--------------------------------------
1. Implement :meth:`_unrelaxed_densities` and :meth:`_perturbed_unrelaxed_densities` below, drawing
   on the CISD density machinery that already exists on :class:`~pycc.ciwfn.CIwfn`
   (``_cisd_densities``, ``_solve_cpci``, ``_perturbed_cisd_corr_opdm``, ``_perturbed_cisd_tpdm``,
   ``_solve_dz_dR``).  The crux is the DENSITY CONVENTION (see each hook's docstring): the base
   wants the *correlation* 1-PDM (no HF ``2*delta_oo`` block) and the *cumulant* 2-PDM (no HF 2-RDM
   block), matching :meth:`~pycc.correlatedderivs.CorrelatedDerivs._lagrangian`.  ``CIwfn._corr_QGX``
   already strips exactly those HF blocks off ``CIwfn._zvector``'s full ``Q``/``G`` -- study it.
2. Validate.  ``CIwfn``'s current ``dipole_derivatives`` / ``hessian`` / ``atomic_axial_tensors`` /
   ``velocity_dipole_derivatives`` are already the perturbed-relaxed-density (2n+1-style) results
   (their ``route='explicit'`` argument is a vestigial, inert label -- the body never branches on
   it), so they are a ready-made numerical oracle for the new base-driven ``CIderiv`` output.  Also
   check the SO==spatial keystone and a tight finite-difference oracle, per pycc convention.
3. Register the driver: uncomment ``register_deriv(CIwfn, CIderiv)`` in ``pycc/__init__.py`` so the
   :mod:`pycc.properties` facade routes CISD through ``CIderiv`` instead of the transitional
   on-``CIwfn`` code.  Then retire the now-redundant per-property CISD derivative code on ``CIwfn``
   (keeping only the genuinely CISD-specific AAT / VG-APT pieces), and drop the inert ``route``
   arguments.

ORBITAL GAUGE
-------------
CISD is invariant to occupied-occupied and virtual-virtual orbital rotations, so it takes the
NON-CANONICAL perturbed-MO approach (the ``U^x_ij = -1/2 S^(x)_ij`` / ``U^x_ab = -1/2 S^(x)_ab``
orthonormality gauge; the oo/vv dependent-pair rotations vanish) -- the same choice as MP2 and CCSD,
and the opposite of CCSD(T).  This is already the default: the inherited
:attr:`~pycc.correlatedderivs.CorrelatedDerivs.perturbed_mo_gauge` returns ``'non-canonical'`` for
any wavefunction whose ``model`` is not ``CCSD(T)``, and ``CIwfn.model`` is ``'CISD'``, so no
override is needed here.  The programmer should nonetheless CONFIRM this holds for their CISD variant
(a non-invariant CI truncation would need the canonical route and an override).
"""

from __future__ import annotations

from .correlatedderivs import CorrelatedDerivs


class CIderiv(CorrelatedDerivs):
    """CISD correlation derivative-property driver -- SCAFFOLD / STUB (Phase 4).

    Constructed from a converged :class:`~pycc.ciwfn.CIwfn`.  Inherits the full orbital-response and
    property-assembly machinery from :class:`~pycc.correlatedderivs.CorrelatedDerivs`; a programmer
    supplies the two CISD density hooks (:meth:`_unrelaxed_densities`,
    :meth:`_perturbed_unrelaxed_densities`) to bring it to life.  See the module docstring for the
    migration roadmap and the density-convention notes.
    """

    def __init__(self, ciwfn) -> None:
        super().__init__(ciwfn)
        self.ciwfn = ciwfn                             # alias: this class uses .ciwfn, the base .wfn

    def _unrelaxed_densities(self):
        """Leaf hook: the unrelaxed reduced densities ``(D, Gam)`` as full-MO arrays -- the CISD
        analogue of :meth:`~pycc.mpderiv.MPderiv._unrelaxed_densities` /
        :meth:`~pycc.ccderiv.CCderiv._unrelaxed_densities`.

        Must return:

        * ``D``   -- the **correlation** 1-PDM (``nmo x nmo``): the Doo/Dov/Dvo/Dvv correlation
          blocks only, WITHOUT the reference ``2*delta_ij`` occupied block (the base adds the
          reference contribution separately via the SCF ``HFwfn``).
        * ``Gam`` -- the **cumulant** 2-PDM (``nmo^4``), WITHOUT the closed-shell HF 2-RDM block
          (``2 d_ik d_jl - d_il d_jk``), carrying the permutational symmetry expected by
          :meth:`~pycc.correlatedderivs.CorrelatedDerivs._lagrangian` (whose 2-PDM term is
          ``4 sum_rst <pr||st> Gam_qrst``).

        IMPLEMENTATION POINTERS.  ``CIwfn._cisd_densities`` returns ``(D_pq, D_pq_corr, D_pqrs)`` and
        ``CIwfn._zvector`` builds the full ``Q``/``G``; ``CIwfn._corr_QGX`` already strips the HF
        blocks off those to give the correlation-only ``(Q_corr, G_corr, X_corr)``.  Reconciling
        those CISD conventions to the base's ``(D, Gam)`` contract above is the main task -- compare
        against how ``MPderiv._unrelaxed_densities`` assembles its full-MO ``(D, Gam)`` from the MP2
        seeds.
        """
        raise NotImplementedError(
            "CIderiv._unrelaxed_densities is a Phase-4 stub. Return the CISD correlation 1-PDM D "
            "and cumulant 2-PDM Gam (full-MO, HF blocks removed) in the CorrelatedDerivs convention "
            "-- see this method's docstring and CIwfn._cisd_densities / CIwfn._corr_QGX."
        )

    def _perturbed_unrelaxed_densities(self, pert, df, deri, dL):
        """Leaf hook: the first-order response ``(d_x gamma, d_x Gamma)`` of the unrelaxed reduced
        densities to ``pert`` (full-MO arrays), in the same convention as :meth:`_unrelaxed_densities`
        -- the CISD analogue of the MP2 closed form / the CC iterative response.

        ``pert`` is a :class:`~pycc.cphf.Perturbation`.  ``df``/``deri``/``dL`` are the CPHF-folded
        perturbed integrals the base has already built (canonical per
        :attr:`~pycc.correlatedderivs.CorrelatedDerivs.perturbed_mo_gauge`); a CISD implementation
        will likely solve its own perturbed CI response from ``pert`` and may ignore them, exactly as
        ``MPderiv._perturbed_unrelaxed_densities`` ignores them for the closed-form MP2 response.

        IMPLEMENTATION POINTERS.  ``CIwfn._solve_cpci(pert)`` solves the perturbed CI coefficients
        ``(dc1, dc2, dc0v)``; ``CIwfn._perturbed_cisd_corr_opdm(dc1, dc2)`` and
        ``CIwfn._perturbed_cisd_tpdm(dc1, dc2, dc0v)`` build the perturbed 1-PDM / 2-PDM from them.
        Return those in the base convention (HF blocks removed), matching what
        :meth:`_unrelaxed_densities` returns for the unperturbed case.
        """
        raise NotImplementedError(
            "CIderiv._perturbed_unrelaxed_densities is a Phase-4 stub. Return the first-order "
            "response (d_x gamma, d_x Gamma) of the CISD correlation densities -- see this method's "
            "docstring and CIwfn._solve_cpci / _perturbed_cisd_corr_opdm / _perturbed_cisd_tpdm."
        )
