"""
cideriv.py: CISD analytic-derivative property driver.

`CIderiv` is the CISD leaf of the :class:`~pycc.correlatedderivs.CorrelatedDerivs` hierarchy,
matching the `MPwfn`/`MPderiv` and `CCwfn`/`CCderiv` split: `CIwfn` holds only wavefunction
quantities (amplitude solve, energy, raw densities); this class supplies the two density hooks
(`_unrelaxed_densities`, `_perturbed_unrelaxed_densities`), from which the base's `dipole`,
`gradient`, `polarizability`, length-gauge `apt`, and `hessian` follow unchanged. Only the
genuinely CISD-specific wave-function-overlap properties - the AAT (`_correlation_aat`) and the
velocity-gauge APT (`_correlation_velocity_dipole_derivatives`) - are custom code, together with the
coupled-perturbed-CI (CPCI) response machinery they (and the two density hooks) are built from.

CPCI/MAGNETIC/VECPOT MACHINERY
-------------------------------
`_cpci_ints`/`_solve_cpci`/`_solve_cpci_ints`/`_cpci_raw`, the magnetic/vector-potential integral
builders, and the raw perturbed-density builders live here (not on `CIwfn`). 
`_solve_cpci_ints` is the core iterative solve, taking already built perturbed integrals directly;
`_solve_cpci` is the `Perturbation`-keyed, cached entry point used by the AAT/VG-APT overlap 
code (magnetic/vecpot/nuclear, via `_cpci_ints`).

"""

from __future__ import annotations

import time

import numpy as np

from .correlatedderivs import CorrelatedDerivs
from .cphf import perturbation_label
from .utils import title, iteration, converged


class CIderiv(CorrelatedDerivs):
    """CISD correlation derivative-property driver. Constructed from a converged CIwfn (aliased
    `self.ci`). `dipole`/`gradient`/`polarizability`/ length-gauge `apt`/
    `hessian` are inherited from `CorrelatedDerivs`, driven by the two density hooks below.
    The AAT (`_correlation_aat`) and velocity-gauge APT (`_correlation_velocity_dipole_derivatives`)
    are custom wave-function-overlap constructions. Spatial (closed-shell RHF); frozen-core aware throughout (the correlation
    amplitudes/densities stay in the active space while every orbital response spans the full
    occupied space - see `_cpci_ints`), matching MPderiv."""

    def __init__(self, ciwfn) -> None:
        super().__init__(ciwfn)
        self.ci = ciwfn                                # alias: this class uses .ci, the base .wfn

    def _cisd_symmetrize(self, D_pqrs_raw):
        return 0.25 * (D_pqrs_raw + D_pqrs_raw.transpose(1, 0, 3, 2)
                       + D_pqrs_raw.transpose(2, 3, 0, 1) + D_pqrs_raw.transpose(3, 2, 1, 0))

    # ---- unrelaxed correlation densities ----
    # The unrelaxed CISD one- and two-particle densities are pure functions of the converged
    # amplitudes (``self.ci.c1``/``c2``) and nothing outside the derivative code consumes them, so
    # they live here on the derivative driver rather than on ``CIwfn`` -- the same split MPderiv
    # documents and CC realizes through pycc.ccdensity.  The *normalization* they are built from
    # (``CIwfn._normalized_amplitudes``) is a wavefunction-level quantity and stays on ``CIwfn``.

    def _cisd_densities(self):
        """``(D_pq, D_pq_corr, D_pqrs)``: full 1-PDM, correlation-only 1-PDM, and 2-PDM,
        cached on this driver (``_ci_dens``), so a second ``CIderiv`` on the same
        wavefunction rebuilds them rather than sharing the first driver's copy.

        The reference block ``2 delta_ij`` spans the FULL occupied space (frozen core +
        active, ``slice(0, o.stop)``) - the correlation blocks below are active-only
        (placed by ``o``/``v``), but the doubly occupied core still carries its two
        electrons, so ``Tr(D) = 2 (nfzc + no)``. ``D_corr`` subtracts the same reference
        block back off and is therefore the pure correlation density in either case."""
        if getattr(self, '_ci_dens', None) is None:
            ci = self.ci
            c = self.contract
            o, v, nmo = ci.o, ci.v, ci.nmo
            ndocc = o.stop                      # nfzc + no: the full occupied space
            n0, n1, n2, tau_n = ci._normalized_amplitudes()

            D = np.zeros((nmo, nmo), dtype=n1.dtype)
            for i in range(ndocc):
                D[i, i] += 2.0
            D[o, o] -= 2.0 * c('ja,ia->ij', n1.conj(), n1)
            D[o, o] -= 2.0 * c('jkab,ikab->ij', tau_n.conj(), n2)
            D[v, v] += 2.0 * c('ia,ib->ab', n1.conj(), n1)
            D[v, v] += 2.0 * c('ijac,ijbc->ab', tau_n.conj(), n2)
            D[o, v] += (2.0 * n0 * n1 + 2.0 * c('jb,ijab->ia', n1.conj(), 2.0 * n2 - n2.swapaxes(2, 3)))
            D[v, o] += (2.0 * n0 * n1.conj().T + 2.0 * c('ijab,jb->ai', (2.0 * n2 - n2.swapaxes(2, 3)).conj(), n1))
            D_corr = D.copy()
            for i in range(ndocc):
                D_corr[i, i] -= 2.0

            G = np.zeros((nmo, nmo, nmo, nmo), dtype=n1.dtype)
            G[o, o, o, o] += c('klab,ijab->ijkl', n2, tau_n)
            G[v, v, v, v] += c('ijab,ijcd->abcd', n2, tau_n)
            G[o, v, v, o] += 4.0 * c('ja,ib->iabj', n1, n1)
            G[o, v, o, v] -= 2.0 * c('ja,ib->iajb', n1, n1)
            G[v, o, o, v] += 2.0 * c('jkac,ikbc->aijb', tau_n, tau_n)
            G[v, o, v, o] -= 4.0 * c('jkac,ikbc->aibj', n2, n2)
            G[v, o, v, o] += 2.0 * c('jkac,ikcb->aibj', n2, n2)
            G[v, o, v, o] += 2.0 * c('jkca,ikbc->aibj', n2, n2)
            G[v, o, v, o] -= 4.0 * c('jkca,ikcb->aibj', n2, n2)
            G[o, o, v, v] += n0 * tau_n
            tau_swp = (2.0 * n2.swapaxes(0, 2).swapaxes(1, 3) - n2.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3))
            G[v, v, o, o] += np.conjugate(tau_swp) * n0
            G[v, o, v, v] += 2.0 * c('ja,ijcb->aibc', n1, tau_n)
            G[o, v, o, o] -= 2.0 * c('kjab,ib->iajk', tau_n, n1)
            G[v, v, v, o] += 2.0 * c('jiab,jc->abci', tau_n, n1)
            G[o, o, o, v] -= 2.0 * c('kb,ijba->ijka', n1, tau_n)

            self._ci_dens = (D, D_corr, G)
        return self._ci_dens

    def _unrelaxed_densities(self):
        """CISD unrelaxed reduced densities (D, Gam) as full-MO arrays."""
        _D_pq, D_pq_corr, D_pqrs = self._cisd_densities()
        Gam = self._cisd_symmetrize(D_pqrs)
        return D_pq_corr, Gam

    def _perturbed_unrelaxed_densities(self, pert, df, deri, dL):
        """First-order response (d_x D, d_x Gam) of the CISD unrelaxed reduced densities to
        pert, in the same convention as _unrelaxed_densities. Solves the coupled-perturbed-CI
        response directly from the base's own full-occupied-space (frozen-core aware)
        perturbed integrals. """
        # Through the shared store-backed record (not _solve_cpci_ints directly), so the AAT and
        # VG-APT can read back what this solve produced instead of repeating it.  df/deri are
        # handed in: the base already built them, and after the gauge fix in _cpci_ints they are
        # the same integrals the AAT path would build for itself.
        dc1, dc2, dc0v = self._cpci_dc(pert, df=df, deri=deri)
        dD_corr = self._perturbed_cisd_corr_opdm(dc1, dc2)
        dG_raw = self._perturbed_cisd_tpdm(dc1, dc2, dc0v)
        dGam = self._cisd_symmetrize(dG_raw)
        return dD_corr, dGam

    # coupled-perturbed-CI response 

    def _cpci_eri(self, pert, gauge='non-canonical'):
        """The ``nmo^4`` perturbed two-electron integrals ``dERI`` for a cphf.Perturbation --
        fetched on demand and **not** cached here.

        Deliberately uncached.  Only one of this class's nine ``_cpci_ints`` call sites wants
        ``dERI`` (the CPCI solve); the other eight want ``U`` alone.  Caching it alongside would
        hold one ``nmo^4`` per perturbation, i.e. ``(3N + 3) * nmo^4`` for one AAT or VG-APT --
        33 tensors for a ten-atom molecule, against the three that ``_mag_int`` / ``_mom_int``
        hold.  Nothing is recomputed by fetching instead: the ``'nuclear'`` branch reads
        :meth:`~pycc.cphf.CPHF.perturbed_eri`, which is backed by the on-disk ``DerivStore``, and
        the field branches read the (still RAM-cached) magnetic / momentum engines.
        """
        ncore = self.ci.o.stop - self.ci.no
        cphf = self._full_occ_cphf()
        if pert.kind == 'nuclear':
            # Take the perturbed-MO gauge from the driver, NOT from this method's default: the
            # base's second-derivative path passes it (correlatedderivs.py, _perturbed_relaxed_
            # density), so letting it default here would solve the same CPCI equations from
            # different integrals on a driver whose gauge is not the default.  Must stay in step
            # with the identical line in _cpci_ints -- dF and dERI have to come from one gauge.
            canonical = self.perturbed_mo_gauge == 'canonical'
            return np.asarray(cphf.perturbed_eri(pert, ncore, canonical=canonical))  # DerivStore-backed
        if pert.kind == 'magnetic':
            return np.asarray(cphf.magnetic_eri(pert.comp, ncore, gauge))
        if pert.kind == 'vecpot':
            return -np.asarray(cphf.momentum_eri(pert.comp, ncore, gauge))       # +Del -> -Del
        raise ValueError(f"unknown perturbation kind {pert.kind!r}")

    def _cpci_ints(self, pert, gauge='non-canonical'):
        """(dF, U) for a cphf.Perturbation -- both ``nmo x nmo``.  The ``nmo^4`` ``dERI`` that used
        to ride along in this tuple is now :meth:`_cpci_eri`, fetched on demand.  Used for
        CIwfn's own nuclear response - NOT for the field/nuclear perturbations that feed the
        base's second-derivative machinery, which instead thread the base's own
        full-occupied-space (df, deri) directly into _solve_cpci_ints.

        Frozen-core aware, matching MPderiv: every branch takes its orbital response from the
        base's full-occupied-space CPHF (`_full_occ_cphf`, core + active) with `ncore`, so the
        nuclear side carries the non-redundant core<->active-occupied rotation and the
        core-virtual CPHF response that the active-only `ci.cphf` cannot supply - and so the
        nuclear and magnetic/vecpot sides of the overlap span the SAME occupied space. For
        `nfzc = 0` the full-occupied CPHF coincides with `ci.cphf`, so all-electron results are
        unchanged. Sharing `_full_occ_cphf` with the base also shares its cached (nmo^4)
        perturbed-ERI store across the AAT/VG-APT and the inherited second-derivative
        properties."""
        if getattr(self, '_cpci_ints_cache', None) is None:
            self._cpci_ints_cache = {}
        # The perturbed-MO gauge is in the key as well as the field gauge: it is a driver
        # property, but ``_gauge_override`` can change it between calls on one driver.
        canonical = self.perturbed_mo_gauge == 'canonical'
        key = (pert, gauge, canonical)
        if key in self._cpci_ints_cache:
            return self._cpci_ints_cache[key]
        ncore = self.ci.o.stop - self.ci.no
        if pert.kind == 'nuclear':
            cphf = self._full_occ_cphf()
            dF = np.asarray(cphf.perturbed_fock(pert, ncore, canonical=canonical))
            U = np.asarray(cphf.full_U(pert, ncore, canonical=canonical))
            result = (dF, U)
        elif pert.kind == 'magnetic':
            U, dF = self._full_occ_cphf().magnetic_ints(pert.comp, ncore, gauge)
            result = (dF, U)
        elif pert.kind == 'vecpot':
            U, dF = self._full_occ_cphf().momentum_ints(pert.comp, ncore, gauge)
            result = (-dF, -U)                 # +Del -> -Del (see docstring)
        else:
            raise ValueError(f"unknown perturbation kind {pert.kind!r}")
        self._cpci_ints_cache[key] = result
        return result

    def _solve_cpci_ints(self, dF, dERI, imaginary=False, maxiter=100, diis_start=2, diis_max=8,
                          e_convergence=1e-11, d_convergence=1e-11, label=None):
        """Core coupled-perturbed-CI iterative solve given already-built perturbed integrals
        (dF, dERI) - the CISD analog of CCderiv._perturbed_amplitudes, factored out so both
        CIderiv's own _cpci_ints-driven entry point (_solve_cpci, below) and the base's
        full-occupied-space (df, deri) (via _perturbed_unrelaxed_densities) can drive
        the same solve. Returns (dc1, dc2, dc0v, dt1, dt2): the true-normalized response
        (dc1, dc2, dc0v) and the raw intermediate-normalized response (dt1, dt2).

        `imaginary=True` for magnetic/vecpot-type perturbations (dc0v forced to 0 - the real
        normalization response vanishes by time-reversal symmetry for an imaginary perturbation);
        `False` (default) for real perturbations (nuclear/field), where dc0v is generally
        nonzero."""
        from .utils import helper_diis
        ci = self.ci
        c = self.contract
        o, v = ci.o, ci.v
        t1, t2 = ci.c1, ci.c2
        F, ERI = np.asarray(ci.H.F), np.asarray(ci.H.ERI)
        Dia, Dijab = ci.Dia, ci.Dijab
        E_cisd = ci.eci
        n0 = ci._normalized_amplitudes()[0]
        dF, dERI = np.asarray(dF), np.asarray(dERI)

        D_pq, D_pq_corr, D_pqrs = self._cisd_densities()
        dE = c('pq,pq->', dF, D_pq) + c('pqrs,pqrs->', dERI, D_pqrs)

        dt1 = -(dE * t1).astype(complex)
        dt1 = dt1 - c('ji,ja->ia', dF[o, o], t1)
        dt1 = dt1 + c('ab,ib->ia', dF[v, v], t1)
        dt1 = dt1 + c('jabi,jb->ia', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t1)
        dt1 = dt1 + c('jb,ijab->ia', dF[o, v], 2.0 * t2 - t2.swapaxes(2, 3))
        dt1 = dt1 + c('ajbc,ijbc->ia', 2.0 * dERI[v, o, v, v] - dERI.swapaxes(2, 3)[v, o, v, v], t2)
        dt1 = dt1 - c('kjib,kjab->ia', 2.0 * dERI[o, o, o, v] - dERI.swapaxes(2, 3)[o, o, o, v], t2)
        dt1 = dt1 / Dia

        dt2 = -(dE * t2).astype(complex)
        dt2 = dt2 + c('abcj,ic->ijab', dERI[v, v, v, o], t1)
        dt2 = dt2 + c('abic,jc->ijab', dERI[v, v, o, v], t1)
        dt2 = dt2 - c('kbij,ka->ijab', dERI[o, v, o, o], t1)
        dt2 = dt2 - c('akij,kb->ijab', dERI[v, o, o, o], t1)
        dt2 = dt2 + c('ac,ijcb->ijab', dF[v, v], t2)
        dt2 = dt2 + c('bc,ijac->ijab', dF[v, v], t2)
        dt2 = dt2 - c('ki,kjab->ijab', dF[o, o], t2)
        dt2 = dt2 - c('kj,ikab->ijab', dF[o, o], t2)
        dt2 = dt2 + c('klij,klab->ijab', dERI[o, o, o, o], t2)
        dt2 = dt2 + c('abcd,ijcd->ijab', dERI[v, v, v, v], t2)
        dt2 = dt2 - c('kbcj,ikca->ijab', dERI[o, v, v, o], t2)
        dt2 = dt2 + c('kaci,kjcb->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
        dt2 = dt2 - c('kbic,kjac->ijab', dERI[o, v, o, v], t2)
        dt2 = dt2 - c('kaci,kjbc->ijab', dERI[o, v, v, o], t2)
        dt2 = dt2 + c('kbcj,ikac->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
        dt2 = dt2 - c('kajc,ikcb->ijab', dERI[o, v, o, v], t2)
        dt2 = dt2 / Dijab

        dE_proj = (2.0 * c('ia,ia->', t1, dF[o, v])
                   + c('ijab,ijab->', t2, 2.0 * dERI[o, o, v, v] - dERI.swapaxes(2, 3)[o, o, v, v])
                   + 2.0 * c('ia,ia->', dt1, F[o, v])
                   + c('ijab,ijab->', dt2, 2.0 * ERI[o, o, v, v] - ERI.swapaxes(2, 3)[o, o, v, v]))

        diis = helper_diis(dt1, dt2, diis_max, getattr(ci, 'precision', 1e-12))

        name = "CISD perturbed amplitudes" + (" (%s)" % label if label else "")
        print(title(name))
        t0 = time.time()
        for niter in range(1, maxiter + 1):
            dE_proj_old = dE_proj
            dt1_old, dt2_old = dt1.copy(), dt2.copy()

            # singles residual - driving terms (dF/dERI acting on t1/t2)
            dRt1 = dF.copy().swapaxes(0, 1)[o, v].astype(complex)
            dRt1 = dRt1 - dE_proj * t1
            dRt1 = dRt1 - c('ji,ja->ia', dF[o, o], t1)
            dRt1 = dRt1 + c('ab,ib->ia', dF[v, v], t1)
            dRt1 = dRt1 + c('jabi,jb->ia', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t1)
            dRt1 = dRt1 + c('jb,ijab->ia', dF[o, v], 2.0 * t2 - t2.swapaxes(2, 3))
            dRt1 = dRt1 + c('ajbc,ijbc->ia', 2.0 * dERI[v, o, v, v] - dERI.swapaxes(2, 3)[v, o, v, v], t2)
            dRt1 = dRt1 - c('kjib,kjab->ia', 2.0 * dERI[o, o, o, v] - dERI.swapaxes(2, 3)[o, o, o, v], t2)
            # singles residual - response terms (F/ERI acting on dt1/dt2)
            dRt1 = dRt1 - E_cisd * dt1
            dRt1 = dRt1 - c('ji,ja->ia', F[o, o], dt1)
            dRt1 = dRt1 + c('ab,ib->ia', F[v, v], dt1)
            dRt1 = dRt1 + c('jabi,jb->ia', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dt1)
            dRt1 = dRt1 + c('jb,ijab->ia', F[o, v], 2.0 * dt2 - dt2.swapaxes(2, 3))
            dRt1 = dRt1 + c('ajbc,ijbc->ia', 2.0 * ERI[v, o, v, v] - ERI.swapaxes(2, 3)[v, o, v, v], dt2)
            dRt1 = dRt1 - c('kjib,kjab->ia', 2.0 * ERI[o, o, o, v] - ERI.swapaxes(2, 3)[o, o, o, v], dt2)

            # doubles residual - driving terms
            dRt2 = dERI.copy().swapaxes(0, 2).swapaxes(1, 3)[o, o, v, v].astype(complex)
            dRt2 = dRt2 - dE_proj * t2
            dRt2 = dRt2 + c('abcj,ic->ijab', dERI[v, v, v, o], t1)
            dRt2 = dRt2 + c('abic,jc->ijab', dERI[v, v, o, v], t1)
            dRt2 = dRt2 - c('kbij,ka->ijab', dERI[o, v, o, o], t1)
            dRt2 = dRt2 - c('akij,kb->ijab', dERI[v, o, o, o], t1)
            dRt2 = dRt2 + c('ac,ijcb->ijab', dF[v, v], t2)
            dRt2 = dRt2 + c('bc,ijac->ijab', dF[v, v], t2)
            dRt2 = dRt2 - c('ki,kjab->ijab', dF[o, o], t2)
            dRt2 = dRt2 - c('kj,ikab->ijab', dF[o, o], t2)
            dRt2 = dRt2 + c('klij,klab->ijab', dERI[o, o, o, o], t2)
            dRt2 = dRt2 + c('abcd,ijcd->ijab', dERI[v, v, v, v], t2)
            dRt2 = dRt2 - c('kbcj,ikca->ijab', dERI[o, v, v, o], t2)
            dRt2 = dRt2 + c('kaci,kjcb->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
            dRt2 = dRt2 - c('kbic,kjac->ijab', dERI[o, v, o, v], t2)
            dRt2 = dRt2 - c('kaci,kjbc->ijab', dERI[o, v, v, o], t2)
            dRt2 = dRt2 + c('kbcj,ikac->ijab', 2.0 * dERI[o, v, v, o] - dERI.swapaxes(2, 3)[o, v, v, o], t2)
            dRt2 = dRt2 - c('kajc,ikcb->ijab', dERI[o, v, o, v], t2)
            # doubles residual - response terms
            dRt2 = dRt2 - E_cisd * dt2
            dRt2 = dRt2 + c('abcj,ic->ijab', ERI[v, v, v, o], dt1)
            dRt2 = dRt2 + c('abic,jc->ijab', ERI[v, v, o, v], dt1)
            dRt2 = dRt2 - c('kbij,ka->ijab', ERI[o, v, o, o], dt1)
            dRt2 = dRt2 - c('akij,kb->ijab', ERI[v, o, o, o], dt1)
            dRt2 = dRt2 + c('ac,ijcb->ijab', F[v, v], dt2)
            dRt2 = dRt2 + c('bc,ijac->ijab', F[v, v], dt2)
            dRt2 = dRt2 - c('ki,kjab->ijab', F[o, o], dt2)
            dRt2 = dRt2 - c('kj,ikab->ijab', F[o, o], dt2)
            dRt2 = dRt2 + c('klij,klab->ijab', ERI[o, o, o, o], dt2)
            dRt2 = dRt2 + c('abcd,ijcd->ijab', ERI[v, v, v, v], dt2)
            dRt2 = dRt2 - c('kbcj,ikca->ijab', ERI[o, v, v, o], dt2)
            dRt2 = dRt2 + c('kaci,kjcb->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dt2)
            dRt2 = dRt2 - c('kbic,kjac->ijab', ERI[o, v, o, v], dt2)
            dRt2 = dRt2 - c('kaci,kjbc->ijab', ERI[o, v, v, o], dt2)
            dRt2 = dRt2 + c('kbcj,ikac->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dt2)
            dRt2 = dRt2 - c('kajc,ikcb->ijab', ERI[o, v, o, v], dt2)

            dt1 = dt1 + dRt1 / Dia
            dt2 = dt2 + dRt2 / Dijab

            diis.add_error_vector(dt1, dt2)
            if niter >= diis_start:
                dt1, dt2 = diis.extrapolate(dt1, dt2)

            dE_proj = (2.0 * c('ia,ia->', t1, dF[o, v])
                       + c('ijab,ijab->', t2, 2.0 * dERI[o, o, v, v] - dERI.swapaxes(2, 3)[o, o, v, v])
                       + 2.0 * c('ia,ia->', dt1, F[o, v])
                       + c('ijab,ijab->', dt2, 2.0 * ERI[o, o, v, v] - ERI.swapaxes(2, 3)[o, o, v, v]))

            delta_dE = abs(dE_proj - dE_proj_old)
            rms_dt1 = np.sqrt(np.sum((dt1 - dt1_old) ** 2))
            rms_dt2 = np.sqrt(np.sum((dt2 - dt2_old) ** 2))
            print(iteration(niter, de=delta_dE, rms=np.sqrt(abs(rms_dt1) ** 2 + abs(rms_dt2) ** 2)))
            if niter > 1 and (delta_dE < e_convergence and rms_dt1 < d_convergence
                              and rms_dt2 < d_convergence):
                print(converged(name, time.time() - t0))
                break

        dc0 = self._cisd_dn0(dt1, dt2)
        if imaginary:
            dc0v = 0.0
            dc1 = n0 * dt1
            dc2 = n0 * dt2
        else:
            # real perturbation (nuclear/field): the response is real - return real arrays so
            # downstream densities stay real (no ComplexWarning casts in the base assembly)
            dt1, dt2 = dt1.real, dt2.real
            dc0 = dc0.real if hasattr(dc0, 'real') else dc0
            dc0v = dc0
            dc1 = dc0 * t1 + n0 * dt1
            dc2 = dc0 * t2 + n0 * dt2

        return dc1, dc2, dc0v, dt1, dt2

    # ---- the perturbed-CI record: one per (perturbation, perturbed-MO gauge), on disk ----
    # Both consumers of the CPCI solve go through here.  The base's second-derivative machinery
    # reaches it from _perturbed_unrelaxed_densities (passing the (df, deri) it has already built);
    # the AAT / VG-APT overlap code reaches it from _solve_cpci, which builds its own.  Before this
    # they were separate solves with separate RAM caches, so a VCD re-solved all 3N nuclear
    # perturbations for the AAT that the Hessian had already solved.  See
    # docs/DERIVATIVES_PLAN_2026-06.md section 14.

    def _cpci_ctx(self, pert, gauge):
        """The store ctx for a CPCI record: the perturbed-MO gauge **that produced it**.

        One member, not two.  :meth:`_solve_cpci_ints` takes no gauge argument, so all gauge
        dependence enters through ``dF``/``dERI``, and each perturbation kind draws those from
        exactly one source: the real (nuclear / field) perturbations from the driver's
        :attr:`~pycc.correlatedderivs.CorrelatedDerivs.perturbed_mo_gauge`, the imaginary
        (magnetic / vecpot) ones from ``gauge``.  Carrying the other one would not be inert, it
        would be a spurious miss: a cross-check run at a non-default magnetic gauge would discard
        all 3N nuclear records, which is exactly the saving this record exists to deliver.

        ``_uid`` is deliberately absent (the ``'pt2'`` precedent, mpderiv.py).  A ``CIwfn`` fixes
        its model at construction and ``CIderiv.__init__`` takes no other argument, so the only
        per-driver variability is the gauge above; two independently built drivers give
        bit-identical amplitudes.  Dropping it lets ``pycc.hessian(pycc.CIderiv(ci))`` followed by
        ``pycc.aat(pycc.CIderiv(ci))`` share records instead of re-solving."""
        real = pert.kind in ('nuclear', 'field')
        return (self.perturbed_mo_gauge if real else gauge,)

    def _cpci_record(self, pert, gauge='non-canonical', df=None, deri=None, **kwargs):
        """``(dt1, dt2, dc0v)`` for ``pert``, memoized in the :class:`~pycc.derivatives.DerivStore`.

        ``dt2`` is the four-index (``o^2 v^2``) member; ``dt1`` and the scalar ``dc0v`` are the
        companions that make the record self-contained, since :meth:`_cpci_dc` needs ``dc0v`` and
        recomputing it would need ``dt1``.  ``df``/``deri`` let a caller that has already built the
        perturbed integrals hand them in rather than have the builder fetch them again."""
        def _build():
            dF, dERI = df, deri
            if dF is None:
                dF, _U = self._cpci_ints(pert, gauge)
                dERI = self._cpci_eri(pert, gauge)
            _dc1, _dc2, dc0v, dt1, dt2 = self._solve_cpci_ints(
                np.asarray(dF), np.asarray(dERI),
                imaginary=pert.kind in ('magnetic', 'vecpot'),
                label=perturbation_label(pert, self.ci.ref.molecule()), **kwargs)
            return dt1, dt2, np.asarray(dc0v)
        return self.wfn.derivatives.store.get_or_compute_group(
            'cpci', pert, _build, ('dt1', 'dt2', 'dc0v'), ctx=self._cpci_ctx(pert, gauge))

    def _cpci_dc(self, pert, gauge='non-canonical', df=None, deri=None, **kwargs):
        r"""``(dc1, dc2, dc0v)``: the true-normalized perturbed CI coefficients, rebuilt from the
        stored record.  ``dc`` is an exact linear function of ``dt``::

            dc1 = dc0v * c1 + n0 * dt1,    dc2 = dc0v * c2 + n0 * dt2

        and the single formula covers **both** branches of :meth:`_solve_cpci_ints`, because
        ``dc0v`` is zero for an imaginary perturbation.  So no ``imaginary`` flag is threaded here,
        and ``dc1``/``dc2`` are never stored: they are two array expressions over quantities the
        wavefunction already holds, against a CPCI solve to recompute."""
        dt1, dt2, dc0v = self._cpci_record(pert, gauge, df=df, deri=deri, **kwargs)
        dc0v = dc0v[()] if isinstance(dc0v, np.ndarray) else dc0v
        n0 = self.ci._normalized_amplitudes()[0]
        return dc0v * self.ci.c1 + n0 * dt1, dc0v * self.ci.c2 + n0 * dt2, dc0v

    def _solve_cpci(self, pert, gauge='non-canonical', **kwargs):
        """Coupled-perturbed CI keyed by :class:`~pycc.cphf.Perturbation`, returning
        ``(dc1, dc2, dc0v)`` -- the entry point for the AAT / VG-APT overlap code below.

        A thin reader over :meth:`_cpci_dc`, so it shares the on-disk record with the base's
        second-derivative machinery: on a Hessian-then-AAT run the nuclear perturbations are solved
        once, not twice.  The former ``_cpci_cache`` / ``_cpci_raw_cache`` RAM dicts are gone; they
        grew monotonically for the driver's lifetime at ``o^2 v^2`` per entry and held two
        normalizations of the same information."""
        return self._cpci_dc(pert, gauge, **kwargs)

    def _cpci_raw(self, pert, gauge='non-canonical'):
        """The raw (``t``-normalized) perturbed amplitudes ``(dt1, dt2)`` for ``pert``."""
        dt1, dt2, _dc0v = self._cpci_record(pert, gauge)
        return dt1, dt2

    # raw perturbed correlation-density builders (true-normalized)

    def _cisd_dn0(self, dn1, dn2):
        """First derivative of the normalization factor ``n0`` along a perturbation, the
        derivative counterpart of :meth:`CIwfn._normalized_amplitudes`::

            dn0/dx = -n0^3 (2 c1_ia dc1_ia + (2 c2 - c2.swap)_ijab dc2_ijab)

        Consumed by :meth:`_solve_cpci_ints`, which needs it to carry the perturbed normalization
        into ``dc1``/``dc2``."""
        ci = self.ci
        c = self.contract
        t1, t2 = ci.c1, ci.c2
        tau = 2.0 * t2 - t2.swapaxes(2, 3)
        n0 = ci._normalized_amplitudes()[0]
        return -n0**3 * (2.0 * c('ia,ia->', t1.conj(), dn1) + c('ijab,ijab->', tau.conj(), dn2))

    def _perturbed_cisd_corr_opdm(self, dc1, dc2):
        ci = self.ci
        c = self.contract
        o, v, nmo = ci.o, ci.v, ci.nmo
        n0, n1, n2, tau_n = ci._normalized_amplitudes()
        dtau_n = 2.0 * dc2 - dc2.swapaxes(2, 3)
        sigma = n2 - n2.swapaxes(2, 3)
        dsigma = dc2 - dc2.swapaxes(2, 3)

        dD = np.zeros((nmo, nmo), dtype=dc1.dtype)
        dD[o, o] -= 2.0 * c('ja,ia->ij', dc1, n1) + 2.0 * c('ja,ia->ij', n1, dc1)
        dD[o, o] -= 2.0 * c('jkab,ikab->ij', dtau_n, n2) + 2.0 * c('jkab,ikab->ij', tau_n, dc2)
        dD[v, v] += 2.0 * c('ia,ib->ab', dc1, n1) + 2.0 * c('ia,ib->ab', n1, dc1)
        dD[v, v] += 2.0 * c('ijac,ijbc->ab', dtau_n, n2) + 2.0 * c('ijac,ijbc->ab', tau_n, dc2)
        dD[o, v] += 2.0 * dc1
        dD[o, v] += 2.0 * c('jb,ijab->ia', dc1, sigma) + 2.0 * c('jb,ijab->ia', n1, dsigma)
        dD[v, o] = dD[o, v].T
        return dD

    def _perturbed_cisd_tpdm(self, dc1, dc2, dc0v):
        ci = self.ci
        c = self.contract
        o, v, nmo = ci.o, ci.v, ci.nmo
        n0, n1, n2, tau_n = ci._normalized_amplitudes()
        dtau_n = 2.0 * dc2 - dc2.swapaxes(2, 3)

        dG = np.zeros((nmo, nmo, nmo, nmo), dtype=dc1.dtype)
        dG[o, o, o, o] = c('klab,ijab->ijkl', dc2, tau_n) + c('klab,ijab->ijkl', n2, dtau_n)
        dG[v, v, v, v] = c('ijab,ijcd->abcd', dc2, tau_n) + c('ijab,ijcd->abcd', n2, dtau_n)
        dG[o, v, v, o] = 4.0 * (c('ja,ib->iabj', dc1, n1) + c('ja,ib->iabj', n1, dc1))
        dG[o, v, o, v] = -2.0 * (c('ja,ib->iajb', dc1, n1) + c('ja,ib->iajb', n1, dc1))
        dG[v, o, o, v] = 2.0 * (c('jkac,ikbc->aijb', dtau_n, tau_n) + c('jkac,ikbc->aijb', tau_n, dtau_n))
        dG[v, o, v, o] = (
            -4.0 * (c('jkac,ikbc->aibj', dc2, n2) + c('jkac,ikbc->aibj', n2, dc2))
            + 2.0 * (c('jkac,ikcb->aibj', dc2, n2) + c('jkac,ikcb->aibj', n2, dc2))
            + 2.0 * (c('jkca,ikbc->aibj', dc2, n2) + c('jkca,ikbc->aibj', n2, dc2))
            - 4.0 * (c('jkca,ikcb->aibj', dc2, n2) + c('jkca,ikcb->aibj', n2, dc2)))
        dG[o, o, v, v] = dc0v * tau_n + n0 * dtau_n
        tau_swp = 2.0 * n2.swapaxes(0, 2).swapaxes(1, 3) - n2.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3)
        dtau_swp = 2.0 * dc2.swapaxes(0, 2).swapaxes(1, 3) - dc2.swapaxes(2, 3).swapaxes(0, 2).swapaxes(1, 3)
        dG[v, v, o, o] = dc0v * tau_swp + n0 * dtau_swp
        dG[v, o, v, v] = 2.0 * (c('ja,ijcb->aibc', dc1, tau_n) + c('ja,ijcb->aibc', n1, dtau_n))
        dG[o, v, o, o] = -2.0 * (c('kjab,ib->iajk', dtau_n, n1) + c('kjab,ib->iajk', tau_n, dc1))
        dG[v, v, v, o] = 2.0 * (c('jiab,jc->abci', dtau_n, n1) + c('jiab,jc->abci', tau_n, dc1))
        dG[o, o, o, v] = -2.0 * (c('kb,ijba->ijka', dc1, tau_n) + c('kb,ijba->ijka', n1, dtau_n))
        return dG

    def _build_Dtilde(self, dc1, dc2, dc0):
        c = self.contract
        n0, n1, n2, tau_n = self.ci._normalized_amplitudes()
        o, v, nmo = self.ci.o, self.ci.v, self.ci.nmo

        R = np.zeros((nmo, nmo), dtype=complex)
        R[o, o] -= 2.0 * c('ja,ia->ij', n1, dc1)
        R[o, o] -= 2.0 * c('jkab,ikab->ij', tau_n, dc2)
        R[v, v] += 2.0 * c('ia,ib->ab', n1, dc1)
        R[v, v] += 2.0 * c('ijac,ijbc->ab', tau_n, dc2)
        R[o, v] += 2.0 * n0 * dc1 + 2.0 * dc0 * n1
        R[o, v] += 2.0 * c('jb,ijab->ia', n1, 2.0 * dc2 - dc2.swapaxes(2, 3))
        R[v, o] += 2.0 * c('ijab,jb->ai', tau_n, dc1)
        return R

    def _aat_dc_normalized(self, pert, gauge='non-canonical'):
        dc1, dc2, dc0v = self._solve_cpci(pert, gauge)
        return dc0v, dc1, dc2

    def compute_Icc_AATs(self, gauge='non-canonical'):
        """Term 1 of the AAT: direct state-vector overlap <dPsi_R|dPsi_H>."""
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        I_cc = np.zeros((3 * natom, 3))
        magH = {b: self._aat_dc_normalized(Perturbation('magnetic', b), gauge) for b in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, dn1_R, dn2_R = self._aat_dc_normalized(pR, gauge)
            for beta in range(3):
                _, dn1_H, dn2_H = magH[beta]
                term = 2.0 * c('ia,ia->', dn1_R.conj(), dn1_H)
                term = term + c('ijab,ijab->', (2.0 * dn2_R - dn2_R.swapaxes(2, 3)).conj(), dn2_H)
                I_cc[la, beta] = term.real
        return I_cc

    def compute_Iphic_AATs(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        AAT_phic = np.zeros((3 * natom, 3), dtype=complex)
        magH = {b: self._aat_dc_normalized(Perturbation('magnetic', b), gauge) for b in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, U_R = self._cpci_ints(pR, gauge)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                dn0_H, dn1_H, dn2_H = magH[beta]
                R_pq = self._build_Dtilde(dn1_H, dn2_H, dn0_H)
                AAT_phic[la, beta] = c('pq,qp->', R_pq, Ur_eff)
        return AAT_phic.real

    def compute_Iphiphi_AATs(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        _, D_pq, _ = self._cisd_densities()   # correlation-only 1-PDM (true-normalized)
        I_pp = np.zeros((3 * natom, 3))
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, U_R = self._cpci_ints(pR, gauge)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                _, U_H = self._cpci_ints(Perturbation('magnetic', beta), gauge)
                I_pp[la, beta] = c('pq,pq->', D_pq, U_H.T @ Ur_eff).real
        return I_pp

    def compute_Icphi_AATs(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        AAT_cphi = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            dn0_R, dn1_R, dn2_R = self._aat_dc_normalized(pR, gauge)
            R_pq = self._build_Dtilde(dn1_R, dn2_R, dn0_R)
            for beta in range(3):
                _, U_H = self._cpci_ints(Perturbation('magnetic', beta), gauge)
                AAT_cphi[la, beta] = c('pq,pq->', R_pq, U_H)
        return AAT_cphi.real

    def aat(self, origin=None, orbital_gauge: str = 'non-canonical') -> "PropertyComponents":
        """Atomic axial tensors (AATs, for VCD) as a :class:`pycc.PropertyComponents`.
        See :func:`pycc.aat`."""
        from . import properties
        return properties.aat(self, origin=origin, orbital_gauge=orbital_gauge)

    def _correlation_aat(self, gauge: str = 'non-canonical') -> np.ndarray:
        """CISD correlation AAT, shape (natom, 3, 3): the four electronic overlap blocks with the
        correlation 1-PDM in Iphiphi. The SCF reference and the nuclear term (Z_A/4) eps_abc R_c
        are supplied by the pycc.aat facade (HFwfn.aat + _nuclear_aat), matching
        MPderiv.

        `gauge` selects the redundant magnetic oo/vv orbital response (see
        `CPHF.magnetic_ints`); the AAT is invariant to it. `'non-canonical'` (default, matching
        MPderiv and the pycc.aat facade) leaves the redundant within-space rotations at zero and
        fills only the non-redundant core<->active-occupied block from the canonical condition,
        avoiding the near-degenerate divides of `'canonical'` among close-lying core orbitals.
        Frozen-core aware."""
        natom = self.ci.ref.molecule().natom()
        total = (self.compute_Icc_AATs(gauge) + self.compute_Iphic_AATs(gauge)
                 + self.compute_Iphiphi_AATs(gauge) + self.compute_Icphi_AATs(gauge))
        return total.reshape(natom, 3, 3)

    def compute_Icc_VG_APT(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        I_cc = np.zeros((3 * natom, 3), dtype=complex)
        vecA = {g: self._aat_dc_normalized(Perturbation('vecpot', g), gauge) for g in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            _, dn1_R, dn2_R = self._aat_dc_normalized(Perturbation('nuclear', (A, beta_)), gauge)
            for gamma in range(3):
                _, dn1_A, dn2_A = vecA[gamma]
                I_cc[la, gamma] = (2.0 * c('ia,ia->', dn1_R.conj(), dn1_A)
                                   + c('ijab,ijab->',
                                       (2.0 * dn2_R - dn2_R.swapaxes(2, 3)).conj(), dn2_A))
        return I_cc

    def compute_Icphi_VG_APT(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        Icphi = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            dn0_R, dn1_R, dn2_R = self._aat_dc_normalized(Perturbation('nuclear', (A, beta_)), gauge)
            D_tilde_R = self._build_Dtilde(dn1_R, dn2_R, dn0_R)
            for gamma in range(3):
                _, U_A = self._cpci_ints(Perturbation('vecpot', gamma), gauge)
                Icphi[la, gamma] = c('pq,pq->', D_tilde_R, U_A)
        return Icphi

    def compute_Iphic_VG_APT(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        Iphic = np.zeros((3 * natom, 3), dtype=complex)
        vecA = {g: self._aat_dc_normalized(Perturbation('vecpot', g), gauge) for g in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, U_R = self._cpci_ints(pR, gauge)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                dn0_A, dn1_A, dn2_A = vecA[gamma]
                D_tilde_A = self._build_Dtilde(dn1_A, dn2_A, dn0_A)
                Iphic[la, gamma] = c('pq,qp->', D_tilde_A.conj(), Ur_eff)
        return Iphic

    def compute_Iphiphi_VG_APT(self, gauge='non-canonical'):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        _, D_pq, _ = self._cisd_densities()   # correlation-only 1-PDM (true-normalized)
        I_pp = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            _, U_R = self._cpci_ints(Perturbation('nuclear', (A, beta_)), gauge)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                _, U_A = self._cpci_ints(Perturbation('vecpot', gamma), gauge)
                I_pp[la, gamma] = c('pq,pq->', D_pq, U_A.T @ Ur_eff)
        return I_pp

    def _correlation_velocity_dipole_derivatives(self, gauge: str = 'non-canonical') -> np.ndarray:
        """CISD correlation velocity-gauge APT, shape (natom, 3, 3): -2 times the four overlap
        blocks with the correlation 1-PDM in Iphiphi. The SCF reference and the Z_A delta nuclear
        term are supplied by the pycc.apt(gauge='velocity') facade, matching MPderiv.

        `gauge` selects the redundant momentum oo/vv orbital response (see
        `CPHF.momentum_ints`), `'non-canonical'` (default) or `'canonical'`; the VG APT is
        invariant to it. Frozen-core aware."""
        natom = self.ci.ref.molecule().natom()
        overlap_total = (self.compute_Icc_VG_APT(gauge) + self.compute_Icphi_VG_APT(gauge)
                          + self.compute_Iphic_VG_APT(gauge) + self.compute_Iphiphi_VG_APT(gauge))
        return (-2.0 * overlap_total).real.reshape(natom, 3, 3)

    def dboc(self):
        r"""Electronic diagonal Born-Oppenheimer correction (DBOC, a.u.) for the
        true-normalized CISD wavefunction (Gauss et al., JCP 125, 144111 (2006)):
        `E = \sum_{A\alpha} \|\partial_{A\alpha}\psi\|^2 / 2M_A`, bare
        NUCLEAR masses.

        The sectors are the AAT's (``Icc``, ``Icphi`` = ``Iphic``, ``Iphiphi``)
        with both perturbations nuclear; ``Iphiphi`` keeps the 2-RDM contraction
        the AAT's symmetry theorem eliminates.  The one new ingredient is the AO
        contact term `\sum_{pq} D_{pq}(Q - B^TB)_{pq}` (the paper's
        `S^{(x)(y)}` class; both derivatives move the basis functions, so
        it has no AAT analog), with the `Q` part via the kinetic sum rule
        `\sum_X Q^{(X)} = 2T_{AA}` (`Derivatives.overlap_dd_sum`)
        and `B` the half-derivative overlap. Matches the paper's Table I(a)
        CISD column to its printed precision.

        Takes **no gauge argument**, unlike its neighbours on this class (``aat``,
        ``_correlation_velocity_dipole_derivatives``, the ``compute_*`` sectors).  Theirs select
        the redundant oo/vv response of an *imaginary* perturbation (magnetic / momentum); the
        DBOC builds only ``Perturbation('nuclear', ...)``, so there is no imaginary perturbation
        and that argument had no consumer -- it was bitwise inert, which is worse than noise,
        since it advertised a cross-check that could not fail.  The real canonical vs
        non-canonical choice for nuclear perturbations is the *perturbed-MO* gauge, which comes
        from the driver (:attr:`~pycc.correlatedderivs.CorrelatedDerivs.perturbed_mo_gauge`) and
        which the DBOC is invariant to -- see
        ``test_cisd_dboc_perturbed_mo_gauge_invariance``.  Matches ``HFwfn.dboc``,
        ``MPderiv.dboc`` and ``MPderiv.dboc_mp1``, none of which takes one either."""
        from .cphf import Perturbation
        ci = self.ci
        c = self.contract
        o, v = ci.o, ci.v
        nmo = ci.nmo
        mol = ci.ref.molecule()
        natom = mol.natom()
        U_ME = 1822.888486209
        ME_U = 5.48579909065e-4
        n0, n1, n2, _ = ci._normalized_amplitudes()
        Dfull = np.asarray(self._cisd_densities()[0]).real
        Dcorr, Gam = self._unrelaxed_densities()
        Dcorr, Gam = np.asarray(Dcorr).real, np.asarray(Gam).real
        Dref = Dfull - Dcorr

        # 2-PDM in the <E_pq E_rs> pairing: stored Gam + separable reference terms
        def sep(A, B):
            return (c('pq,rs->pqrs', A, B)
                    - 0.5 * c('ps,rq->pqrs', A, B))

        Gfull = (2.0 * Gam.transpose(0, 2, 1, 3)
                 + sep(Dref, Dref) + sep(Dref, Dcorr) + sep(Dcorr, Dref))

        E = 0.0
        for A in range(natom):
            w = 1.0 / (2.0 * (mol.mass(A) - mol.Z(A) * ME_U) * U_ME)
            # Q part of the contact term via the kinetic sum rule (once per atom)
            E += w * np.sum(Dfull * np.asarray(ci.derivatives.overlap_dd_sum(A)))
            for cart in range(3):
                p = Perturbation('nuclear', (A, cart))
                dF, U = self._cpci_ints(p)
                # Through the shared store-backed record, so a DBOC run after a Hessian (or an
                # AAT) reads the 3N nuclear amplitudes instead of re-solving them.  The record's
                # builder reaches the same integrals this method would have built for itself.
                dc1, dc2, dc0v = self._cpci_dc(p)
                dc1 = np.nan_to_num(np.asarray(dc1).real)
                dc2 = np.nan_to_num(np.asarray(dc2).real)
                dc0v = float(np.nan_to_num(np.asarray(dc0v).real))
                half_S = np.asarray(
                    ci.derivatives.overlap_half(A)[cart]).T
                Ueff = np.asarray(U).real + half_S
                Icc = (dc0v * dc0v + 2.0 * np.sum(dc1 * dc1)
                       + np.sum((2.0 * dc2 - dc2.swapaxes(2, 3)) * dc2))
                R = np.asarray(self._build_Dtilde(dc1, dc2, dc0v)).real
                T = np.zeros((nmo, nmo))
                T[o, o] = R[o, o].T
                T[v, v] = R[v, v].T
                T[o, v] = (2.0 * dc0v * n1
                           + 2.0 * c('jb,ijab->ia', dc1, 2.0 * n2 - n2.swapaxes(2, 3)))
                T[v, o] = (2.0 * n0 * dc1.T
                           + 2.0 * c('ijab,jb->ai', 2.0 * dc2 - dc2.swapaxes(2, 3), n1))
                Icphi = np.sum(T * Ueff)
                Iphic = Icphi
                Iphiphi = (c('pq,pq->', Dfull, Ueff @ Ueff.T)
                           - c('pqrs,pq,rs->', Gfull, Ueff, Ueff))
                contact = -np.sum(Dfull * (half_S.T @ half_S))
                E += w * (Icc + Icphi + Iphic + Iphiphi + contact)
        return E
