"""
cideriv.py: CISD analytic-derivative property driver.

`CIderiv` is the CISD leaf of the :class:`~pycc.correlatedderivs.CorrelatedDerivs` hierarchy,
matching the `MPwfn`/`MPderiv` and `CCwfn`/`CCderiv` split: `CIwfn` holds only wavefunction
quantities (amplitude solve, energy, raw densities); this class supplies the two density hooks
(`_unrelaxed_densities`, `_perturbed_unrelaxed_densities`) and inherits `relaxed_dipole`,
`gradient`, `polarizability`, `dipole_derivatives`, and `hessian` from the base unchanged. Only
the genuinely CISD-specific wave-function-overlap properties - `atomic_axial_tensors` and
`velocity_dipole_derivatives` - are custom code, together with the coupled-perturbed-CI (CPCI)
response machinery they (and the two density hooks) are built from.

DENSITY HOOKS
-------------
Both hooks return the correlation 1-PDM and the Klein-four-symmetrized PURE cumulant 2-PDM (a
0.25*(4-permutation) average, NO correlation-density cross term, no HF block -- see
`_cisd_symmetrize` for the convention derivation), true-normalized. The base's generalized-Fock
Lagrangian is built on the Fock-convention partitioning E_corr = Tr(D f) + Tr(Gam ERI) (like
MP2's densities); the mean-field coupling of the 1-PDM is supplied by the Lagrangian's own
termA/termB, not folded into Gam.

`_perturbed_unrelaxed_densities` solves the CPCI response directly from the base's own canonical,
full-occupied-space perturbed integrals (`df`, `deri`) via `_solve_cpci_ints` -- the CISD analog of
`CCderiv._perturbed_amplitudes`. This is the piece that used to be dead code: the base's second-
derivative path was previously bypassed by CISD-specific overrides of `dipole_derivatives`/`hessian`/
`polarizability`, which have been removed in favor of the base's inherited implementations.

CPCI/MAGNETIC/VECPOT MACHINERY
-------------------------------
`_cpci_ints`/`_solve_cpci`/`_solve_cpci_ints`/`_cpci_raw`, the magnetic/vector-potential integral
builders, and the raw perturbed-density builders live here (not on `CIwfn`) - they are about
derivatives, not the wavefunction. `_solve_cpci_ints` is the core iterative solve, taking already-
built perturbed integrals directly; `_solve_cpci` is the `Perturbation`-keyed, cached entry point
used by the AAT/VG-APT overlap code (magnetic/vecpot/nuclear, via `_cpci_ints`).

"""

from __future__ import annotations

import numpy as np

from .correlatedderivs import CorrelatedDerivs


class CIderiv(CorrelatedDerivs):
    """CISD correlation derivative-property driver. Constructed from a converged CIwfn (aliased
    `self.ci`). `relaxed_dipole`/`gradient`/`polarizability`/`dipole_derivatives`/
    `hessian` are inherited from `CorrelatedDerivs`, driven by the two density hooks below.
    `atomic_axial_tensors`/`velocity_dipole_derivatives` are custom wave-function-overlap
    constructions. Spatial (closed-shell RHF), all-electron only."""

    def __init__(self, ciwfn) -> None:
        super().__init__(ciwfn)
        self.ci = ciwfn                                # alias: this class uses .ci, the base .wfn

    def _cisd_symmetrize(self, D_pqrs_raw):
        return 0.25 * (D_pqrs_raw + D_pqrs_raw.transpose(1, 0, 3, 2)
                       + D_pqrs_raw.transpose(2, 3, 0, 1) + D_pqrs_raw.transpose(3, 2, 1, 0))

    def _unrelaxed_densities(self):
        """CISD unrelaxed reduced densities (D, Gam) as full-MO arrays."""
        _D_pq, D_pq_corr, D_pqrs = self.ci._cisd_densities()
        Gam = self._cisd_symmetrize(D_pqrs)
        return D_pq_corr, Gam

    def _perturbed_unrelaxed_densities(self, pert, df, deri, dL):
        """First-order response (d_x D, d_x Gam) of the CISD unrelaxed reduced densities to
        pert, in the same convention as _unrelaxed_densities. Solves the coupled-perturbed-CI
        response directly from the base's own canonical, full-occupied-space perturbed integrals. """
        dc1, dc2, dc0v, _dt1, _dt2 = self._solve_cpci_ints(df, deri, imaginary=False)
        dD_corr = self._perturbed_cisd_corr_opdm(dc1, dc2)
        dG_raw = self._perturbed_cisd_tpdm(dc1, dc2, dc0v)
        dGam = self._cisd_symmetrize(dG_raw)
        return dD_corr, dGam

    # coupled-perturbed-CI response 

    def _psi4_mints(self):
        """psi4 MintsHelper + C as psi4.core.Matrix"""
        if getattr(self, '_mints_cache', None) is None:
            import psi4
            mints = psi4.core.MintsHelper(self.ci.H.basisset)
            C_p4 = psi4.core.Matrix.from_array(np.asarray(self.ci.C))
            self._mints_cache = (mints, C_p4)
        return self._mints_cache

    def _build_magnetic_ints(self, beta):
        ct = self.contract
        nbf = self.ci.nmo
        o, v = self.ci.o, self.ci.v
        t = slice(0, nbf)
        no, nv = self.ci.no, self.ci.nv
        ERI = np.asarray(self.ci.H.ERI)
        F = np.asarray(self.ci.H.F)
        C = np.asarray(self.ci.C)
        eps = np.diag(F)
        mints, _ = self._psi4_mints()

        # A-matrix (antisymmetric perturbation)
        A_mag = -(2 * ERI - ERI.swapaxes(2, 3)) + (2 * ERI - ERI.swapaxes(2, 3)).swapaxes(1, 3)
        A_mag = A_mag.swapaxes(1, 2)
        G_mag = (ct('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no)) + A_mag[v, o, v, o])
        G_mag = np.linalg.inv(G_mag.reshape(nv * no, nv * no))

        # Magnetic dipole integrals
        L_AO = mints.ao_angular_momentum()
        h_mag = ct('mp,mn,nq->pq', C.conj(), -0.5 * L_AO[beta].np, C)

        # U_H  (full 4-block solve - exactly as in working code)
        U_H = np.zeros((nbf, nbf), dtype=complex)
        B_vo = h_mag[v, o]
        U_H[v, o] += (G_mag @ B_vo.reshape(nv * no)).reshape(nv, no)
        U_H[o, v] += U_H[v, o].T

        D_oo = (eps[o] - eps[o].reshape(-1, 1)) + np.eye(no)
        B_oo = (-h_mag[o, o].copy() + ct('em,iejm->ij', U_H[v, o], A_mag.swapaxes(1, 2)[o, v, o, o]))
        U_H[o, o] += B_oo / D_oo

        D_vv = (eps[v] - eps[v].reshape(-1, 1)) + np.eye(nv)
        B_vv = (-h_mag[v, v].copy() + ct('em,aebm->ab', U_H[v, o], A_mag.swapaxes(1, 2)[v, v, v, o]))
        U_H[v, v] += B_vv / D_vv

        for j in range(no):
            U_H[j, j] = 0
        for cc in range(no, nbf):
            U_H[cc, cc] = 0

        # dF/dH
        dF = np.zeros((nbf, nbf), dtype=complex)
        dF[o, o] -= h_mag[o, o].copy()
        dF[o, o] += (U_H[o, o] * eps[o].reshape(-1, 1) - U_H[o, o].swapaxes(0, 1) * eps[o])
        dF[o, o] += ct('em,iejm->ij', U_H[v, o], A_mag.swapaxes(1, 2)[o, v, o, o])

        dF[v, v] -= h_mag[v, v].copy()
        dF[v, v] += (U_H[v, v] * eps[v].reshape(-1, 1) - U_H[v, v].swapaxes(0, 1) * eps[v])
        dF[v, v] += ct('em,aebm->ab', U_H[v, o], A_mag.swapaxes(1, 2)[v, v, v, o])

        # dERI/dH
        dERI = np.zeros(ERI.shape, dtype=complex)
        dERI += ct('tr,pqts->pqrs', U_H[:, t], ERI[t, t, :, t])
        dERI += ct('ts,pqrt->pqrs', U_H[:, t], ERI[t, t, t, :])
        dERI -= ct('tp,tqrs->pqrs', U_H[:, t], ERI[:, t, t, t])
        dERI -= ct('tq,ptrs->pqrs', U_H[:, t], ERI[t, :, t, t])

        return dF, dERI, U_H

    def _build_vecpot_ints(self, gamma):
        ct = self.contract
        nbf = self.ci.nmo
        o, v = self.ci.o, self.ci.v
        t = slice(0, nbf)
        no, nv = self.ci.no, self.ci.nv
        ERI = np.asarray(self.ci.H.ERI)
        F = np.asarray(self.ci.H.F)
        C = np.asarray(self.ci.C)
        eps = np.diag(F)
        mints, _ = self._psi4_mints()

        # A-matrix for antisymmetric perturbation (same as magnetic)
        A_mag = -(2 * ERI - ERI.swapaxes(2, 3)) + (2 * ERI - ERI.swapaxes(2, 3)).swapaxes(1, 3)
        A_mag = A_mag.swapaxes(1, 2)
        G_mag = (ct('ab,ij,aibj->aibj', np.eye(nv), np.eye(no), F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no)) + A_mag[v, o, v, o])
        G_mag = np.linalg.inv(G_mag.reshape(nv * no, nv * no))

        # Linear momentum integrals: p_gamma = -i * nabla_gamma
        nabla_AO = mints.ao_nabla()
        h_A = ct('mp,mn,nq->pq', C.conj(), -nabla_AO[gamma].np, C)

        # Solve CPHF for U^A (identical structure to U^H)
        U_A = np.zeros((nbf, nbf), dtype=complex)
        B_vo = h_A[v, o]
        U_A[v, o] += (G_mag @ B_vo.reshape(nv * no)).reshape(nv, no)
        U_A[o, v] += U_A[v, o].T

        D_oo = (eps[o] - eps[o].reshape(-1, 1)) + np.eye(no)
        B_oo = (-h_A[o, o].copy() + ct('em,iejm->ij', U_A[v, o], A_mag.swapaxes(1, 2)[o, v, o, o]))
        U_A[o, o] += B_oo / D_oo

        D_vv = (eps[v] - eps[v].reshape(-1, 1)) + np.eye(nv)
        B_vv = (-h_A[v, v].copy() + ct('em,aebm->ab', U_A[v, o], A_mag.swapaxes(1, 2)[v, v, v, o]))
        U_A[v, v] += B_vv / D_vv

        for j in range(no):
            U_A[j, j] = 0
        for cc in range(no, nbf):
            U_A[cc, cc] = 0

        # dF/dA_gamma
        dF = np.zeros((nbf, nbf), dtype=complex)
        dF[o, o] -= h_A[o, o].copy()
        dF[o, o] += (U_A[o, o] * eps[o].reshape(-1, 1) - U_A[o, o].swapaxes(0, 1) * eps[o])
        dF[o, o] += ct('em,iejm->ij', U_A[v, o], A_mag.swapaxes(1, 2)[o, v, o, o])

        dF[v, v] -= h_A[v, v].copy()
        dF[v, v] += (U_A[v, v] * eps[v].reshape(-1, 1) - U_A[v, v].swapaxes(0, 1) * eps[v])
        dF[v, v] += ct('em,aebm->ab', U_A[v, o], A_mag.swapaxes(1, 2)[v, v, v, o])

        # dERI/dA_gamma (orbital response only, same sign pattern as magnetic)
        dERI = np.zeros(ERI.shape, dtype=complex)
        dERI += ct('tr,pqts->pqrs', U_A[:, t], ERI[t, t, :, t])
        dERI += ct('ts,pqrt->pqrs', U_A[:, t], ERI[t, t, t, :])
        dERI -= ct('tp,tqrs->pqrs', U_A[:, t], ERI[:, t, t, t])
        dERI -= ct('tq,ptrs->pqrs', U_A[:, t], ERI[t, :, t, t])

        return dF, dERI, U_A

    def _cpci_ints(self, pert):
        """(dF, dERI, U) for a cphf.Perturbation. Used for magnetic/vecpot (AAT/VG-APT) and for
        CIwfn's own (non-canonical, active-space) nuclear response - NOT for the field/nuclear
        perturbations that feed the base's second-derivative machinery, which instead thread the
        base's own canonical, full-occupied-space (df, deri) directly into _solve_cpci_ints."""
        if getattr(self, '_cpci_ints_cache', None) is None:
            self._cpci_ints_cache = {}
        if pert in self._cpci_ints_cache:
            return self._cpci_ints_cache[pert]
        cphf = self.ci.cphf
        if pert.kind == 'nuclear':
            dF = np.asarray(cphf.perturbed_fock(pert))
            dERI = np.asarray(cphf.perturbed_eri(pert))
            U = np.asarray(cphf.full_U(pert))
            result = (dF, dERI, U)
        elif pert.kind == 'magnetic':
            result = self._build_magnetic_ints(pert.comp)
        elif pert.kind == 'vecpot':
            result = self._build_vecpot_ints(pert.comp)
        elif pert.kind == 'field':
            dF = np.asarray(cphf.perturbed_fock(pert))
            dERI = np.asarray(cphf.perturbed_eri(pert))   # zero for electric field
            U = np.asarray(cphf.full_U(pert))
            result = (dF, dERI, U)
        else:
            raise ValueError(f"unknown perturbation kind {pert.kind!r}")
        self._cpci_ints_cache[pert] = result
        return result

    def _solve_cpci_ints(self, dF, dERI, imaginary=False, maxiter=100, diis_start=2, diis_max=8,
                          e_convergence=1e-11, d_convergence=1e-11):
        """Core coupled-perturbed-CI iterative solve given already-built perturbed integrals
        (dF, dERI) - the CISD analog of CCderiv._perturbed_amplitudes, factored out so both
        CIderiv's own _cpci_ints-driven entry point (_solve_cpci, below) and the base's
        canonical/full-occupied-space (df, deri) (via _perturbed_unrelaxed_densities) can drive
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

        D_pq, D_pq_corr, D_pqrs = ci._cisd_densities()
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

        for iteration in range(1, maxiter + 1):
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
            if iteration >= diis_start:
                dt1, dt2 = diis.extrapolate(dt1, dt2)

            dE_proj = (2.0 * c('ia,ia->', t1, dF[o, v])
                       + c('ijab,ijab->', t2, 2.0 * dERI[o, o, v, v] - dERI.swapaxes(2, 3)[o, o, v, v])
                       + 2.0 * c('ia,ia->', dt1, F[o, v])
                       + c('ijab,ijab->', dt2, 2.0 * ERI[o, o, v, v] - ERI.swapaxes(2, 3)[o, o, v, v]))

            delta_dE = abs(dE_proj - dE_proj_old)
            rms_dt1 = np.sqrt(np.sum((dt1 - dt1_old) ** 2))
            rms_dt2 = np.sqrt(np.sum((dt2 - dt2_old) ** 2))
            if iteration > 1 and (delta_dE < e_convergence and rms_dt1 < d_convergence
                                   and rms_dt2 < d_convergence):
                break

        dc0 = ci._cisd_dn0(dt1, dt2)
        if imaginary:
            dc0v = 0.0
            dc1 = n0 * dt1
            dc2 = n0 * dt2
        else:
            dc0v = dc0
            dc1 = dc0 * t1 + n0 * dt1
            dc2 = dc0 * t2 + n0 * dt2

        return dc1, dc2, dc0v, dt1, dt2

    def _solve_cpci(self, pert, maxiter=100, diis_start=2, diis_max=8,
                     e_convergence=1e-11, d_convergence=1e-11):
        """Coupled-perturbed CI, entry point keyed by Perturbation - used by the AAT/VG-APT
        overlap code below (magnetic/vecpot/nuclear via _cpci_ints). Cached by pert. Returns
        (dc1, dc2, dc0v). The base's second-derivative machinery does NOT go through this -
        see _perturbed_unrelaxed_densities, which calls _solve_cpci_ints directly with the
        base's own canonical (df, deri)."""
        if getattr(self, '_cpci_cache', None) is None:
            self._cpci_cache = {}
        if pert in self._cpci_cache:
            return self._cpci_cache[pert]
        dF, dERI, U = self._cpci_ints(pert)
        imaginary = pert.kind in ('magnetic', 'vecpot')
        dc1, dc2, dc0v, dt1, dt2 = self._solve_cpci_ints(
            dF, dERI, imaginary=imaginary, maxiter=maxiter, diis_start=diis_start,
            diis_max=diis_max, e_convergence=e_convergence, d_convergence=d_convergence)
        if getattr(self, '_cpci_raw_cache', None) is None:
            self._cpci_raw_cache = {}
        self._cpci_raw_cache[pert] = (dt1, dt2)
        result = (dc1, dc2, dc0v)
        self._cpci_cache[pert] = result
        return result

    def _cpci_raw(self, pert):
        if getattr(self, '_cpci_raw_cache', None) is None or pert not in self._cpci_raw_cache:
            self._solve_cpci(pert)
        return self._cpci_raw_cache[pert]

    # raw perturbed correlation-density builders (true-normalized)

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

    def _aat_dc_normalized(self, pert):
        dc1, dc2, dc0v = self._solve_cpci(pert)
        return dc0v, dc1, dc2

    def compute_Icc_AATs(self):
        """Term 1 of the AAT: direct state-vector overlap <dPsi_R|dPsi_H>."""
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        I_cc = np.zeros((3 * natom, 3))
        magH = {b: self._aat_dc_normalized(Perturbation('magnetic', b)) for b in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, dn1_R, dn2_R = self._aat_dc_normalized(pR)
            for beta in range(3):
                _, dn1_H, dn2_H = magH[beta]
                term = 2.0 * c('ia,ia->', dn1_R.conj(), dn1_H)
                term = term + c('ijab,ijab->', (2.0 * dn2_R - dn2_R.swapaxes(2, 3)).conj(), dn2_H)
                I_cc[la, beta] = term.real
        return I_cc

    def compute_Iphic_AATs(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        AAT_phic = np.zeros((3 * natom, 3), dtype=complex)
        magH = {b: self._aat_dc_normalized(Perturbation('magnetic', b)) for b in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, _, U_R = self._cpci_ints(pR)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                dn0_H, dn1_H, dn2_H = magH[beta]
                R_pq = self._build_Dtilde(dn1_H, dn2_H, dn0_H)
                AAT_phic[la, beta] = c('pq,qp->', R_pq, Ur_eff)
        return AAT_phic.real

    def compute_Iphiphi_AATs(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        _, D_pq, _ = self.ci._cisd_densities()   # correlation-only 1-PDM (true-normalized)
        I_pp = np.zeros((3 * natom, 3))
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, _, U_R = self._cpci_ints(pR)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                _, _, U_H = self._cpci_ints(Perturbation('magnetic', beta))
                I_pp[la, beta] = c('pq,pq->', D_pq, U_H.T @ Ur_eff).real
        return I_pp

    def compute_Icphi_AATs(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        AAT_cphi = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            dn0_R, dn1_R, dn2_R = self._aat_dc_normalized(pR)
            R_pq = self._build_Dtilde(dn1_R, dn2_R, dn0_R)
            for beta in range(3):
                _, _, U_H = self._cpci_ints(Perturbation('magnetic', beta))
                AAT_cphi[la, beta] = c('pq,pq->', R_pq, U_H)
        return AAT_cphi.real

    def atomic_axial_tensors(self, gauge: str = 'canonical') -> np.ndarray:
        """CISD correlation AAT, shape (natom, 3, 3): the four electronic overlap blocks with the
        correlation 1-PDM in Iphiphi. The SCF reference and the nuclear term (Z_A/4) eps_abc R_c
        are supplied by the pycc.aat facade (HFwfn.atomic_axial_tensors + _nuclear_aat), matching
        MPderiv."""
        natom = self.ci.ref.molecule().natom()
        total = (self.compute_Icc_AATs() + self.compute_Iphic_AATs() + self.compute_Iphiphi_AATs() + self.compute_Icphi_AATs())
        return total.reshape(natom, 3, 3)

    def compute_Icc_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        I_cc = np.zeros((3 * natom, 3), dtype=complex)
        vecA = {g: self._aat_dc_normalized(Perturbation('vecpot', g)) for g in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            _, dn1_R, dn2_R = self._aat_dc_normalized(Perturbation('nuclear', (A, beta_)))
            for gamma in range(3):
                _, dn1_A, dn2_A = vecA[gamma]
                I_cc[la, gamma] = (2.0 * c('ia,ia->', dn1_R.conj(), dn1_A) + c('ijab,ijab->', (2.0 * dn2_R - dn2_R.swapaxes(2, 3)).conj(), dn2_A))
        return I_cc

    def compute_Icphi_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        Icphi = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            dn0_R, dn1_R, dn2_R = self._aat_dc_normalized(Perturbation('nuclear', (A, beta_)))
            D_tilde_R = self._build_Dtilde(dn1_R, dn2_R, dn0_R)
            for gamma in range(3):
                _, _, U_A = self._cpci_ints(Perturbation('vecpot', gamma))
                Icphi[la, gamma] = c('pq,pq->', D_tilde_R, U_A)
        return Icphi

    def compute_Iphic_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        Iphic = np.zeros((3 * natom, 3), dtype=complex)
        vecA = {g: self._aat_dc_normalized(Perturbation('vecpot', g)) for g in range(3)}
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            pR = Perturbation('nuclear', (A, beta_))
            _, _, U_R = self._cpci_ints(pR)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                dn0_A, dn1_A, dn2_A = vecA[gamma]
                D_tilde_A = self._build_Dtilde(dn1_A, dn2_A, dn0_A)
                Iphic[la, gamma] = c('pq,qp->', D_tilde_A.conj(), Ur_eff)
        return Iphic

    def compute_Iphiphi_VG_APT(self):
        from .cphf import Perturbation
        c = self.contract
        natom = self.ci.derivatives.natom
        _, D_pq, _ = self.ci._cisd_densities()   # correlation-only 1-PDM (true-normalized)
        I_pp = np.zeros((3 * natom, 3), dtype=complex)
        for la in range(3 * natom):
            A, beta_ = divmod(la, 3)
            _, _, U_R = self._cpci_ints(Perturbation('nuclear', (A, beta_)))
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                _, _, U_A = self._cpci_ints(Perturbation('vecpot', gamma))
                I_pp[la, gamma] = c('pq,pq->', D_pq, U_A.T @ Ur_eff)
        return I_pp

    def velocity_dipole_derivatives(self, gauge: str = 'canonical') -> np.ndarray:
        """CISD correlation velocity-gauge APT, shape (natom, 3, 3): -2 times the four overlap
        blocks with the correlation 1-PDM in Iphiphi. The SCF reference and the Z_A delta nuclear
        term are supplied by the pycc.apt(gauge='velocity') facade, matching MPderiv."""
        natom = self.ci.ref.molecule().natom()
        overlap_total = (self.compute_Icc_VG_APT() + self.compute_Icphi_VG_APT() + self.compute_Iphic_VG_APT() + self.compute_Iphiphi_VG_APT())
        return (-2.0 * overlap_total).real.reshape(natom, 3, 3)
