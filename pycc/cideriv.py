"""
cideriv.py: CISD analytic-derivative property driver.

`CIderiv` is the CISD leaf of the :class:`~pycc.correlatedderivs.CorrelatedDerivs` hierarchy.

* `relaxed_dipole` and `gradient` are inherited from the base, driven by the two density hooks
  below (:meth:`_unrelaxed_densities` / :meth:`_perturbed_unrelaxed_densities`, built from
  `CIwfn._corr_QGX()`'s true-normalized correlation 1-/2-PDM).
* `polarizability`, `dipole_derivatives`, and `hessian` are custom, using `CIwfn`'s own
  equilibrium Z-vector (`_zvector`/`_corr_QGX`/`_solve_dz_dR`) rather than the base's generic
  machinery.
* `atomic_axial_tensors` and `velocity_dipole_derivatives` are custom wave-function-overlap
  constructions the base does not provide at all.

CPCI/magnetic/vecpot machinery (`_cpci_ints`/`_solve_cpci`/`_cpci_raw`, the magnetic/vecpot
integral builders) and the density builders (`_perturbed_cisd_corr_opdm`/`_perturbed_cisd_tpdm`/
`_corr_QGX`/`_zvector`) stay on `CIwfn` and are called from here.

"""

from __future__ import annotations

import numpy as np

from .correlatedderivs import CorrelatedDerivs


class CIderiv(CorrelatedDerivs):
    """CISD correlation derivative-property driver. Constructed from a converged CIwfn
    (aliased `self.ci`). See the module docstring for which properties are inherited vs custom.
    Spatial (closed-shell RHF), all-electron only."""

    def __init__(self, ciwfn) -> None:
        super().__init__(ciwfn)
        self.ci = ciwfn                                # alias: this class uses .ci, the base .wfn

    def _unrelaxed_densities(self):
        """(D, Gam) from CIwfn._corr_QGX(). Feeds relaxed_dipole/gradient only."""
        Q_corr, G_corr, _X_corr = self.ci._corr_QGX()
        return Q_corr, G_corr

    def _perturbed_unrelaxed_densities(self, pert, df=None, deri=None, dL=None):
        """First-order response (d_x D, d_x Gam) to pert. Feeds base-driven gradient/relaxed_dipole
        second-derivative uses; dipole_derivatives/hessian/polarizability use their own _solve_dz_dR below instead."""
        ci = self.ci
        no = ci.no
        dc1, dc2, dc0v = ci._solve_cpci(pert)

        dD_corr = ci._perturbed_cisd_corr_opdm(dc1, dc2)
        dG_raw = ci._perturbed_cisd_tpdm(dc1, dc2, dc0v)

        dG_sym = 0.25 * (dG_raw + dG_raw.transpose(1, 0, 3, 2) + dG_raw.transpose(2, 3, 0, 1) + dG_raw.transpose(3, 2, 1, 0))
        dGam = dG_sym.copy()
        dG_cross = np.zeros_like(dGam)
        for i in range(no):
            dG_cross[i, :, i, :] += 2.0 * dD_corr
            dG_cross[:, i, :, i] += 2.0 * dD_corr
            dG_cross[i, :, :, i] -= dD_corr.T
            dG_cross[:, i, i, :] -= dD_corr
        dGam = dGam + 0.5 * dG_cross

        return dD_corr, dGam

    # length-gauge APT 

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        """CISD correlation static polarizability (a.u.), shape (3, 3):
        alpha[a,b] = -d^2 E_corr / dF_a dF_b. Both routes are custom,
        using CIwfn's own equilibrium Z-vector.

        route='2n+1' (default): differentiate the relaxed dipole a second
        time. See _polarizability_2n1().
        route='explicit': differentiate the energy twice, reusing the
        Hessian's T0-T5/Z-vector machinery for field-field pairs. See
        _polarizability_explicit()."""
        if route == '2n+1':
            return self._polarizability_2n1()
        if route == 'explicit':
            return self._polarizability_explicit()
        raise ValueError(f"unknown polarizability route {route!r} (use '2n+1' or 'explicit')")

    def _polarizability_2n1(self) -> np.ndarray:
        """2n+1 route: uses CIwfn's own equilibrium Z-vector and perturbed
        Z-vector solve, matching dipole_derivatives()/hessian()."""
        from .cphf import Perturbation
        c = self.contract
        ci = self.ci
        o, v = ci.o, ci.v

        zv = ci._zvector()
        Drel = zv['Q_relaxed'].copy()
        for i in range(ci.no):
            Drel[i, i] -= 2.0
        mu = [np.asarray(ci.H.mu[a]) for a in range(3)]

        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            U_b = np.asarray(ci.cphf.full_U(pert))
            dc1, dc2, dc0v = ci._solve_cpci(pert)
            dc1, dc2 = dc1.real, dc2.real
            dc0v = dc0v.real if hasattr(dc0v, 'real') else dc0v
            dz, dD_corr = self._solve_dz_dR(pert, dc1, dc2, dc0v)
            dDrel = dD_corr.copy()
            dDrel[v, o] += dz
            dDrel[o, v] += dz.T
            for a in range(3):
                rot = U_b.T @ mu[a] + mu[a] @ U_b
                alpha[a, b] = (c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot))
        return alpha

    def _polarizability_explicit(self) -> np.ndarray:
        """Explicit route: reuses the Hessian's Z-vector machinery
        (build_xi_ab/build_X_deriv/build_cpci_term/_hess_build_Y/_hess_X_tilde)
        for field-field pairs - all second-derivative skeletons vanish
        for a static field, so no U^{fg} solve is needed."""
        from .cphf import Perturbation
        ct = self.contract
        ci = self.ci
        o, v, no, nv = ci.o, ci.v, ci.no, ci.nv
        F = np.asarray(ci.H.F)
        eps = np.diag(F)
        nmo = ci.nmo

        zv = ci._zvector()
        Q, G, _ = ci._corr_QGX()
        X_t = self._hess_X_tilde()
        Y = self._hess_build_Y()
        ERI = np.asarray(ci.H.ERI)
        z = zv['z_ai']
        n0 = ci._normalized_amplitudes()[0]

        ERI_M = ERI.transpose(0, 2, 1, 3)
        W_M = 2 * ERI_M - ERI_M.transpose(0, 2, 1, 3)
        A_M = 4 * ERI_M - ERI_M.transpose(0, 2, 1, 3) - ERI_M.transpose(0, 3, 2, 1)

        mu = [np.asarray(ci.H.mu[a]) for a in range(3)]
        zero_g = np.zeros((nmo, nmo, nmo, nmo))
        zero_S = np.zeros((nmo, nmo))

        # First-order field quantities: U^a, field-CPCI raw amplitudes,
        # X_deriv[a] = -Q_corr.mu_a, Fa_M[a] = -mu_a (g-dependent piece is 0)
        U_all, dT1_all, dT2_all, X_deriv, Fa_M = {}, {}, {}, {}, {}
        for a in range(3):
            pert = Perturbation('field', a)
            U_all[a] = np.asarray(ci.cphf.full_U(pert))
            dT1_all[a], dT2_all[a] = ci._cpci_raw(pert)
            h_a_skel = -mu[a]
            X_deriv[a] = self.build_X_deriv(h_a_skel, zero_g)
            Fa_M[a] = h_a_skel.copy()
            # Aa_M[a] is identically zero for a field (built from g^f = 0)

        alpha = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                U_a, U_b = U_all[a], U_all[b]

                # T0: skeleton density contraction - vanishes (h^{fg}=g^{fg}=0)
                t0_val = 0.0

                # T2: Z-vector replacement (no U^{fg} solved - same as hessian())
                xi = self.build_xi_ab(U_a, U_b, zero_S, zero_S, zero_S)
                t2_xiXt = -ct('ij,ji->', xi, X_t)
                # F_ab_M = 0 (h^{fg} = g^{fg} = 0)
                Bab = np.zeros((nmo, nmo))
                Bab -= ct('ij,j->ij', xi, eps)
                Bab -= ct('kl,ijkl->ij', xi[o, o], W_M[:, :, o, o])
                Bab += ct('ki,kj->ij', U_a, Fa_M[b])
                Bab += ct('ki,kj->ij', U_b, Fa_M[a])
                Bab += ct('kj,ik->ij', U_a, Fa_M[b])
                Bab += ct('kj,ik->ij', U_b, Fa_M[a])
                Bab += ct('ki,kj,k->ij', U_a, U_b, eps)
                Bab += ct('ki,kj,k->ij', U_b, U_a, eps)
                UaUbT = U_a[:, o] @ U_b[:, o].T
                Bab += ct('kl,ijkl->ij', UaUbT, A_M)
                temp5_b = ct('lm,kjlm->kj', U_b[:, o], A_M[:, :, :, o])
                temp5_a = ct('lm,kjlm->kj', U_a[:, o], A_M[:, :, :, o])
                Bab += ct('ki,kj->ij', U_a, temp5_b)
                Bab += ct('ki,kj->ij', U_b, temp5_a)
                temp6_b = ct('lm,iklm->ik', U_b[:, o], A_M[:, :, :, o])
                temp6_a = ct('lm,iklm->ik', U_a[:, o], A_M[:, :, :, o])
                Bab += ct('kj,ik->ij', U_a, temp6_b)
                Bab += ct('kj,ik->ij', U_b, temp6_a)
                # Aa_M[a]/[b] terms dropped - identically zero for a field
                t2_zvec = 2.0 * ct('ai,ai->', Bab[v, o], z)
                t2_val = t2_xiXt + t2_zvec

                # T3+T4: orbital-response Lagrangian + Y tensor (Y equilibrium, reused)
                t3_val = 2.0 * (ct('ij,ij->', U_b, X_deriv[a])
                                + ct('ij,ij->', U_a, X_deriv[b]))
                t4_val = 2.0 * ct('ij,kl,ijkl->', U_a, U_b, Y)

                # T5: CPCI amplitude response (field-perturbed raw amplitudes)
                t5_val = n0**2 * self.build_cpci_term(
                    dT1_all[a], dT2_all[a], dT1_all[b], dT2_all[b]).real

                # alpha = -d^2 E_corr / dF dG (hessian() returns +d^2E/dXdY
                # with no extra sign, so the field-field analog needs the
                # explicit minus here)
                alpha[a, b] = -(t0_val + t2_val + t3_val + t4_val + t5_val).real

        alpha = 0.5 * (alpha + alpha.T)
        return alpha

    def dipole_derivatives(self, route: str = 'explicit') -> np.ndarray:
        """CISD CORRELATION-ONLY LG-APT, shape (natom, 3, 3), [A, beta, alpha].
        """
        from .cphf import Perturbation
        c = self.contract
        o, v = self.ci.o, self.ci.v
        zv = self.ci._zvector()
        # Correlation-only relaxed density: Q_relaxed minus the HF 2*delta_oo.
        # T3/T3z are already pure correlation (amplitude/Z-vector response).
        Qt = zv['Q_relaxed'].copy()
        for i in range(self.ci.no):
            Qt[i, i] -= 2.0
        natom = self.ci.derivatives.natom
        h_E = [np.asarray(self.ci.H.mu[f]) for f in range(3)]

        APT = np.zeros((natom, 3, 3))
        for A in range(natom):
            dip = self.ci.derivatives.dipole(A)
            for beta in range(3):
                pert = Perturbation('nuclear', (A, beta))
                U_R = np.asarray(self.ci.cphf.full_U(pert))
                dc1, dc2, dc0v = self.ci._solve_cpci(pert)
                dc1, dc2 = dc1.real, dc2.real
                dc0v = dc0v.real if hasattr(dc0v, 'real') else dc0v
                dz, dD_corr = self._solve_dz_dR(pert, dc1, dc2, dc0v)

                for alpha in range(3):
                    h_RE = np.asarray(dip[alpha * 3 + beta])
                    T1 = c('ij,ij->', Qt, h_RE)
                    T2 = (c('rp,pq,rq->', U_R, Qt, h_E[alpha]) + c('pq,ps,sq->', Qt, h_E[alpha], U_R))
                    T3 = c('pq,pq->', dD_corr, h_E[alpha])
                    T3z = 2.0 * c('ai,ai->', dz, h_E[alpha][v, o])
                    APT[A, beta, alpha] = (T1 + T2 + T3 + T3z).real
        return APT

    def _solve_dz_dR(self, pert, dc1, dc2, dc0v):
        c = self.contract
        o, v, no, nv = self.ci.o, self.ci.v, self.ci.no, self.ci.nv
        zv = self.ci._zvector()
        z = zv['z_ai']

        dF, dERI, _ = self.ci._cpci_ints(pert)
        dD_corr = self.ci._perturbed_cisd_corr_opdm(dc1, dc2)
        dD_pqrs = self.ci._perturbed_cisd_tpdm(dc1, dc2, dc0v)

        o_ = self.ci.o
        dh_mo = dF - c('piqi->pq', 2.0 * dERI[:, o_, :, o_] - dERI.swapaxes(2, 3)[:, o_, :, o_])
        dG_sym = 0.25 * (dD_pqrs + dD_pqrs.transpose(1, 0, 3, 2) + dD_pqrs.transpose(2, 3, 0, 1) + dD_pqrs.transpose(3, 2, 1, 0))
        dG = dG_sym.copy()
        dG_cross = np.zeros_like(dG)
        for i in range(no):
            dG_cross[i, :, i, :] += 2.0 * dD_corr
            dG_cross[:, i, :, i] += 2.0 * dD_corr
            dG_cross[i, :, :, i] -= dD_corr.T
            dG_cross[:, i, i, :] -= dD_corr
        dG = dG + 0.5 * dG_cross

        ERI = np.asarray(self.ci.H.ERI)
        dX = c('jm,im->ij', dD_corr, zv['h_mo']) + c('jm,im->ij', zv['Q'], dh_mo)
        dX = dX + 2.0 * c('jmkl,imkl->ij', dG, ERI) + 2.0 * c('jmkl,imkl->ij', zv['G'], dERI)

        db = -(dX[v, o] - dX[o, v].T)

        dL = 2.0 * dERI - dERI.swapaxes(2, 3)
        dG_mat_vovo = (dL[v, o, o, v].transpose(0, 2, 3, 1)
                       + dL[v, v, o, o].transpose(0, 2, 1, 3))
        dG_mat_vovo = dG_mat_vovo + c('ab,ij->aibj', dF[v, v], np.eye(no))
        dG_mat_vovo = dG_mat_vovo - c('ab,ij->aibj', np.eye(nv), dF[o, o])
        dGz = c('aibj,bj->ai', dG_mat_vovo, z)

        rhs = db - dGz
        dz_ov = self.ci.cphf.solve(rhs.T)
        dz = dz_ov.T
        return dz, dD_corr


    # Hessian

    def _hess_build_Y(self):
        """Y_ijkl double-UU tensor, correlation densities only (linear in Q/G,
        so the HF part belongs to the SCF reference Hessian)."""
        ct = self.contract
        zv = self.ci._zvector()
        Q, G, _ = self.ci._corr_QGX()
        h_mo = zv['h_mo']
        ERI = np.asarray(self.ci.H.ERI)
        Y = ct('jl,ik->ijkl', Q, h_mo)
        Y = Y + 2.0 * ct('jlmn,ikmn->ijkl', G, ERI)
        Y = Y + 2.0 * ct('jmln,imkn->ijkl', G, ERI)
        Y = Y + 2.0 * ct('jmnl,imnk->ijkl', G, ERI)
        return Y

    def _hess_X_tilde(self):
        """Correlation-only X_tilde: X_corr (linear in the corr densities) with
        the Z-vector-corrected vo block. z itself is solved from the FULL
        Lagrangian in _zvector() (unchanged from the validated code); it is a
        pure correlation quantity since the HF part of X satisfies Brillouin
        (X_HF[v,o] - X_HF[o,v].T = 0) and contributes nothing to the rhs."""
        ct = self.contract
        o, v, no, nv = self.ci.o, self.ci.v, self.ci.no, self.ci.nv
        zv = self.ci._zvector()
        _, _, X_corr = self.ci._corr_QGX()
        F, ERI = np.asarray(self.ci.H.F), np.asarray(self.ci.H.ERI)
        A = (2.0 * ERI - ERI.swapaxes(2, 3)) + (2.0 * ERI - ERI.swapaxes(2, 3)).swapaxes(1, 3)
        A = A.swapaxes(1, 2)
        G_mat = (ct('ab,ij->aibj', np.eye(nv), np.eye(no)) * (F[v, v].reshape(nv, 1, nv, 1) - F[o, o].reshape(1, no, 1, no)) + A[v, o, v, o])
        X_tilde = X_corr.copy()
        X_tilde[v, o] += ct('aibj,bj->ai', G_mat, zv['z_ai'])
        return X_tilde

    def build_xi_ab(self, U_a, U_b, S_a, S_b, S_ab):
        c = self.contract
        xi = S_ab.copy()
        xi = xi + c('im,jm->ij', U_a, U_b) + c('im,jm->ij', U_b, U_a)
        xi = xi - c('im,jm->ij', S_a, S_b) - c('im,jm->ij', S_b, S_a)
        return xi

    def build_X_deriv(self, h_a_skel, g_a_skel):
        """Skeleton-derivative Lagrangian, correlation densities only."""
        ct = self.contract
        Q, G, _ = self.ci._corr_QGX()
        Xa = ct('jm,im->ij', Q, h_a_skel)
        Xa = Xa + 2.0 * ct('jmkl,imkl->ij', G, g_a_skel)
        return Xa

    def build_cpci_term(self, dc1_a, dc2_a, dc1_b, dc2_b):
        """The pure linear CI-Jacobian action on dc1_b/dc2_b."""
        c = self.contract
        o, v = self.ci.o, self.ci.v
        F, ERI = np.asarray(self.ci.H.F), np.asarray(self.ci.H.ERI)
        E_tot = self.ci.eci

        sig1 = -E_tot * dc1_b
        sig1 = sig1 - c('ji,ja->ia', F[o, o], dc1_b)
        sig1 = sig1 + c('ab,ib->ia', F[v, v], dc1_b)
        sig1 = sig1 + c('jabi,jb->ia', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dc1_b)
        sig1 = sig1 + c('jb,ijab->ia', F[o, v], 2.0 * dc2_b - dc2_b.swapaxes(2, 3))
        sig1 = sig1 + c('ajbc,ijbc->ia', 2.0 * ERI[v, o, v, v] - ERI.swapaxes(2, 3)[v, o, v, v], dc2_b)
        sig1 = sig1 - c('kjib,kjab->ia', 2.0 * ERI[o, o, o, v] - ERI.swapaxes(2, 3)[o, o, o, v], dc2_b)

        sig2 = -E_tot * dc2_b
        sig2 = sig2 + c('abcj,ic->ijab', ERI[v, v, v, o], dc1_b)
        sig2 = sig2 + c('abic,jc->ijab', ERI[v, v, o, v], dc1_b)
        sig2 = sig2 - c('kbij,ka->ijab', ERI[o, v, o, o], dc1_b)
        sig2 = sig2 - c('akij,kb->ijab', ERI[v, o, o, o], dc1_b)
        sig2 = sig2 + c('ac,ijcb->ijab', F[v, v], dc2_b)
        sig2 = sig2 + c('bc,ijac->ijab', F[v, v], dc2_b)
        sig2 = sig2 - c('ki,kjab->ijab', F[o, o], dc2_b)
        sig2 = sig2 - c('kj,ikab->ijab', F[o, o], dc2_b)
        sig2 = sig2 + c('klij,klab->ijab', ERI[o, o, o, o], dc2_b)
        sig2 = sig2 + c('abcd,ijcd->ijab', ERI[v, v, v, v], dc2_b)
        sig2 = sig2 - c('kbcj,ikca->ijab', ERI[o, v, v, o], dc2_b)
        sig2 = sig2 + c('kaci,kjcb->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dc2_b)
        sig2 = sig2 - c('kbic,kjac->ijab', ERI[o, v, o, v], dc2_b)
        sig2 = sig2 - c('kaci,kjbc->ijab', ERI[o, v, v, o], dc2_b)
        sig2 = sig2 + c('kbcj,ikac->ijab', 2.0 * ERI[o, v, v, o] - ERI.swapaxes(2, 3)[o, v, v, o], dc2_b)
        sig2 = sig2 - c('kajc,ikcb->ijab', ERI[o, v, o, v], dc2_b)

        return -2.0 * (2.0 * c('ia,ia->', dc1_a, sig1) + c('ijab,ijab->', 2.0 * dc2_a - dc2_a.swapaxes(2, 3), sig2)).real

    def hessian(self, route: str = 'explicit') -> np.ndarray:
        """CISD CORRELATION-ONLY nuclear Hessian, shape (3*natom, 3*natom):
        the T0-T5 assembly with the correlation densities in the density with linear
        terms (T0, xi-X_tilde, T3, T4, the HF blocks of those terms are the
        SCF reference Hessian) and the unchanged pure-correlation Z-vector (Bab.z)
        and CPCI (T5) terms; nuclear-repulsion block dropped. Reference and
        nuclear are supplied by the pycc.hessian facade, matching MPwfn's
        contract."""
        from .cphf import Perturbation
        ct = self.contract
        o, v, no = self.ci.o, self.ci.v, self.ci.no
        mints, C_p4 = self.ci._psi4_mints()
        mol = self.ci.ref.molecule()
        natom = mol.natom()
        ndof = 3 * natom
        F = np.asarray(self.ci.H.F)
        eps = np.diag(F)

        zv = self.ci._zvector()
        Q, G, _ = self.ci._corr_QGX()   # correlation densities (T0/T3/T4/xi-X)
        X_t = self._hess_X_tilde()
        Y = self._hess_build_Y()
        ERI = np.asarray(self.ci.H.ERI)
        z = zv['z_ai']
        n0 = self.ci._normalized_amplitudes()[0]

        ERI_M = ERI.transpose(0, 2, 1, 3)
        W_M = 2 * ERI_M - ERI_M.transpose(0, 2, 1, 3)
        A_M = 4 * ERI_M - ERI_M.transpose(0, 2, 1, 3) - ERI_M.transpose(0, 3, 2, 1)

        def to_mulliken(g_dirac):
            return g_dirac.transpose(0, 2, 1, 3)

        h_skel_1, g_skel_1, S_skel_1 = {}, {}, {}
        U_all, dT1_all, dT2_all = {}, {}, {}

        for N1 in range(natom):
            T_core = mints.mo_oei_deriv1('KINETIC', N1, C_p4, C_p4)
            V_core = mints.mo_oei_deriv1('POTENTIAL', N1, C_p4, C_p4)
            S_core = mints.mo_oei_deriv1('OVERLAP', N1, C_p4, C_p4)
            E_core = mints.mo_tei_deriv1(N1, C_p4, C_p4, C_p4, C_p4)
            for a in range(3):
                la = 3 * N1 + a
                h_skel_1[la] = T_core[a].np + V_core[a].np
                S_skel_1[la] = S_core[a].np
                g_skel_1[la] = E_core[a].np.swapaxes(1, 2)
                pert = Perturbation('nuclear', (N1, a))
                U_all[la] = np.asarray(self.ci.cphf.full_U(pert))
                dT1_all[la], dT2_all[la] = self.ci._cpci_raw(pert)

        X_deriv = {la: self.build_X_deriv(h_skel_1[la], g_skel_1[la]) for la in range(ndof)}

        Fa_M, Aa_M = {}, {}
        for la in range(ndof):
            g_a_M = to_mulliken(g_skel_1[la])
            Fa = h_skel_1[la].copy()
            Fa += (ct('ijkk->ij', 2 * g_a_M[:, :, o, o]) - ct('ikjk->ij', g_a_M[:, o, :, o]))
            Fa_M[la] = Fa
            Aa_M[la] = (4 * g_a_M - g_a_M.transpose(0, 2, 1, 3) - g_a_M.transpose(0, 3, 2, 1))

        Hessian = np.zeros((ndof, ndof))

        def _assemble_ab(a, b, h_ab_sk, S_ab_sk, g_ab_sk, N1, N2):
            la, lb = 3 * N1 + a, 3 * N2 + b
            U_a, U_b = U_all[la], U_all[lb]

            # T1: skeleton density contraction
            t0_val = ct('ij,ij->', Q, h_ab_sk)
            t0_val += ct('pqrs,pqrs->', G, g_ab_sk)

            # T2: Z-vector replacement
            xi = self.build_xi_ab(U_a, U_b, S_skel_1[la], S_skel_1[lb], S_ab_sk)
            t2_xiXt = -ct('ij,ji->', xi, X_t)
            g_ab_sk_M = to_mulliken(g_ab_sk)
            F_ab_M = (h_ab_sk + ct('ijkk->ij', 2 * g_ab_sk_M[:, :, o, o]) - ct('ikjk->ij', g_ab_sk_M[:, o, :, o]))
            Bab = F_ab_M.copy()
            Bab -= ct('ij,j->ij', xi, eps)
            Bab -= ct('kl,ijkl->ij', xi[o, o], W_M[:, :, o, o])
            Bab += ct('ki,kj->ij', U_a, Fa_M[lb])
            Bab += ct('ki,kj->ij', U_b, Fa_M[la])
            Bab += ct('kj,ik->ij', U_a, Fa_M[lb])
            Bab += ct('kj,ik->ij', U_b, Fa_M[la])
            Bab += ct('ki,kj,k->ij', U_a, U_b, eps)
            Bab += ct('ki,kj,k->ij', U_b, U_a, eps)
            UaUbT = U_a[:, o] @ U_b[:, o].T
            Bab += ct('kl,ijkl->ij', UaUbT, A_M)
            temp5_b = ct('lm,kjlm->kj', U_b[:, o], A_M[:, :, :, o])
            temp5_a = ct('lm,kjlm->kj', U_a[:, o], A_M[:, :, :, o])
            Bab += ct('ki,kj->ij', U_a, temp5_b)
            Bab += ct('ki,kj->ij', U_b, temp5_a)
            temp6_b = ct('lm,iklm->ik', U_b[:, o], A_M[:, :, :, o])
            temp6_a = ct('lm,iklm->ik', U_a[:, o], A_M[:, :, :, o])
            Bab += ct('kj,ik->ij', U_a, temp6_b)
            Bab += ct('kj,ik->ij', U_b, temp6_a)
            Bab += ct('kl,ijkl->ij', U_a[:, o], Aa_M[lb][:, :, :, o])
            Bab += ct('kl,ijkl->ij', U_b[:, o], Aa_M[la][:, :, :, o])
            t2_zvec = 2.0 * ct('ai,ai->', Bab[v, o], z)
            t2_val = t2_xiXt + t2_zvec

            # T3+T4: orbital-response Lagrangian + Y tensor
            t3_val = 2.0 * (ct('ij,ij->', U_b, X_deriv[la])+ ct('ij,ij->', U_a, X_deriv[lb]))
            t4_val = 2.0 * ct('ij,kl,ijkl->', U_a, U_b, Y)

            # T5: CPCI amplitude response
            t5_val = n0**2 * self.build_cpci_term(
                dT1_all[la], dT2_all[la], dT1_all[lb], dT2_all[lb]).real

            return la, lb, (t0_val + t2_val + t3_val + t4_val + t5_val).real

        for N1 in range(natom):
            for N2 in range(natom):
                T_ab = mints.mo_oei_deriv2('KINETIC', N1, N2, C_p4, C_p4)
                V_ab = mints.mo_oei_deriv2('POTENTIAL', N1, N2, C_p4, C_p4)
                S_ab = mints.mo_oei_deriv2('OVERLAP', N1, N2, C_p4, C_p4)
                E_ab = mints.mo_tei_deriv2(N1, N2, C_p4, C_p4, C_p4, C_p4)

                h_ab = {(a, b): T_ab[3 * a + b].np + V_ab[3 * a + b].np for a in range(3) for b in range(3)}
                S_ab_np = {(a, b): S_ab[3 * a + b].np for a in range(3) for b in range(3)}
                g_ab = {(a, b): E_ab[3 * a + b].np.swapaxes(1, 2) for a in range(3) for b in range(3)}

                for a in range(3):
                    for b in range(3):
                        la, lb, val = _assemble_ab(
                            a, b, h_ab[(a, b)], S_ab_np[(a, b)], g_ab[(a, b)], N1, N2)
                        Hessian[la, lb] = val

                del T_ab, V_ab, S_ab, E_ab, h_ab, S_ab_np, g_ab

        Hessian = 0.5 * (Hessian + Hessian.T)
        return Hessian

    # CISD-specific properties (overlap formulations; not part of the base)
    # AAT and the velocity-gauge APT are wave-function-overlap constructions (Krishnan, Shumberger &
    # Crawford)

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
        dc1, dc2, dc0v = self.ci._solve_cpci(pert)
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
            _, _, U_R = self.ci._cpci_ints(pR)
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
            _, _, U_R = self.ci._cpci_ints(pR)
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for beta in range(3):
                _, _, U_H = self.ci._cpci_ints(Perturbation('magnetic', beta))
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
                _, _, U_H = self.ci._cpci_ints(Perturbation('magnetic', beta))
                AAT_cphi[la, beta] = c('pq,pq->', R_pq, U_H)
        return AAT_cphi.real

    def atomic_axial_tensors(self, gauge: str = 'canonical') -> np.ndarray:
        """CISD correlation AAT, shape (natom, 3, 3): the four electronic overlap blocks with the
        correlation 1-PDM in Iphiphi. The SCF reference and the nuclear term (Z_A/4) eps_abc R_c
        are supplied by the pycc.aat facade (HFwfn.atomic_axial_tensors + _nuclear_aat), matching
        MPderiv's contract."""
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
                _, _, U_A = self.ci._cpci_ints(Perturbation('vecpot', gamma))
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
            _, _, U_R = self.ci._cpci_ints(pR)
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
            _, _, U_R = self.ci._cpci_ints(Perturbation('nuclear', (A, beta_)))
            half_S = np.asarray(self.ci.derivatives.overlap_half(A)[beta_])
            Ur_eff = U_R + half_S.T
            for gamma in range(3):
                _, _, U_A = self.ci._cpci_ints(Perturbation('vecpot', gamma))
                I_pp[la, gamma] = c('pq,pq->', D_pq, U_A.T @ Ur_eff)
        return I_pp

    def velocity_dipole_derivatives(self, gauge: str = 'canonical') -> np.ndarray:
        """CISD correlation velocity-gauge APT, shape (natom, 3, 3): -2 times the four overlap
        blocks with the correlation 1-PDM in Iphiphi. The SCF reference and the Z_A delta nuclear
        term are supplied by the pycc.apt(gauge='velocity') facade, matching MPderiv's contract."""
        natom = self.ci.ref.molecule().natom()
        overlap_total = (self.compute_Icc_VG_APT() + self.compute_Icphi_VG_APT() + self.compute_Iphic_VG_APT() + self.compute_Iphiphi_VG_APT())
        return (-2.0 * overlap_total).real.reshape(natom, 3, 3)