"""Shared base for correlated analytic-derivative property drivers (MP2, CC; CI to follow).

`CorrelatedDerivs` owns the method-agnostic orbital-response and assembly machinery -- the pieces
that depend only on the reduced densities and the SCF reference, not on how the correlated
wavefunction was obtained.  Method-specific subclasses (`MPderiv`, `CCderiv`) supply the reduced
densities and their first-order responses.  See docs/DERIVATIVES_PLAN_2026-06.md section 9 for the
base/leaf split and the phased plan; more machinery moves here in later phases.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np


#: Result of the unperturbed orbital-response (Z-vector) solve.  ``Drel``/``Gam`` are the relaxed
#: 1-PDM and cumulant 2-PDM; the remaining fields are the byproducts the perturbed (2n+1) machinery
#: reuses: the unrelaxed 1-PDM ``D``, the ov Z-vector amplitudes ``z``, the MO orbital-Hessian
#: solver handle ``mo_hessian`` (the reference ``HFwfn`` on the spatial path, the inline matrix ``G``
#: on the spin-orbital path), the frozen-core core<->active divide ``Pco``, and the
#: canonical-perturbed-MO oo/vv dependent-pair rotations ``Poo``/``Pvv`` (``None`` unless the gauge
#: is canonical).
OrbitalResponse = namedtuple('OrbitalResponse', 'Drel Gam D z mo_hessian Pco Poo Pvv')

#: Result of a perturbed (2n+1) orbital-response solve for one perturbation ``x``.  ``dDrel`` is the
#: first-order response of the relaxed 1-PDM; ``dGam`` the response of the unrelaxed cumulant 2-PDM;
#: ``dW`` the response of the energy-weighted density ``W = I'(Drel, Gam)``.  All three fall out of a
#: single perturbed solve -- the polarizability needs only ``dDrel``; the APT/Hessian
#: (nuclear-skeleton) assemblies also contract ``dGam`` and ``dW`` against the perturbed integrals.
PerturbedResponse = namedtuple('PerturbedResponse', 'dDrel dGam dW')


class CorrelatedDerivs:
    """Base class for correlated derivative-property drivers.

    Holds the correlated wavefunction and the method-agnostic derivative machinery.  A subclass
    (`MPderiv`, `CCderiv`) is constructed from a converged correlated wavefunction and supplies the
    method-specific reduced densities and their perturbed responses.
    """

    def __init__(self, wfn) -> None:
        self.wfn = wfn
        self.contract = wfn.contract
        self._ref_hf = None

    def _reference_hf(self):
        """All-electron :class:`~pycc.hfwfn.HFwfn` for the SCF reference (cached) -- supplies the
        reference contribution the :mod:`pycc.properties` facade pairs with the correlation part,
        and the CPHF orbital Hessian used by the Z-vector solve."""
        if self._ref_hf is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.wfn.ref, orbital_basis=self.wfn.orbital_basis)
        return self._ref_hf

    @property
    def perturbed_mo_gauge(self):
        """The active occ-occ / virt-virt perturbed-orbital gauge, ``'canonical'`` or
        ``'non-canonical'`` (docs/cc_gradients_orbital_response.tex, sec. Canonical Perturbed
        Orbitals).  ``'non-canonical'`` uses the orthonormality conditions ``U^x_ij = -1/2 S^(x)_ij``
        (occ) and ``U^x_ab = -1/2 S^(x)_ab`` (vir), leaving the oo/vv dependent-pair rotations to
        vanish -- valid when the correlation energy is invariant to oo/vv rotations (MP2, CCSD).
        ``'canonical'`` keeps the perturbed occ/vir blocks canonical (``d f_ij/dx = 0``, ``i != j``),
        carrying the oo/vv dependent-pair rotations ``P_oo``/``P_vv`` explicitly; it is the choice
        for the CCSD(T) gradient, where building the (T) contributions to ``Doo``/``Dvv`` from their
        diagonal oo/vv blocks alone saves an O(N^7) step.  (The two routes give the same result; for
        oo/vv-invariant methods the pairs vanish and the canonical recipe reduces to the
        non-canonical one.)  Defaults to ``'canonical'`` for CCSD(T), ``'non-canonical'`` otherwise;
        a future non-canonical-(T) route (building the full (T) density) would override this.  The
        frozen-core core<->active-occupied divide ``P_co`` is always canonical, independent of this
        choice."""
        return 'canonical' if getattr(self.wfn, 'model', '').upper() == 'CCSD(T)' else 'non-canonical'

    def _full_occ_cphf(self):
        """A CPHF over the **full** occupied space (frozen core + active) in the wavefunction's own
        MO ordering (cached).  The perturbed-response (2n+1) routes need the orbital response over the
        full occupied space (core<->active and core-virtual), which the active-only ``wfn.cphf``
        can't supply; building it here -- rather than borrowing an all-electron ``HFwfn`` -- keeps
        the spin-orbital ordering consistent with the densities.  For ``nfzc=0`` it coincides with
        ``wfn.cphf``.  CPHF depends only on the shared reference/orbitals/integrals, so building it on
        ``self.wfn`` is equivalent for MP2 (``self.wfn`` is the MPwfn) and CC (the CCwfn shares those
        with its ``cc.mp``)."""
        if getattr(self, '_focphf', None) is None:
            from .cphf import CPHF
            self._focphf = CPHF(self.wfn, full_occ=True)
        return self._focphf

    # ---- generalized-Fock orbital Lagrangian I'(D, Gam) ----
    # The Gauss/Stanton/Bartlett orbital-gradient Lagrangian: a method-agnostic function of a
    # full-MO 1-PDM ``D`` and cumulant 2-PDM ``Gam`` (both supplied by the leaf) plus the SCF
    # reference integrals/Fock.  Its occupied-virtual antisymmetric part ``X_ai = I'_ia - I'_ai``
    # drives the Z-vector; evaluated at the relaxed density it is the energy-weighted density ``W``.
    # Basis-dispatched: antisymmetrized ``<pq||rs>`` (spin-orbital) vs the spin-adapted ``L``
    # (closed-shell).  ``Gam`` must carry the proper 2-PDM permutational symmetry (the caller's
    # densities -- MP2 seeds / :meth:`ccdensity.gradient_densities` -- ensure this).

    def _lagrangian(self, D, Gam) -> np.ndarray:
        """Generalized-Fock orbital Lagrangian ``I'(D, Gam)`` (``nmo x nmo``)

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (w_rpsq + w_rqsp)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        with the two-electron kernel ``w`` = the antisymmetrized ``<pq||rs>`` (spin-orbital,
        :meth:`_so_lagrangian`) or the spin-adapted ``L`` (closed-shell, :meth:`_spatial_lagrangian`),
        dispatched on the orbital basis."""
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_lagrangian(D, Gam)
        return self._spatial_lagrangian(D, Gam)

    def _so_lagrangian(self, D, Gam) -> np.ndarray:
        """Spin-orbital generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) from a full-MO
        1-PDM ``D`` and cumulant 2-PDM ``Gam``

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (<rp||sq> + <rq||sp>)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        The 1-PDM term's column index runs over the full occupied space ``ofull`` (= core +
        active), so the frozen-core rows/columns are built (for ``nfzc=0`` this is the whole
        occupied space)."""
        nmo = self.wfn.nmo
        ofull = slice(0, self.wfn.o.stop)
        ERI = np.asarray(self.wfn.H.ERI)
        eps = np.diag(np.asarray(self.wfn.H.F))
        termA = eps[:, None] * (D + D.T)                       # f_pp (D_pq + D_qp)
        termB = np.zeros((nmo, nmo))
        termB[:, ofull] = (self.contract('rs,rpsq->pq', D, ERI[:, :, :, ofull])
                           + self.contract('rs,rqsp->pq', D, ERI[:, ofull, :, :]))
        termC = 4.0 * self.contract('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    def _spatial_lagrangian(self, D, Gam) -> np.ndarray:
        """Spin-adapted (closed-shell) generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) -- the
        closed-shell analogue of :meth:`_so_lagrangian`

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (L_rpsq + L_rqsp)
                           + 4 sum_rst <pr|st> Gamma_qrst ]

        with the spin-adapted ``L_pqrs = 2 <pq|rs> - <pq|sr>`` (= ``H.L``) carrying the closed-shell
        spin sum in the two-electron 1-PDM term, and the bare ``<pr|st>`` (= ``H.ERI``) with
        ``Gamma`` in the 2-PDM term.  ``ofull`` is the full occupied space (core + active)."""
        nmo = self.wfn.nmo
        ofull = slice(0, self.wfn.nfzc + self.wfn.no)
        ERI = np.asarray(self.wfn.H.ERI)
        L = np.asarray(self.wfn.H.L)
        eps = np.diag(np.asarray(self.wfn.H.F))
        termA = eps[:, None] * (D + D.T)
        termB = np.zeros((nmo, nmo))
        termB[:, ofull] = (self.contract('rs,rpsq->pq', D, L[:, :, :, ofull])
                           + self.contract('rs,rqsp->pq', D, L[:, ofull, :, :]))
        termC = 4.0 * self.contract('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    # ---- first-order response dI' of the generalized-Fock Lagrangian ----
    # The field/nuclear derivative of I'(D, Gam), method-agnostic given the perturbed integrals
    # (df = d_x f, deri = d_x <pq|rs>, dL = 2 deri - deri.swap) and the density responses (dD, dGam).
    # Its ov-antisymmetric part is the perturbed Z-vector RHS; evaluated at the relaxed density (with
    # its response) it is the perturbed energy-weighted density d_x W.  The termA derivative is the
    # FULL Fock matrix product df @ (D + D.T) -- not the diagonal d_x(eps) stencil of the unperturbed
    # form (valid only at F=0); the off-diagonal df couples the ov/core-active blocks of a relaxed D
    # (zero for an unrelaxed MP2 D, nonzero for CC's Dov/Dvo).

    def _perturbed_lagrangian(self, df, deri, dL, D, dD, Gam, dGam) -> np.ndarray:
        """Spin-adapted first-order response ``dI'`` (``nmo x nmo``) given the perturbed integrals
        and density responses

            dI'_pq = -1/2 [ df @ (D + D.T) + eps_p (dD_pq + dD_qp)
                            + delta_{q in ofull} sum_rs ( dD_rs L_rpsq + D_rs dL_rpsq
                                                          + dD_rs L_rqsp + D_rs dL_rqsp )
                            + 4 sum_rst ( deri_prst Gamma_qrst + <pr|st> dGamma_qrst ) ]

        with ``L`` (= ``H.L``) and its derivative ``dL`` in the 1-PDM term, and ``<pr|st>`` (= ``H.ERI``)
        with ``Gamma``/``dGamma`` in the 2-PDM term.  The caller supplies the perturbed integrals and
        the (unrelaxed or relaxed) densities + responses."""
        nmo, ofull = self.wfn.nmo, slice(0, self.wfn.o.stop)
        ERI = np.asarray(self.wfn.H.ERI)
        L = np.asarray(self.wfn.H.L)
        eps = np.diag(np.asarray(self.wfn.H.F))
        c = self.contract
        dA = df @ (D + D.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, L[:, :, :, ofull]) + c('rs,rpsq->pq', D, dL[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, L[:, ofull, :, :]) + c('rs,rqsp->pq', D, dL[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    def _so_perturbed_lagrangian(self, df, deri, D, dD, Gam, dGam) -> np.ndarray:
        """Spin-orbital first-order response ``dI'`` -- the antisymmetrized-integral analogue of
        :meth:`_perturbed_lagrangian` (the ``<pq||rs>`` derivative ``deri`` in the 1-PDM term
        in place of ``L``/``dL``)."""
        nmo, ofull = self.wfn.nmo, slice(0, self.wfn.o.stop)
        ERI = np.asarray(self.wfn.H.ERI)
        eps = np.diag(np.asarray(self.wfn.H.F))
        c = self.contract
        dA = df @ (D + D.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, ERI[:, :, :, ofull]) + c('rs,rpsq->pq', D, deri[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, ERI[:, ofull, :, :]) + c('rs,rqsp->pq', D, deri[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    # ---- unperturbed relaxed density and orbital-response (Z-vector) ----
    # Given the leaf's unrelaxed reduced densities, the relaxed 1-PDM adds the orbital-relaxation
    # blocks driven by the Lagrangian's ov-antisymmetric part:
    #
    #     Drel = D  +  P_co (core<->active-occ)  +  P_oo/P_vv (extra oo/vv, e.g. (T))  -  z (ov),
    #
    # with the Z-vector  A z = X,  X_ai = I'_ia - I'_ai (over the full occupied space).  The
    # non-redundant core<->active-occupied rotation is a direct divide P_co = (I'_ci - I'_ic) /
    # (eps_c - eps_i) (the SCF energy is invariant to occ-occ rotations), coupled into X.  The orbital
    # Hessian A is the all-electron SCF Hessian: borrowed from the reference HFwfn CPHF (spatial) or
    # built inline (spin-orbital, G_ia,jb = <aj||ib> + <ab||ij> + delta_ij delta_ab (eps_a - eps_i)).
    # Method-agnostic given the two leaf hooks below.

    def _unrelaxed_densities(self):
        """Leaf hook: the unrelaxed reduced densities ``(D, Gam)`` as full-MO arrays -- the 1-PDM
        (``nmo x nmo``, occupied/virtual diagonal blocks) and cumulant 2-PDM (``nmo^4``).  Supplied
        by the method (MP2 amplitude seeds / CC :meth:`ccdensity.gradient_densities`)."""
        raise NotImplementedError

    def _orbital_response(self):
        """Spatial (closed-shell) unperturbed orbital-response (Z-vector) solve (cached, frozen-core
        aware), returning an :class:`OrbitalResponse` record.  The relaxed 1-PDM

            Drel = D + P_co + P_oo + P_vv - z,   X_ai = I'_ia - I'_ai,   A z = X,

        with the generalized-Fock Lagrangian ``I'`` from :meth:`_spatial_lagrangian` (spin-adapted
        ``L``), the frozen-core divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`` coupled into ``X``,
        the canonical-perturbed-MO oo/vv rotations ``P_oo``/``P_vv`` (populated only for
        :attr:`perturbed_mo_gauge` ``== 'canonical'``), and the ov Z-vector ``A z = X`` solved with
        the orbital Hessian ``A`` from the all-electron reference ``HFwfn`` CPHF (``mo_hessian`` =
        that ``HFwfn``; occupied space = the full ``ndocc``).  ``z`` is indexed ``(I, a)`` over the
        full occupied space.  The record's byproducts (``z``, ``mo_hessian``, ``Pco``, ``Poo``,
        ``Pvv``, ``D``) are reused by the perturbed (2n+1) machinery."""
        if getattr(self, '_orbresp', None) is None:
            nmo, nfzc, no = self.wfn.nmo, self.wfn.nfzc, self.wfn.no
            o, v = self.wfn.o, self.wfn.v
            co = slice(0, nfzc)
            ofull = slice(0, nfzc + no)
            eps = np.diag(np.asarray(self.wfn.H.F))
            L = np.asarray(self.wfn.H.L)
            c = self.contract
            D, Gam = self._unrelaxed_densities()
            D = np.asarray(D)
            Ip = self._spatial_lagrangian(D, Gam)
            Drel = D.copy()
            Pco = Poo = Pvv = None
            if nfzc:
                Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
                Drel[co, o] += Pco
                Drel[o, co] += Pco.T
            X = Ip[ofull, v] - Ip[v, ofull].T
            if nfzc:
                zjc = -Pco.T                                   # z_jc, active-occupied x core
                X = X - (c('jc,ajic->ia', zjc, L[v, o, ofull, co])
                         + c('jc,acij->ia', zjc, L[v, co, ofull, o]))
            # Canonical perturbed MOs: carry the off-diagonal oo/vv orbital response as the
            # dependent-pair rotations kappa_oo/kappa_vv (added to Drel, coupled into X).  This is
            # the CCSD(T) choice -- the (T) contributions to Doo/Dvv are then built from their
            # diagonal oo/vv blocks alone, saving an O(N^7) step.  For oo/vv-invariant methods the
            # pairs vanish, so the non-canonical default (MP2, CCSD) simply skips them.
            if self.perturbed_mo_gauge == 'canonical':
                Poo = self._dependent_pairs(Ip[o, o], eps[o])
                Pvv = self._dependent_pairs(Ip[v, v], eps[v])
                Drel[o, o] += Poo
                Drel[v, v] += Pvv
                X = X + (c('kl,akil->ia', Poo, L[v, o, ofull, o])
                         + c('bc,ibac->ia', Pvv, L[ofull, v, v, v]))
            hf = self._reference_hf()
            zia = hf.cphf.solve(X)                              # (I,a) over full occ
            Drel[v, ofull] += -zia.T
            Drel[ofull, v] += -zia
            self._orbresp = OrbitalResponse(Drel, Gam, D, zia, hf, Pco, Poo, Pvv)
        return self._orbresp

    def _so_orbital_response(self):
        """Spin-orbital unperturbed orbital-response (Z-vector) solve (cached, frozen-core aware),
        returning an :class:`OrbitalResponse` record.  The relaxed 1-PDM

            Drel = D + P_co + P_oo + P_vv - z,   X_ai = I'_ia - I'_ai,   A z = X,

        with the generalized-Fock Lagrangian ``I'`` from :meth:`_so_lagrangian` (antisymmetrized
        ``<pq||rs>``), the frozen-core divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`` coupled into
        ``X``, the canonical-perturbed-MO oo/vv rotations ``P_oo``/``P_vv`` (populated only for
        :attr:`perturbed_mo_gauge` ``== 'canonical'``), and the ov Z-vector ``A z = X`` solved with
        the orbital Hessian ``A = G`` built inline (``G_ia,jb = <aj||ib> + <ab||ij> + delta_ij
        delta_ab (eps_a - eps_i)``, ``mo_hessian`` = that ``G``) -- there is no all-electron
        spin-orbital ``HFwfn`` CPHF to borrow (it orders the spins differently from the densities).
        ``z`` is indexed ``(I, a)`` over the full occupied space.  The record's byproducts (``z``,
        ``mo_hessian``, ``Pco``, ``Poo``, ``Pvv``, ``D``) are reused by the perturbed (2n+1)
        machinery.  **UHF only** -- raises for ROHF (the semicanonical response does not reproduce
        the restricted ROHF response)."""
        if getattr(self, '_so_orbresp', None) is None:
            if self.wfn.cphf.is_rohf:
                raise NotImplementedError(
                    "The spin-orbital correlated relaxed gradient/dipole is not implemented for "
                    "ROHF references (the semicanonical response does not reproduce the restricted "
                    "ROHF response); RHF and UHF are supported.")
            nmo, nfzc, nv = self.wfn.nmo, self.wfn.nfzc, self.wfn.nv
            o, v, co = self.wfn.o, self.wfn.v, self.wfn.co
            ofull = slice(0, o.stop)
            nof = o.stop
            ERI = np.asarray(self.wfn.H.ERI)
            eps = np.diag(np.asarray(self.wfn.H.F))
            c = self.contract
            D, Gam = self._unrelaxed_densities()
            D = np.asarray(D)
            Ip = self._so_lagrangian(D, Gam)
            Drel = D.copy()
            Pco = Poo = Pvv = None
            if nfzc:
                Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
                Drel[co, o] += Pco
                Drel[o, co] += Pco.T
            X = Ip[ofull, v] - Ip[v, ofull].T
            if nfzc:
                zjc = -Pco.T                                   # z_jc, active-occupied x core
                X = X - (c('jc,ajic->ia', zjc, ERI[v, o, ofull, co])
                         + c('jc,acij->ia', zjc, ERI[v, co, ofull, o]))
            # Canonical perturbed MOs: carry the off-diagonal oo/vv orbital response as the
            # dependent-pair rotations kappa_oo/kappa_vv (added to Drel, coupled into X).  This is
            # the CCSD(T) choice -- the (T) contributions to Doo/Dvv are then built from their
            # diagonal oo/vv blocks alone, saving an O(N^7) step.  For oo/vv-invariant methods the
            # pairs vanish, so the non-canonical default (MP2, CCSD) simply skips them.
            if self.perturbed_mo_gauge == 'canonical':
                Poo = self._dependent_pairs(Ip[o, o], eps[o])
                Pvv = self._dependent_pairs(Ip[v, v], eps[v])
                Drel[o, o] += Poo
                Drel[v, v] += Pvv
                X = X + (c('kl,akil->ia', Poo, ERI[v, o, ofull, o])
                         + c('bc,ibac->ia', Pvv, ERI[ofull, v, v, v]))
            G = (c('ajib->iajb', ERI[v, ofull, ofull, v])
                 + c('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof * nv, nof * nv)
            G[np.diag_indices(nof * nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
            zia = np.linalg.solve(G, X.reshape(-1)).reshape(nof, nv)
            Drel[v, ofull] += -zia.T
            Drel[ofull, v] += -zia
            self._so_orbresp = OrbitalResponse(Drel, Gam, D, zia, G, Pco, Poo, Pvv)
        return self._so_orbresp

    def _relaxed_density(self):
        """Relaxed 1-PDM ``Drel`` and cumulant 2-PDM ``Gam`` (``Tr(Drel mu)`` gives the correlation
        dipole; ``Drel``/``Gam`` feed the gradient), dispatched on the orbital basis.  The full
        orbital-response byproducts are available from :meth:`_orbital_response` /
        :meth:`_so_orbital_response`."""
        rec = self._so_orbital_response() if self.wfn.orbital_basis == 'spinorbital' else self._orbital_response()
        return rec.Drel, rec.Gam

    def _so_relaxed_density(self):
        """Spin-orbital relaxed 1-PDM and 2-PDM -- ``(Drel, Gam)`` from :meth:`_so_orbital_response`."""
        rec = self._so_orbital_response()
        return rec.Drel, rec.Gam

    # ---- first-order response of the relaxed density (perturbed Z-vector) ----
    # d_x Drel differentiates the relaxed-density build once more.  Given the leaf's perturbed
    # unrelaxed densities (dDg, dGam), the assembly is method-agnostic: the perturbed Lagrangian dI'
    # (with the perturbed integrals df/deri/dL), the perturbed frozen-core divide d_x P_co (a
    # Sylvester relation), the perturbed canonical-MO oo/vv rotations d_x P_oo/d_x P_vv (gauge-gated),
    # and the perturbed ov Z-vector z^x = A^{-1}(dX - A^x z) reusing the unperturbed orbital Hessian A
    # and z from the OrbitalResponse record.

    def _perturbed_unrelaxed_densities(self, pert, df, deri, dL):
        """Leaf hook: the first-order response ``(d_x gamma, d_x Gamma)`` of the unrelaxed reduced
        densities to ``pert`` (full-MO arrays).  MP2 supplies the closed-form response
        (:meth:`MPderiv._perturbed_densities`); CC supplies the iterative perturbed-amplitude /
        perturbed-Lambda response.  ``df``/``deri``/``dL`` are the CPHF-folded perturbed integrals
        (canonical per :attr:`perturbed_mo_gauge`) the CC iterative solve consumes; the MP2 closed
        form recomputes its own and ignores them."""
        raise NotImplementedError

    def _perturbed_relaxed_density(self, pert):
        """Spatial perturbed (2n+1) orbital response for ``pert`` -- a :class:`PerturbedResponse`
        ``(dDrel, dGam, dW)``.  The relaxed-1-PDM response (``nmo x nmo``) is

            d_x Drel = d_x D + d_x P_co + d_x P_oo + d_x P_vv - z^x,   A z^x = dX - A^x z,

        with the perturbed unrelaxed density ``d_x D`` (:meth:`_perturbed_unrelaxed_densities`), the
        perturbed Lagrangian ``dI'`` (:meth:`_perturbed_lagrangian`) giving the perturbed
        Z-vector RHS ``dX_ai = dI'_ia - dI'_ai``, the perturbed frozen-core Sylvester divide
        ``d_x P_co = [d_x(I'_ci - I'_ic) - df_cd P_di + P_cj df_ji] / (eps_c - eps_i)`` coupled into
        ``dX``, the perturbed canonical-MO oo/vv rotations ``d_x P_oo``/``d_x P_vv``
        (:meth:`_perturbed_dependent_pairs`, populated only for :attr:`perturbed_mo_gauge` ``==
        'canonical'``) coupled into ``dX``, and the perturbed ov Z-vector ``z^x`` reusing the
        unperturbed orbital Hessian ``A`` and ``z`` from :meth:`_orbital_response` (``A^x z`` the
        perturbed-Hessian response).  The perturbed integrals ``df``/``deri`` are canonical per
        :attr:`perturbed_mo_gauge`.  The same solve yields the unrelaxed cumulant-2-PDM response
        ``dGam`` and the perturbed energy-weighted density ``dW = d_x I'(Drel, Gam)`` (the perturbed
        Lagrangian evaluated at the *relaxed* density), so the APT/Hessian assemblies need no extra
        perturbed solve."""
        wfn = self.wfn
        o, v = wfn.o, wfn.v
        co = slice(0, wfn.nfzc)
        ofull = slice(0, o.stop)
        ncore = o.stop - wfn.no
        c = self.contract
        L = np.asarray(wfn.H.L)
        eps = np.diag(np.asarray(wfn.H.F))
        canonical = self.perturbed_mo_gauge == 'canonical'
        rec = self._orbital_response()
        z, hf, Pco, Poo0, Pvv0, D0, Gam0, Drel0 = (rec.z, rec.mo_hessian, rec.Pco, rec.Poo, rec.Pvv,
                                                   rec.D, rec.Gam, rec.Drel)
        cphf = self._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore, canonical=canonical))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore, canonical=canonical))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        dDg, dGam = self._perturbed_unrelaxed_densities(pert, df, deri, dL)
        dDg, dGam = np.asarray(dDg), np.asarray(dGam)
        dIp = self._perturbed_lagrangian(df, deri, dL, D0, dDg, Gam0, dGam)
        dX = dIp[ofull, v] - dIp[v, ofull].T
        dPco = None
        if wfn.nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, L[v, o, ofull, co]) + c('jc,acij->ia', dzjc, L[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, dL[v, o, ofull, co]) + c('jc,acij->ia', zjc, dL[v, co, ofull, o]))
        dPoo = dPvv = None
        if canonical:
            dfd = np.diag(df)                              # canonical df diagonal = the perturbed gaps
            dPoo = self._perturbed_dependent_pairs(dIp[o, o], Poo0, eps[o], dfd[o])
            dPvv = self._perturbed_dependent_pairs(dIp[v, v], Pvv0, eps[v], dfd[v])
            dX = dX + (c('kl,akil->ia', dPoo, L[v, o, ofull, o]) + c('kl,akil->ia', Poo0, dL[v, o, ofull, o])
                       + c('bc,ibac->ia', dPvv, L[ofull, v, v, v]) + c('bc,ibac->ia', Pvv0, dL[ofull, v, v, v]))
        Axz = (c('ajib,jb->ia', dL[v, ofull, ofull, v], z) + c('abij,jb->ia', dL[v, v, ofull, ofull], z)
               + c('ab,ib->ia', df[v, v], z) - c('ij,ja->ia', df[ofull, ofull], z))
        zx = np.asarray(hf.cphf.solve(dX - Axz))
        dDrel = dDg.copy()
        if wfn.nfzc:
            dDrel[co, o] += dPco
            dDrel[o, co] += dPco.T
        if canonical:
            dDrel[o, o] += dPoo
            dDrel[v, v] += dPvv
        dDrel[v, ofull] += -zx.T
        dDrel[ofull, v] += -zx
        dW = self._perturbed_lagrangian(df, deri, dL, Drel0, dDrel, Gam0, dGam)
        return PerturbedResponse(dDrel, dGam, dW)

    def _so_perturbed_relaxed_density(self, pert):
        """Spin-orbital perturbed (2n+1) orbital response -- the spin-orbital analogue of
        :meth:`_perturbed_relaxed_density`, returning the same :class:`PerturbedResponse`
        ``(dDrel, dGam, dW)`` (antisymmetrized ``<pq||rs>`` derivatives ``deri`` in the couplings;
        the inline orbital Hessian ``G`` from :meth:`_so_orbital_response` solved with
        ``numpy.linalg.solve``)."""
        wfn = self.wfn
        o, v, nv = wfn.o, wfn.v, wfn.nv
        co = wfn.co
        ofull = slice(0, o.stop)
        nof = o.stop
        ncore = o.stop - wfn.no
        c = self.contract
        ERI = np.asarray(wfn.H.ERI)
        eps = np.diag(np.asarray(wfn.H.F))
        canonical = self.perturbed_mo_gauge == 'canonical'
        rec = self._so_orbital_response()
        z, G, Pco, Poo0, Pvv0, D0, Gam0, Drel0 = (rec.z, rec.mo_hessian, rec.Pco, rec.Poo, rec.Pvv,
                                                  rec.D, rec.Gam, rec.Drel)
        cphf = self._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore, canonical=canonical))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore, canonical=canonical))
        dDg, dGam = self._perturbed_unrelaxed_densities(pert, df, deri, None)
        dDg, dGam = np.asarray(dDg), np.asarray(dGam)
        dIp = self._so_perturbed_lagrangian(df, deri, D0, dDg, Gam0, dGam)
        dX = dIp[ofull, v] - dIp[v, ofull].T
        dPco = None
        if wfn.nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, ERI[v, o, ofull, co]) + c('jc,acij->ia', dzjc, ERI[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, deri[v, o, ofull, co]) + c('jc,acij->ia', zjc, deri[v, co, ofull, o]))
        dPoo = dPvv = None
        if canonical:
            dfd = np.diag(df)
            dPoo = self._perturbed_dependent_pairs(dIp[o, o], Poo0, eps[o], dfd[o])
            dPvv = self._perturbed_dependent_pairs(dIp[v, v], Pvv0, eps[v], dfd[v])
            dX = dX + (c('kl,akil->ia', dPoo, ERI[v, o, ofull, o]) + c('kl,akil->ia', Poo0, deri[v, o, ofull, o])
                       + c('bc,ibac->ia', dPvv, ERI[ofull, v, v, v]) + c('bc,ibac->ia', Pvv0, deri[ofull, v, v, v]))
        Axz = (c('ajib,jb->ia', deri[v, ofull, ofull, v], z) + c('abij,jb->ia', deri[v, v, ofull, ofull], z)
               + c('ab,ib->ia', df[v, v], z) - c('ij,ja->ia', df[ofull, ofull], z))
        zx = np.linalg.solve(G, (dX - Axz).reshape(-1)).reshape(nof, nv)
        dDrel = dDg.copy()
        if wfn.nfzc:
            dDrel[co, o] += dPco
            dDrel[o, co] += dPco.T
        if canonical:
            dDrel[o, o] += dPoo
            dDrel[v, v] += dPvv
        dDrel[v, ofull] += -zx.T
        dDrel[ofull, v] += -zx
        dW = self._so_perturbed_lagrangian(df, deri, Drel0, dDrel, Gam0, dGam)
        return PerturbedResponse(dDrel, dGam, dW)

    # ---- first-derivative properties: relaxed dipole and nuclear gradient ----
    # Both are contractions of the relaxed density against the property integrals, method-agnostic
    # given (Drel, Gam) and the energy-weighted density W = I'(Drel).  The reference (SCF) and
    # nuclear contributions are kept separate and summed by the pycc.properties facade.

    def relaxed_dipole(self) -> np.ndarray:
        """Correlation contribution to the electronic dipole moment (a.u.), shape ``(3,)``

            mu_a^corr = sum_pq Drel_pq (mu_a)_pq,

        the relaxed 1-PDM contracted with the MO dipole integrals (``H.mu = -e r``).  A static
        field does not move the AO basis, so there is no energy-weighted-density or 2-PDM term (only
        the gradient has those); the orbital relaxation (and, for CCSD(T), the canonical-MO oo/vv
        response) rides inside ``Drel``.  The reference (SCF) dipole is kept separate; the total is
        their sum.  Basis-aware (dispatches via :meth:`_relaxed_density`)."""
        Drel, _ = self._relaxed_density()
        c = self.contract
        return np.array([c('pq,pq->', Drel, np.asarray(self.wfn.H.mu[a])) for a in range(3)])

    def gradient(self) -> np.ndarray:
        """Correlation contribution to the analytic nuclear energy gradient (a.u.), shape
        ``(natom, 3)``

            dE_corr/dX = sum_pq Drel_pq f^(X)_pq + sum_pqrs Gamma_pqrs <pq|rs>^(X)
                         + sum_pq W_pq S^(X)_pq,

        with the relaxed 1-PDM ``Drel``, cumulant 2-PDM ``Gamma`` (:meth:`_relaxed_density`), and the
        energy-weighted density ``W = I'(Drel)`` (:meth:`_lagrangian`).  ``f^(X) = h^(X) + sum_m
        L[p,m,q,m]^(X)`` is the closed-shell skeleton Fock derivative (``m`` over the full occupied
        space), and ``S^(X)``/``<pq|rs>^(X)`` are the skeleton derivative integrals from
        ``wfn.derivatives`` (chemist ``(pq|rs)^(X)``, converted to physicist here) -- no
        per-perturbation CPHF solve.  Spatial (closed-shell RHF) path; the spin-orbital path is
        :meth:`_so_gradient`.  The reference (SCF) gradient is kept separate."""
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_gradient()
        ofull = slice(0, self.wfn.o.stop)                # full occupied (core + active)
        Drel, Gam = self._relaxed_density()
        W = self._lagrangian(Drel, Gam)
        c = self.contract
        d = self.wfn.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.core(atom); Sx = d.overlap(atom); ERIx = d.eri(atom)   # chemist (pq|rs)^X
            for cart in range(3):
                phys = ERIx[cart].transpose(0, 2, 1, 3)                   # -> physicist <pq|rs>^X
                Lx = 2.0 * phys - phys.transpose(0, 1, 3, 2)
                fx = hx[cart] + c('pmqm->pq', Lx[:, ofull, :, ofull])     # skeleton Fock deriv (full occ)
                grad[atom, cart] = (c('pq,pq->', Drel, fx)
                                    + c('pqrs,pqrs->', Gam, phys)
                                    + c('pq,pq->', W, Sx[cart]))
        return grad

    def _so_gradient(self) -> np.ndarray:
        """Spin-orbital correlation gradient -- the spin-orbital analogue of :meth:`gradient` with
        the antisymmetrized ``<pq||rs>^(X)`` from ``wfn.derivatives.so_*`` and ``f^(X) = h^(X) +
        sum_m <pm||qm>^(X)`` (``m`` over the full occupied space)."""
        ofull = slice(0, self.wfn.o.stop)                # full occupied (core + active)
        Drel, Gam = self._so_relaxed_density()
        W = self._lagrangian(Drel, Gam)
        c = self.contract
        d = self.wfn.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.so_core(atom); Sx = d.so_overlap(atom); ERIx = d.so_eri(atom)
            for cart in range(3):
                fx = hx[cart] + c('pmqm->pq', ERIx[cart][:, ofull, :, ofull])   # skeleton Fock deriv
                grad[atom, cart] = (c('pq,pq->', Drel, fx)
                                    + c('pqrs,pqrs->', Gam, ERIx[cart])
                                    + c('pq,pq->', W, Sx[cart]))
        return grad

    # ---- second-derivative properties: polarizability, APT (dipole derivatives), Hessian ----
    # All three are the asymmetric (2n+1) route: differentiate a relaxed-density first derivative a
    # second time, using only first-order responses (the perturbed relaxed density / energy-weighted
    # density and U^y -- no second-order CPHF U^{xy}).  Method-agnostic given the orbital-response
    # record and the PerturbedResponse hook; these are the public correlation-property API (the
    # pycc.properties facade calls them by name).  The reference (SCF) and nuclear parts stay
    # separate and are summed by the facade.  A leaf overrides one of these only to add
    # method-specific behavior (e.g. CCderiv.polarizability's model / (T)-intermediate guards).

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        """Correlation contribution to the static (omega=0) dipole polarizability (a.u.), shape
        ``(3, 3)``: ``alpha_corr_ab = -d^2 E_corr / dF_a dF_b``, via the 2n+1 route (frozen-core
        aware; spin-orbital and spin-adapted paths).  Differentiating the relaxed dipole
        ``d_b E = -Tr(D_rel mu_b)`` (field skeleton ``f^(b) = -mu_b``) a second time::

            alpha_ab = sum_pq d_a D_rel_pq (mu_b)_pq
                     + sum_pq D_rel_pq [ (U^a).T mu_b + mu_b U^a ]_pq

        The first term is the perturbed relaxed density (:meth:`_perturbed_relaxed_density`, carrying
        the perturbed Z-vector and, for frozen core, the perturbed core-active divide); the second is
        the MO dipole rotating under the field (``U^a`` over the full occupied space -- ``ncore``
        canonical core-active block, gauge per :attr:`perturbed_mo_gauge`).  No second-order CPHF
        ``U^{ab}`` -- only first-order responses.

        ``route`` accepts only ``'2n+1'`` (the sole route; the argument is retained for a uniform
        property signature).  The reference part is kept separate (:meth:`HFwfn.polarizability`) and
        summed with this correlation part by :func:`pycc.polarizability`."""
        if route != '2n+1':
            raise ValueError(f"unknown polarizability route {route!r} (only '2n+1')")
        from .cphf import Perturbation
        wfn = self.wfn
        c = self.contract
        ncore = wfn.o.stop - wfn.no
        canonical = self.perturbed_mo_gauge == 'canonical'
        cphf = self._full_occ_cphf()
        if wfn.orbital_basis == 'spinorbital':
            Drel = self._so_orbital_response().Drel
            popdm = self._so_perturbed_relaxed_density
        else:
            Drel = self._orbital_response().Drel
            popdm = self._perturbed_relaxed_density
        mu = [np.asarray(wfn.H.mu[a]) for a in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            dDrel = popdm(pert).dDrel
            Ub = np.asarray(cphf.full_U(pert, ncore, canonical=canonical))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def dipole_derivatives(self, route: str = '2n+1-field') -> np.ndarray:
        """Correlation contribution to the atomic polar tensors (nuclear dipole derivatives, a.u.),
        shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` =
        ``d(mu_alpha)/d(X_{A,beta}) = -d^2 E_corr / dF_alpha dX_{A,beta}`` -- the mixed field/nuclear
        analog of :meth:`polarizability`, via the 2n+1 route (both spin paths, frozen-core aware).

        ``route='2n+1-field'`` (default) or ``'2n+1-nuclear'``; both give the same tensor, and
        ``'2n+1-field'`` is cheaper (3 field solves vs ``3N`` nuclear).  The nuclear ``Z_A`` and SCF
        reference terms are kept separate and summed with this correlation part by :func:`pycc.apt`.

        Nuclear side -- differentiate the relaxed dipole ``Tr(D_rel mu_a)`` w.r.t. the nucleus (the
        field gradient has no ``S^X``/2e-skeleton term, so no energy-weighted density appears)::

            P[X,a] = Tr(d_X D_rel mu_a) + Tr(D_rel [mu_a^X + rotate(U^X, mu_a)]).

        Field side -- differentiate the relaxed nuclear gradient
        ``E^X = sum D_rel f^X + sum Gamma <pq||rs>^X + sum W S^X`` w.r.t. the field::

            P[X,a] = -[ sum d_a D_rel f^X + sum D_rel d_a f^X + sum d_a Gamma <>^X
                        + sum Gamma d_a <>^X + sum d_a W S^X + sum W d_a S^X ],

        with the 3 field responses ``d_a D_rel``, ``d_a Gamma``, and the perturbed energy-weighted
        density ``d_a W`` all from one :class:`PerturbedResponse` per field
        (:meth:`_perturbed_relaxed_density`).  The field-derivatives of the nuclear skeletons carry
        the orbital rotation ``rotate(U^a, .)`` plus, for ``d_a f^X``, the occupied-sum response and
        the ``-mu_a^X`` mixed skeleton (the field enters ``h``).  Both routes give the same tensor;
        ``'2n+1-field'`` is cheaper (3 field responses vs ``3N`` nuclear)."""
        if route not in ('2n+1-nuclear', '2n+1-field'):
            raise ValueError(f"unknown dipole-derivative route {route!r} "
                             "(use '2n+1-nuclear' or '2n+1-field')")
        from .cphf import Perturbation
        wfn = self.wfn
        c = self.contract
        so = wfn.orbital_basis == 'spinorbital'
        o = wfn.o
        ofull = slice(0, o.stop)
        ncore = o.stop - wfn.no
        canonical = self.perturbed_mo_gauge == 'canonical'
        cphf = self._full_occ_cphf()
        d = wfn.derivatives
        natom = d.natom
        rec = self._so_orbital_response() if so else self._orbital_response()
        Drel = rec.Drel
        popdm = self._so_perturbed_relaxed_density if so else self._perturbed_relaxed_density
        mu = [np.asarray(wfn.H.mu[a]) for a in range(3)]
        P = np.zeros((natom, 3, 3))

        if route == '2n+1-nuclear':
            for A in range(natom):
                dip = d.so_dipole(A) if so else d.dipole(A)          # [alpha*3 + beta]
                for beta in range(3):
                    pX = Perturbation('nuclear', (A, beta))
                    dDrel = popdm(pX).dDrel
                    UX = np.asarray(cphf.full_U(pX, ncore, canonical=canonical))
                    for alpha in range(3):
                        dmu = np.asarray(dip[alpha * 3 + beta])       # skeleton d(mu_a)/dX_beta
                        rot = UX.T @ mu[alpha] + mu[alpha] @ UX
                        P[A, beta, alpha] = (c('pq,pq->', dDrel, mu[alpha])
                                             + c('pq,pq->', Drel, dmu + rot))
            return P

        # route == '2n+1-field'
        Gam = rec.Gam
        W = self._lagrangian(Drel, Gam)
        field = [Perturbation('field', a) for a in range(3)]
        resp = [popdm(field[a]) for a in range(3)]                    # one perturbed solve per field
        dDrel = [r.dDrel for r in resp]
        dGamF = [r.dGam for r in resp]
        dW = [r.dW for r in resp]                                     # perturbed energy-weighted density
        U = [np.asarray(cphf.full_U(field[a], ncore, canonical=canonical)) for a in range(3)]

        def rot1(Um, M):
            return Um.T @ M + M @ Um

        def rot4(Um, T):
            return (c('tp,tqrs->pqrs', Um, T) + c('tq,ptrs->pqrs', Um, T)
                    + c('tr,pqts->pqrs', Um, T) + c('ts,pqrt->pqrs', Um, T))

        for A in range(natom):
            hx = d.so_core(A) if so else d.core(A)
            Sx = d.so_overlap(A) if so else d.overlap(A)
            dip = d.so_dipole(A) if so else d.dipole(A)
            if so:
                eriF = [np.asarray(e) for e in d.so_eri(A)]          # <pq||rs>^X (Fock and Gamma)
                gamX = eriF
            else:
                phys = [np.asarray(ch).transpose(0, 2, 1, 3) for ch in d.eri(A)]  # <pq|rs>^X (Gamma)
                eriF = [2.0 * p - p.transpose(0, 1, 3, 2) for p in phys]          # L^X (Fock)
                gamX = phys
            for beta in range(3):
                fX = np.asarray(hx[beta]) + c('pmqm->pq', eriF[beta][:, ofull, :, ofull])
                SX = np.asarray(Sx[beta])
                gX, eX = gamX[beta], eriF[beta]
                for alpha in range(3):
                    Um = U[alpha]
                    muX = np.asarray(dip[alpha * 3 + beta])
                    occ = (c('rm,prqm->pq', Um[:, ofull], eX[:, :, :, ofull])
                           + c('rm,pmqr->pq', Um[:, ofull], eX[:, ofull, :, :]))
                    dfX = rot1(Um, fX) - muX + occ
                    P[A, beta, alpha] = -(c('pq,pq->', dDrel[alpha], fX) + c('pq,pq->', Drel, dfX)
                                          + c('pqrs,pqrs->', dGamF[alpha], gX) + c('pqrs,pqrs->', Gam, rot4(Um, gX))
                                          + c('pq,pq->', dW[alpha], SX) + c('pq,pq->', W, rot1(Um, SX)))
        return P

    def hessian(self, route: str = '2n+1') -> np.ndarray:
        """Correlation contribution to the molecular (nuclear) Hessian (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3+a, B*3+b)`` = ``d^2 E_corr / dX_{Aa} dX_{Bb}`` -- the
        nuclear-nuclear analog of :meth:`polarizability` / :meth:`dipole_derivatives`, via the 2n+1
        route (both spin paths, frozen-core aware).  Differentiate the relaxed nuclear gradient
        ``E^X = sum D_rel f^X + sum Gamma <pq||rs>^X + sum W S^X`` w.r.t. a second nucleus ``Y``::

            H[X,Y] = sum d_Y D_rel f^X + sum D_rel d_Y f^X + sum d_Y Gamma <>^X
                     + sum Gamma d_Y <>^X + sum d_Y W S^X + sum W d_Y S^X,

        the nuclear-nuclear analog of the ``'2n+1-field'`` APT (:meth:`dipole_derivatives`).
        Only ``3N`` first-order solves -- the perturbed relaxed density ``d_Y D_rel``, the perturbed
        energy-weighted density ``d_Y W``, and ``d_Y Gamma`` all from one :class:`PerturbedResponse`
        per nucleus (:meth:`_perturbed_relaxed_density`), plus ``U^Y`` (:meth:`CPHF.full_U`).

        The field-derivatives of the nuclear skeletons carry (i) the full second integral skeletons
        ``f^{XY}``/``<>^{XY}``/``S^{XY}`` (:meth:`CPHF.nuclear_hessian_skeletons`, cached per atom pair -- all
        nonzero here, unlike the field case where only ``-mu^X`` survived), and (ii) the ``U^Y``
        orbital rotation of the ``X`` skeletons.  The rotations are hoisted off the ``O(N^2)`` pair
        loop onto the (per-``Y``) densities via ``sum A rot(U,B) = sum rot(U^T,A) B``:
        ``Dtil = U D + D U^T``, ``Wtil`` likewise, ``Gtil = rotate4(U^T, Gamma)``, and the Fock
        skeleton's occupied-sum response as the per-``X`` intermediate ``J^X`` contracted with
        ``U^Y`` (so no ``O(N^2)`` four-index rotation).

        ``route`` accepts only ``'2n+1'`` (retained for a uniform property signature).  The reference
        and nuclear parts are separate and summed with this correlation part by :func:`pycc.hessian`."""
        if route != '2n+1':
            raise ValueError(f"unknown hessian route {route!r} (only '2n+1')")
        from .cphf import Perturbation
        wfn = self.wfn
        c = self.contract
        so = wfn.orbital_basis == 'spinorbital'
        ofull = slice(0, wfn.o.stop)
        ncore = wfn.o.stop - wfn.no
        canonical = self.perturbed_mo_gauge == 'canonical'
        cphf = self._full_occ_cphf()
        d = wfn.derivatives
        natom = d.natom
        nc = 3 * natom
        rec = self._so_orbital_response() if so else self._orbital_response()
        Drel, Gam = rec.Drel, rec.Gam
        W = self._lagrangian(Drel, Gam)
        popdm = self._so_perturbed_relaxed_density if so else self._perturbed_relaxed_density
        pert = [Perturbation('nuclear', (A, ct)) for A in range(natom) for ct in range(3)]

        def rot4(Um, T):
            return (c('tp,tqrs->pqrs', Um, T) + c('tq,ptrs->pqrs', Um, T)
                    + c('tr,pqts->pqrs', Um, T) + c('ts,pqrt->pqrs', Um, T))

        # first-order responses + hoisted per-Y rotated densities (sum A rot(U,B) = sum rot(U^T,A) B)
        resp = [popdm(p) for p in pert]                              # one perturbed solve per nucleus
        dDrel = [r.dDrel for r in resp]
        dGamN = [r.dGam for r in resp]
        dW = [r.dW for r in resp]
        U = [np.asarray(cphf.full_U(p, ncore, canonical=canonical)) for p in pert]
        Dtil = [U[i] @ Drel + Drel @ U[i].T for i in range(nc)]
        Wtil = [U[i] @ W + W @ U[i].T for i in range(nc)]
        Gtil = [rot4(U[i].T, Gam) for i in range(nc)]

        # per-X first skeletons; J^X carries the Fock skeleton's occupied-sum rotation response
        fX, gamX, SX, JX = [], [], [], []
        for p in pert:
            A, ct = p.comp
            hx = np.asarray((d.so_core(A) if so else d.core(A))[ct])
            Sx = np.asarray((d.so_overlap(A) if so else d.overlap(A))[ct])
            if so:
                eF = np.asarray(d.so_eri(A)[ct])
                gm = eF
            else:
                ph = np.asarray(d.eri(A)[ct]).transpose(0, 2, 1, 3)     # <pq|rs>^X (Gamma)
                eF = 2.0 * ph - ph.transpose(0, 1, 3, 2)                # L^X (Fock)
                gm = ph
            fX.append(hx + c('pmqm->pq', eF[:, ofull, :, ofull]))
            gamX.append(gm)
            SX.append(Sx)
            JX.append(c('pq,prqm->rm', Drel, eF[:, :, :, ofull])
                      + c('pq,pmqr->rm', Drel, eF[:, ofull, :, :]))

        H = np.zeros((nc, nc))
        for iy, py in enumerate(pert):
            Ay, cy = py.comp
            for ix, px in enumerate(pert):
                Ax, cx = px.comp
                blk = cphf.nuclear_hessian_skeletons(Ax, Ay)             # raw second skeletons (no U^{XY})
                core2 = blk['core'][cx * 3 + cy]
                ov2 = blk['overlap'][cx * 3 + cy]
                e2 = blk['eri'][cx * 3 + cy]
                L2 = e2 if so else 2.0 * e2 - e2.swapaxes(2, 3)
                f2 = core2 + c('pmqm->pq', L2[:, ofull, :, ofull])       # f^{XY}
                H[ix, iy] = (c('pq,pq->', dDrel[iy] + Dtil[iy], fX[ix]) + c('pq,pq->', Drel, f2)
                             + c('pqrs,pqrs->', dGamN[iy] + Gtil[iy], gamX[ix]) + c('pqrs,pqrs->', Gam, e2)
                             + c('pq,pq->', dW[iy] + Wtil[iy], SX[ix]) + c('pq,pq->', W, ov2)
                             + float(np.sum(U[iy][:, ofull] * JX[ix])))
        return H

    @staticmethod
    def _dependent_pairs(Iblock, eps_block, thresh=1e-8):
        """Canonical dependent-pair rotation ``P_mn = (I'_mn - I'_nm)/(eps_m - eps_n)`` for a square
        occ-occ or virt-virt Lagrangian block ``Iblock`` and its orbital energies ``eps_block``.
        Numerator-gated (``|Delta I'| < thresh`` -> 0), skipping the diagonal (``m=n``) and
        near-degenerate pairs.  ``P`` is symmetric (numerator and denominator both antisymmetric).

        This is the frozen-core core<->active-occupied divide generalized to an arbitrary square
        block; it also supplies the active oo/vv rotations of the canonical perturbed-MO gauge
        (:attr:`perturbed_mo_gauge`, used by :meth:`_orbital_response` / :meth:`_so_orbital_response`)."""
        num = np.asarray(Iblock) - np.asarray(Iblock).T
        den = eps_block[:, None] - eps_block[None, :]
        P = np.zeros_like(num)
        m = np.abs(num) > thresh
        P[m] = num[m] / den[m]
        return P

    @staticmethod
    def _perturbed_dependent_pairs(dIblock, Pblock0, eps_block, dfdiag_block, thresh=1e-8):
        """Field derivative ``dP`` of :meth:`_dependent_pairs` (quotient rule):
        ``dP_mn = (dI'_mn - dI'_nm)/(eps_m - eps_n) - P0_mn (df_mm - df_nn)/(eps_m - eps_n)`` -- the
        second term the canonical-``df``-diagonal denominator derivative, using the unperturbed
        ``Pblock0``.  Gated on ``|eps_m - eps_n| > thresh`` (diagonal + near-degenerate -> 0)."""
        dnum = np.asarray(dIblock) - np.asarray(dIblock).T
        gap = eps_block[:, None] - eps_block[None, :]
        dgap = dfdiag_block[:, None] - dfdiag_block[None, :]
        dP = np.zeros_like(dnum)
        m = np.abs(gap) > thresh
        dP[m] = (dnum[m] - np.asarray(Pblock0)[m] * dgap[m]) / gap[m]
        return dP
