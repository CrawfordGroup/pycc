"""Shared base for correlated analytic-derivative property drivers (MP2, CC; CI to follow).

`CorrelatedDerivs` owns the orbital-response and assembly machinery shared across methods.  Most of
it depends only on the reduced densities and the SCF reference; the one method-dependent choice --
the perturbed-MO gauge (:attr:`CorrelatedDerivs.perturbed_mo_gauge`, ``'canonical'`` for CCSD(T) and
``'non-canonical'`` otherwise) -- is encapsulated here and selects the orbital-response variant.
The canonical gauge carries the extra oo/vv dependent-pair rotations, so the CCSD(T) orbital
response (and hence its gradient) is genuinely not the same computation as CCSD's -- the machinery
is shared, but it is not method-*agnostic*.  Method-specific subclasses (`MPderiv`, `CCderiv`)
supply the reduced densities and their first-order responses.  See docs/DERIVATIVES_PLAN_2026-06.md
section 9 for the base/leaf split and the phased plan; more machinery moves here in later phases.
"""

from __future__ import annotations

import contextlib
import os
import time
import tempfile
from collections import namedtuple

import numpy as np

from .derivatives import atom_label
from .timing import timer, timed, progress


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
#: ``dI`` the response of the energy-weighted density ``I = I'(Drel, Gam)``.  All three fall out of a
#: single perturbed solve -- the polarizability needs only ``dDrel``; the APT/Hessian
#: (nuclear-skeleton) assemblies also contract ``dGam`` and ``dI`` against the perturbed integrals.
PerturbedResponse = namedtuple('PerturbedResponse', 'dDrel dGam dI')


class CorrelatedDerivs:
    """Base class for correlated derivative-property drivers.

    Holds the correlated wavefunction and the shared derivative machinery, parameterized by the
    method-determined perturbed-MO gauge (:attr:`perturbed_mo_gauge`; canonical for CCSD(T), which
    changes the orbital response).  A subclass (`MPderiv`, `CCderiv`) is constructed from a converged
    correlated wavefunction and supplies the method-specific reduced densities and their perturbed
    responses.
    """

    #: process-lifetime counter yielding a distinct id per driver instance (deterministic, no RNG),
    #: used to namespace this object's method-dependent perturbed responses in the shared store.
    _obj_counter = 0

    def __init__(self, wfn) -> None:
        """Bind the converged correlated wavefunction and its (device-aware) contraction
        backend, and initialize the cached SCF-reference HFwfn handle (see
        :meth:`_reference_hf`)."""
        self.wfn = wfn
        self.contract = wfn.contract
        self._ref_hf = None
        CorrelatedDerivs._obj_counter += 1
        self._uid = CorrelatedDerivs._obj_counter

    def _reference_hf(self):
        """All-electron :class:`~pycc.hfwfn.HFwfn` for the SCF reference (cached) -- supplies the
        reference contribution the :mod:`pycc.properties` facade pairs with the correlation part,
        and the CPHF orbital Hessian used by the Z-vector solve."""
        if self._ref_hf is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.wfn.ref, orbital_basis=self.wfn.orbital_basis, quiet=True)
        return self._ref_hf

    # ---- public property API: each returns the PropertyComponents decomposition
    # (nuclear + reference + correlation).  The module-level pycc.<property> facades are thin
    # wrappers around these; the reference block comes from the SCF HFwfn (_reference_hf), the
    # correlation block from this driver's _correlation_* method. ----

    def dipole(self) -> "PropertyComponents":
        """Electric-dipole moment as a :class:`pycc.PropertyComponents`.  See :func:`pycc.dipole`."""
        from . import properties
        return properties.dipole(self)

    def gradient(self) -> "PropertyComponents":
        """Analytic energy gradient as a :class:`pycc.PropertyComponents`.  See :func:`pycc.gradient`."""
        from . import properties
        return properties.gradient(self)

    def polarizability(self, omega: float = 0.0, relaxed: bool = None,
                       units: str = 'Eh') -> "PropertyComponents":
        """Dipole polarizability as a :class:`pycc.PropertyComponents`.  See :func:`pycc.polarizability`."""
        from . import properties
        return properties.polarizability(self, omega=omega, relaxed=relaxed, units=units)

    def hessian(self) -> "PropertyComponents":
        """Molecular (nuclear) Hessian as a :class:`pycc.PropertyComponents`.  See :func:`pycc.hessian`."""
        from . import properties
        return properties.hessian(self)

    def apt(self, gauge: str = 'length', route: str = '2n+1-field',
            orbital_gauge: str = 'non-canonical') -> "PropertyComponents":
        """Atomic polar tensors as a :class:`pycc.PropertyComponents`.  See :func:`pycc.apt`."""
        from . import properties
        return properties.apt(self, gauge=gauge, route=route, orbital_gauge=orbital_gauge)

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
        if getattr(self, '_gauge_override', None) is not None:
            return self._gauge_override   # test/validation override (e.g. force canonical for CCSD)
        return 'canonical' if getattr(self.wfn, 'model', '').upper() == 'CCSD(T)' else 'non-canonical'

    def _full_occ_cphf(self):
        """A CPHF over the **full** occupied space (frozen core + active) in the wavefunction's own
        MO ordering (cached).  The perturbed-response (2n+1) routes need the orbital response over the
        full occupied space (core<->active and core-virtual), which the active-only ``wfn.cphf``
        can't supply; building it here -- rather than borrowing an all-electron ``HFwfn`` -- keeps
        the spin-orbital ordering consistent with the densities.  For ``nfzc=0`` it coincides with
        ``wfn.cphf``.  CPHF depends only on the shared reference/orbitals/integrals, so building it on
        ``self.wfn`` works uniformly for the MPwfn and the CCwfn."""
        if getattr(self, '_focphf', None) is None:
            from .cphf import CPHF
            self._focphf = CPHF(self.wfn, full_occ=True)
        return self._focphf

    # ---- generalized-Fock orbital Lagrangian I'(D, Gam) ----
    # The Gauss/Stanton/Bartlett orbital-gradient Lagrangian: a method-agnostic function of a
    # full-MO 1-PDM ``D`` and cumulant 2-PDM ``Gam`` (both supplied by the leaf) plus the SCF
    # reference integrals/Fock.  Its occupied-virtual antisymmetric part ``X_ai = I'_ia - I'_ai``
    # drives the Z-vector; evaluated at the relaxed density it is the energy-weighted density ``I``.
    # Basis-dispatched: antisymmetrized ``<pq||rs>`` (spin-orbital) vs the spin-adapted ``L``
    # (closed-shell).  ``Gam`` must carry the proper 2-PDM permutational symmetry (the caller's
    # densities -- MP2 seeds / :meth:`ccdensity.gradient_densities` -- ensure this).

    def _lagrangian(self, D, Gam) -> np.ndarray:
        r"""Generalized-Fock orbital Lagrangian ``I'(D, Gam)`` (``nmo x nmo``); repeated
        indices summed, ``q`` over the full occupied space in the 1-PDM term::

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} D_rs (w_rpsq + w_rqsp)
                           + 4 <pr||st> Gamma_qrst ]

        .. math::

            I'_{pq} = -\tfrac{1}{2}\big[ f_{pp} (D_{pq} + D_{qp})
                + \delta_{q\in o_\mathrm{full}}\, D_{rs}(w_{rpsq} + w_{rqsp})
                + 4\,\langle pr\Vert st\rangle\,\Gamma_{qrst} \big]

        with the two-electron kernel ``w`` = the antisymmetrized ``<pq||rs>`` (spin-orbital,
        :meth:`_so_lagrangian`) or the spin-adapted ``L`` (closed-shell, :meth:`_spatial_lagrangian`),
        dispatched on the orbital basis."""
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_lagrangian(D, Gam)
        return self._spatial_lagrangian(D, Gam)

    def _so_lagrangian(self, D, Gam) -> np.ndarray:
        r"""Spin-orbital generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) from a full-MO
        1-PDM ``D`` and cumulant 2-PDM ``Gam`` (repeated indices summed)::

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} D_rs (<rp||sq> + <rq||sp>)
                           + 4 <pr||st> Gamma_qrst ]

        .. math::

            I'_{pq} = -\tfrac{1}{2}\big[ f_{pp} (D_{pq} + D_{qp})
                + \delta_{q\in o_\mathrm{full}}\, D_{rs}(\langle rp\Vert sq\rangle + \langle rq\Vert sp\rangle)
                + 4\,\langle pr\Vert st\rangle\,\Gamma_{qrst} \big]

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
        r"""Spin-adapted (closed-shell) generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) -- the
        closed-shell analogue of :meth:`_so_lagrangian` (repeated indices summed)::

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} D_rs (L_rpsq + L_rqsp)
                           + 4 <pr|st> Gamma_qrst ]

        .. math::

            I'_{pq} = -\tfrac{1}{2}\big[ f_{pp} (D_{pq} + D_{qp})
                + \delta_{q\in o_\mathrm{full}}\, D_{rs}(L_{rpsq} + L_{rqsp})
                + 4\,\langle pr|st\rangle\,\Gamma_{qrst} \big]

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
    # its response) it is the perturbed energy-weighted density d_x I.  The termA derivative is the
    # FULL Fock matrix product df @ (D + D.T) -- not the diagonal d_x(eps) stencil of the unperturbed
    # form (valid only at F=0); the off-diagonal df couples the ov/core-active blocks of a relaxed D
    # (zero for an unrelaxed MP2 D, nonzero for CC's Dov/Dvo).

    def _perturbed_lagrangian(self, df, deri, dL, D, dD, Gam, dGam) -> np.ndarray:
        r"""Spin-adapted first-order response ``dI'`` (``nmo x nmo``) given the perturbed
        integrals and density responses (``d`` = the full ``d/dx`` derivative; repeated
        indices summed)::

            dI'_pq = -1/2 [ df @ (D + D.T) + eps_p (dD_pq + dD_qp)
                            + delta_{q in ofull} ( dD_rs L_rpsq + D_rs dL_rpsq
                                                   + dD_rs L_rqsp + D_rs dL_rqsp )
                            + 4 ( deri_prst Gamma_qrst + <pr|st> dGamma_qrst ) ]

        .. math::

            \begin{aligned}
            \partial_x I'_{pq} = -\tfrac{1}{2}\big[ &(\partial_x f\,(D + D^{T}))_{pq}
                + f_{pp}(\partial_x D_{pq} + \partial_x D_{qp}) \\
            &+ \delta_{q\in o_\mathrm{full}}(\partial_x D_{rs} L_{rpsq} + D_{rs}\,\partial_x L_{rpsq}
                + \partial_x D_{rs} L_{rqsp} + D_{rs}\,\partial_x L_{rqsp}) \\
            &+ 4(\partial_x \langle pr|st\rangle\,\Gamma_{qrst} + \langle pr|st\rangle\,\partial_x \Gamma_{qrst}) \big]
            \end{aligned}

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
        r"""Spin-orbital first-order response ``dI'`` -- the antisymmetrized-integral analogue of
        :meth:`_perturbed_lagrangian` (the ``<pq||rs>`` derivative ``deri`` in the 1-PDM term in
        place of ``L``/``dL``); ``d`` = the full ``d/dx`` derivative, repeated indices summed::

            dI'_pq = -1/2 [ df @ (D + D.T) + eps_p (dD_pq + dD_qp)
                            + delta_{q in ofull} ( dD_rs <rp||sq> + D_rs deri_rpsq
                                                   + dD_rs <rq||sp> + D_rs deri_rqsp )
                            + 4 ( deri_prst Gamma_qrst + <pr||st> dGamma_qrst ) ]

        .. math::

            \begin{aligned}
            \partial_x I'_{pq} = -\tfrac{1}{2}\big[ &(\partial_x f\,(D + D^{T}))_{pq}
                + f_{pp}(\partial_x D_{pq} + \partial_x D_{qp}) \\
            &+ \delta_{q\in o_\mathrm{full}}(\partial_x D_{rs}\langle rp\Vert sq\rangle + D_{rs}\,\partial_x \langle rp\Vert sq\rangle
                + \partial_x D_{rs}\langle rq\Vert sp\rangle + D_{rs}\,\partial_x \langle rq\Vert sp\rangle) \\
            &+ 4(\partial_x \langle pr\Vert st\rangle\,\Gamma_{qrst} + \langle pr\Vert st\rangle\,\partial_x \Gamma_{qrst}) \big]
            \end{aligned}
        """
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
    # with the Z-vector  G z = X,  X_ai = I'_ia - I'_ai (over the full occupied space).  The
    # non-redundant core<->active-occupied rotation is a direct divide P_co = (I'_ci - I'_ic) /
    # (eps_c - eps_i) (the SCF energy is invariant to occ-occ rotations), coupled into X.  The orbital
    # Hessian G is the all-electron SCF Hessian: borrowed from the reference HFwfn CPHF (spatial) or
    # built inline (spin-orbital, G_ia,jb = <aj||ib> + <ab||ij> + delta_ij delta_ab (eps_a - eps_i)).
    # Method-agnostic given the two leaf hooks below.

    def _unrelaxed_densities(self):
        """Leaf hook: the unrelaxed reduced densities ``(D, Gam)`` as full-MO arrays -- the 1-PDM
        (``nmo x nmo``, occupied/virtual diagonal blocks) and cumulant 2-PDM (``nmo^4``).  Supplied
        by the method (MP2 amplitude seeds / CC :meth:`ccdensity.gradient_densities`)."""
        raise NotImplementedError

    @timed("relaxed density (Z-vector)")
    def _orbital_response(self):
        r"""Spatial (closed-shell) unperturbed orbital-response (Z-vector) solve (cached, frozen-core
        aware), returning an :class:`OrbitalResponse` record.  The relaxed 1-PDM::

            Drel = D + P_co + P_oo + P_vv - z,   X_ai = I'_ia - I'_ai,   G z = X

        .. math::

            D^\mathrm{rel} = D + P_\mathrm{co} + P_\mathrm{oo} + P_\mathrm{vv} - z,
            \qquad X_{ai} = I'_{ia} - I'_{ai}, \qquad G z = X

        with the generalized-Fock Lagrangian ``I'`` from :meth:`_spatial_lagrangian` (spin-adapted
        ``L``), the frozen-core divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`` coupled into ``X``,
        the canonical-perturbed-MO oo/vv rotations ``P_oo``/``P_vv`` (populated only for
        :attr:`perturbed_mo_gauge` ``== 'canonical'``), and the ov Z-vector ``G z = X`` solved with
        the orbital Hessian ``G`` from the all-electron reference ``HFwfn`` CPHF (``mo_hessian`` =
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
            # dependent-pair rotations P_oo/P_vv (added to Drel, coupled into X).  This is
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

    @timed("relaxed density (Z-vector)")
    def _so_orbital_response(self):
        r"""Spin-orbital unperturbed orbital-response (Z-vector) solve (cached, frozen-core aware),
        returning an :class:`OrbitalResponse` record.  The relaxed 1-PDM::

            Drel = D + P_co + P_oo + P_vv - z,   X_ai = I'_ia - I'_ai,   G z = X

        .. math::

            D^\mathrm{rel} = D + P_\mathrm{co} + P_\mathrm{oo} + P_\mathrm{vv} - z,
            \qquad X_{ai} = I'_{ia} - I'_{ai}, \qquad G z = X

        with the generalized-Fock Lagrangian ``I'`` from :meth:`_so_lagrangian` (antisymmetrized
        ``<pq||rs>``), the frozen-core divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)`` coupled into
        ``X``, the canonical-perturbed-MO oo/vv rotations ``P_oo``/``P_vv`` (populated only for
        :attr:`perturbed_mo_gauge` ``== 'canonical'``), and the ov Z-vector ``G z = X`` solved with
        the orbital Hessian ``G`` built inline (``G_ia,jb = <aj||ib> + <ab||ij> + delta_ij
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
            # dependent-pair rotations P_oo/P_vv (added to Drel, coupled into X).  This is
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
    # unrelaxed densities (dDg, dGam), the assembly is shared across methods (gauge-parameterized,
    # so CCSD(T)'s canonical oo/vv pairs enter but the code is common): the perturbed Lagrangian dI'
    # (with the perturbed integrals df/deri/dL), the perturbed frozen-core divide d_x P_co (a
    # Sylvester relation), the perturbed canonical-MO oo/vv rotations d_x P_oo/d_x P_vv (gauge-gated),
    # and the perturbed ov Z-vector z^x = G^{-1}(dX - G^x z) reusing the unperturbed orbital Hessian G
    # and z from the OrbitalResponse record.

    def _perturbed_unrelaxed_densities(self, pert, df, deri, dL):
        """Leaf hook: the first-order response ``(d_x gamma, d_x Gamma)`` of the unrelaxed reduced
        densities to ``pert`` (full-MO arrays).  MP2 supplies the closed-form response
        (:meth:`MPderiv._perturbed_unrelaxed_densities`); CC supplies the iterative
        perturbed-amplitude / perturbed-Lambda response.  ``df``/``deri``/``dL`` are the CPHF-folded perturbed integrals
        (canonical per :attr:`perturbed_mo_gauge`) the CC iterative solve consumes; the MP2 closed
        form recomputes its own and ignores them."""
        raise NotImplementedError

    def _perturbed_relaxed_density(self, pert):
        r"""Spatial perturbed (2n+1) orbital response for ``pert`` -- a :class:`PerturbedResponse`
        ``(dDrel, dGam, dI)``.  The relaxed-1-PDM response (``nmo x nmo``) is::

            d_x Drel = d_x D + d_x P_co + d_x P_oo + d_x P_vv - z^x,   G z^x = dX - G^x z

        .. math::

            \partial_x D^\mathrm{rel} = \partial_x D + \partial_x P_\mathrm{co}
                + \partial_x P_\mathrm{oo} + \partial_x P_\mathrm{vv} - z^{x},
            \qquad G z^{x} = \partial_x X - G^{x} z

        with the perturbed unrelaxed density ``d_x D`` (:meth:`_perturbed_unrelaxed_densities`), the
        perturbed Lagrangian ``dI'`` (:meth:`_perturbed_lagrangian`) giving the perturbed
        Z-vector RHS ``dX_ai = dI'_ia - dI'_ai``, the perturbed frozen-core Sylvester divide
        ``d_x P_co = [d_x(I'_ci - I'_ic) - df_cd P_di + P_cj df_ji] / (eps_c - eps_i)`` coupled into
        ``dX``, the perturbed canonical-MO oo/vv rotations ``d_x P_oo``/``d_x P_vv``
        (:meth:`_perturbed_dependent_pairs`, populated only for :attr:`perturbed_mo_gauge` ``==
        'canonical'``) coupled into ``dX``, and the perturbed ov Z-vector ``z^x`` reusing the
        unperturbed orbital Hessian ``G`` and ``z`` from :meth:`_orbital_response` (``G^x z`` the
        perturbed-Hessian response).  The perturbed integrals ``df``/``deri`` are canonical per
        :attr:`perturbed_mo_gauge`.  The same solve yields the unrelaxed cumulant-2-PDM response
        ``dGam`` and the perturbed energy-weighted density ``dI = d_x I'(Drel, Gam)`` (the perturbed
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
        dI = self._perturbed_lagrangian(df, deri, dL, Drel0, dDrel, Gam0, dGam)
        return PerturbedResponse(dDrel, dGam, dI)

    def _so_perturbed_relaxed_density(self, pert):
        """Spin-orbital perturbed (2n+1) orbital response -- the spin-orbital analogue of
        :meth:`_perturbed_relaxed_density`, returning the same :class:`PerturbedResponse`
        ``(dDrel, dGam, dI)`` (antisymmetrized ``<pq||rs>`` derivatives ``deri`` in the couplings;
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
        dI = self._so_perturbed_lagrangian(df, deri, Drel0, dDrel, Gam0, dGam)
        return PerturbedResponse(dDrel, dGam, dI)

    @timed("perturbed density")
    def _relaxed_response(self, pert):
        """Per-perturbation :class:`PerturbedResponse` ``(dDrel, dGam, dI)`` for ``pert``,
        dispatched to the spatial or spin-orbital solve and memoized in the persistent store
        (:attr:`Derivatives.store`, disk when enabled, RAM otherwise).  The record is
        method-dependent -- it consumes this driver's amplitudes through
        :meth:`_perturbed_unrelaxed_densities` -- so it is keyed on the driver-instance id
        ``_uid`` alongside the perturbation, gauge, and route.  A second property call on the
        *same* driver then reads the record back (skipping the perturbed-amplitude / Z-vector
        solve and the ``nmo^4`` cumulant-response build), while never colliding with a different
        driver's response (e.g. CCSD vs CCSD(T)) on the same wavefunction."""
        so = self.wfn.orbital_basis == 'spinorbital'
        popdm = self._so_perturbed_relaxed_density if so else self._perturbed_relaxed_density
        ctx = (self._uid, self.perturbed_mo_gauge, 'so' if so else 'sp')
        parts = self.wfn.derivatives.store.get_or_compute_group(
            'resp', pert, lambda: tuple(popdm(pert)), ('dDrel', 'dGam', 'dI'), ctx=ctx)
        return PerturbedResponse(*parts)

    # ---- first-derivative properties: relaxed dipole and nuclear gradient ----
    # Both are contractions of the relaxed density against the property integrals, method-agnostic
    # given (Drel, Gam) and the energy-weighted density I = I'(Drel).  The reference (SCF) and
    # nuclear contributions are kept separate and summed by the pycc.properties facade.

    def _correlation_dipole(self) -> np.ndarray:
        r"""Correlation contribution to the electronic dipole moment (a.u.), shape ``(3,)``
        (repeated indices summed)::

            mu_a^corr = Drel_pq (mu_a)_pq

        .. math::

            \mu_a^\mathrm{corr} = D^\mathrm{rel}_{pq}\,(\mu_a)_{pq}

        the relaxed 1-PDM contracted with the MO dipole integrals (``H.mu = -e r``).  A static
        field does not move the AO basis, so there is no energy-weighted-density or 2-PDM term (only
        the gradient has those); the orbital relaxation (and, for CCSD(T), the canonical-MO oo/vv
        response) rides inside ``Drel``.  The reference (SCF) dipole is kept separate; the total is
        their sum.  Basis-aware (dispatches via :meth:`_relaxed_density`)."""
        Drel, _ = self._relaxed_density()
        c = self.contract
        return np.array([c('pq,pq->', Drel, np.asarray(self.wfn.H.mu[a])) for a in range(3)])

    def _correlation_gradient(self) -> np.ndarray:
        r"""Correlation contribution to the analytic nuclear energy gradient (a.u.), shape
        ``(natom, 3)`` (repeated indices summed)::

            dE_corr/dX = Drel_pq f^(X)_pq + Gamma_pqrs <pq|rs>^(X) + W_pq S^(X)_pq

        .. math::

            \frac{\partial E_\mathrm{corr}}{\partial X} = D^\mathrm{rel}_{pq}\,f^{(X)}_{pq}
                + \Gamma_{pqrs}\,\langle pq|rs\rangle^{(X)} + W_{pq}\,S^{(X)}_{pq}

        with the relaxed 1-PDM ``Drel``, cumulant 2-PDM ``Gamma`` (:meth:`_relaxed_density`), and the
        energy-weighted density ``I = I'(Drel)`` (:meth:`_lagrangian`).  ``f^(X) = h^(X) + sum_m
        L[p,m,q,m]^(X)`` is the closed-shell skeleton Fock derivative (``m`` over the full occupied
        space), and ``S^(X)``/``<pq|rs>^(X)`` are the skeleton derivative integrals from
        ``wfn.derivatives`` (chemist ``(pq|rs)^(X)``, converted to physicist here) -- no
        per-perturbation CPHF solve.  Spatial (closed-shell RHF) path; the spin-orbital path is
        :meth:`_so_gradient`.  The reference (SCF) gradient is kept separate."""
        if self.wfn.orbital_basis == 'spinorbital':
            return self._so_correlation_gradient()
        ofull = slice(0, self.wfn.o.stop)                # full occupied (core + active)
        Drel, Gam = self._relaxed_density()
        I = self._lagrangian(Drel, Gam)
        c = self.contract
        d = self.wfn.derivatives
        grad = np.zeros((d.natom, 3))
        route = getattr(self, '_skel_eri_route', 'aod')  # default; 'mo' is the per-atom-MO opt-out
        if route == 'aod':
            # AO-density route (plan doc s.10): fold the 2-e part of Drel*f^(X) into Gam_eff, back-
            # transform it to AO ONCE, and contract the raw ao_tei_deriv1 directly -- no per-atom MO
            # transform and no f^(X) build (the 1-e remainder Drel*h^(X) + I*S^(X) stays in the cheap
            # MO OEI blocks).  Reuses the Hessian's _effective_2pdm_ao: ao_tei_deriv1 is fully
            # permutationally symmetric, so that builder's completion + ket-swap are exact no-ops.
            GeffAO = self._effective_2pdm_ao(Drel, Gam)               # 1*nao^4, built once
            for atom in range(d.natom):
                hx = d.core(atom); Sx = d.overlap(atom); ao1 = d.ao_eri1(atom)   # 3 spatial AO, held
                for cart in range(3):
                    grad[atom, cart] = (c('pq,pq->', Drel, hx[cart])
                                        + c('mnls,mnls->', GeffAO, ao1[cart])
                                        + c('pq,pq->', I, Sx[cart]))
        else:
            for atom in range(d.natom):
                hx = d.core(atom); Sx = d.overlap(atom); ERIx = d.eri(atom)   # physicist <pq|rs>^(X)
                for cart in range(3):
                    phys = ERIx[cart]
                    Lx = 2.0 * phys - phys.transpose(0, 1, 3, 2)
                    fx = hx[cart] + c('pmqm->pq', Lx[:, ofull, :, ofull])  # skeleton Fock deriv (full occ)
                    grad[atom, cart] = (c('pq,pq->', Drel, fx)
                                        + c('pqrs,pqrs->', Gam, phys)
                                        + c('pq,pq->', I, Sx[cart]))
        return grad

    def _so_correlation_gradient(self) -> np.ndarray:
        r"""Spin-orbital correlation gradient -- the spin-orbital analogue of :meth:`gradient`
        with the antisymmetrized ``<pq||rs>^(X)`` from ``wfn.derivatives.so_*`` (``m`` over the
        full occupied space; repeated indices summed)::

            dE_corr/dX = Drel_pq f^(X)_pq + Gamma_pqrs <pq||rs>^(X) + W_pq S^(X)_pq,
            f^(X) = h^(X) + <pm||qm>^(X)

        .. math::

            \frac{\partial E_\mathrm{corr}}{\partial X} = D^\mathrm{rel}_{pq}\,f^{(X)}_{pq}
                + \Gamma_{pqrs}\,\langle pq\Vert rs\rangle^{(X)} + W_{pq}\,S^{(X)}_{pq},
            \qquad f^{(X)}_{pq} = h^{(X)}_{pq} + \langle pm\Vert qm\rangle^{(X)}
        """
        ofull = slice(0, self.wfn.o.stop)                # full occupied (core + active)
        Drel, Gam = self._so_relaxed_density()
        I = self._lagrangian(Drel, Gam)
        c = self.contract
        d = self.wfn.derivatives
        grad = np.zeros((d.natom, 3))
        route = getattr(self, '_skel_eri_route', 'aod')  # default; 'mo' is the per-atom-MO opt-out
        if route == 'aod':
            # AO-density route (plan doc s.10): the spin-orbital analogue, reusing
            # _effective_2pdm_ao_so (which folds the spin blocks + <pq||rs> antisymmetrization onto
            # the one spatial ao_tei_deriv1).  No per-atom SO transform and no f^(X) build.
            GeffAO = self._effective_2pdm_ao_so(Drel, Gam)           # 1*nao^4, built once
            for atom in range(d.natom):
                hx = d.so_core(atom); Sx = d.so_overlap(atom); ao1 = d.ao_eri1(atom)
                for cart in range(3):
                    grad[atom, cart] = (c('pq,pq->', Drel, hx[cart])
                                        + c('mnls,mnls->', GeffAO, ao1[cart])
                                        + c('pq,pq->', I, Sx[cart]))
        else:
            for atom in range(d.natom):
                hx = d.so_core(atom); Sx = d.so_overlap(atom); ERIx = d.so_eri(atom)
                for cart in range(3):
                    fx = hx[cart] + c('pmqm->pq', ERIx[cart][:, ofull, :, ofull])  # skeleton Fock deriv
                    grad[atom, cart] = (c('pq,pq->', Drel, fx)
                                        + c('pqrs,pqrs->', Gam, ERIx[cart])
                                        + c('pq,pq->', I, Sx[cart]))
        return grad

    # ---- second-derivative properties: polarizability, APT (dipole derivatives), Hessian ----
    # All three are the asymmetric (2n+1) route: differentiate a relaxed-density first derivative a
    # second time, using only first-order responses (the perturbed relaxed density / energy-weighted
    # density and U^y -- no second-order CPHF U^{xy}).  Method-agnostic given the orbital-response
    # record and the PerturbedResponse hook; these are the public correlation-property API (the
    # pycc.properties facade calls them by name).  The reference (SCF) and nuclear parts stay
    # separate and are summed by the facade.  A leaf overrides one of these only to add
    # method-specific behavior (e.g. CCderiv.polarizability's model / (T)-intermediate guards).

    def _correlation_polarizability(self) -> np.ndarray:
        r"""Correlation contribution to the static (omega=0) dipole polarizability (a.u.), shape
        ``(3, 3)``: ``alpha_corr_ab = -d^2 E_corr / dF_a dF_b``, via the 2n+1 route (frozen-core
        aware; spin-orbital and spin-adapted paths).  Differentiating the relaxed dipole
        ``d_b E = -Tr(D_rel mu_b)`` (field skeleton ``f^(b) = -mu_b``) a second time::

            alpha_ab = d_a D_rel_pq (mu_b)_pq + D_rel_pq [ (U^a).T mu_b + mu_b U^a ]_pq

        .. math::

            \alpha_{ab} = \partial_a D^\mathrm{rel}_{pq}\,(\mu_b)_{pq}
                + D^\mathrm{rel}_{pq}\,[(U^a)^{T}\mu_b + \mu_b U^a]_{pq}

        The first term is the perturbed relaxed density (:meth:`_perturbed_relaxed_density`, carrying
        the perturbed Z-vector and, for frozen core, the perturbed core-active divide); the second is
        the MO dipole rotating under the field (``U^a`` over the full occupied space -- ``ncore``
        canonical core-active block, gauge per :attr:`perturbed_mo_gauge`).  No second-order CPHF
        ``U^{ab}`` -- only first-order responses.

        The reference part is kept separate (:meth:`HFwfn.polarizability`) and summed with this
        correlation part by :func:`pycc.polarizability`."""
        from .cphf import Perturbation
        wfn = self.wfn
        c = self.contract
        ncore = wfn.o.stop - wfn.no
        canonical = self.perturbed_mo_gauge == 'canonical'
        cphf = self._full_occ_cphf()
        if wfn.orbital_basis == 'spinorbital':
            Drel = self._so_orbital_response().Drel
        else:
            Drel = self._orbital_response().Drel
        mu = [np.asarray(wfn.H.mu[a]) for a in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            pert = Perturbation('field', b)
            dDrel = self._relaxed_response(pert).dDrel
            Ub = np.asarray(cphf.full_U(pert, ncore, canonical=canonical))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def _correlation_dipole_derivatives(self, route: str = '2n+1-field') -> np.ndarray:
        r"""Correlation contribution to the atomic polar tensors (nuclear dipole derivatives, a.u.),
        shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` =
        ``d(mu_alpha)/d(X_{A,beta}) = -d^2 E_corr / dF_alpha dX_{A,beta}`` -- the mixed field/nuclear
        analog of :meth:`polarizability`, via the 2n+1 route (both spin paths, frozen-core aware).

        ``route='2n+1-field'`` (default) or ``'2n+1-nuclear'``; both give the same tensor, but which
        is cheaper depends on context.  **Standalone**, ``'2n+1-field'`` is usually cheaper (it solves
        the perturbed response along only the 3 field components, versus ``3N`` nuclear), though the
        margin is method-dependent: clear for CCSD, whose perturbed responses are iterative (measured
        ~1.5-2x on H2O/H2O2/CH4 in cc-pVDZ), but a near-wash for MP2, whose closed-form responses are
        offset by the field route's own ``3N`` nuclear skeleton-ERI builds.  **In an IR/VCD
        spectrum**, though, the Hessian is computed first and has already solved and cached the
        ``3N`` nuclear perturbed responses on this same driver (the DerivStore / CPHF nuclear-response
        caches); there ``'2n+1-nuclear'`` reuses them for free -- no new perturbed solves -- while
        ``'2n+1-field'`` still pays 3 fresh field solves, so ``'2n+1-nuclear'`` is the cheaper choice
        (measured ~8-16x for MP2 and ~22-47x for CCSD in cc-pVDZ, the margin growing with basis).
        The default suits the standalone case; a spectrum workflow may prefer ``'2n+1-nuclear'``.  The
        nuclear ``Z_A`` and SCF reference terms are kept separate and summed with this correlation
        part by :func:`pycc.apt`.

        Nuclear side -- differentiate the relaxed dipole ``Tr(D_rel mu_a)`` w.r.t. the nucleus (the
        field gradient has no ``S^(X)``/2e-skeleton term, so no energy-weighted density appears)::

            P[X,a] = Tr(d_X D_rel mu_a) + Tr(D_rel [mu_a^(X) + rotate(U^X, mu_a)])

        .. math::

            P[X,a] = \mathrm{Tr}(\partial_X D^\mathrm{rel}\,\mu_a)
                + \mathrm{Tr}(D^\mathrm{rel}\,[\mu_a^{(X)} + \mathrm{rotate}(U^X, \mu_a)])

        Field side -- differentiate the relaxed nuclear gradient
        ``dE/dX = D_rel f^(X) + Gamma <pq|rs>^(X) + I S^(X)`` w.r.t. the field::

            P[X,a] = -[ d_a D_rel f^(X) + D_rel d_a f^(X) + d_a Gamma <pq|rs>^(X)
                        + Gamma d_a <pq|rs>^(X) + d_a I S^(X) + I d_a S^(X) ]

        .. math::

            \begin{aligned}
            P[X,a] = -\big[ &\partial_a D^\mathrm{rel}_{pq} f^{(X)}_{pq} + D^\mathrm{rel}_{pq}\,\partial_a f^{(X)}_{pq}
                + \partial_a \Gamma_{pqrs}\,\langle pq|rs\rangle^{(X)} \\
            &+ \Gamma_{pqrs}\,\partial_a \langle pq|rs\rangle^{(X)} + \partial_a I_{pq}\,S^{(X)}_{pq}
                + I_{pq}\,\partial_a S^{(X)}_{pq} \big]
            \end{aligned}

        with the 3 field responses ``d_a D_rel``, ``d_a Gamma``, and the perturbed energy-weighted
        density ``d_a I`` all from one :class:`PerturbedResponse` per field
        (:meth:`_perturbed_relaxed_density`).  The orbital-response terms are assembled in the
        canonical orbital-response form (the ``2 U^Y X~^(X) + S^(Y) I~''^(X) + P^(X) f^(Y)`` line
        shared with :meth:`_correlation_hessian`): with the field skeleton
        ``f^(a) = -mu``, ``S^(a) = 0``, ``<>^(a) = 0``, the only surviving pieces are ``2 U^a_bi
        X~^(X)_bi`` and ``P^(X)_pq f^(a)_pq`` (from :meth:`_skeleton_lagrangian` and
        :meth:`_augment_with_canonical_pair_rotations`), plus the fixed-density mixed skeleton
        ``D_rel f^(Xa)`` with ``f^(Xa) = -mu^(X)`` (the field enters ``h``).  Both routes give the same
        tensor; see the route note above for the cost trade-off (``'2n+1-field'`` cheaper standalone,
        ``'2n+1-nuclear'`` cheaper when a Hessian has already cached the nuclear responses)."""
        if route not in ('2n+1-nuclear', '2n+1-field'):
            raise ValueError(f"unknown dipole-derivative route {route!r} "
                             "(use '2n+1-nuclear' or '2n+1-field')")
        from .cphf import Perturbation
        wfn = self.wfn
        c = self.contract
        so = wfn.orbital_basis == 'spinorbital'
        o, v = wfn.o, wfn.v
        ofull = slice(0, o.stop)
        ncore = o.stop - wfn.no
        canonical = self.perturbed_mo_gauge == 'canonical'
        cphf = self._full_occ_cphf()
        d = wfn.derivatives
        natom = d.natom
        rec = self._so_orbital_response() if so else self._orbital_response()
        Drel = rec.Drel
        mu = [np.asarray(wfn.H.mu[a]) for a in range(3)]
        P = np.zeros((natom, 3, 3))

        if route == '2n+1-nuclear':
            for A in range(natom):
                dip = d.so_dipole(A) if so else d.dipole(A)          # [alpha*3 + beta]
                for beta in range(3):
                    pX = Perturbation('nuclear', (A, beta))
                    dDrel = self._relaxed_response(pX).dDrel
                    UX = np.asarray(cphf.full_U(pX, ncore, canonical=canonical))
                    for alpha in range(3):
                        dmu = np.asarray(dip[alpha * 3 + beta])       # skeleton d(mu_a)/dX_beta
                        rot = UX.T @ mu[alpha] + mu[alpha] @ UX
                        P[A, beta, alpha] = (c('pq,pq->', dDrel, mu[alpha])
                                             + c('pq,pq->', Drel, dmu + rot))
            return P

        # route == '2n+1-field'
        Gam = rec.Gam
        I = self._lagrangian(Drel, Gam)
        field = [Perturbation('field', a) for a in range(3)]
        resp = [self._relaxed_response(field[a]) for a in range(3)]   # one perturbed solve per field
        dDrel = [r.dDrel for r in resp]
        dGamF = [r.dGam for r in resp]                                # F = field-perturbation response
        dI = [r.dI for r in resp]                                     # perturbed energy-weighted density
        U = [np.asarray(cphf.full_U(field[a], ncore, canonical=canonical)) for a in range(3)]

        for A in range(natom):
            hx = d.so_core(A) if so else d.core(A)
            Sx = d.so_overlap(A) if so else d.overlap(A)
            dip = d.so_dipole(A) if so else d.dipole(A)
            if so:
                eriL = [np.asarray(e) for e in d.so_eri(A)]          # <pq||rs>^(X) (Fock and Gamma)
                eriX = eriL
            else:
                phys = [np.asarray(ch) for ch in d.eri(A)]                        # <pq|rs>^(X) (Gamma)
                eriL = [2.0 * p - p.transpose(0, 1, 3, 2) for p in phys]          # L^X (Fock)
                eriX = phys
            for beta in range(3):
                fX = np.asarray(hx[beta]) + c('pmqm->pq', eriL[beta][:, ofull, :, ofull])
                SX = np.asarray(Sx[beta])
                eriXb, eriLb = eriX[beta], eriL[beta]                 # this Cartesian's X-skeletons
                # Reference-doc form (eq:d2E-canon-final): build the nuclear (X) orbital-response
                # carriers once per (A, beta) from the X-skeletons, then contract with the field (F)
                # per alpha.  The field skeleton is f^(F) = -mu, S^(F) = 0, <>^(F) = 0, so the only
                # surviving orbital-response terms are 2 U^F X~^(X) and P^(X) f^(F); the fixed-density
                # mixed skeleton is D~ f^(XF) with f^(XF) = d(-mu)/dX = -muX (no <>^(XF)/S^(XF)).
                Ip, xov, i2 = self._skeleton_lagrangian(fX, SX, eriLb, eriXb, Drel, Gam, I)
                if canonical or ncore:
                    Xt, _, Pf = self._augment_with_canonical_pair_rotations(Ip, xov, i2)
                else:
                    Xt, Pf = xov, None
                for alpha in range(3):
                    Um = U[alpha]
                    muX = np.asarray(dip[alpha * 3 + beta])
                    orb = 2.0 * c('ai,ai->', Um[v, ofull], Xt[v, ofull])          # 2 U^F_ai X~^(X)_ai
                    if Pf is not None:
                        orb = orb - c('pq,pq->', Pf, mu[alpha])                   # P^(X)_pq f^(F)_pq, f^(F)=-mu
                    P[A, beta, alpha] = -(c('pq,pq->', dDrel[alpha], fX)          # 2n+1 D response
                                          + c('pqrs,pqrs->', dGamF[alpha], eriXb)   # 2n+1 Gamma response
                                          + c('pq,pq->', dI[alpha], SX)           # 2n+1 I response
                                          - c('pq,pq->', Drel, muX)               # D~ f^(XF), f^(XF) = -muX
                                          + orb)
        return P

    @contextlib.contextmanager
    def _offload_assembly_idle_tensors(self):
        """Free the whole-run ``nmo^4`` tensors the two-pass Hessian assembly never reads, for the
        duration of the block, and restore them on exit.  The assembly contracts only ``Gam`` (pass 1)
        and the per-pair derivative tensors, so the baseline MO ERIs (``H.ERI`` and, on the spatial
        path, ``H.L``) and the raw CISD unrelaxed 2-PDM (``_ci_dens[2]``, used only to build ``Gam``
        and the perturbed-CI ``dE`` in the setup) sit idle -- 2-3 ``nmo^4`` arrays.  Spilling them
        drops the resident floor from ~4 to ~1 ``nmo^4`` (52 -> 13 GiB at cc-pVTZ; peak 168 -> 129 GiB).

        Each spilled array is written to a temp file and reloaded on exit (``H.ERI`` and the 2-PDM are
        genuine work -- the ``O(N^5)`` transform and the density build -- worth restoring rather than
        recomputing).  ``H.L = 2<pq|rs> - <pq|sr>`` is a linear combination of ``H.ERI``, so it is
        dropped and recomputed on restore.  The local reference is dropped after each spill so the RAM
        is actually reclaimed (nulling the attribute is not enough).  Exception-safe: the ``finally``
        always restores and deletes the temp files."""
        H = self.wfn.H
        restores = []

        def _spill(array, restore):
            """Write ``array`` to a temp file; register ``restore(reloaded_array)`` for the exit."""
            fd, path = tempfile.mkstemp(suffix='.npy')
            os.close(fd)
            np.save(path, np.asarray(array))
            restores.append((restore, path))

        # Baseline MO ERIs: spill H.ERI, drop H.L (recomputed from H.ERI on restore).
        eri = getattr(H, 'ERI', None)
        if eri is not None:
            has_L = getattr(H, 'L', None) is not None
            def _restore_eri(a):
                H.ERI = a
                if has_L:
                    H.L = 2.0 * a - a.swapaxes(2, 3)
            _spill(eri, _restore_eri)
            H.ERI = None
            if has_L:
                H.L = None
        del eri

        # CISD raw unrelaxed 2-PDM (absent for MP2/CC).  Its ONLY reference is the _ci_dens cache;
        # keep the two nmo^2 1-PDMs resident, spill the nmo^4 2-PDM, and restore the full triple.
        dens = getattr(self.wfn, '_ci_dens', None)
        if dens is not None and dens[2] is not None:
            def _restore_g(a, head=dens[:2]):     # head (D, D_corr) captured now; G is not held here
                self.wfn._ci_dens = (head[0], head[1], a)
            _spill(dens[2], _restore_g)
            self.wfn._ci_dens = (dens[0], dens[1], None)
        del dens

        try:
            yield
        finally:
            for restore, path in restores:
                restore(np.load(path))
                os.remove(path)

    @timed("two-particle density (AO)")
    def _effective_2pdm_ao(self, Drel, Gam):
        r"""Effective 2-PDM back-transformed to the AO chemist basis, for the AO-density Hessian
        skeleton (plan doc s.10).  Folds the two-electron part of ``D_rel * f^(XY)`` into the
        cumulant so the whole two-electron skeleton is a single density contraction
        ``Gam_eff^AO . (mu nu|la si)^(XY)``, and ``f^(XY)`` is never built::

            Gam_D[a,b,c,d] = 2 D_rel[a,c] P[b,d] - D_rel[a,d] P[b,c]   (physicist; P = occ. projector)
            Gam_eff        = Gam + Gam_D

        ``Gam_eff`` is bra<->ket symmetrized (so Psi4's raw, bra<->ket-doubled ``ao_tei_deriv2`` is
        exact without ``_complete_deriv2``), back-transformed to AO, then ket-swapped
        (``transpose(0,1,3,2)``) to invert ``mo_eri_helper``'s internal reorder so it contracts
        directly against the raw AO integral.  (Validated against the CC correlation energy for the
        fold and the MO route for the transform.)

        The same ``Gam_eff^AO`` also serves the AO-density *gradient* (:meth:`gradient`) contracted
        against the first-derivative ``ao_tei_deriv1``: that integral is fully permutationally
        symmetric, so the bra<->ket symmetrization and ket-swap here are exact no-ops on it."""
        wfn = self.wfn
        c = self.contract
        C = np.asarray(wfn.C)
        nmo = Gam.shape[0]
        Pocc = np.zeros((nmo, nmo))
        np.fill_diagonal(Pocc, (np.arange(nmo) < wfn.o.stop).astype(float))
        GamD = 2.0 * c('ac,bd->abcd', Drel, Pocc) - c('ad,bc->abcd', Drel, Pocc)
        G = (Gam + GamD).swapaxes(1, 2)                      # physicist <ab|cd> -> chemist (ab|cd)
        G = 0.5 * (G + G.transpose(2, 3, 0, 1))              # bra<->ket symmetrize (chemist)
        G = c('pqrs,mp->mqrs', G, C)                         # back-transform each index to the AO basis
        G = c('mqrs,nq->mnrs', G, C)
        G = c('mnrs,lr->mnls', G, C)
        G = c('mnls,ks->mnlk', G, C)
        return G.transpose(0, 1, 3, 2)                       # invert mo_eri_helper's ket reorder

    def _effective_2pdm_ao_so(self, Drel, Gam):
        r"""Spin-orbital effective 2-PDM back-transformed to the *spatial* AO chemist basis, for the
        AO-density Hessian skeleton (plan doc s.10) -- the spin-orbital analogue of
        :meth:`_effective_2pdm_ao`.  Folds the two-electron part of ``D_rel * f^(XY)`` (``f^(XY) =
        h^(XY) + <pm||qm>^(XY)``, ``m`` over the full occupied space) into the antisymmetrized
        cumulant so the whole two-electron skeleton is one density contraction
        ``Gam_eff . <pq||rs>^(XY)``::

            Gam_D[a,b,c,d] = D_rel[a,c] P[b,d]     (physicist; P = occ. projector; NO factor 2 and
            Gam_eff        = Gam + Gam_D            no explicit exchange -- <pq||rs> is antisymmetrized)

        Because ``<pq||rs>`` is built per spin block from the spin-free *spatial* AO second-derivative
        integral (:meth:`Derivatives.so_eri2_mo_component`), ``Gam_eff`` is back-transformed onto that
        one spatial AO tensor, summing the four same-spin combinations.  Mirroring the SO forward
        transform's steps in reverse: the ket antisymmetrization is folded into the density
        (``Ga = Gam_eff - Gam_eff.transpose(0,1,3,2)``, so it contracts the plain ``<pq|rs>``), the
        physicist->chemist swap becomes ``transpose(0,2,1,3)``, the bra<->ket average
        (:func:`_complete_deriv2`) becomes the chemist symmetrization, each spin block is
        back-transformed with the spin-blocked ``Ca``/``Cb``, and the trailing ket swap
        (``transpose(0,1,3,2)``) inverts ``mo_eri_helper``'s reorder so the result contracts directly
        against the raw spatial ``ao_tei_deriv2``.  (Validated per component against
        :meth:`Derivatives.so_eri2_mo_component`.)

        The same ``Gam_eff^AO`` also serves the AO-density spin-orbital *gradient*
        (:meth:`_so_gradient`) contracted against the first-derivative ``ao_tei_deriv1`` (fully
        permutationally symmetric, so the symmetrization and ket-swap here are exact no-ops on it)."""
        c = self.contract
        d = self.wfn.derivatives
        nso = Gam.shape[0]
        Pocc = np.zeros((nso, nso))
        np.fill_diagonal(Pocc, (np.arange(nso) < self.wfn.o.stop).astype(float))
        GamD = c('ac,bd->abcd', Drel, Pocc)                 # SO fold (antisym handles exchange; no *2)
        Ga = (Gam + GamD)
        Ga = Ga - Ga.transpose(0, 1, 3, 2)                  # fold in the <pq||rs> ket antisymmetrization
        Gc = Ga.transpose(0, 2, 1, 3)                       # physicist <pq|rs> -> chemist (pr|qs)
        Gd = 0.5 * (Gc + Gc.transpose(2, 3, 0, 1))          # bra<->ket symmetrize (chemist)
        shape, sel = d._so_eri_blocks(('all', 'all', 'all', 'all'))
        nao = np.asarray(sel[0][0][1]).shape[0]
        GdAO = np.zeros((nao, nao, nao, nao))
        for s12 in (0, 1):
            p1, C1 = sel[0][s12]
            p2, C2 = sel[1][s12]
            if not (p1.size and p2.size):
                continue
            C1a, C2a = np.asarray(C1), np.asarray(C2)
            for s34 in (0, 1):
                p3, C3 = sel[2][s34]
                p4, C4 = sel[3][s34]
                if not (p3.size and p4.size):
                    continue
                C3a, C4a = np.asarray(C3), np.asarray(C4)
                blk = Gd[np.ix_(p1, p2, p3, p4)]            # this spin combo's SO density block
                t = c('pqrs,mp->mqrs', blk, C1a)            # back-transform each index to spatial AO
                t = c('mqrs,nq->mnrs', t, C2a)
                t = c('mnrs,lr->mnls', t, C3a)
                t = c('mnls,ks->mnlk', t, C4a)
                GdAO += t
        return GdAO.transpose(0, 1, 3, 2)                   # invert mo_eri_helper's ket reorder

    def _hessian_blocks(self):
        r"""The ``(reference, correlation)`` electronic molecular-Hessian blocks (a.u.), each shape
        ``(3*natom, 3*natom)``.  The correlation block is the contribution below; the reference block
        is the SCF electronic Hessian, computed HERE (not by a separate ``HFwfn`` pass) so its
        ao-dependent skeleton shares this assembly's ``ao_tei_deriv2`` -- the dominant cost is paid
        once, not twice (plan doc s.11.2).  :func:`pycc.hessian` adds the nuclear-repulsion second
        derivative and packs the :class:`PropertyComponents`.

        Correlation contribution to the molecular (nuclear) Hessian, indexed
        ``(A*3+a, B*3+b)`` = ``d^2 E_corr / dX_{Aa} dX_{Bb}`` -- the
        nuclear-nuclear analog of :meth:`polarizability` / :meth:`apt`, via the 2n+1
        route (both spin paths, frozen-core aware).  Differentiate the relaxed nuclear gradient
        ``dE/dX = D_rel f^(X) + Gamma <pq||rs>^(X) + I S^(X)`` w.r.t. a second nucleus ``Y``::

            H[X,Y] = d_Y D_rel f^(X) + D_rel d_Y f^(X) + d_Y Gamma <pq|rs>^(X)
                     + Gamma d_Y <pq|rs>^(X) + d_Y I S^(X) + I d_Y S^(X)

        .. math::

            \begin{aligned}
            H[X,Y] = &\partial_Y D^\mathrm{rel}_{pq} f^{(X)}_{pq} + D^\mathrm{rel}_{pq}\,\partial_Y f^{(X)}_{pq}
                + \partial_Y \Gamma_{pqrs}\,\langle pq|rs\rangle^{(X)} \\
            &+ \Gamma_{pqrs}\,\partial_Y \langle pq|rs\rangle^{(X)} + \partial_Y W_{pq}\,S^{(X)}_{pq}
                + W_{pq}\,\partial_Y S^{(X)}_{pq}
            \end{aligned}

        the nuclear-nuclear analog of the ``'2n+1-field'`` APT (:meth:`apt`).
        Only ``3N`` first-order solves -- the perturbed relaxed density ``d_Y D_rel``, the perturbed
        energy-weighted density ``d_Y I``, and ``d_Y Gamma`` all from one :class:`PerturbedResponse`
        per nucleus (:meth:`_perturbed_relaxed_density`), plus ``U^Y`` (:meth:`CPHF.full_U`).

        The mixed second derivative assembles three groups: (i) the fixed-density second integral
        skeletons contracted with the unperturbed relaxed densities, ``D~ f^(XY) + Gamma <>^(XY) +
        I S^(XY)`` (:meth:`Derivatives.nuclear_hessian_skeletons`, cached per atom pair -- all nonzero
        here, unlike the field case where only ``-mu^(X)`` survives); (ii) the orbital response
        ``2 U^Y_ai X~^(X)_ai + S^(Y)_pq I~''^(X)_pq + P^(X)_pq f^(Y)_pq``, built per ``X`` from the
        skeleton Lagrangian (:meth:`_skeleton_lagrangian`, :meth:`_augment_with_canonical_pair_rotations`);
        and (iii) the 2n+1 density response ``d_Y D~ f^(X) + d_Y Gamma <>^(X) + d_Y I S^(X)``.

        Local naming: a trailing ``X``/``x`` marks a *skeleton integral derivative* (``fX`` = ``f^(X)``,
        ``SX`` = ``S^(X)``, ``erix``/``erX`` = ``<pq|rs>^(X)`` spatial / ``<pq||rs>^(X)`` spin-orbital) --
        never a density derivative.  ``wx`` is the Fock-building companion of ``erix``: the
        spin-adapted ``L^(X) = 2<pq|rs>^(X) - <pq|sr>^(X)`` (closed-shell), or ``erix`` itself when
        the integrals are already antisymmetrized (spin-orbital).  (:meth:`apt` builds
        the same two kernels under the names ``eriX``/``eriL``.)  ``X~``/``I~''`` (``Xx``/``I2x``) and
        ``P^(x)`` (``Pf_x``) are the dependent-pair-augmented skeleton carriers from
        :meth:`_augment_with_canonical_pair_rotations`; ``D~`` = ``Drel`` is the relaxed 1-PDM.
        A trailing ``N``/``F`` on a perturbed-response array names the *perturbation* it
        responds to -- nuclear here (``dGamN`` = ``d_Y Gamma``), field in the ``'2n+1-field'`` APT
        (``dGamF``); both are the ``dGam`` of :class:`PerturbedResponse`.

        The reference block is returned alongside (computed here to share ``ao_tei_deriv2``); the
        nuclear-repulsion second derivative is added by :func:`pycc.hessian`."""
        from .cphf import Perturbation
        wfn = self.wfn
        c = self.contract
        so = wfn.orbital_basis == 'spinorbital'
        o, v, ofull = wfn.o, wfn.v, slice(0, wfn.o.stop)
        ncore = wfn.o.stop - wfn.no
        co = slice(0, ncore)                                  # frozen core (independent core<->active rot.)
        eps = np.diag(np.asarray(wfn.H.F))                    # orbital energies (dependent pairs)
        w = np.asarray(wfn.H.ERI if so else wfn.H.L)          # orbital-Hessian weight (<pq||rs> / L)
        # The orbital response uses the reference-doc form (eq:d2E-noncanon line 2, or
        # eq:d2E-canon-final when canonical) via the skeleton Lagrangian I'^(x).  The gauge follows
        # perturbed_mo_gauge (canonical for CCSD(T)); the relaxed densities Drel/dDrel already fold in
        # P_oo/P_vv (they are D~), so the canonical branch adds only the *orbital*-response
        # augmentation (Sum P^(x) f^(y), X~^(x), I~''^(x)).  To exercise canonical for a CCSD wfn
        # (valid, oo/vv-invariant) set self._gauge_override = 'canonical' on a fresh driver.
        canonical = self.perturbed_mo_gauge == 'canonical'
        cphf = self._full_occ_cphf()
        d = wfn.derivatives
        natom = d.natom
        nc = 3 * natom
        rec = self._so_orbital_response() if so else self._orbital_response()
        Drel, Gam = rec.Drel, rec.Gam
        I = self._lagrangian(Drel, Gam)
        pert = [Perturbation('nuclear', (A, ct)) for A in range(natom) for ct in range(3)]

        # first-order responses.  The large per-perturbation nmo^4 quantities live in the persistent
        # store (wfn.derivatives.store): _relaxed_response persists each dGam and _eri_cached persists
        # the per-atom eri stacks, so the assembly reads them back one atom pair at a time rather than
        # holding 3*natom-long in-RAM lists.  Only the small quantities (dDrel/dI/U and the nmo^2
        # fX/SX/Xx/I2x/Pf_x) stay resident.
        dDrel, dI, U = [], [], []
        t_stage = time.time()
        with timer("perturbed wave functions"):
            for i, p in enumerate(pert):
                r = self._relaxed_response(p)                        # one perturbed solve; persists dGam
                dDrel.append(r.dDrel)
                dI.append(r.dI)
                U.append(np.asarray(cphf.full_U(p, ncore, canonical=canonical)))
                progress("Hessian perturbed wave functions", i + 1, len(pert), t_stage,
                         "%s %s" % (atom_label(d.mol, p.comp[0]), "xyz"[p.comp[1]]))

        # per-X first skeletons.  wx = the 1-PDM two-electron kernel (L^(x) closed-shell /
        # <pq||rs>^(x) spin-orbital); erix = the 2-PDM ERI skeleton (<pq|rs>^(x) closed-shell /
        # <pq||rs>^(x) SO -- in SO the single antisymmetrized <pq||rs>^(x) serves both).  The per-X
        # skeleton Lagrangian I'^(x) gives X^(x)/I''^(x) (X~^(x)/I~''^(x)/P^(x) when canonical or
        # frozen-core).  Reading d.eri/d.so_eri here also warms the store.
        with timer("first-derivative integrals"):
            fX, SX, Xx, I2x, Pf_x = [], [], [], [], []
            for i, p in enumerate(pert):
                A, ct = p.comp
                hx = np.asarray((d.so_core(A) if so else d.core(A))[ct])
                Sx = np.asarray((d.so_overlap(A) if so else d.overlap(A))[ct])
                if so:
                    erix = np.asarray(d.so_eri(A)[ct]); wx = erix          # <pq||rs>^(X): both kernels
                else:
                    erix = np.asarray(d.eri(A)[ct])                        # <pq|rs>^(X) (2-PDM / Gamma)
                    wx = 2.0 * erix - erix.transpose(0, 1, 3, 2)           # L^X (1-PDM / Fock)
                fX.append(hx + c('pmqm->pq', wx[:, ofull, :, ofull]))
                SX.append(Sx)
                Ip, xov, i2 = self._skeleton_lagrangian(fX[i], Sx, wx, erix, Drel, Gam, I)
                if canonical or ncore:        # independent core<->active (FC) and/or redundant oo/vv (canon.)
                    xt, it, pf = self._augment_with_canonical_pair_rotations(Ip, xov, i2)
                    Xx.append(xt); I2x.append(it); Pf_x.append(pf)
                else:
                    Xx.append(xov); I2x.append(i2); Pf_x.append(None)
        # ---- assembly: two atom-pair sweeps, one per nmo^4 working set ----
        # The correlation Hessian is a sum of two contributions with DISJOINT nmo^4 inputs:
        #   (1) the fixed-density second skeleton  Gam*<pq||rs>^(XY) (+ Drel*f2 + I*S2), which
        #       contracts the 2nd-deriv integrals against the UNPERTURBED Drel/Gam/I; and
        #   (2) the density response  dGam^(Y)*<pq||rs>^(X) (+ orbital response), which contracts the
        #       1st-deriv integrals (eriX) against the density derivatives (dGam).
        # The 2nd-deriv block never meets eriX/dGam in a contraction, so the assembly sweeps the atom
        # pairs TWICE and holds only one nmo^4 working set at a time (peak max(9, 6+6) instead of
        # 9+6+6).  Each pair still computes its 2nd-deriv skeletons ONCE (cache=False, so _d2int never
        # accumulates).  The skeleton scalar s = Drel.f2 + Gam.e2 + I.ov2 is symmetric in the pair
        # (d2/dA dB = d2/dB dA), shared by H[ix,iy] and its transpose; only the response half differs.
        # The per-pair dGam/eriX are read from the store one pair at a time (never a 3*natom RAM list).
        def _dGs(j):        # bare cumulant response dGam[j] (the doc form needs no U^y rotation)
            return self._relaxed_response(pert[j]).dGam

        def _resp(i, j, dGam_j, erX_i):   # 2n+1 density response + doc orbital response (line 2)
            orb = (2.0 * c('ai,ai->', U[j][v, ofull], Xx[i][v, ofull])  # 2 U^y_ai X(~)^(x)_ai
                   + c('pq,pq->', SX[j], I2x[i]))                       # S^(y)_pq I(~)''^(x)_pq
            if Pf_x[i] is not None:
                orb = orb + c('pq,pq->', Pf_x[i], fX[j])               # + Sum P^(x)_pq f^(y)_pq
            return (c('pq,pq->', dDrel[j], fX[i]) + c('pqrs,pqrs->', dGam_j, erX_i)
                    + c('pq,pq->', dI[j], SX[i]) + orb)

        def _skel_scalar_from(core2, ov2, e2):     # fixed-density second-skeleton scalar s from one
            L2 = e2 if so else 2.0 * e2 - e2.swapaxes(2, 3)             # (cx, cy) block triple
            f2 = core2 + c('pmqm->pq', L2[:, ofull, :, ofull])          # f^(XY)
            return c('pq,pq->', Drel, f2) + c('pqrs,pqrs->', Gam, e2) + c('pq,pq->', I, ov2)

        H = np.zeros((nc, nc))
        route = getattr(self, '_skel_eri_route', 'aod')   # default; 'mo'/'ao' are opt-outs
        # Reference (SCF) electronic Hessian.  For the 'aod' route we accumulate its ao-DEPENDENT
        # skeleton into Href here in Pass 1, contracting the SAME ao_tei_deriv2 blocks this correlation
        # assembly already computes -- so a correlated Hessian generates the (by far most expensive)
        # 2nd-derivative TEIs ONCE, not once here and again in the reference HFwfn.  The ao-INDEPENDENT
        # reference CPHF response is added after the assembly by delegating to HFwfn._hessian_response
        # (no ao_tei_deriv2).  The 'mo'/'ao' opt-out routes do not fold Href; they fall back to the
        # reference HFwfn's own (unshared) electronic Hessian.  (Href skeleton == HFwfn._hessian_skeleton
        # by construction: same pair loop, same shared ao, same 2 D D - D D contraction.)
        Href = np.zeros((nc, nc))

        # Spill the whole-run nmo^4 tensors the two passes never read (baseline MO ERIs H.ERI/H.L and
        # the raw CISD 2-PDM) to disk for the assembly; only Gam (pass 1) + derivative tensors stay.
        with self._offload_assembly_idle_tensors():
            # Pass 1 -- fixed-density second skeleton  s = Drel*f^(XY) + Gam*<pq||rs>^(XY) + I*S^(XY),
            # contracted against the UNPERTURBED Drel/Gam/I (the 1st-deriv ERIs and density
            # derivatives are not touched here).  Two routes for the 2nd-deriv ERI:
            #   'mo' (opt-out): all 9 MO blocks resident per pair (9*nmo^4 spatial / 9*16*nmo^4 SO).
            #   'ao' (default): hold the 9 *spatial* AO blocks and transform ONE Cartesian pair at a
            #       time to the MO integral (eri2_mo_component / so_eri2_mo_component), so only 1 MO
            #       block is live -- floor ~= 9 AO + 1 MO + Gam instead of 9 AO + 9 MO + Gam (the SO
            #       saving is ~10x larger, its MO block being 16*nmo^4).  Same spatial AO source for
            #       both bases; each per-component block is bit-identical to the 'mo' route.
            if route == 'aod':
                # AO-density route (plan doc s.10): fold the 2-e part of Drel*f^(XY) into Gam_eff,
                # back-transform it to the spatial AO basis ONCE, and contract the raw ao_tei_deriv2
                # directly -- no per-pair MO transform and no f^(XY) build.  Floor ~= 9 AO + 1
                # Gam_eff^AO.  Both spin paths share the spin-free spatial ao_tei_deriv2: the SO
                # builder folds the spin blocks and the ket antisymmetrization into the back-transform
                # (_effective_2pdm_ao_so), so its Gam_eff^AO is the same nao^4 spatial tensor (the SO
                # saving over the per-component 'ao' route is ~16x, its MO block being 16*nmo^4).  The
                # one-electron remainder (Drel*h^(XY) + I*S^(XY)) stays in the cheap MO/SO OEI blocks.
                # The SCF reference skeleton (2 D D - D D of the same ao block) rides along here so the
                # ao_tei_deriv2 blocks feed BOTH densities (plan doc s.11.2).
                eps_o_ref = eps[ofull]                                   # SCF orbital energies (occ)
                if so:
                    GeffAO = self._effective_2pdm_ao_so(Drel, Gam)        # 1*nao^4, built once
                    core2, overlap2 = d.so_core2, d.so_overlap2
                    Ca = np.asarray(wfn.H.Ca); Cb = np.asarray(wfn.H.Cb)  # C1-flattened alpha/beta MOs
                    na, nb = wfn.ref.nalpha(), wfn.ref.nbeta()
                    Da_ref = Ca[:, :na] @ Ca[:, :na].T                    # alpha occupied AO density
                    Db_ref = Cb[:, :nb] @ Cb[:, :nb].T                    # beta occupied AO density
                    Dt_ref = Da_ref + Db_ref                              # total AO density
                else:
                    GeffAO = self._effective_2pdm_ao(Drel, Gam)           # 1*nao^4, built once
                    core2, overlap2 = d.core2, d.overlap2
                    Co_ref = np.asarray(wfn.C)[:, ofull]                  # occupied MO coefficients
                    Dref = Co_ref @ Co_ref.T                              # occupied AO density
                with timer("second-derivative integral terms"):
                    t_stage = time.time()
                    n_pair = natom * (natom + 1) // 2
                    for a1 in range(natom):
                        for a2 in range(a1, natom):
                            core2s = [np.asarray(m) for m in core2(a1, a2)]      # 9 OEI (nmo^2 / nso^2)
                            ov2s = [np.asarray(m) for m in overlap2(a1, a2)]
                            ao_eri = d.ao_eri2(a1, a2)                            # 9 spatial AO, held
                            with timer("skeleton contractions"):
                                for cx in range(3):
                                    for cy in range(3):
                                        comp = cx * 3 + cy
                                        aoc = ao_eri[comp]
                                        s = (c('mnls,mnls->', GeffAO, aoc)
                                             + c('pq,pq->', Drel, core2s[comp]) + c('pq,pq->', I, ov2s[comp]))
                                        # SCF reference skeleton (shared ao): Coulomb - exchange + OEI traces
                                        h2oo = core2s[comp][ofull, ofull]; S2oo = ov2s[comp][ofull, ofull]
                                        if so:
                                            two_e_ref = 0.5 * (c('mn,ls,mnls->', Dt_ref, Dt_ref, aoc)
                                                               - c('ms,nl,mnls->', Da_ref, Da_ref, aoc)
                                                               - c('ms,nl,mnls->', Db_ref, Db_ref, aoc))
                                            s_ref = (c('ii->', h2oo) + two_e_ref
                                                     - c('i,ii->', eps_o_ref, S2oo))
                                        else:
                                            two_e_ref = (2.0 * c('mn,ls,mnls->', Dref, Dref, aoc)
                                                         - c('ml,ns,mnls->', Dref, Dref, aoc))
                                            s_ref = (2.0 * np.trace(h2oo) + two_e_ref
                                                     - 2.0 * c('i,ii->', eps_o_ref, S2oo))
                                        ix, iy = a1 * 3 + cx, a2 * 3 + cy
                                        H[ix, iy] += s; Href[ix, iy] += s_ref
                                        if a1 != a2:                         # s, s_ref symmetric in the pair
                                            H[iy, ix] += s; Href[iy, ix] += s_ref
                            with timer("release second-derivative block"):
                                del core2s, ov2s, ao_eri
                            progress("Hessian second-derivative integrals",
                                     a1 * natom - a1 * (a1 - 1) // 2 + (a2 - a1) + 1, n_pair,
                                     t_stage, "%s-%s" % (atom_label(d.mol, a1),
                                                         atom_label(d.mol, a2)))
                del GeffAO
            else:
                for a1 in range(natom):
                    for a2 in range(a1, natom):
                        if route == 'mo':
                            blk = d.nuclear_hessian_skeletons(a1, a2, cache=False)
                            core2s, ov2s, ao_eri = blk['core'], blk['overlap'], None
                            eri2s = blk['eri']
                        else:
                            if so:
                                core2s = [np.asarray(m) for m in d.so_core2(a1, a2)]     # 9 SO OEI (small)
                                ov2s = [np.asarray(m) for m in d.so_overlap2(a1, a2)]
                            else:
                                core2s = [np.asarray(m) for m in d.core2(a1, a2)]        # 9 MO OEI (nmo^2)
                                ov2s = [np.asarray(m) for m in d.overlap2(a1, a2)]
                            ao_eri = d.ao_eri2(a1, a2)                                    # 9 spatial AO, held
                            eri2s = None
                        for cx in range(3):
                            for cy in range(3):
                                comp = cx * 3 + cy
                                if route == 'mo':
                                    e2 = eri2s[comp]
                                elif so:
                                    e2 = d.so_eri2_mo_component(ao_eri[comp])   # 1 SO block at a time
                                else:
                                    e2 = d.eri2_mo_component(ao_eri[comp])      # 1 MO block at a time
                                s = _skel_scalar_from(core2s[comp], ov2s[comp], e2)
                                if route != 'mo':
                                    del e2
                                ix, iy = a1 * 3 + cx, a2 * 3 + cy
                                H[ix, iy] += s
                                if a1 != a2:
                                    H[iy, ix] += s                         # s is symmetric in the pair
                        del core2s, ov2s, ao_eri, eri2s                    # free before pass 2 loads eriX/dGam

            # Pass 2 -- density response  dGam^(Y)*<pq||rs>^(X) + orbital response.  This term has NO
            # 2nd-deriv integrals, hence no atom-pair structure: it is a full nc x nc coordinate
            # contraction.  eriX^(X) is a per-atom 3-stack (read once per atom from the store);
            # dGam^(Y) is one nmo^4 per nuclear coordinate, streamed back from the store one
            # coordinate at a time (both were persisted by the setup loops -- neither is re-solved or
            # recomputed here).  Peak residency is one atom's 3-stack eriX + one dGam = 4*nmo^4, so
            # pass 1's 9*nmo^4 2nd-deriv block is the binding peak, not this pass.  Each H element
            # gets the same response addition as the fused loop, so H is unchanged.
            t_stage = time.time()
            with timer("density-response terms"):
                for a in range(natom):
                    erX = [np.asarray(m) for m in (d.so_eri(a) if so else d.eri(a))]  # 3*nmo^4 over iy
                    for iy in range(nc):
                        dGam_y = _dGs(iy)                               # 1*nmo^4, streamed from store
                        for cx in range(3):
                            ix = a * 3 + cx
                            with timer("response contractions"):
                                H[ix, iy] += _resp(ix, iy, dGam_y, erX[cx])
                    del erX
                    progress("Hessian density response", a + 1, natom, t_stage,
                             atom_label(d.mol, a))

        # Reference (SCF) electronic block.  'aod': the ao-dependent skeleton is in Href (built above
        # from this driver's shared ao_tei_deriv2); add the ao-independent CPHF response by delegation.
        # 'mo'/'ao' opt-outs did not fold Href -> use the reference HFwfn's own electronic Hessian.
        hf = self._reference_hf()
        reference = (Href + np.asarray(hf._hessian_response())) if route == 'aod' \
            else np.asarray(hf._hessian_electronic())
        return reference, H

    @staticmethod
    def _dependent_pairs(Iblock, eps_block, thresh=1e-8):
        r"""Canonical dependent-pair rotation for a square occ-occ or virt-virt Lagrangian
        block ``Iblock`` and its orbital energies ``eps_block``::

            P_mn = (I'_mn - I'_nm) / (eps_m - eps_n)

        .. math::

            P_{mn} = \frac{I'_{mn} - I'_{nm}}{\epsilon_m - \epsilon_n}

        Gated on the MO-energy **gap** (``|eps_m - eps_n| <= thresh`` -> 0), which skips the diagonal
        (``m=n``) and any near-degenerate pair; the divide is taken for every other pair.  ``P`` is
        symmetric (numerator and denominator both antisymmetric).

        The gap gate (not a numerator gate) is required for consistency with the derivative
        :meth:`_perturbed_dependent_pairs`, which gates the same way.  Gating instead on a small
        *numerator* would hard-zero small-but-nonzero rotations whose derivative ``dP = dnum/gap`` is
        nonzero, leaving ``P`` and ``dP`` inconsistent; that makes the (T) density non-smooth in
        geometry/field and is a genuine gradient/Hessian error at low-symmetry (C1) references, where
        such small-nonzero numerators occur (a numerator gate is only harmless at symmetric geometries,
        where the affected pairs are symmetry-zero).  A near-degeneracy (small gap) is the one place the
        divide is ill-conditioned; there the numerator vanishes by the same symmetry, so gating it to
        zero is the correct regularization.

        Supplies the *redundant* active oo/vv canonical rotations of the canonical perturbed-MO gauge
        (:attr:`perturbed_mo_gauge`; used by :meth:`_orbital_response` / :meth:`_so_orbital_response`
        and the Hessian :meth:`_augment_with_canonical_pair_rotations`).  The same divide also fixes the
        *independent* (non-redundant)
        frozen-core core<->active-occupied rotation ``P_co``, but that off-diagonal block is built
        separately as an ungated *direct* divide (its gap is always large, so the degeneracy skip is
        unnecessary) -- see :meth:`_orbital_response`."""
        num = np.asarray(Iblock) - np.asarray(Iblock).T
        den = eps_block[:, None] - eps_block[None, :]
        P = np.zeros_like(num)
        m = np.abs(den) > thresh
        P[m] = num[m] / den[m]
        return P

    @staticmethod
    def _perturbed_dependent_pairs(dIblock, Pblock0, eps_block, dfdiag_block, thresh=1e-8):
        r"""Field derivative ``dP`` of :meth:`_dependent_pairs` (quotient rule)::

            dP_mn = (dI'_mn - dI'_nm)/(eps_m - eps_n) - P0_mn (df_mm - df_nn)/(eps_m - eps_n)

        .. math::

            \partial_x P_{mn} = \frac{\partial_x I'_{mn} - \partial_x I'_{nm}}{\epsilon_m - \epsilon_n}
                - P^{0}_{mn}\,\frac{\partial_x f_{mm} - \partial_x f_{nn}}{\epsilon_m - \epsilon_n}

        The second term is the canonical-``df``-diagonal denominator derivative, using the unperturbed
        ``Pblock0``.  Gated on ``|eps_m - eps_n| > thresh`` (diagonal + near-degenerate -> 0)."""
        dnum = np.asarray(dIblock) - np.asarray(dIblock).T
        gap = eps_block[:, None] - eps_block[None, :]
        dgap = dfdiag_block[:, None] - dfdiag_block[None, :]
        dP = np.zeros_like(dnum)
        m = np.abs(gap) > thresh
        dP[m] = (dnum[m] - np.asarray(Pblock0)[m] * dgap[m]) / gap[m]
        return dP

    # ---- shared per-perturbation orbital-response builders (used by both hessian() and
    #      apt() when they run in the reference-doc form) ----

    @timed("orbital Lagrangian")
    def _skeleton_lagrangian(self, fXx, SXx, wx, erix, Drel, Gam, I):
        r"""Skeleton-perturbed orbital Lagrangian ``I'^(x)`` for one perturbation ``x`` -- the
        integral-derivative half of :meth:`_perturbed_lagrangian`, evaluated at *fixed* (unperturbed)
        relaxed densities ``Drel``/``Gam``/``I``.  Returns the triple ``(I'^(x), X^(x), I''^(x))``
        with the occupied-virtual orbital-response driver ``X^(x)_ai = I'^(x)_ia - I'^(x)_ai`` and the
        energy-weighted rewrite ``I''^(x)`` (``I'^(x)`` with its virtual-occupied block transposed into
        the occupied-virtual position)::

            I'^(x)_pq = -1/2 [ Drel_qr f^(x)_pr + Drel_rq f^(x)_rp
                               + delta_{q in ofull} Drel_rs ( w^(x)_rpsq + w^(x)_rqsp )
                               + 4 <pr||st>^(x) Gam_qrst + I_qr S^(x)_pr + I_rq S^(x)_rp ]

        The kernels are the per-perturbation skeleton integral derivatives: ``fXx`` = ``f^(x)``,
        ``SXx`` = ``S^(x)``, ``wx`` = the 1-PDM two-electron kernel (spin-adapted ``L^(x)`` closed-shell
        / antisymmetrized ``<pq||rs>^(x)`` spin-orbital), ``erix`` = the 2-PDM ERI skeleton
        (``<pq|rs>^(x)`` closed-shell / ``<pq||rs>^(x)`` spin-orbital).  Perturbation-agnostic: the
        caller supplies the nuclear or field skeletons."""
        c = self.contract
        wfn = self.wfn
        v, ofull = wfn.v, slice(0, wfn.o.stop)
        termA = c('qr,pr->pq', Drel, fXx) + c('rq,rp->pq', Drel, fXx)
        termB = np.zeros_like(fXx)
        termB[:, ofull] = (c('rs,rpsq->pq', Drel, wx[:, :, :, ofull])
                           + c('rs,rqsp->pq', Drel, wx[:, ofull, :, :]))
        termC = 4.0 * c('prst,qrst->pq', erix, Gam)
        termD = c('qr,pr->pq', I, SXx) + c('rq,rp->pq', I, SXx)
        Ip = -0.5 * (termA + termB + termC + termD)
        I2 = Ip.copy(); I2[v, ofull] = Ip[ofull, v].T
        return Ip, Ip.T - Ip, I2

    def _augment_with_canonical_pair_rotations(self, Ip, Xov, I2):
        r"""Add the closed-form (canonical Brillouin) orbital-rotation contributions to the skeleton
        ``X^(x)``/``I''^(x)`` of :meth:`_skeleton_lagrangian`, for the rotations the CPHF
        occupied-virtual solve does *not* provide (the redundant active occ-occ/virt-virt rotations of
        the canonical gauge, and the independent frozen-core core<->active-occupied rotation).

        Two kinds of rotation enter, both fixed by the canonical condition ``d_x f_pq = 0`` and sharing
        the divide ``P^(x)_pq = (I'^(x)_pq - I'^(x)_qp)/(eps_p - eps_q)``:

        * the INDEPENDENT (non-redundant) core<->active-occupied rotation ``P^(x)_ci`` -- the energy is
          not invariant to core<->active mixing, so it is ALWAYS present when there is a frozen
          core; built here as an ungated *direct* divide (its gap is always large, so
          the degeneracy skip is unnecessary), matching the density's ``Pco``;
        * the REDUNDANT (dependent) active occ-occ / virt-virt rotations ``P^(x)_ij``/``P^(x)_ab`` --
          present only in the canonical gauge (CCSD(T)); for CCSD they vanish by invariance and the
          non-canonical ``-1/2 S`` is used instead.  Built via the gap-gated :meth:`_dependent_pairs`
          -- the *same* routine that forms the unperturbed ``P``, here fed the perturbed (skeleton)
          Lagrangian ``I'^(x)`` instead of ``I'``, so ``P`` and ``P^(x)`` share one gate on the MO-energy
          gap (see :meth:`_dependent_pairs`; a numerator gate here would key ``P^(x)`` on the
          *perturbed* numerator, inconsistent with both ``P`` and the ``dP`` of
          :meth:`_perturbed_dependent_pairs`).

        ``P^(x)`` is folded into the three orbital-response carriers (occupied pair-sums ``k,l`` run
        over the full occupied space; ``A_pqrs = w_pqrs + w_psrq`` with ``w`` the unperturbed
        orbital-Hessian weight, matching :meth:`cphf.CPHF.full_U`)::

            X~^(x)_ai = X^(x)_ai + 1/2 [ Sum_kl P^(x)_kl A_kali + Sum_de P^(x)_de A_daei ]
            I~''^(x)_ij = I''^(x)_ij - P^(x)_ij eps_j
                          - 1/2 [ Sum_kl P^(x)_kl A_kilj + Sum_de P^(x)_de A_diej ]
            I~''^(x)_ab = I''^(x)_ab - P^(x)_ab eps_b

        Returns ``(X~^(x), I~''^(x), P^(x))`` -- the augmented occupied-virtual Z-vector driver, the
        augmented energy-weighted skeleton density, and the full-MO ``P^(x)`` (the latter for the
        leading ``Sum P^(x)_pq f^(y)_pq`` term -- the dependent-pair rotation contracted with the
        *second*-perturbation skeleton Fock, which has no first-derivative counterpart)."""
        c = self.contract
        wfn = self.wfn
        so = wfn.orbital_basis == 'spinorbital'
        o, v, ofull = wfn.o, wfn.v, slice(0, wfn.o.stop)
        ncore = wfn.o.stop - wfn.no
        co = slice(0, ncore)
        eps = np.diag(np.asarray(wfn.H.F))
        w = np.asarray(wfn.H.ERI if so else wfn.H.L)          # unperturbed orbital-Hessian weight
        canonical = self.perturbed_mo_gauge == 'canonical'
        Pof = np.zeros_like(Ip[ofull, ofull])                 # P^(x) over the full occupied space
        Pvv = np.zeros_like(Ip[v, v])
        if ncore:                                             # INDEPENDENT core<->active-occ (always)
            Pof[co, o] = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            Pof[o, co] = Pof[co, o].T
        if canonical:                                         # REDUNDANT active oo/vv (CCSD(T) gauge)
            Pof[o, o] = self._dependent_pairs(Ip[o, o], eps[o])
            Pvv = self._dependent_pairs(Ip[v, v], eps[v])
        Pf = np.zeros_like(Ip); Pf[ofull, ofull] = Pof; Pf[v, v] = Pvv
        # X~^(x)_ai (eq:Xtilde): occupied pair-sum over the FULL occupied space
        Xk = (c('kl,kali->ai', Pof, w[ofull, v, ofull, ofull])
              + c('kl,kila->ai', Pof, w[ofull, ofull, ofull, v]))
        Xd = c('de,daei->ai', Pvv, w[v, v, v, ofull]) + c('de,diea->ai', Pvv, w[v, ofull, v, v])
        Xt = Xov.copy(); Xt[v, ofull] = Xov[v, ofull] + 0.5 * (Xk + Xd)
        # I~''^(x) (eq:Idouble-tilde): full occupied-occupied block
        Aoo = (c('kl,kilj->ij', Pof, w[ofull, ofull, ofull, ofull])
               + c('kl,kjli->ij', Pof, w[ofull, ofull, ofull, ofull]))
        Avv = c('de,diej->ij', Pvv, w[v, ofull, v, ofull]) + c('de,djei->ij', Pvv, w[v, ofull, v, ofull])
        It = I2.copy()
        It[ofull, ofull] = I2[ofull, ofull] - Pof * eps[ofull][None, :] - 0.5 * (Aoo + Avv)
        It[v, v] = I2[v, v] - Pvv * eps[v][None, :]
        return Xt, It, Pf
