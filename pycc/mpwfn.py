"""
mpwfn.py: Moller-Plesset perturbation-theory (MP2) wavefunction.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from .wavefunction import Wavefunction
from .utils import diag, clone

if TYPE_CHECKING:
    from ._typing import Tensor


class MPwfn(Wavefunction):
    """An MP2 wavefunction built on the shared :class:`Wavefunction` base.

    Holds the energy denominators and the first-order (MP2) doubles amplitudes,
    derived from the base's seeded MO integrals, and computes the MP2 correlation
    energy. It is also the canonical home for that denominator/amplitude code:
    ``ccwfn`` composes one of these (via :meth:`from_wavefunction`) to obtain its
    energy denominators and CC initial guess without building the Hamiltonian twice.

    Attributes
    ----------
    eps_occ, eps_vir : Tensor
        occupied / virtual orbital energies (the diagonal of the seeded Fock)
    Dijab : Tensor
        two-electron energy denominator, eps_i + eps_j - eps_a - eps_b
    t2 : Tensor
        first-order (MP2) doubles amplitudes, ERI[o,o,v,v] / Dijab (i.e. <ij|ab>
        spatial, or the antisymmetrized <ij||ab> in the spin-orbital path)
    Dia, t1 : Tensor
        (spin-orbital path only) the singles denominator and first-order singles
        t1 = f_ia / Dia, nonzero for a non-canonical (ROHF) reference
    emp2 : float
        the MP2 correlation energy (set by :meth:`compute_energy`)
    """

    def __init__(self, scf_wfn: Any, **kwargs) -> None:
        super().__init__(scf_wfn, **kwargs)
        self._build_mp2()

    @classmethod
    def from_wavefunction(cls, wfn: "Wavefunction") -> "MPwfn":
        """Build an MPwfn that reuses ``wfn``'s already-constructed base (reference,
        orbitals, seeded integrals, device manager) -- no second integral transform.
        """
        mp = cls._from_shared_base(wfn)
        mp._build_mp2()
        return mp

    def _build_mp2(self) -> None:
        """Build the energy denominators and MP2 doubles amplitudes from the seeded
        MO integrals. The ERI oovv block is staged onto the compute device with
        clone(..., device1) so the divide lands where the amplitudes live.

        Basis-agnostic: ``t2 = <ij|ab> / Dijab`` (spatial) and
        ``t2 = <ij||ab> / Dijab`` (spin-orbital) are the same expression in the
        respective ``H.ERI``, and the denominator from the Fock diagonal is too.
        """
        o = self.o
        v = self.v
        self.eps_occ = diag(self.H.F)[o]
        self.eps_vir = diag(self.H.F)[v]
        self.Dijab = (self.eps_occ.reshape(-1, 1, 1, 1) + self.eps_occ.reshape(-1, 1, 1)
                      - self.eps_vir.reshape(-1, 1) - self.eps_vir)
        self.t2 = clone(self.H.ERI[o, o, v, v], device=self.device1) / self.Dijab

        # First-order singles. For a semicanonical (e.g. ROHF) reference the occ-vir
        # Fock block f_ia is nonzero, so t1 = f_ia / Dia contributes to MP2; for a
        # canonical RHF/UHF reference f_ia = 0, so t1 vanishes. Only the spin-orbital
        # path uses these (the spatial energy is the spin-adapted t2.L form).
        if self.orbital_basis == 'spinorbital':
            self.Dia = self.eps_occ.reshape(-1, 1) - self.eps_vir
            self.t1 = clone(self.H.F[o, v], device=self.device1) / self.Dia

    def compute_energy(self) -> "Tensor":
        """Compute and return the MP2 correlation energy.

        Spatial (spin-adapted) path: ``E = t2_ijab L_ijab``. Spin-orbital path:
        ``E = f_ia t1_ia + 1/4 <ij||ab> t2_ijab`` -- no ``L`` exists, the 1/4 accounts
        for the unrestricted sum over the antisymmetrized doubles, and the singles term
        is nonzero only for a non-canonical (e.g. ROHF) reference.
        """
        o = self.o
        v = self.v
        if self.orbital_basis == 'spatial':
            self.emp2 = self.contract('ijab,ijab->', self.t2, self.H.L[o, o, v, v])
        else:
            self.emp2 = (0.25 * self.contract('ijab,ijab->', self.t2, self.H.ERI[o, o, v, v])
                         + self.contract('ia,ia->', self.H.F[o, v], self.t1))
        return self.emp2

    # ---- spin-orbital MP2 relaxed-gradient densities ----
    # The orbital-response (Z-vector) machinery follows the spin-orbital CC gradient
    # formulation (Gauss, Stanton & Bartlett, JCP 95, 2623 (1991)): the correlation
    # one- and two-particle densities feed an orbital-gradient Lagrangian I'_pq whose
    # occupied-virtual antisymmetric part drives the Z-vector, giving the relaxed
    # off-diagonal density. Spin-orbital only (the formulae apply verbatim there).

    def _so_mp2_corr_opdm(self):
        """Spin-orbital unrelaxed MP2 one-particle correlation density blocks
        ``(Doo, Dvv)``: ``Doo = -1/2 t_imef t_jmef``, ``Dvv = 1/2 t_mnbe t_mnae``. The
        1/2 is the normalization that makes the densities close the energy
        (``Tr(F Doo) + Tr(F Dvv) = -E_MP2``)."""
        c = self.contract
        t2 = self.t2
        Doo = -0.5 * c('imef,jmef->ij', t2, t2)
        Dvv = 0.5 * c('mnbe,mnae->ab', t2, t2)
        return Doo, Dvv

    def _so_mp2_cumulant(self) -> np.ndarray:
        """Spin-orbital MP2 cumulant 2-PDM ``Gamma_ijab = 1/4 t2`` in the ``oovv``/``vvoo``
        blocks -- the only blocks that contribute (determined from the MP2 energy
        Lagrangian, in which Lambda and T2 enter linearly). Built over the full MO space
        (``self.nmo``); the active ``o``/``v`` slices place the amplitudes."""
        o, v = self.o, self.v
        nmo = self.nmo
        t2 = np.asarray(self.t2)
        Gam = np.zeros((nmo, nmo, nmo, nmo))
        Gam[o, o, v, v] = 0.25 * t2
        Gam[v, v, o, o] = 0.25 * t2.transpose(2, 3, 0, 1)
        return Gam

    def _so_mp2_lagrangian(self, D, Gam) -> np.ndarray:
        """Spin-orbital generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) from a full-MO
        1-PDM ``D`` and cumulant 2-PDM ``Gamma``

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q in ofull} sum_rs D_rs (<rp||sq> + <rq||sp>)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        The 1-PDM term's column index runs over the full occupied space ``ofull`` (= core +
        active), so the frozen-core rows/columns are built (for ``nfzc=0`` this is the whole
        occupied space). Its occupied-virtual antisymmetric part ``X_ai = I'_ia - I'_ai`` is
        the Z-vector RHS; evaluated at the relaxed ``D`` it is the energy-weighted density."""
        nmo = self.nmo
        ofull = slice(0, self.o.stop)
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        termA = eps[:, None] * (D + D.T)                       # f_pp (D_pq + D_qp)
        termB = np.zeros((nmo, nmo))
        termB[:, ofull] = (self.contract('rs,rpsq->pq', D, ERI[:, :, :, ofull])
                           + self.contract('rs,rqsp->pq', D, ERI[:, ofull, :, :]))
        termC = 4.0 * self.contract('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    def _so_mp2_relaxed_densities(self):
        """Spin-orbital MP2 relaxed 1-PDM ``D_r`` (``nmo x nmo``), cumulant ``Gamma``, and
        energy-weighted density ``W``, with the orbital response over the full occupied space
        (frozen-core aware) -- the spin-orbital analog of :meth:`_mp2_relaxed_densities`.
        With the antisymmetrized ``<pq||rs>`` the frozen-core recipe applies directly (no
        spin-adaptation):

          * core <-> active-occupied: a direct divide ``P_co = (I'_ci - I'_ic)/(eps_c -
            eps_i)`` (the HF energy is invariant to occupied-occupied rotations, so the
            orbital Hessian is the orbital-energy difference);
          * occupied <-> virtual (incl. core-virtual): the Z-vector ``A z = X`` over the full
            ``ndocc x nv`` space, with ``P_co`` coupled into the RHS
            (``X_ai -= sum_jc[<aj||ic> + <ac||ij>] z_jc``).

        The Z-vector uses the full-occupied spin-orbital orbital Hessian, built here (the
        active-space CPHF is ``self.cphf``; there is no all-electron spin-orbital HFwfn to
        borrow it from, as the spatial path does). ``W = I'(D_r)``. For ``nfzc=0`` the core
        blocks are empty and this reduces to the all-electron relaxed density."""
        if self.orbital_basis != 'spinorbital':
            raise NotImplementedError(
                "The spin-orbital MP2 relaxed gradient requires the spin-orbital path.")
        if self.cphf.is_rohf:
            raise NotImplementedError(
                "The spin-orbital MP2 gradient is not implemented for ROHF references "
                "(the semicanonical response does not reproduce the restricted ROHF "
                "response). RHF and UHF are supported.")
        nmo, nfzc, nv = self.nmo, self.nfzc, self.nv
        o, v, co = self.o, self.v, self.co
        ofull = slice(0, o.stop)
        nof = o.stop
        eps = np.diag(np.asarray(self.H.F))
        ERI = np.asarray(self.H.ERI)

        Doo, Dvv = self._so_mp2_corr_opdm()
        Gam = self._so_mp2_cumulant()
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        Ip = self._so_mp2_lagrangian(D, Gam)

        if nfzc:
            Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            D[co, o] = Pco
            D[o, co] = Pco.T

        X = Ip[ofull, v] - Ip[v, ofull].T
        if nfzc:
            zjc = -Pco.T                                   # z_jc, active-occupied x core
            X = X - (self.contract('jc,ajic->ia', zjc, ERI[v, o, ofull, co])
                     + self.contract('jc,acij->ia', zjc, ERI[v, co, ofull, o]))

        # Full-occupied spin-orbital orbital Hessian A_{ia,jb} = <aj||ib> + <ab||ij>
        # + (eps_a - eps_i) delta; solve A z = X over the full ndocc x nv space.
        G = (self.contract('ajib->iajb', ERI[v, ofull, ofull, v])
             + self.contract('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof * nv, nof * nv)
        G[np.diag_indices(nof * nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
        z = np.linalg.solve(G, X.reshape(-1)).reshape(nof, nv).T
        D[v, ofull] = -z
        D[ofull, v] = -z.T

        W = self._so_mp2_lagrangian(D, Gam)
        return D, Gam, W

    def mp2_relaxed_opdm(self) -> np.ndarray:
        """Relaxed MP2 one-particle correlation density (``nmo x nmo``): the unrelaxed
        ``Doo``/``Dvv`` plus the orbital-relaxation blocks from the Z-vector solve. Both the
        spin-orbital and spin-adapted closed-shell paths are frozen-core aware
        (:meth:`_so_mp2_relaxed_densities` / :meth:`_mp2_relaxed_densities`)."""
        if self.orbital_basis == 'spinorbital':
            D, Gam, W = self._so_mp2_relaxed_densities()
        else:
            D, Gam, W, hf = self._mp2_relaxed_densities()
        return D

    def relaxed_dipole(self) -> np.ndarray:
        """MP2 correlation contribution to the electronic dipole moment (a.u.), shape
        ``(3,)`` (x, y, z).

        The relaxed (orbital-response) correlation one-particle density contracted with
        the MO dipole integrals, ``mu_a^corr = sum_pq D_pq (mu_a)_pq`` (:meth:`mp2_relaxed_opdm`,
        ``H.mu`` = ``-e r``). This is the *correlation* dipole only: the reference (SCF) dipole
        is kept separate (:meth:`HFwfn.dipole`) -- the total MP2 dipole is their sum.
        Basis-aware (the relaxed density dispatches on the orbital basis) and frozen-core aware.

        Validated against a finite field of (E_MP2 - E_SCF)."""
        D = self.mp2_relaxed_opdm()
        return np.array([self.contract('pq,pq->', D, np.asarray(self.H.mu[a]))
                         for a in range(3)])

    # ---- explicit-derivative property route (notes.pdf) ----
    # The first derivative of the correlation energy w.r.t. a perturbation, via the
    # full (CPHF-folded) derivatives of f and <pq||rs> from the shared CPHF engine,
    # contracted with the *unrelaxed* correlation densities (the orbital relaxation rides
    # inside d_x f / d_x <pq||rs>, not in a relaxed density). This is the building block of
    # the analytic MP2 polarizability (the second derivative); for an electric-field
    # perturbation the first derivative is the (negative) correlation dipole, which must
    # reproduce :meth:`relaxed_dipole`. Spin-orbital path first.

    def _corr_energy_deriv(self, pert) -> float:
        """First derivative of the MP2 correlation energy along ``pert`` (a
        :class:`~pycc.cphf.Perturbation`)::

            dE_corr/dx = sum_pq gamma_pq d_x f_pq + sum_pqrs Gamma_pqrs d_x <pq||rs>

        with the unrelaxed correlation 1-PDM ``gamma`` (``Doo``/``Dvv``) and the 2PDM
        ``Gamma`` (``oovv``/``vvoo``), and the explicit derivatives from
        :meth:`CPHF.perturbed_fock` / :meth:`CPHF.perturbed_eri`. Spin-orbital path only."""
        if self.orbital_basis != 'spinorbital':
            raise NotImplementedError(
                "the explicit-derivative correlation gradient is implemented for the "
                "spin-orbital path only so far.")
        nmo, o, v = self.nmo, self.o, self.v
        Doo, Dvv = self._so_mp2_corr_opdm()
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = np.asarray(Doo)
        gamma[v, v] = np.asarray(Dvv)
        Gam = self._so_mp2_cumulant()
        df = self.cphf.perturbed_fock(pert)
        deri = self.cphf.perturbed_eri(pert)
        return (self.contract('pq,pq->', gamma, df)
                + self.contract('pqrs,pqrs->', Gam, deri))

    def _corr_dipole_explicit(self) -> np.ndarray:
        """MP2 correlation electronic dipole (a.u.), shape ``(3,)``, via the
        explicit-derivative route -- ``mu_a = -dE_corr/dF_a`` from :meth:`_corr_energy_deriv`
        for the three electric-field perturbations. An independent cross-check of
        :meth:`relaxed_dipole` (same number, computed without the relaxed density / Z-vector)
        and the validation that the CPHF perturbed-derivative engine is correct."""
        from .cphf import Perturbation
        return np.array([-self._corr_energy_deriv(Perturbation('field', a))
                         for a in range(3)])

    def _corr_gradient_explicit(self) -> np.ndarray:
        """MP2 correlation contribution to the nuclear gradient (a.u.), shape
        ``(natom, 3)``, via the explicit-derivative route: ``dE_corr/dX_Ac`` from
        :meth:`_corr_energy_deriv` for each nuclear perturbation. The "simple but
        inefficient" form -- one nuclear CPHF solve per perturbation (``3*natom``) and a full
        perturbed-integral build, instead of the single Z-vector of :meth:`gradient`. An
        independent cross-check of the relaxed-density correlation gradient (:meth:`gradient`,
        which is now correlation-only). Spin-orbital, all-electron path."""
        from .cphf import Perturbation
        natom = self.derivatives.natom
        g = np.zeros((natom, 3))
        for atom in range(natom):
            for c in range(3):
                g[atom, c] = self._corr_energy_deriv(Perturbation('nuclear', (atom, c)))
        return g

    # ---- spin-adapted (closed-shell RHF) MP2 relaxed-gradient densities ----
    # The closed-shell analogue of the spin-orbital densities: the spin sum is carried by
    # the spin-adapted lambda ``l2 = 2(2 t2 - t2.swap)`` and the spin-adapted ``L`` (= H.L,
    # 2<pq|rs>-<pq|sr>) in the two-electron 1-PDM term, with the cumulant ``Gamma = 2 t2 -
    # t2.swap``. Validated against the spin-orbital path (same physics) and Psi4.

    def _mp2_corr_opdm(self):
        """Spin-adapted unrelaxed MP2 one-particle correlation density blocks ``(Doo,
        Dvv)``: ``Doo = -t_imef l2_jmef``, ``Dvv = t_mnbe l2_mnae``, with the spin-adapted
        lambda ``l2 = 2(2 t2 - t2.swap)`` (the factor-2 carries the closed-shell spin sum)."""
        c = self.contract
        t2 = self.t2
        l2 = 2.0 * (2.0 * t2 - t2.swapaxes(2, 3))
        Doo = -c('imef,jmef->ij', t2, l2)
        Dvv = c('mnbe,mnae->ab', t2, l2)
        return Doo, Dvv

    def _mp2_cumulant(self) -> np.ndarray:
        """Spin-adapted MP2 cumulant 2-PDM ``Gamma_ijab = 2 t2 - t2.swap`` (``oovv``/``vvoo``).
        Built over the full MO space (``self.nmo``); the active ``o``/``v`` slices place the
        amplitudes, leaving any frozen-core rows/columns zero."""
        o, v = self.o, self.v
        nmo = self.nmo
        t2 = np.asarray(self.t2)
        u = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
        Gam = np.zeros((nmo, nmo, nmo, nmo))
        Gam[o, o, v, v] = u
        Gam[v, v, o, o] = u.transpose(2, 3, 0, 1)
        return Gam

    def _mp2_lagrangian(self, D, Gam) -> np.ndarray:
        """Spin-adapted generalized-Fock Lagrangian ``I'_pq`` (``nmo x nmo``) from a
        full-MO 1-PDM ``D`` and cumulant 2-PDM ``Gamma`` -- the closed-shell analogue of
        :meth:`_so_mp2_lagrangian`, with the spin-adapted ``L`` (= H.L) in the two-electron
        1-PDM term and ``<pr|st>`` (= H.ERI) with ``Gamma`` in the 2-PDM term. The 1-PDM
        term's column index runs over the full occupied space ``ofull`` (= core + active),
        so the frozen-core rows/columns are built (for ``nfzc=0`` this is the whole occupied
        space). Used both for the orbital-gradient Lagrangian (from the unrelaxed ``D``) and
        for the energy-weighted density ``W = I'(D_relaxed)``."""
        nmo = self.nmo
        ofull = slice(0, self.nfzc + self.no)
        ERI = np.asarray(self.H.ERI)
        L = np.asarray(self.H.L)
        eps = np.diag(np.asarray(self.H.F))
        termA = eps[:, None] * (D + D.T)
        termB = np.zeros((nmo, nmo))
        termB[:, ofull] = (self.contract('rs,rpsq->pq', D, L[:, :, :, ofull])
                           + self.contract('rs,rqsp->pq', D, L[:, ofull, :, :]))
        termC = 4.0 * self.contract('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    def _mp2_relaxed_densities(self):
        """Spatial MP2 relaxed 1-PDM ``D_r`` (``nmo x nmo``), cumulant ``Gamma``, and
        energy-weighted density ``W``, with the orbital response over the full occupied
        space (frozen-core aware). Also returns the all-electron ``HFwfn`` whose CPHF
        carries the response (reused for the SCF gradient).

        The unrelaxed correlation density (``Doo``/``Dvv``) lives on the active blocks; the
        orbital response spans the full occupied space:

          * core <-> active-occupied: non-redundant for frozen-core MP2 (these rotations
            move the frozen/active partition) but the HF energy is invariant to any
            occupied-occupied rotation, so their orbital Hessian is just the orbital-energy
            difference -- a direct divide ``P_co = (I'_ci - I'_ic)/(eps_c - eps_i)``, not a
            CPHF solve;
          * occupied <-> virtual (incl. core-virtual): the Z-vector ``A z = X`` over the
            full ``ndocc x nv`` space, with ``P_co`` coupled into the right-hand side, solved
            with the all-electron ``HFwfn`` CPHF (whose occupied space is the full ``ndocc``).

        ``W = I'(D_r)`` is the Lagrangian evaluated at the relaxed density; its core-active
        contribution reproduces the ``z_kc`` energy-weighted-density term. For ``nfzc=0`` the
        core blocks are empty and this reduces to the all-electron relaxed density."""
        from .hfwfn import HFwfn
        nmo, nfzc, no = self.nmo, self.nfzc, self.no
        o, v = self.o, self.v
        co = slice(0, nfzc)
        ofull = slice(0, nfzc + no)
        eps = np.diag(np.asarray(self.H.F))
        L = np.asarray(self.H.L)

        Doo, Dvv = self._mp2_corr_opdm()
        Gam = self._mp2_cumulant()
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        Ip = self._mp2_lagrangian(D, Gam)

        if nfzc:
            Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
            D[co, o] = Pco
            D[o, co] = Pco.T

        hf = HFwfn(self.ref)
        X = Ip[ofull, v] - Ip[v, ofull].T
        if nfzc:
            zjc = -Pco.T                                   # z_jc, active-occupied x core
            X = X - (self.contract('jc,ajic->ia', zjc, L[v, o, ofull, co])
                     + self.contract('jc,acij->ia', zjc, L[v, co, ofull, o]))
        z = hf.cphf.solve(X).T
        D[v, ofull] = -z
        D[ofull, v] = -z.T

        W = self._mp2_lagrangian(D, Gam)
        return D, Gam, W, hf

    # ---- MP2 nuclear gradient ----

    def gradient(self) -> np.ndarray:
        """MP2 **correlation** contribution to the analytic nuclear energy gradient (a.u.),
        shape (natom, 3)

            dE_corr/dX = sum_pq D_pq f^X_pq + sum_pqrs Gamma_pqrs <pq|rs>^X
                         + sum_pq I_pq S^X_pq

        with the relaxed 1-PDM ``D``, cumulant 2-PDM ``Gamma``, and energy-weighted density
        ``W`` (Gauss/Stanton/Bartlett). The **reference (SCF) gradient is kept separate** (as
        for :meth:`relaxed_dipole`): the total MP2 gradient is ``HFwfn(ref).gradient()`` plus
        this. This is the spin-adapted (closed-shell RHF) path, frozen-core aware: the skeleton
        derivative integrals come from ``self.derivatives`` in the full spatial MO basis
        (chemist ``(pq|rs)^X``, converted to physicist here), ``f^X = h^X + sum_m L[p,m,q,m]^X``
        is the closed-shell skeleton Fock derivative with ``m`` over the full occupied space
        (core + active), and the relaxed density/orbital response span the full occupied space
        (:meth:`_mp2_relaxed_densities`). The spin-orbital path is :meth:`_so_gradient`."""
        if self.orbital_basis == 'spinorbital':
            return self._so_gradient()
        ofull = slice(0, self.nfzc + self.no)
        D, Gam, W, _ = self._mp2_relaxed_densities()

        d = self.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.core(atom)
            Sx = d.overlap(atom)
            ERIx = d.eri(atom)                        # chemist (pq|rs)^X
            for c in range(3):
                phys = ERIx[c].transpose(0, 2, 1, 3)  # -> physicist <pq|rs>^X
                Lx = 2.0 * phys - phys.transpose(0, 1, 3, 2)
                fx = hx[c] + self.contract('pmqm->pq', Lx[:, ofull, :, ofull])  # Fock deriv (full occ)
                grad[atom, c] = (self.contract('pq,pq->', D, fx)
                                 + self.contract('pqrs,pqrs->', Gam, phys)
                                 + self.contract('pq,pq->', W, Sx[c]))
        return grad

    def _so_gradient(self) -> np.ndarray:
        """Spin-orbital MP2 **correlation** gradient (the reference gradient kept separate;
        see :meth:`gradient`): the antisymmetrized ``<pq||rs>^X`` from the spin-orbital
        ``self.derivatives.so_*`` (semicanonical gauge), ``f^X = h^X + sum_m <pm||qm>^X`` with
        ``m`` over the full occupied space (frozen-core aware; :meth:`_so_mp2_relaxed_densities`)."""
        ofull = slice(0, self.o.stop)
        D, Gam, W = self._so_mp2_relaxed_densities()

        d = self.derivatives
        grad = np.zeros((d.natom, 3))
        for atom in range(d.natom):
            hx = d.so_core(atom)
            Sx = d.so_overlap(atom)
            ERIx = d.so_eri(atom)
            for c in range(3):
                fx = hx[c] + self.contract('pmqm->pq', ERIx[c][:, ofull, :, ofull])  # Fock deriv (full occ)
                grad[atom, c] = (self.contract('pq,pq->', D, fx)
                                 + self.contract('pqrs,pqrs->', Gam, ERIx[c])
                                 + self.contract('pq,pq->', W, Sx[c]))
        return grad
