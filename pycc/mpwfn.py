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

    # ---- unrelaxed and relaxed correlation densities ----
    # Orbital-response (Z-vector) machinery following the spin-orbital CC gradient formulation
    # (Gauss, Stanton & Bartlett, JCP 95, 2623 (1991)): the correlation one- and two-particle
    # densities feed an orbital-gradient Lagrangian I'_pq whose occupied-virtual antisymmetric
    # part drives the Z-vector, giving the relaxed off-diagonal density. The spin-orbital (_so_)
    # and spin-adapted (closed-shell, unlabeled) paths are paired; the spatial spin sum rides in
    # l2 = 2(2 t2 - t2.swap) and the spin-adapted L (= H.L). W = I'(D_relaxed) is the
    # energy-weighted density.

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

    def _so_mp2_tpdm(self) -> np.ndarray:
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
        D, Gam, _, _, _, _ = self._so_zvector()
        W = self._so_mp2_lagrangian(D, Gam)
        return D, Gam, W

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

    def _mp2_tpdm(self) -> np.ndarray:
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
        D, Gam, _, _, hf, _ = self._zvector()
        W = self._mp2_lagrangian(D, Gam)
        return D, Gam, W, hf

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

    # ---- correlation dipole ----

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

    # ---- first derivatives: explicit-derivative route (notes.pdf) and the nuclear gradient ----
    # The correlation-energy first derivative via the full (CPHF-folded) derivatives of f and
    # <pq||rs> (shared CPHF engine) contracted with the *unrelaxed* densities -- the orbital
    # relaxation rides inside d_x f / d_x <pq||rs>, not in a relaxed density. For an electric
    # field the (negative) correlation dipole reproduces relaxed_dipole. The relaxed-density
    # gradient (sum D f^X + sum Gamma <pq||rs>^X + sum I S^X) is the equivalent Z-vector form.
    # Spatial closed-shell default; spin-orbital via the _so_ route.

    def _full_occ_cphf(self):
        """A CPHF over the **full** occupied space (frozen core + active) in this wavefunction's
        own MO ordering (cached). The explicit-derivative engine needs the orbital response over
        the full occupied space (core<->active and core-virtual), which ``self.cphf`` (active
        only) can't supply; building it here -- rather than borrowing an all-electron ``HFwfn``
        -- keeps the spin-orbital ordering consistent with the densities (the all-electron SO
        ``HFwfn`` orders the spins differently). For ``nfzc=0`` it coincides with ``self.cphf``."""
        if getattr(self, '_focphf', None) is None:
            from .cphf import CPHF
            self._focphf = CPHF(self, full_occ=True)
        return self._focphf

    def _corr_energy_deriv(self, pert) -> float:
        """First derivative of the MP2 correlation energy along ``pert`` (a
        :class:`~pycc.cphf.Perturbation`)::

            dE_corr/dx = sum_pq gamma_pq d_x f_pq + sum_pqrs Gamma_pqrs d_x <pq|rs>

        with the unrelaxed correlation 1-PDM ``gamma`` (``Doo``/``Dvv``) and the 2PDM
        ``Gamma`` (``oovv``/``vvoo``), and the explicit derivatives from
        :meth:`CPHF.perturbed_fock` / :meth:`CPHF.perturbed_eri`. Spin-adapted (closed-shell
        RHF) here; the spin-orbital path is :meth:`_so_corr_energy_deriv`.

        Frozen-core aware with no rearrangement: the densities stay in the active space
        (``Doo``/``Dvv``/``Gamma`` placed at the active ``o``/``v`` blocks of full-MO arrays),
        while the perturbed derivatives are built over the **full occupied space**
        (:meth:`_full_occ_cphf`) with the core<->active block of ``U`` from ``ncore``. The
        contraction against the full ``f``/``<pq|rs>`` derivatives carries the core response."""
        if self.orbital_basis == 'spinorbital':
            return self._so_corr_energy_deriv(pert)
        nmo, o, v = self.nmo, self.o, self.v
        Doo, Dvv = self._mp2_corr_opdm()
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = np.asarray(Doo)
        gamma[v, v] = np.asarray(Dvv)
        Gam = self._mp2_tpdm()
        ncore = o.stop - self.no                    # frozen-core orbitals at the front of o
        cphf = self._full_occ_cphf()
        df = cphf.perturbed_fock(pert, ncore)
        deri = cphf.perturbed_eri(pert, ncore)
        return (self.contract('pq,pq->', gamma, df)
                + self.contract('pqrs,pqrs->', Gam, deri))

    def _so_corr_energy_deriv(self, pert) -> float:
        """Spin-orbital first derivative of the MP2 correlation energy along ``pert`` --
        the antisymmetrized-integral form of :meth:`_corr_energy_deriv` (unrelaxed
        ``gamma`` and 2PDM ``Gamma``, with the spin-orbital perturbed derivatives over the
        full occupied space; ``ncore`` = the frozen spin orbitals at the front of ``o``)."""
        nmo, o, v = self.nmo, self.o, self.v
        Doo, Dvv = self._so_mp2_corr_opdm()
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = np.asarray(Doo)
        gamma[v, v] = np.asarray(Dvv)
        Gam = self._so_mp2_tpdm()
        ncore = o.stop - self.no                    # frozen-core spin orbitals at the front of o
        cphf = self._full_occ_cphf()
        df = cphf.perturbed_fock(pert, ncore)
        deri = cphf.perturbed_eri(pert, ncore)
        return (self.contract('pq,pq->', gamma, df)
                + self.contract('pqrs,pqrs->', Gam, deri))

    def _corr_dipole_explicit(self) -> np.ndarray:
        """MP2 correlation electronic dipole (a.u.), shape ``(3,)``, via the
        explicit-derivative route -- ``mu_a = -dE_corr/dF_a`` from :meth:`_corr_energy_deriv`
        for the three electric-field perturbations. An independent cross-check of
        :meth:`relaxed_dipole` (same number, computed without the relaxed density / Z-vector)
        and the validation that the CPHF perturbed-derivative engine is correct. Basis-aware,
        all-electron."""
        from .cphf import Perturbation
        return np.array([-self._corr_energy_deriv(Perturbation('field', a))
                         for a in range(3)])

    def _corr_gradient_explicit(self) -> np.ndarray:
        """MP2 correlation contribution to the nuclear gradient (a.u.), shape
        ``(natom, 3)``, via the explicit-derivative route: ``dE_corr/dX_Ac`` from
        :meth:`_corr_energy_deriv` for each nuclear perturbation. The "simple but
        inefficient" form -- one nuclear CPHF solve per perturbation (``3*natom``) and a full
        perturbed-integral build, instead of the single Z-vector of :meth:`gradient`. An
        independent cross-check of the relaxed-density correlation gradient (:meth:`gradient`).
        Basis-aware, all-electron."""
        from .cphf import Perturbation
        natom = self.derivatives.natom
        g = np.zeros((natom, 3))
        for atom in range(natom):
            for c in range(3):
                g[atom, c] = self._corr_energy_deriv(Perturbation('nuclear', (atom, c)))
        return g

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

    # ---- perturbed amplitudes / densities (for the second derivatives) ----

    def _perturbed_t2(self, pert) -> np.ndarray:
        """First-order response of the MP2 doubles amplitudes to ``pert`` -- closed form,
        since the MP2 amplitudes are non-iterative::

            t^x_ijab = [ d_x<ij||ab> + sum_c (d_x f_ac t_ijcb + d_x f_bc t_ijac)
                         - sum_k (d_x f_ik t_kjab + d_x f_jk t_ikab) ] / D_ijab

        from the active ``oovv`` block of :meth:`CPHF.perturbed_eri` and the active ``oo``/``vv``
        blocks of :meth:`CPHF.perturbed_fock` (the diagonal of ``d_x f`` recovers ``-t d_x D``;
        the off-diagonal ``oo``/``vv`` blocks are the non-canonical coupling). Basis-agnostic --
        the integral convention rides in ``H.ERI``. Built over the full occupied space
        (frozen-core aware) but indexed on the active amplitudes."""
        o, v = self.o, self.v
        ncore = o.stop - self.no
        cphf = self._full_occ_cphf()
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dfoo, dfvv = df[o, o], df[v, v]
        t2 = np.asarray(self.t2)
        c = self.contract
        num = (deri[o, o, v, v]
               + c('ac,ijcb->ijab', dfvv, t2) + c('bc,ijac->ijab', dfvv, t2)
               - c('ik,kjab->ijab', dfoo, t2) - c('jk,ikab->ijab', dfoo, t2))
        return num / np.asarray(self.Dijab)

    def _perturbed_densities(self, pert):
        """First-order response of the unrelaxed correlation densities to ``pert``: returns
        ``(d_a gamma, d_a Gamma)`` (full-MO arrays), from the perturbed amplitudes
        :meth:`_perturbed_t2` by the product rule -- the same density expressions as the
        unrelaxed densities (:meth:`_so_mp2_corr_opdm`/`_so_mp2_tpdm` and the spatial
        siblings), differentiated. Basis-aware."""
        o, v, nmo = self.o, self.v, self.nmo
        t2 = np.asarray(self.t2)
        ta = self._perturbed_t2(pert)
        c = self.contract
        dgam = np.zeros((nmo, nmo))
        dGam = np.zeros((nmo, nmo, nmo, nmo))
        if self.orbital_basis == 'spinorbital':
            dgam[o, o] = -0.5 * (c('imef,jmef->ij', ta, t2) + c('imef,jmef->ij', t2, ta))
            dgam[v, v] = 0.5 * (c('mnbe,mnae->ab', ta, t2) + c('mnbe,mnae->ab', t2, ta))
            u = 0.25 * ta
        else:
            l2 = 2.0 * (2.0 * t2 - t2.swapaxes(2, 3))
            la = 2.0 * (2.0 * ta - ta.swapaxes(2, 3))
            dgam[o, o] = -(c('imef,jmef->ij', ta, l2) + c('imef,jmef->ij', t2, la))
            dgam[v, v] = (c('mnbe,mnae->ab', ta, l2) + c('mnbe,mnae->ab', t2, la))
            u = 2.0 * ta - ta.swapaxes(2, 3)
        dGam[o, o, v, v] = u
        dGam[v, v, o, o] = u.transpose(2, 3, 0, 1)
        return dgam, dGam

    # ---- 2n+1 route: perturbed relaxed density ----
    # The relaxed-density gradient is already the 2n+1 first derivative; its second derivative
    # (polarizability/APT/Hessian) needs the *response* of the relaxed density, whose new piece
    # is the perturbed Z-vector z^x (same orbital Hessian as the gradient, perturbed RHS). The
    # spin-orbital path builds G inline; the spatial analogue carries the spin-adapted L and
    # solves through the all-electron HFwfn CPHF. See mp2_2n1_perturbed.tex / DERIVATIVES_PLAN.

    def _so_zvector(self):
        """Unperturbed spin-orbital Z-vector data (cached, frozen-core aware): the relaxed
        1-PDM ``D_rel``, cumulant ``Gam``, unrelaxed ``D_u``, the Z-vector amplitudes ``z``
        (indexed ``(I,a)`` over the full occupied space), the orbital Hessian ``G``
        (``nof*nv`` square), and the core-active block ``P_co`` (``None`` for ``nfzc=0``) --
        shared by the relaxed density (:meth:`_so_mp2_relaxed_densities` delegates here) and
        its perturbed response."""
        if getattr(self, '_so_zvec', None) is None:
            nmo, nfzc, nv = self.nmo, self.nfzc, self.nv
            o, v, co = self.o, self.v, self.co
            ofull = slice(0, o.stop)
            nof = o.stop
            ERI = np.asarray(self.H.ERI)
            eps = np.diag(np.asarray(self.H.F))
            c = self.contract
            Doo, Dvv = self._so_mp2_corr_opdm()
            Gam = np.asarray(self._so_mp2_tpdm())
            Du = np.zeros((nmo, nmo))
            Du[o, o] = np.asarray(Doo)
            Du[v, v] = np.asarray(Dvv)
            Ip = self._so_mp2_lagrangian(Du, Gam)
            Drel = Du.copy()
            Pco = None
            if nfzc:
                Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
                Drel[co, o] = Pco
                Drel[o, co] = Pco.T
            X = Ip[ofull, v] - Ip[v, ofull].T
            if nfzc:
                zjc = -Pco.T                                   # z_jc, active-occupied x core
                X = X - (c('jc,ajic->ia', zjc, ERI[v, o, ofull, co])
                         + c('jc,acij->ia', zjc, ERI[v, co, ofull, o]))
            G = (c('ajib->iajb', ERI[v, ofull, ofull, v])
                 + c('abij->iajb', ERI[v, v, ofull, ofull])).reshape(nof * nv, nof * nv)
            G[np.diag_indices(nof * nv)] += (eps[v][None, :] - eps[ofull][:, None]).reshape(-1)
            zia = np.linalg.solve(G, X.reshape(-1)).reshape(nof, nv)
            Drel[v, ofull] = -zia.T
            Drel[ofull, v] = -zia
            self._so_zvec = (Drel, Gam, Du, zia, G, Pco)
        return self._so_zvec

    def _so_perturbed_lagrangian(self, pert) -> np.ndarray:
        """First-order response ``d_x I'`` of the GSB orbital Lagrangian (``nmo x nmo``),
        spin-orbital, using the **unrelaxed** densities and their responses (this is the
        Z-vector RHS derivative). Differentiates :meth:`_so_mp2_lagrangian` term by term with
        the perturbed integrals (:meth:`CPHF.perturbed_fock`/`perturbed_eri`) and the unrelaxed
        density responses (:meth:`_perturbed_densities`)."""
        nmo, o, v = self.nmo, self.o, self.v
        ofull = slice(0, o.stop)
        ncore = o.stop - self.no
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        c = self.contract
        cphf = self._full_occ_cphf()
        _, _, Du, _, _, _ = self._so_zvector()
        Gam = np.asarray(self._so_mp2_tpdm())
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        deps = np.diag(df)
        dDg, dGam = self._perturbed_densities(pert)
        dD = np.asarray(dDg)
        dGam = np.asarray(dGam)
        dA = deps[:, None] * (Du + Du.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, ERI[:, :, :, ofull])
                        + c('rs,rpsq->pq', Du, deri[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, ERI[:, ofull, :, :])
                        + c('rs,rqsp->pq', Du, deri[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    def _so_perturbed_relaxed_opdm(self, pert) -> np.ndarray:
        """First-order response ``d_x D_rel`` of the relaxed MP2 1-PDM (``nmo x nmo``),
        spin-orbital, frozen-core aware: the unrelaxed oo/vv responses
        (:meth:`_perturbed_densities`), the perturbed core-active divide ``d_x P_co``, and the
        **perturbed Z-vector** ``z^x`` in the ov/vo blocks over the full occupied space. ``z^x``
        solves the same orbital Hessian ``G`` as the gradient's Z-vector with the perturbed RHS
        ``X^x - A^x z`` (``A^x z`` uses the full non-canonical ``d_x f`` vv/oo blocks).

        ``d_x P_co`` is the derivative of the divide ``(eps_c - eps_i) P_ci = I'_ci - I'_ic``,
        a Sylvester relation ``(eps_c - eps_i) d_x P_ci = d_x(I'_ci - I'_ic) - sum_d d_x f_cd
        P_di + sum_j P_cj d_x f_ji``: the diagonal ``d=c``/``j=i`` terms are the
        ``-P_ci(d_x eps_c - d_x eps_i)`` orbital-energy shift, and the off-diagonal ``j != i``
        active-active ``d_x f`` couples the block because a field leaves the active occupied
        space non-canonical. ``P_co`` also feeds the ov RHS coupling (perturbed here too)."""
        nmo, nfzc, nv = self.nmo, self.nfzc, self.nv
        o, v, co = self.o, self.v, self.co
        ofull = slice(0, o.stop)
        nof = o.stop
        ncore = o.stop - self.no
        c = self.contract
        cphf = self._full_occ_cphf()
        _, _, _, zia, G, Pco = self._so_zvector()
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dIp = self._so_perturbed_lagrangian(pert)
        dDg, _ = self._perturbed_densities(pert)
        dDg = np.asarray(dDg)
        dD = np.zeros((nmo, nmo))
        dD[o, o] = dDg[o, o]
        dD[v, v] = dDg[v, v]
        dX = dIp[ofull, v] - dIp[v, ofull].T
        if nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            dD[co, o] = dPco
            dD[o, co] = dPco.T
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, ERI[v, o, ofull, co]) + c('jc,acij->ia', dzjc, ERI[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, deri[v, o, ofull, co]) + c('jc,acij->ia', zjc, deri[v, co, ofull, o]))
        Axz = (c('ajib,jb->ia', deri[v, ofull, ofull, v], zia) + c('abij,jb->ia', deri[v, v, ofull, ofull], zia)
               + c('ab,ib->ia', df[v, v], zia) - c('ij,ja->ia', df[ofull, ofull], zia))
        zx = np.linalg.solve(G, (dX - Axz).reshape(-1)).reshape(nof, nv)
        dD[v, ofull] = -zx.T
        dD[ofull, v] = -zx
        return dD

    def _zvector(self):
        """Spatial (closed-shell) unperturbed Z-vector data (cached, frozen-core aware):
        relaxed 1-PDM ``D_rel``, cumulant ``Gam``, unrelaxed ``D_u``, the Z-vector amplitudes
        ``z`` (indexed ``(I,a)`` over the full occupied space), the all-electron ``HFwfn``
        whose CPHF carries the orbital Hessian, and the core-active block ``P_co`` (``None``
        for ``nfzc=0``). :meth:`_mp2_relaxed_densities` delegates here."""
        if getattr(self, '_zvec', None) is None:
            from .hfwfn import HFwfn
            nmo, nfzc, no = self.nmo, self.nfzc, self.no
            o, v = self.o, self.v
            co = slice(0, nfzc)
            ofull = slice(0, nfzc + no)
            eps = np.diag(np.asarray(self.H.F))
            L = np.asarray(self.H.L)
            c = self.contract
            Doo, Dvv = self._mp2_corr_opdm()
            Gam = np.asarray(self._mp2_tpdm())
            Du = np.zeros((nmo, nmo))
            Du[o, o] = np.asarray(Doo)
            Du[v, v] = np.asarray(Dvv)
            Ip = self._mp2_lagrangian(Du, Gam)
            Drel = Du.copy()
            Pco = None
            if nfzc:
                Pco = (Ip[co, o] - Ip[o, co].T) / (eps[co][:, None] - eps[o][None, :])
                Drel[co, o] = Pco
                Drel[o, co] = Pco.T
            hf = HFwfn(self.ref)
            X = Ip[ofull, v] - Ip[v, ofull].T
            if nfzc:
                zjc = -Pco.T                                   # z_jc, active-occupied x core
                X = X - (c('jc,ajic->ia', zjc, L[v, o, ofull, co])
                         + c('jc,acij->ia', zjc, L[v, co, ofull, o]))
            zia = hf.cphf.solve(X)                              # (I,a) over full occ
            Drel[v, ofull] = -zia.T
            Drel[ofull, v] = -zia
            self._zvec = (Drel, Gam, Du, zia, hf, Pco)
        return self._zvec

    def _perturbed_lagrangian(self, pert) -> np.ndarray:
        """Spatial first-order response ``d_x I'`` of the GSB orbital Lagrangian (``nmo x nmo``),
        the closed-shell analogue of :meth:`_so_perturbed_lagrangian`: the spin-adapted ``L``
        (and its derivative ``dL = 2 d_x<pq|rs> - d_x<pq|sr>``) in the two-electron 1-PDM term,
        ``<pr|st>`` (H.ERI) with ``Gamma`` in the 2-PDM term."""
        nmo, o, v = self.nmo, self.o, self.v
        ofull = slice(0, o.stop)
        ncore = o.stop - self.no
        ERI = np.asarray(self.H.ERI)
        L = np.asarray(self.H.L)
        eps = np.diag(np.asarray(self.H.F))
        c = self.contract
        cphf = self._full_occ_cphf()
        _, _, Du, _, _, _ = self._zvector()
        Gam = np.asarray(self._mp2_tpdm())
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        deps = np.diag(df)
        dDg, dGam = self._perturbed_densities(pert)
        dD = np.asarray(dDg)
        dGam = np.asarray(dGam)
        dA = deps[:, None] * (Du + Du.T) + eps[:, None] * (dD + dD.T)
        dB = np.zeros((nmo, nmo))
        dB[:, ofull] = (c('rs,rpsq->pq', dD, L[:, :, :, ofull])
                        + c('rs,rpsq->pq', Du, dL[:, :, :, ofull])
                        + c('rs,rqsp->pq', dD, L[:, ofull, :, :])
                        + c('rs,rqsp->pq', Du, dL[:, ofull, :, :]))
        dC = 4.0 * (c('prst,qrst->pq', deri, Gam) + c('prst,qrst->pq', ERI, dGam))
        return -0.5 * (dA + dB + dC)

    def _perturbed_relaxed_opdm(self, pert) -> np.ndarray:
        """Spatial first-order response ``d_x D_rel`` of the relaxed MP2 1-PDM (``nmo x nmo``),
        frozen-core aware: the unrelaxed oo/vv responses, the perturbed core-active divide
        ``d_x P_co`` (the Sylvester derivative -- see :meth:`_so_perturbed_relaxed_opdm`), and
        the perturbed Z-vector ``z^x`` in the ov/vo blocks over the full occupied space. ``z^x``
        reuses the all-electron HFwfn CPHF (same orbital Hessian as the gradient's Z-vector)
        with the perturbed RHS ``X^x - A^x z``; ``A^x z`` uses ``dL`` and the full non-canonical
        ``d_x f`` vv/oo blocks, and (for ``nfzc>0``) the perturbed ``P_co`` RHS coupling."""
        nmo, nfzc, no = self.nmo, self.nfzc, self.no
        o, v = self.o, self.v
        co = slice(0, nfzc)
        ofull = slice(0, nfzc + no)
        ncore = o.stop - self.no
        c = self.contract
        cphf = self._full_occ_cphf()
        _, _, _, zia, hf, Pco = self._zvector()
        L = np.asarray(self.H.L)
        eps = np.diag(np.asarray(self.H.F))
        df = np.asarray(cphf.perturbed_fock(pert, ncore))
        deri = np.asarray(cphf.perturbed_eri(pert, ncore))
        dL = 2.0 * deri - deri.swapaxes(2, 3)
        dIp = self._perturbed_lagrangian(pert)
        dDg, _ = self._perturbed_densities(pert)
        dDg = np.asarray(dDg)
        dD = np.zeros((nmo, nmo))
        dD[o, o] = dDg[o, o]
        dD[v, v] = dDg[v, v]
        dX = dIp[ofull, v] - dIp[v, ofull].T
        if nfzc:
            gap = eps[co][:, None] - eps[o][None, :]
            dPco = (dIp[co, o] - dIp[o, co].T - df[co, co] @ Pco + Pco @ df[o, o]) / gap
            dD[co, o] = dPco
            dD[o, co] = dPco.T
            zjc, dzjc = -Pco.T, -dPco.T
            dX = dX - (c('jc,ajic->ia', dzjc, L[v, o, ofull, co]) + c('jc,acij->ia', dzjc, L[v, co, ofull, o])
                       + c('jc,ajic->ia', zjc, dL[v, o, ofull, co]) + c('jc,acij->ia', zjc, dL[v, co, ofull, o]))
        Axz = (c('ajib,jb->ia', dL[v, ofull, ofull, v], zia) + c('abij,jb->ia', dL[v, v, ofull, ofull], zia)
               + c('ab,ib->ia', df[v, v], zia) - c('ij,ja->ia', df[ofull, ofull], zia))
        zx = hf.cphf.solve(dX - Axz)
        dD[v, ofull] = -zx.T
        dD[ofull, v] = -zx
        return dD

    # ---- second derivatives: polarizability, APT (dipole derivatives), Hessian ----

    def polarizability(self, route: str = 'explicit') -> np.ndarray:
        """MP2 **correlation** contribution to the static (omega=0) dipole polarizability
        (a.u.), shape ``(3, 3)``: ``alpha_corr_ab = -d^2 E_corr / dF_a dF_b``.

        ``route='explicit'`` (default) -- the explicit second-derivative route (``notes.pdf``
        Eq. 15)::

            d_ab E_corr = sum_pq [ d_a gamma_pq d_b f_pq + gamma_pq d_ab f_pq ]
                        + sum_pqrs [ d_a Gamma_pqrs d_b <pq||rs> + Gamma_pqrs d_ab <pq||rs> ]

        with the unrelaxed densities ``gamma``/``Gamma``, their responses ``d_a gamma``/
        ``d_a Gamma`` (:meth:`_perturbed_densities`), the first perturbed derivatives
        (:meth:`CPHF.perturbed_fock`/`perturbed_eri`), and the second perturbed derivatives
        (:meth:`CPHF.perturbed_fock2`/`perturbed_eri2`, which carry the second-order CPHF
        response ``U^{ab}``). Basis-aware; all-electron or frozen-core.

        ``route='2n+1'`` -- the 2n+1 route (:meth:`_polarizability_2n1`): differentiate the
        relaxed-density gradient, using only the first-order perturbed *relaxed* density
        (:meth:`_so_perturbed_relaxed_opdm` / :meth:`_perturbed_relaxed_opdm`, which carry the
        perturbed Z-vector). Frozen-core aware (both spin-orbital and spin-adapted paths);
        reproduces the explicit route to ~machine precision (an independent cross-check, and
        the efficient route for the APT/Hessian).

        The reference part is kept separate (:meth:`HFwfn.polarizability`); the total is their
        sum. Electric field only (the second-order engine assumes a perturbation-independent AO
        basis)."""
        if route == '2n+1':
            return self._polarizability_2n1()
        if route != 'explicit':
            raise ValueError(f"unknown polarizability route {route!r} (use 'explicit' or '2n+1')")
        from .cphf import Perturbation
        o, v, nmo = self.o, self.v, self.nmo
        ncore = o.stop - self.no
        if self.orbital_basis == 'spinorbital':
            Doo, Dvv = self._so_mp2_corr_opdm()
            Gam = self._so_mp2_tpdm()
        else:
            Doo, Dvv = self._mp2_corr_opdm()
            Gam = self._mp2_tpdm()
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = np.asarray(Doo)
        gamma[v, v] = np.asarray(Dvv)
        cphf = self._full_occ_cphf()
        pert = [Perturbation('field', a) for a in range(3)]
        dgam, dGam = zip(*(self._perturbed_densities(pert[a]) for a in range(3)))
        c = self.contract
        alpha = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                dbf = cphf.perturbed_fock(pert[b], ncore)
                dbe = cphf.perturbed_eri(pert[b], ncore)
                d2f = cphf.perturbed_fock2(pert[a], pert[b], ncore)
                d2e = cphf.perturbed_eri2(pert[a], pert[b], ncore)
                e2 = (c('pq,pq->', dgam[a], dbf) + c('pq,pq->', gamma, d2f)
                      + c('pqrs,pqrs->', dGam[a], dbe) + c('pqrs,pqrs->', Gam, d2e))
                alpha[a, b] = -e2
        return alpha

    def _polarizability_2n1(self) -> np.ndarray:
        """MP2 correlation polarizability via the 2n+1 route (frozen-core aware; spin-orbital
        and spin-adapted paths).

        Differentiating the relaxed-density gradient in a field: ``d_b E = -Tr(D_rel mu_b)``
        (field skeleton ``f^(b) = -mu_b``), so::

            alpha_ab = sum_pq d_a D_rel_pq (mu_b)_pq
                     + sum_pq D_rel_pq [ (U^a).T mu_b + mu_b U^a ]_pq

        The first term is the perturbed relaxed density (:meth:`_so_perturbed_relaxed_opdm`,
        carrying the perturbed Z-vector and, for frozen core, the perturbed core-active divide);
        the second is the MO dipole rotating under the field (``d_a f^(b) = rotate(U^a, -mu_b)``,
        with ``U^a`` over the full occupied space -- ``ncore`` canonical core-active block). No
        second-order CPHF ``U^{ab}`` -- only first-order responses. Reproduces the explicit
        route to ~machine precision."""
        from .cphf import Perturbation
        ncore = self.o.stop - self.no
        cphf = self._full_occ_cphf()
        c = self.contract
        if self.orbital_basis == 'spinorbital':
            Drel = self._so_zvector()[0]
            perturbed_opdm = self._so_perturbed_relaxed_opdm
        else:
            Drel = self._zvector()[0]
            perturbed_opdm = self._perturbed_relaxed_opdm
        mu = [np.asarray(self.H.mu[a]) for a in range(3)]
        field = [Perturbation('field', b) for b in range(3)]
        alpha = np.zeros((3, 3))
        for b in range(3):
            dDrel = perturbed_opdm(field[b])
            Ub = np.asarray(cphf._full_U(field[b], ncore))
            for a in range(3):
                rot = Ub.T @ mu[a] + mu[a] @ Ub
                alpha[a, b] = c('pq,pq->', dDrel, mu[a]) + c('pq,pq->', Drel, rot)
        return alpha

    def dipole_derivatives(self) -> np.ndarray:
        """MP2 **correlation** contribution to the atomic polar tensors (nuclear dipole
        derivatives, a.u.), shape ``(natom, 3, 3)`` indexed ``[A, beta, alpha]`` =
        ``d(mu_alpha)/d(X_{A,beta}) = -d^2 E_corr / dF_alpha dX_{A,beta}`` -- the mixed
        field/nuclear analog of :meth:`polarizability`, via the explicit route (Eq. 15)::

            d_{F X} E_corr = sum_pq [ d_X gamma_pq d_F f_pq + gamma_pq d_{F X} f_pq ]
                           + sum_pqrs [ d_X Gamma d_F <pq||rs> + Gamma d_{F X} <pq||rs> ]

        with the unrelaxed densities ``gamma``/``Gamma``, the **nuclear** density responses
        ``d_X gamma``/``d_X Gamma`` (:meth:`_perturbed_densities`), the field first derivatives
        (:meth:`CPHF.perturbed_fock`/`perturbed_eri`), and the mixed second derivatives
        (:meth:`CPHF.perturbed_fock2`/`perturbed_eri2`, which carry the second-order response
        ``U^{FX}`` and the ``-mu^X`` mixed skeleton). Electronic only; the nuclear ``Z_A`` term
        is in the reference (:meth:`HFwfn.dipole_derivatives`), so the total is their sum
        (:meth:`total_dipole_derivatives`). Basis-aware. Validated against a finite nuclear
        displacement of the analytic dipole (``test_068``)."""
        from .cphf import Perturbation
        o, v, nmo = self.o, self.v, self.nmo
        ncore = o.stop - self.no
        if self.orbital_basis == 'spinorbital':
            Doo, Dvv = self._so_mp2_corr_opdm()
            Gam = self._so_mp2_tpdm()
        else:
            Doo, Dvv = self._mp2_corr_opdm()
            Gam = self._mp2_tpdm()
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = np.asarray(Doo)
        gamma[v, v] = np.asarray(Dvv)
        Gam = np.asarray(Gam)
        cphf = self._full_occ_cphf()
        c = self.contract
        natom = self.derivatives.natom
        field = [Perturbation('field', a) for a in range(3)]
        dFf = [np.asarray(cphf.perturbed_fock(field[a], ncore)) for a in range(3)]
        dFe = [np.asarray(cphf.perturbed_eri(field[a], ncore)) for a in range(3)]
        P = np.zeros((natom, 3, 3))
        for A in range(natom):
            for beta in range(3):
                pX = Perturbation('nuclear', (A, beta))
                dgX, dGX = self._perturbed_densities(pX)
                dgX = np.asarray(dgX)
                dGX = np.asarray(dGX)
                for alpha in range(3):
                    d2f = np.asarray(cphf.perturbed_fock2(field[alpha], pX, ncore))
                    d2e = np.asarray(cphf.perturbed_eri2(field[alpha], pX, ncore))
                    e2 = (c('pq,pq->', dgX, dFf[alpha]) + c('pq,pq->', gamma, d2f)
                          + c('pqrs,pqrs->', dGX, dFe[alpha]) + c('pqrs,pqrs->', Gam, d2e))
                    P[A, beta, alpha] = -e2
        return P

    def hessian(self) -> np.ndarray:
        """MP2 **correlation** contribution to the molecular (nuclear) Hessian (a.u.), shape
        ``(3*natom, 3*natom)`` indexed ``(A*3+a, B*3+b)`` = ``d^2 E_corr / dX_{Aa} dX_{Bb}`` --
        the nuclear-nuclear analog of :meth:`polarizability`/:meth:`dipole_derivatives`, via
        the explicit route (Eq. 15)::

            d_{XY} E_corr = sum_pq [ d_X gamma d_Y f + gamma d_{XY} f ]
                          + sum_pqrs [ d_X Gamma d_Y <pq||rs> + Gamma d_{XY} <pq||rs> ]

        with the nuclear density responses ``d_X gamma``/``d_X Gamma``
        (:meth:`_perturbed_densities`), the nuclear first derivatives
        (:meth:`CPHF.perturbed_fock`/`perturbed_eri`), and the nuclear-nuclear second
        derivatives (:meth:`CPHF.perturbed_fock2`/`perturbed_eri2`, which carry ``U^{XY}``, the
        ``xi^{XY}`` ``S^{XY}``/``S^X S^Y`` overlap terms, and the ``h^{XY}``/``<pq||rs>^{XY}``
        skeletons). The reference part is separate (:meth:`HFwfn.hessian`); total is their sum
        (:meth:`total_hessian`). Basis-aware.

        The explicit route solves ``U^{XY}`` for each of the ``3N(3N+1)/2`` unique nuclear
        pairs -- the ``O(N^2)`` cost the 2n+1 (Z-vector interchange) route would avoid; fine for
        a reference implementation on small molecules. Validated against a finite difference of
        the analytic gradient (``test_069``)."""
        from .cphf import Perturbation
        o, v, nmo = self.o, self.v, self.nmo
        ncore = o.stop - self.no
        if self.orbital_basis == 'spinorbital':
            Doo, Dvv = self._so_mp2_corr_opdm()
            Gam = self._so_mp2_tpdm()
        else:
            Doo, Dvv = self._mp2_corr_opdm()
            Gam = self._mp2_tpdm()
        gamma = np.zeros((nmo, nmo))
        gamma[o, o] = np.asarray(Doo)
        gamma[v, v] = np.asarray(Dvv)
        Gam = np.asarray(Gam)
        cphf = self._full_occ_cphf()
        c = self.contract
        natom = self.derivatives.natom
        nc = 3 * natom
        pert = [Perturbation('nuclear', (A, cart)) for A in range(natom) for cart in range(3)]
        dXf = [np.asarray(cphf.perturbed_fock(pert[i], ncore)) for i in range(nc)]
        dXe = [np.asarray(cphf.perturbed_eri(pert[i], ncore)) for i in range(nc)]
        dg, dG = zip(*(self._perturbed_densities(pert[i]) for i in range(nc)))
        dg = [np.asarray(x) for x in dg]
        dG = [np.asarray(x) for x in dG]
        H = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(i, nc):                      # d_{XY} f / <> symmetric in X<->Y
                d2f = np.asarray(cphf.perturbed_fock2(pert[i], pert[j], ncore))
                d2e = np.asarray(cphf.perturbed_eri2(pert[i], pert[j], ncore))
                gd2f = c('pq,pq->', gamma, d2f)
                Gd2e = c('pqrs,pqrs->', Gam, d2e)
                H[i, j] = (c('pq,pq->', dg[i], dXf[j]) + gd2f
                           + c('pqrs,pqrs->', dG[i], dXe[j]) + Gd2e)
                if j != i:
                    H[j, i] = (c('pq,pq->', dg[j], dXf[i]) + gd2f
                               + c('pqrs,pqrs->', dG[j], dXe[i]) + Gd2e)
        return H

    # ---- total (reference + correlation) properties ----
    # The correlation methods above are the correlation contribution only; these add the
    # all-electron SCF reference (HFwfn, frozen_core=False) for the full molecular property.

    def _reference_hf(self):
        """The all-electron :class:`HFwfn` for the SCF reference (cached), supplying the
        reference dipole and gradient for the total MP2 properties."""
        if getattr(self, '_ref_hf', None) is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.ref, orbital_basis=self.orbital_basis)
        return self._ref_hf

    def total_dipole(self) -> np.ndarray:
        """Total MP2 electric-dipole moment (a.u.), shape ``(3,)``: the SCF reference dipole
        (:meth:`HFwfn.dipole`) plus the MP2 correlation dipole (:meth:`relaxed_dipole`)."""
        return self._reference_hf().dipole() + self.relaxed_dipole()

    def total_gradient(self) -> np.ndarray:
        """Total MP2 analytic nuclear gradient (a.u.), shape ``(natom, 3)``: the SCF reference
        gradient (:meth:`HFwfn.gradient`) plus the MP2 correlation gradient (:meth:`gradient`)."""
        return self._reference_hf().gradient() + self.gradient()

    def total_dipole_derivatives(self) -> np.ndarray:
        """Total MP2 atomic polar tensors (nuclear dipole derivatives, a.u.), shape
        ``(natom, 3, 3)`` indexed ``[A, beta, alpha]``: the SCF reference APTs
        (:meth:`HFwfn.dipole_derivatives`, which carry the ``Z_A`` nuclear term) plus the MP2
        correlation APTs (:meth:`dipole_derivatives`)."""
        return self._reference_hf().dipole_derivatives() + self.dipole_derivatives()

    def total_hessian(self) -> np.ndarray:
        """Total MP2 molecular (nuclear) Hessian (a.u.), shape ``(3*natom, 3*natom)``: the SCF
        reference Hessian (:meth:`HFwfn.hessian`, which carries the nuclear-repulsion term) plus
        the MP2 correlation Hessian (:meth:`hessian`)."""
        return self._reference_hf().hessian() + self.hessian()
