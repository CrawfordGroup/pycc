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
    from .cphf import CPHF


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

    @property
    def cphf(self) -> "CPHF":
        """RHF coupled-perturbed-Hartree-Fock orbital-response solver for this
        reference, built lazily and cached.

        Exposes the orbital Hessian and the linear solve ``G z = B`` that the relaxed
        MP2 gradient uses as its Z-vector solver. The orbital Hessian is reference-level
        (built from ``H.L`` and the orbital energies), so it is identical to the one
        ``HFwfn`` builds; this is an MP2-local accessor for now -- a promotion to the
        :class:`Wavefunction` base can follow when the CC gradients arrive and a shared
        derivative layer is lifted out. Spatial-RHF only (the orbital Hessian needs the
        spin-adapted ``H.L``)."""
        if getattr(self, '_cphf', None) is None:
            if self.orbital_basis != 'spatial':
                raise NotImplementedError(
                    "CPHF orbital response is implemented for the spatial RHF path only.")
            from .cphf import CPHF
            self._cphf = CPHF(self)
        return self._cphf

    # ---- MP2 relaxed-gradient densities (spatial RHF) ----

    def _mp2_lambda(self) -> "Tensor":
        """First-order (MP2) Lambda doubles in the spatial spin-adapted convention,
        ``l2 = 2 (2 t2 - t2^{ab<->ba})``. The leading factor of 2 is the Lambda
        normalization that is part of the spin-adaptation -- the same convention used
        by ``cclambda``/``ccdensity`` -- so the MP2 densities below close the energy."""
        return 2.0 * (2.0 * self.t2 - self.t2.swapaxes(2, 3))

    def mp2_opdm_corr(self):
        """Unrelaxed MP2 one-particle correlation density blocks ``(Doo, Dvv)``,
        spatial RHF (the orbital-relaxation ``ov`` block comes later from the Z-vector).

        ``Doo_ij = - t2_imef l2_jmef`` and ``Dvv_ab = t2_mnbe l2_mnae`` -- the MP2 limit
        (``t1=l1=0``) of the spin-adapted CC one-particle density (``ccdensity``)."""
        c = self.contract
        t2 = self.t2
        l2 = self._mp2_lambda()
        Doo = -c('imef,jmef->ij', t2, l2)
        Dvv = c('mnbe,mnae->ab', t2, l2)
        return Doo, Dvv

    def mp2_tpdm_oovv(self) -> "Tensor":
        """Non-separable MP2 two-particle density ``oovv`` block,
        ``Gamma_ijab = 2 (2 t2 - t2^{ab<->ba}) + l2`` -- the first-order (energy-carrying)
        block of the spin-adapted two-particle density (``ccdensity.build_Doovv``, MP2
        limit). The separable / HF cross pieces of the full 2-PDM are assembled later
        (gradient phase)."""
        t2 = self.t2
        return 2.0 * (2.0 * t2 - t2.swapaxes(2, 3)) + self._mp2_lambda()

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

    def _so_orbital_hessian(self):
        """Spin-orbital orbital Hessian ``A_{ai,bj} = (eps_a - eps_i) delta_ab delta_ij
        + <ab||ij> + <aj||ib>`` as an ``(nv*no, nv*no)`` matrix (row/col flattened
        a-major: ``a*no + i``). The MP2 Z-vector solver; the SO analogue of the CPHF
        orbital Hessian (its two-electron part is the orbital-Hessian structure)."""
        o, v = self.o, self.v
        no, nv = self.no, self.nv
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        diag = (np.einsum('ab,ij->aibj', np.eye(nv), np.eye(no))
                * (eps[v][:, None, None, None] - eps[o][None, :, None, None]))
        A = diag + np.einsum('abij->aibj', ERI[v, v, o, o]) + np.einsum('ajib->aibj', ERI[v, o, o, v])
        return A.reshape(nv * no, nv * no)

    def _so_mp2_orbital_lagrangian(self, Doo, Dvv):
        """Occupied-virtual orbital-gradient ``X_ai = I'_ia - I'_ai`` (shape (nv, no)),
        the Z-vector right-hand side, from the spin-orbital GSB Lagrangian

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q,occ} sum_rs D_rs (<rp||sq> + <rq||sp>)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        with the correlation 1-PDM ``D`` (Doo/Dvv) and the cumulant 2-PDM
        ``Gamma_ijab = 1/4 t2`` (``oovv``/``vvoo`` only)."""
        o, v = self.o, self.v
        no, nv = self.no, self.nv
        nmo = no + nv
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        t2 = np.asarray(self.t2)
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        Gam = np.zeros((nmo, nmo, nmo, nmo))
        Gam[o, o, v, v] = 0.25 * t2
        Gam[v, v, o, o] = 0.25 * t2.transpose(2, 3, 0, 1)

        termA = eps[:, None] * (D + D.T)                       # f_pp (D_pq + D_qp)
        termB = np.zeros((nmo, nmo))                           # only q in occ
        termB[:, o] = (np.einsum('rs,rpsq->pq', D, ERI[:, :, :, o])
                       + np.einsum('rs,rqsp->pq', D, ERI[:, o, :, :]))
        termC = 4.0 * np.einsum('prst,qrst->pq', ERI, Gam)
        Ip = -0.5 * (termA + termB + termC)
        return Ip[o, v].T - Ip[v, o]                           # X_ai

    def mp2_relaxed_opdm(self) -> np.ndarray:
        """Spin-orbital relaxed MP2 one-particle correlation density (``nmo x nmo``):
        the unrelaxed ``Doo``/``Dvv`` plus the orbital-relaxation off-diagonal blocks
        ``D_ai = D_ia = -z_ai`` from the Z-vector solve ``A z = X``. Spin-orbital only."""
        if self.orbital_basis != 'spinorbital':
            raise NotImplementedError(
                "The MP2 relaxed-gradient density is implemented for the spin-orbital path.")
        o, v = self.o, self.v
        no, nv = self.no, self.nv
        nmo = no + nv
        Doo, Dvv = self._so_mp2_corr_opdm()
        X = self._so_mp2_orbital_lagrangian(Doo, Dvv)         # (nv, no)
        A = self._so_orbital_hessian()
        z = np.linalg.solve(A, X.reshape(-1)).reshape(nv, no)  # z_ai
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        D[v, o] = -z
        D[o, v] = -z.T
        return D
