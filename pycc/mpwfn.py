"""
mpwfn.py: Moller-Plesset perturbation-theory (MP2) wavefunction.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import psi4

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

    def _so_mp2_cumulant(self) -> np.ndarray:
        """Spin-orbital MP2 cumulant 2-PDM ``Gamma_ijab = 1/4 t2`` in the ``oovv``/``vvoo``
        blocks -- the only blocks that contribute (determined from the MP2 energy
        Lagrangian, in which Lambda and T2 enter linearly)."""
        o, v = self.o, self.v
        nmo = self.no + self.nv
        t2 = np.asarray(self.t2)
        Gam = np.zeros((nmo, nmo, nmo, nmo))
        Gam[o, o, v, v] = 0.25 * t2
        Gam[v, v, o, o] = 0.25 * t2.transpose(2, 3, 0, 1)
        return Gam

    def _so_mp2_lagrangian(self, Doo, Dvv, Gam) -> np.ndarray:
        """Spin-orbital orbital-gradient Lagrangian matrix ``I'_pq`` (``nmo x nmo``)

            I'_pq = -1/2 [ f_pp (D_pq + D_qp)
                           + delta_{q,occ} sum_rs D_rs (<rp||sq> + <rq||sp>)
                           + 4 sum_rst <pr||st> Gamma_qrst ]

        from the correlation 1-PDM ``D`` (Doo/Dvv) and cumulant 2-PDM ``Gamma``. Its
        occupied-virtual antisymmetric part ``X_ai = I'_ia - I'_ai`` is the Z-vector RHS."""
        o, v = self.o, self.v
        nmo = self.no + self.nv
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        termA = eps[:, None] * (D + D.T)                       # f_pp (D_pq + D_qp)
        termB = np.zeros((nmo, nmo))                           # only q in occ
        termB[:, o] = (np.einsum('rs,rpsq->pq', D, ERI[:, :, :, o])
                       + np.einsum('rs,rqsp->pq', D, ERI[:, o, :, :]))
        termC = 4.0 * np.einsum('prst,qrst->pq', ERI, Gam)
        return -0.5 * (termA + termB + termC)

    def _so_mp2_zvector(self):
        """Solve the spin-orbital MP2 Z-vector. Returns ``(Doo, Dvv, Gam, Ip, z)``: the
        correlation densities, the cumulant 2-PDM, the Lagrangian ``I'_pq``, and the
        orbital relaxation ``z_ai`` (``A z = X``, ``X_ai = I'_ia - I'_ai``)."""
        if self.orbital_basis != 'spinorbital':
            raise NotImplementedError(
                "The MP2 relaxed gradient is implemented for the spin-orbital path.")
        o, v = self.o, self.v
        no, nv = self.no, self.nv
        Doo, Dvv = self._so_mp2_corr_opdm()
        Gam = self._so_mp2_cumulant()
        Ip = self._so_mp2_lagrangian(Doo, Dvv, Gam)
        X = Ip[o, v].T - Ip[v, o]                              # X_ai
        A = self._so_orbital_hessian()
        z = np.linalg.solve(A, X.reshape(-1)).reshape(nv, no)  # z_ai
        return Doo, Dvv, Gam, Ip, z

    def mp2_relaxed_opdm(self) -> np.ndarray:
        """Spin-orbital relaxed MP2 one-particle correlation density (``nmo x nmo``):
        the unrelaxed ``Doo``/``Dvv`` plus the orbital-relaxation off-diagonal blocks
        ``D_ai = D_ia = -z_ai`` from the Z-vector solve. Spin-orbital only."""
        o, v = self.o, self.v
        nmo = self.no + self.nv
        Doo, Dvv, Gam, Ip, z = self._so_mp2_zvector()
        D = np.zeros((nmo, nmo))
        D[o, o] = np.asarray(Doo)
        D[v, v] = np.asarray(Dvv)
        D[v, o] = -z
        D[o, v] = -z.T
        return D

    def _so_energy_weighted_opdm(self, Ip, z) -> np.ndarray:
        """Spin-orbital MP2 energy-weighted density ``I_pq`` (the gradient's overlap-
        derivative weight), from the Lagrangian ``I'`` and the Z-vector (GSB notes):

            I_ij = I'_ij + sum_ak z_ak (<ai||kj> + <aj||ki>),   I_ab = I'_ab,
            I_ia = I_ai = I'_ia + z_ai eps_i."""
        o, v = self.o, self.v
        nmo = self.no + self.nv
        ERI = np.asarray(self.H.ERI)
        eps = np.diag(np.asarray(self.H.F))
        I = np.zeros((nmo, nmo))
        I[o, o] = (Ip[o, o] + np.einsum('ak,aikj->ij', z, ERI[v, o, o, o])
                   + np.einsum('ak,ajki->ij', z, ERI[v, o, o, o]))
        I[v, v] = Ip[v, v]
        I[o, v] = Ip[o, v] + (z * eps[o][None, :]).T
        I[v, o] = Ip[o, v].T + z * eps[o][None, :]
        return I

    # ---- spin-orbital MP2 nuclear gradient ----

    def _so_oei_deriv(self, mints, kind, atom):
        """Spin-orbital one-electron derivative integral (block-diagonal in spin) for
        ``atom``; returns the three (x, y, z) ``nmo x nmo`` arrays. Built from the
        spatial MO derivative integrals in the semicanonical MO gauge."""
        nmo = self.no + self.nv
        spin = np.asarray(self.spin)
        spat = np.asarray(self.spat)
        a = np.where(spin == 0)[0]
        b = np.where(spin == 1)[0]
        sa, sb = spat[a], spat[b]
        Ca = psi4.core.Matrix.from_array(self.H.Ca)
        Cb = psi4.core.Matrix.from_array(self.H.Cb)
        Da = mints.mo_oei_deriv1(kind, atom, Ca, Ca)
        Db = mints.mo_oei_deriv1(kind, atom, Cb, Cb)
        out = []
        for c in range(3):
            M = np.zeros((nmo, nmo))
            M[np.ix_(a, a)] = np.asarray(Da[c])[np.ix_(sa, sa)]
            M[np.ix_(b, b)] = np.asarray(Db[c])[np.ix_(sb, sb)]
            out.append(M)
        return out

    def _so_eri_deriv(self, mints, atom):
        """Spin-orbital antisymmetrized two-electron derivative integral ``<pq||rs>^x``
        for ``atom``; returns the three (x, y, z) ``nmo^4`` arrays. Built by spin-blocking
        the spatial chemist derivative integrals (semicanonical gauge) then
        antisymmetrizing, mirroring the spin-orbital Hamiltonian's ERI construction."""
        nmo = self.no + self.nv
        spin = np.asarray(self.spin)
        spat = np.asarray(self.spat)
        a = np.where(spin == 0)[0]
        b = np.where(spin == 1)[0]
        sa, sb = spat[a], spat[b]
        Ca = psi4.core.Matrix.from_array(self.H.Ca)
        Cb = psi4.core.Matrix.from_array(self.H.Cb)
        AA = mints.mo_tei_deriv1(atom, Ca, Ca, Ca, Ca)
        BB = mints.mo_tei_deriv1(atom, Cb, Cb, Cb, Cb)
        AB = mints.mo_tei_deriv1(atom, Ca, Ca, Cb, Cb)
        out = []
        for c in range(3):
            chem = np.zeros((nmo, nmo, nmo, nmo))
            chem[np.ix_(a, a, a, a)] = np.asarray(AA[c])[np.ix_(sa, sa, sa, sa)]
            chem[np.ix_(b, b, b, b)] = np.asarray(BB[c])[np.ix_(sb, sb, sb, sb)]
            chem[np.ix_(a, a, b, b)] = np.asarray(AB[c])[np.ix_(sa, sa, sb, sb)]
            chem[np.ix_(b, b, a, a)] = np.asarray(AB[c]).transpose(2, 3, 0, 1)[np.ix_(sb, sb, sa, sa)]
            phys = chem.swapaxes(1, 2)
            out.append(phys - phys.swapaxes(2, 3))
        return out

    def gradient(self) -> np.ndarray:
        """Spin-orbital MP2 analytic nuclear energy gradient (a.u.), shape (natom, 3).

        The SCF (``HFwfn``) gradient plus the correlation contribution

            dE_corr/dX = sum_pq D_pq f^X_pq + sum_pqrs Gamma_pqrs <pq||rs>^X
                         + sum_pq I_pq S^X_pq

        with the relaxed 1-PDM ``D``, cumulant 2-PDM ``Gamma``, and energy-weighted
        density ``I`` (spin-orbital CC gradient formulation, Gauss/Stanton/Bartlett). The
        derivative integrals are built in the semicanonical MO gauge the densities live in;
        ``f^X = h^X + sum_m <pm||qm>^X`` is the skeleton Fock derivative. Spin-orbital only."""
        from .hfwfn import HFwfn
        o = self.o
        Doo, Dvv, Gam, Ip, z = self._so_mp2_zvector()
        D = self.mp2_relaxed_opdm()
        I = self._so_energy_weighted_opdm(Ip, z)

        mints = psi4.core.MintsHelper(self.H.basisset)
        natom = self.H.mol.natom()
        grad = np.zeros((natom, 3))
        for atom in range(natom):
            Tx = self._so_oei_deriv(mints, "KINETIC", atom)
            Vx = self._so_oei_deriv(mints, "POTENTIAL", atom)
            Sx = self._so_oei_deriv(mints, "OVERLAP", atom)
            ERIx = self._so_eri_deriv(mints, atom)
            for c in range(3):
                hx = Tx[c] + Vx[c]
                fx = hx + np.einsum('pmqm->pq', ERIx[c][:, o, :, o])   # skeleton Fock deriv
                grad[atom, c] = (np.einsum('pq,pq->', D, fx)
                                 + np.einsum('pqrs,pqrs->', Gam, ERIx[c])
                                 + np.einsum('pq,pq->', I, Sx[c]))
        return HFwfn(self.ref).gradient() + grad
