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

    The analytic derivative-property code (gradient, polarizability, APT, Hessian, AAT,
    VG-APT) lives on :class:`~pycc.mpderiv.MPderiv`, reached through the cached
    :attr:`deriv` driver; the property methods here are thin delegators kept for the
    historical ``mpwfn.<property>()`` call sites.

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

    # ---- unrelaxed correlation-density seeds ----
    # The unrelaxed one- and two-particle correlation densities (pure functions of the MP2
    # amplitudes) seed the relaxed densities and orbital response.  They are built here, on the
    # wavefunction; MPderiv (the derivative driver) consumes them (self.mp._*_corr_opdm / _*_tpdm)
    # for the Lagrangian, Z-vector, and property derivatives.  The spin-orbital (_so_) and
    # spin-adapted (closed-shell, unlabeled) paths are paired; the spatial spin sum rides in
    # l2 = 2(2 t2 - t2.swap) and the spin-adapted L (= H.L).

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

    def _mp2_normalization(self) -> float:
        """MP2 intermediate normalization ``N = 1/sqrt(1 + <T2|2T2 - T2~>)`` (spin-adapted),
        for the wave-function-overlap AAT (:meth:`MPderiv.atomic_axial_tensors`). The normalized
        doubles are ``c2 = N t2`` and the reference coefficient is ``c0 = N``."""
        t2 = np.asarray(self.t2)
        norm2 = self.contract('ijab,ijab->', t2, 2.0 * t2 - t2.swapaxes(2, 3))
        return 1.0 / np.sqrt(1.0 + norm2)

    def _so_mp2_normalization(self) -> float:
        """Spin-orbital MP2 normalization ``N = 1/sqrt(1 + (1/4)<T2|T2>)`` -- the spin-orbital
        analogue of :meth:`_mp2_normalization` (the ``1/4`` for the antisymmetric double sum).
        Equal to the spin-adapted value on a closed shell."""
        t2 = np.asarray(self.t2)
        norm2 = 0.25 * self.contract('ijab,ijab->', t2, t2)
        return 1.0 / np.sqrt(1.0 + norm2)

    # ---- analytic derivative-property driver (see pycc.mpderiv.MPderiv) ----
    # The analytic derivative-property code lives on MPderiv, the MP2 leaf of CorrelatedDerivs.
    # `deriv` is the cached driver; the thin methods below delegate to it so the historical
    # `mpwfn.<property>()` call sites keep working, and the pycc property facade routes through
    # the registry (pycc/__init__.py) to the same driver.

    @property
    def deriv(self):
        """The cached :class:`~pycc.mpderiv.MPderiv` derivative-property driver for this
        wavefunction (built lazily).  A single instance so its ``_full_occ_cphf`` / Z-vector
        caches are shared across the MP2 property calls and the ``CCderiv`` cross-calls that
        reach it via ``cc.mp.deriv``."""
        if getattr(self, '_deriv', None) is None:
            from .mpderiv import MPderiv
            self._deriv = MPderiv(self)
        return self._deriv

    def relaxed_dipole(self) -> np.ndarray:
        """MP2 correlation dipole -- delegates to :meth:`MPderiv.relaxed_dipole`."""
        return self.deriv.relaxed_dipole()

    def gradient(self) -> np.ndarray:
        """MP2 correlation nuclear gradient -- delegates to :meth:`MPderiv.gradient`."""
        return self.deriv.gradient()

    def polarizability(self, route: str = '2n+1') -> np.ndarray:
        """MP2 correlation polarizability -- delegates to :meth:`MPderiv.polarizability`."""
        return self.deriv.polarizability(route)

    def hessian(self, route: str = '2n+1') -> np.ndarray:
        """MP2 correlation Hessian -- delegates to :meth:`MPderiv.hessian`."""
        return self.deriv.hessian(route)

    def dipole_derivatives(self, route: str = '2n+1-field') -> np.ndarray:
        """MP2 correlation length-gauge APT -- delegates to :meth:`MPderiv.dipole_derivatives`."""
        return self.deriv.dipole_derivatives(route)

    def velocity_dipole_derivatives(self, gauge: str = 'non-canonical') -> np.ndarray:
        """MP2 correlation velocity-gauge APT -- delegates to
        :meth:`MPderiv.velocity_dipole_derivatives`."""
        return self.deriv.velocity_dipole_derivatives(gauge)

    def atomic_axial_tensors(self, gauge: str = 'non-canonical') -> np.ndarray:
        """MP2 correlation AAT -- delegates to :meth:`MPderiv.atomic_axial_tensors`."""
        return self.deriv.atomic_axial_tensors(gauge)

    def _perturbed_unrelaxed_densities(self, pert):
        """First-order unrelaxed correlation-density response -- delegates to
        :meth:`MPderiv._perturbed_unrelaxed_densities`."""
        return self.deriv._perturbed_unrelaxed_densities(pert)

    # ---- reference for the total (reference + correlation) properties ----
    # The property methods above are the correlation contribution only.  The full molecular
    # property (nuclear + SCF reference + correlation) is assembled by the pycc property facade
    # (pycc.dipole/gradient/polarizability/hessian/apt/aat), which pairs each correlation method
    # with the SCF reference below and the separate nuclear term.

    def _reference_hf(self):
        """The all-electron :class:`HFwfn` for the SCF reference (cached), supplying the reference
        (electronic) contribution to the total MP2 properties via the pycc property facade."""
        if getattr(self, '_ref_hf', None) is None:
            from .hfwfn import HFwfn
            self._ref_hf = HFwfn(self.ref, orbital_basis=self.orbital_basis)
        return self._ref_hf
