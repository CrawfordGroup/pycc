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
        r"""Build the energy denominators and MP2 doubles amplitudes from the seeded
        MO integrals. The ERI oovv block is staged onto the compute device with
        clone(..., device1) so the divide lands where the amplitudes live.

        Basis-agnostic -- the same expression in the respective ``H.ERI`` (``<ij|ab>`` spatial,
        the antisymmetrized ``<ij||ab>`` spin-orbital), with the denominator from the Fock
        diagonal::

            D_ijab  = eps_i + eps_j - eps_a - eps_b
            t2_ijab = <ij||ab> / D_ijab
            t1_ia   = f_ia / D_ia          (spin-orbital only)

        .. math::

            \begin{aligned}
            D_{ijab} &= \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b \\
            t^{ab}_{ij} &= \langle ij||ab \rangle / D_{ijab} \\
            t^a_i &= f_{ia} / D_{ia} \qquad \text{(spin-orbital only)}
            \end{aligned}
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
        r"""Compute and return the MP2 correlation energy.

        Spatial (spin-adapted) and spin-orbital paths (no ``L`` exists in the SO path; the ``1/4``
        accounts for the unrestricted sum over the antisymmetrized doubles, and the singles term is
        nonzero only for a non-canonical, e.g. ROHF, reference)::

            E = sum_ijab t2_ijab L_ijab                            (spatial)
            E = sum_ia f_ia t1_ia + 1/4 sum_ijab <ij||ab> t2_ijab  (spin-orbital)

        .. math::

            \begin{aligned}
            E &= \sum_{ijab} t^{ab}_{ij} L_{ijab} && \text{(spatial)} \\
            E &= \sum_{ia} f_{ia} t^a_i + \tfrac{1}{4} \sum_{ijab} \langle ij||ab \rangle t^{ab}_{ij} && \text{(spin-orbital)}
            \end{aligned}
        """
        o = self.o
        v = self.v
        if self.orbital_basis == 'spatial':
            self.emp2 = self.contract('ijab,ijab->', self.t2, self.H.L[o, o, v, v])
        else:
            self.emp2 = (0.25 * self.contract('ijab,ijab->', self.t2, self.H.ERI[o, o, v, v])
                         + self.contract('ia,ia->', self.H.F[o, v], self.t1))
        return self.emp2

    # ---- intermediate normalization ----
    # A wavefunction-level quantity (the norm of the intermediate-normalized MP2 wave function),
    # kept here on MPwfn.  The derivative driver's AAT / VG-APT overlaps read it via self.mp; the
    # unrelaxed correlation-density seeds themselves live on MPderiv.

    def _mp2_normalization(self) -> float:
        r"""MP2 intermediate normalization (spin-adapted), for the wave-function-overlap AAT
        (:meth:`MPderiv.atomic_axial_tensors`). The normalized doubles are ``c2 = N t2`` and the
        reference coefficient is ``c0 = N``::

            N = 1 / sqrt(1 + sum_ijab t2_ijab (2 t2_ijab - t2_ijba))

        .. math::

            \begin{aligned}
            N = \Big(1 + \sum_{ijab} t^{ab}_{ij} \, (2 t^{ab}_{ij} - t^{ba}_{ij})\Big)^{-1/2}
            \end{aligned}
        """
        t2 = np.asarray(self.t2)
        norm2 = self.contract('ijab,ijab->', t2, 2.0 * t2 - t2.swapaxes(2, 3))
        return 1.0 / np.sqrt(1.0 + norm2)

    def _so_mp2_normalization(self) -> float:
        r"""Spin-orbital MP2 normalization -- the spin-orbital analogue of
        :meth:`_mp2_normalization` (the ``1/4`` for the antisymmetric double sum). Equal to the
        spin-adapted value on a closed shell::

            N = 1 / sqrt(1 + 1/4 sum_ijab t2_ijab t2_ijab)

        .. math::

            \begin{aligned}
            N = \Big(1 + \tfrac{1}{4} \sum_{ijab} t^{ab}_{ij} t^{ab}_{ij}\Big)^{-1/2}
            \end{aligned}
        """
        t2 = np.asarray(self.t2)
        norm2 = 0.25 * self.contract('ijab,ijab->', t2, t2)
        return 1.0 / np.sqrt(1.0 + norm2)

    # ---- analytic derivative-property driver (see pycc.mpderiv.MPderiv) ----

    @property
    def deriv(self):
        """The cached :class:`~pycc.mpderiv.MPderiv` derivative-property driver for this
        wavefunction (built lazily).  ``MPderiv`` is the MP2 leaf of
        :class:`~pycc.correlatedderivs.CorrelatedDerivs` and carries the analytic
        derivative-property code; the thin property methods below (:meth:`gradient`,
        :meth:`polarizability`, :meth:`hessian`, ...) delegate to it so the historical
        ``mpwfn.<property>()`` call sites keep working, and the :mod:`pycc.properties` facade routes
        through the registry (``pycc/__init__.py``) to the same driver.  A single cached instance,
        so its ``_full_occ_cphf`` / Z-vector caches are shared across the MP2 property calls and the
        ``CCderiv`` cross-calls that reach it via ``cc.mp.deriv``."""
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

    def polarizability(self) -> np.ndarray:
        """MP2 correlation polarizability -- delegates to :meth:`MPderiv.polarizability`."""
        return self.deriv.polarizability()

    def hessian(self) -> np.ndarray:
        """MP2 correlation Hessian -- delegates to :meth:`MPderiv.hessian`."""
        return self.deriv.hessian()

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
