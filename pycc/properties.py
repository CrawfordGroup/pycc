"""Molecular-property facade: uniform, wavefunction-type-agnostic access to the analytic
derivative properties, each returned as its additive physical decomposition

    total = nuclear + reference + correlation

The same call works for any supported wavefunction (``HFwfn``, ``MPwfn``, ...): the correlation
block is simply zero for an SCF wavefunction.  The pieces are genuinely computed apart -- the
correlation contribution is built from correlation quantities only, never as (total - reference).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PropertyComponents:
    """Additive decomposition of a molecular-property tensor::

        total = nuclear + reference + correlation

    all the same shape.  ``reference`` is the SCF electronic contribution, ``correlation`` the
    post-SCF (e.g. MP2) correction -- an all-zeros array for an SCF wavefunction.  ``origin``
    records the coordinate/gauge origin used by origin-dependent properties (the AAT), else
    ``None``.

    Access is by name (never by position): ``.total`` and ``.electronic`` are derived so they
    can never drift out of sync with the stored pieces; ``.scf`` and ``.hf`` are aliases of
    ``.reference``.
    """

    nuclear: np.ndarray
    reference: np.ndarray
    correlation: np.ndarray
    origin: Optional[Tuple[float, float, float]] = None

    @property
    def total(self) -> np.ndarray:
        """nuclear + reference + correlation."""
        return self.nuclear + self.reference + self.correlation

    @property
    def electronic(self) -> np.ndarray:
        """reference + correlation (the full electronic contribution)."""
        return self.reference + self.correlation

    @property
    def scf(self) -> np.ndarray:
        """Alias of :attr:`reference`."""
        return self.reference

    @property
    def hf(self) -> np.ndarray:
        """Alias of :attr:`reference`."""
        return self.reference


# Levi-Civita symbol eps_{alpha,beta,gamma}
_LEVI_CIVITA = np.zeros((3, 3, 3))
for _a, _b, _g in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI_CIVITA[_a, _b, _g] = 1.0
    _LEVI_CIVITA[_a, _g, _b] = -1.0


def _nuclear_aat(mol, origin) -> np.ndarray:
    """Nuclear contribution to the AAT (a.u.), shape ``(natom, 3, 3)``::

        I^A_{alpha,beta} = (Z_A / 4) sum_gamma eps_{alpha,beta,gamma} (R_{A,gamma} - O_gamma)

    ``origin`` O (in the molecule's frame, bohr); ``None`` -> the coordinate origin (0, 0, 0)."""
    natom = mol.natom()
    R = np.asarray(mol.geometry().np)                    # (natom, 3), bohr, input frame
    O = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    Z = np.array([mol.Z(A) for A in range(natom)])
    return 0.25 * Z[:, None, None] * np.einsum('abg,Ag->Aab', _LEVI_CIVITA, R - O)


def aat(wfn, origin=None) -> PropertyComponents:
    """Atomic axial tensors (AATs, for VCD) as a :class:`PropertyComponents`
    (``nuclear + reference + correlation``), shape ``(natom, 3, 3)`` each, for any supported
    wavefunction type.

    The correlation block is the genuine correlation contribution
    (:meth:`MPwfn.atomic_axial_tensors`, which excludes the reference density), the reference
    block is the independent SCF AAT (:meth:`HFwfn.atomic_axial_tensors`), and the nuclear block
    is ``(Z_A/4) eps R``.  Each block is orbital-gauge invariant, so the decomposition is well
    defined independent of the magnetic oo/vv gauge.

    ``origin`` is the coordinate origin for the nuclear term; ``None`` (default) uses the current
    coordinate origin ``(0, 0, 0)`` in the molecule's input frame (honoring ``no_com`` /
    ``no_reorient``).  The electronic AAT's common gauge origin is psi4's (the input-frame
    origin), which matches the default; a non-default ``origin`` shifts only the nuclear term (the
    matching electronic-gauge shift is a planned follow-up)."""
    from .hfwfn import HFwfn
    from .mpwfn import MPwfn

    mol = wfn.ref.molecule()
    nuclear = _nuclear_aat(mol, origin)
    if isinstance(wfn, MPwfn):
        reference = np.asarray(wfn._reference_hf().atomic_axial_tensors())
        correlation = np.asarray(wfn.atomic_axial_tensors())
    elif isinstance(wfn, HFwfn):
        reference = np.asarray(wfn.atomic_axial_tensors())
        correlation = np.zeros_like(reference)
    else:
        raise TypeError(f"pycc.aat: unsupported wavefunction type {type(wfn).__name__!r}")
    o = (0.0, 0.0, 0.0) if origin is None else tuple(float(x) for x in origin)
    return PropertyComponents(nuclear=nuclear, reference=reference, correlation=correlation, origin=o)
