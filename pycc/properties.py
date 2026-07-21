"""Molecular-property facade: uniform access to the analytic derivative properties, each returned
as its additive physical decomposition

    total = nuclear + reference + correlation

Each property takes a **derivative driver** (``CCderiv``/``MPderiv``/``CIderiv`` -- the object that
owns the solve and the correlation-derivative methods) or, for a reference-only value, a bare
``HFwfn`` (its correlation block is zero).  During the transition to the driver-object API a bare
correlated wavefunction is also accepted and wrapped in its registered driver (the construction the
facade used to do implicitly); this path is temporary and will be removed once callers pass drivers.
The pieces are genuinely computed apart -- the correlation contribution is built from correlation
quantities only, never as (total - reference).
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


# ------------------------------------------------------------------------------------------------
# Nuclear-only contributions (geometry + charge; no wavefunction).  One helper per property whose
# total carries a nuclear term (the gradient/Hessian nuclear terms are the nuclear-repulsion
# derivatives, taken from Derivatives; polarizability has no nuclear term).
# ------------------------------------------------------------------------------------------------

# Levi-Civita symbol eps_{alpha,beta,gamma}
_LEVI_CIVITA = np.zeros((3, 3, 3))
for _a, _b, _g in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI_CIVITA[_a, _b, _g] = 1.0
    _LEVI_CIVITA[_a, _g, _b] = -1.0


def _nuclear_dipole(mol) -> np.ndarray:
    """Nuclear contribution to the electric dipole (a.u.), shape ``(3,)``: ``sum_A Z_A R_A``."""
    geom = np.asarray(mol.geometry().np)
    Z = np.array([mol.Z(A) for A in range(mol.natom())])
    return np.einsum('a,ax->x', Z, geom)


def _nuclear_apt(mol) -> np.ndarray:
    """Nuclear contribution to the APT (length or velocity gauge), shape ``(natom, 3, 3)``
    indexed ``[A, beta, alpha]``: ``Z_A delta_{alpha,beta}``."""
    natom = mol.natom()
    nuc = np.zeros((natom, 3, 3))
    for A in range(natom):
        for a in range(3):
            nuc[A, a, a] = mol.Z(A)
    return nuc


def _nuclear_aat(mol, origin) -> np.ndarray:
    """Nuclear contribution to the AAT (a.u.), shape ``(natom, 3, 3)``::

        I^A_{alpha,beta} = (Z_A / 4) sum_gamma eps_{alpha,beta,gamma} (R_{A,gamma} - O_gamma)

    ``origin`` O (in the molecule's frame, bohr); ``None`` -> the coordinate origin (0, 0, 0)."""
    natom = mol.natom()
    R = np.asarray(mol.geometry().np)                    # (natom, 3), bohr, input frame
    O = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    Z = np.array([mol.Z(A) for A in range(natom)])
    return 0.25 * Z[:, None, None] * np.einsum('abg,Ag->Aab', _LEVI_CIVITA, R - O)


# ------------------------------------------------------------------------------------------------
# Property facade: dispatch on wavefunction type, assemble the PropertyComponents decomposition.
# ------------------------------------------------------------------------------------------------

# Registry mapping a wavefunction class to its downstream derivative-driver class (the strategy
# that carries the correlation-property methods, e.g. CCwfn -> CCderiv).  Populated at import
# (pycc/__init__.py).  A wfn class absent from the registry is treated as carrying its correlation
# methods itself (the current MPwfn, until an MPderiv is split out) -- see _correlated.
_DERIV_REGISTRY: dict = {}


def register_deriv(wfn_cls, deriv_cls) -> None:
    """Register ``deriv_cls`` as the derivative driver for wavefunctions of type ``wfn_cls``."""
    _DERIV_REGISTRY[wfn_cls] = deriv_cls


def _correlated(obj):
    """Resolve ``obj`` to ``(reference_hf, target)``: the SCF-reference ``HFwfn`` carrying the
    reference derivative, and the driver carrying the correlation-derivative methods.

    ``obj`` may be a derivative driver (``CCderiv``/``MPderiv``/``CIderiv``) -- used directly -- or,
    during the transition to the driver-object property API, a correlated wavefunction, which is
    wrapped in its registered driver (``deriv_cls(wfn)``, the same construction the facade did
    implicitly before).  A wavefunction with no registered driver is used as its own target
    (transitional, while a method's derivative code still lives on its wfn class -- CISD)."""
    from .correlatedderivs import CorrelatedDerivs
    if isinstance(obj, CorrelatedDerivs):
        return obj._reference_hf(), obj
    deriv_cls = _DERIV_REGISTRY.get(type(obj))
    if deriv_cls is not None:
        target = deriv_cls(obj)
        return target._reference_hf(), target
    return obj._reference_hf(), obj


def _wfn_of(obj):
    """The underlying wavefunction of ``obj``: ``obj.wfn`` if ``obj`` is a derivative driver, else
    ``obj`` itself (a wavefunction).  Used for the nuclear terms (molecule / nuclear-repulsion
    derivatives), which live on the wavefunction regardless of what the facade was handed."""
    from .correlatedderivs import CorrelatedDerivs
    return obj.wfn if isinstance(obj, CorrelatedDerivs) else obj


def _dispatch(obj, hf_method, corr_method, corr_kwargs=None):
    """Reference (SCF electronic) and correlation blocks of a property, computed apart.  ``obj`` is
    a derivative driver or (transitionally) a correlated wavefunction: the reference is the
    all-electron SCF value and the correlation comes from the driver (:func:`_correlated`), called
    with ``corr_kwargs`` (``route`` / ``gauge`` knobs the SCF reference does not take).  For an
    ``HFwfn`` the reference is the SCF value and the correlation is an all-zeros array."""
    from .hfwfn import HFwfn
    if isinstance(obj, HFwfn):
        reference = np.asarray(getattr(obj, hf_method)())
        return reference, np.zeros_like(reference)
    reference_hf, target = _correlated(obj)
    reference = np.asarray(getattr(reference_hf, hf_method)())
    correlation = np.asarray(getattr(target, corr_method)(**(corr_kwargs or {})))
    return reference, correlation


def dipole(wfn) -> PropertyComponents:
    """Electric-dipole moment as a :class:`PropertyComponents` (``nuclear + reference +
    correlation``, shape ``(3,)`` each) for any supported wavefunction type."""
    reference, correlation = _dispatch(wfn, '_dipole_electronic', 'relaxed_dipole')
    return PropertyComponents(_nuclear_dipole(_wfn_of(wfn).ref.molecule()), reference, correlation)


def gradient(wfn) -> PropertyComponents:
    """Analytic energy gradient as a :class:`PropertyComponents` (``nuclear + reference +
    correlation``, shape ``(natom, 3)`` each).  The nuclear block is the nuclear-repulsion
    derivative ``dV_NN/dX``."""
    reference, correlation = _dispatch(wfn, '_gradient_electronic', 'gradient')
    nuclear = np.asarray(_wfn_of(wfn).derivatives.nuclear_repulsion())
    return PropertyComponents(nuclear, reference, correlation)


def polarizability(wfn, route='2n+1') -> PropertyComponents:
    """Static electric-dipole polarizability as a :class:`PropertyComponents`, shape ``(3, 3)``
    each.  A pure electronic response property: the nuclear block is zero.

    ``route`` selects the MP2 correlation algorithm, ``'2n+1'`` (default -- the O(N)-cheaper 2n+1
    route) or ``'explicit'``; both give the same tensor and it is ignored for an ``HFwfn``."""
    reference, correlation = _dispatch(wfn, 'polarizability', 'polarizability', {'route': route})
    return PropertyComponents(np.zeros((3, 3)), reference, correlation)


def hessian(wfn, route='2n+1') -> PropertyComponents:
    """Molecular (nuclear) Hessian as a :class:`PropertyComponents` (``nuclear + reference +
    correlation``, shape ``(3*natom, 3*natom)`` each).  The nuclear block is the nuclear-
    repulsion second derivative ``d^2 V_NN/dX dY``.

    ``route`` selects the MP2 correlation algorithm, ``'2n+1'`` (default -- the O(N)-cheaper 2n+1
    route) or ``'explicit'``; both give the same matrix and it is ignored for an ``HFwfn``."""
    reference, correlation = _dispatch(wfn, '_hessian_electronic', 'hessian', {'route': route})
    nuclear = np.asarray(_wfn_of(wfn).derivatives.nuclear_repulsion2())
    return PropertyComponents(nuclear, reference, correlation)


def apt(wfn, gauge='length', route='2n+1-field', orbital_gauge='non-canonical') -> PropertyComponents:
    """Atomic polar tensors (nuclear dipole derivatives) as a :class:`PropertyComponents`
    (``nuclear + reference + correlation``, shape ``(natom, 3, 3)`` each), indexed
    ``[A, beta, alpha]``.  The nuclear block is ``Z_A delta_{alpha,beta}``.

    ``gauge='length'`` (default) is the length-gauge APT (:meth:`dipole_derivatives`);
    ``gauge='velocity'`` is the velocity-gauge APT (:meth:`velocity_dipole_derivatives`).

    ``route`` (length gauge only) selects the MP2 correlation algorithm -- ``'2n+1-field'``
    (default -- the O(N)-cheaper route, 3 field solves), ``'2n+1-nuclear'``, or ``'explicit'``;
    all give the same tensor.

    ``orbital_gauge`` (velocity gauge only; **expert only**) selects the redundant momentum
    orbital-rotation gauge, ``'non-canonical'`` (default, numerically stable) or ``'canonical'``.
    The velocity APT is invariant to this choice, so it exists only for verification/debugging.
    Both extra knobs are ignored for an ``HFwfn`` (no correlation)."""
    if gauge == 'length':
        reference, correlation = _dispatch(wfn, '_dipole_derivatives_electronic',
                                           'dipole_derivatives', {'route': route})
    elif gauge == 'velocity':
        reference, correlation = _dispatch(wfn, '_velocity_dipole_derivatives_electronic',
                                           'velocity_dipole_derivatives', {'gauge': orbital_gauge})
    else:
        raise ValueError(f"apt: gauge must be 'length' or 'velocity', got {gauge!r}")
    return PropertyComponents(_nuclear_apt(_wfn_of(wfn).ref.molecule()), reference, correlation)


def aat(wfn, origin=None, orbital_gauge='non-canonical') -> PropertyComponents:
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
    matching electronic-gauge shift is a planned follow-up).

    ``orbital_gauge`` (**expert only**) selects the redundant magnetic orbital-rotation gauge of
    the MP2 correlation AAT, ``'non-canonical'`` (default, numerically stable) or ``'canonical'``.
    The AAT is invariant to this choice, so it exists only for verification/debugging; it is
    ignored for an ``HFwfn`` (no correlation)."""
    from .hfwfn import HFwfn
    from .mpwfn import MPwfn
    from .ciwfn import CIwfn

    mol = wfn.ref.molecule()
    nuclear = _nuclear_aat(mol, origin)
    if isinstance(wfn, MPwfn):
        reference = np.asarray(wfn._reference_hf().atomic_axial_tensors())
        correlation = np.asarray(wfn.atomic_axial_tensors(gauge=orbital_gauge))
    elif isinstance(wfn, HFwfn):
        reference = np.asarray(wfn.atomic_axial_tensors())
        correlation = np.zeros_like(reference)
    elif isinstance(wfn, CIwfn):
        reference = np.asarray(wfn._reference_hf().atomic_axial_tensors())
        correlation = np.asarray(wfn.atomic_axial_tensors(gauge=orbital_gauge))
    else:
        raise TypeError(f"pycc.aat: unsupported wavefunction type {type(wfn).__name__!r}")
    o = (0.0, 0.0, 0.0) if origin is None else tuple(float(x) for x in origin)
    return PropertyComponents(nuclear=nuclear, reference=reference, correlation=correlation, origin=o)
