"""Molecular-property facade: uniform access to the analytic derivative properties, each returned
as its additive physical decomposition

    total = nuclear + reference + correlation

Every property is a method on the object that owns it -- a **derivative driver**
(``CCderiv``/``MPderiv``/``CIderiv``, which owns the solve and the correlation-derivative methods) or,
for a reference-only value, a bare ``HFwfn`` (its correlation block is zero).  ``pycc.hessian(x)`` and
the other module-level names are thin wrappers around ``x.hessian()`` etc., kept for discoverability.
Each returns a :class:`PropertyComponents`; the pieces are genuinely computed apart (the correlation
contribution is built from correlation quantities only, never as ``total - reference``).

Dispatch is explicit: each assembler below reads the reference block from the SCF ``HFwfn`` (its own
if the caller *is* an ``HFwfn``, else the driver's cached ``_reference_hf()``) and the correlation
block from the driver's ``_correlation_*`` method (zero for an ``HFwfn``).  No registry, no
name-keyed lookup.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import qcelemental as qcel

from .utils import field

#: Dipole polarizability atomic units (Bohr^3) to Angstrom^3 (= (a0 in Angstrom)^3), and the photon
#: wavelength (nm) per field frequency omega (Eh): ``lambda = NM_PER_EH / omega`` (= 1e7 nm/cm over
#: the Hartree-to-wavenumber relation).  Sourced from qcelemental so they track the same CODATA
#: revision as every other constant psi4 uses.
ANG3_PER_AU = (qcel.constants.get('bohr radius') * 1e10) ** 3
NM_PER_EH = 1e7 / (qcel.constants.get('hartree-inverse meter relationship') / 100.0)
#: Electric dipole moment atomic units (e a0) to Debye.
DEBYE_PER_AU = qcel.constants.conversion_factor('e * bohr', 'debye')


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

    def report(self, name: str, unit: str = "a.u.", params=None, summary=None) -> "PropertyComponents":
        """Print the property as an SCF / correlation / total breakdown and return ``self``
        (so a facade can ``return components.report(...)``).  ``SCF`` is the full SCF-level value
        ``nuclear + reference`` (what a bare Hartree-Fock calculation gives), ``correlation`` the
        post-SCF correction, and ``total`` their sum -- so ``SCF + correlation = total``.

        ``params`` is an optional list of ``(label, value)`` rows printed above the tensor (e.g. the
        field frequency of a polarizability); ``summary`` an optional list printed below it (e.g. the
        isotropic invariant)."""
        scf = self.nuclear + self.reference
        pre = " " * 17

        def fmt(a):
            return np.array2string(np.asarray(a), precision=8, suppress_small=True,
                                   separator=", ", prefix=pre)

        print("\n" + "=" * 70)
        print("%s  (%s)" % (name, unit))
        print("=" * 70)
        for label, value in (params or []):
            print(field(label, value))
        print("  SCF          : " + fmt(scf))
        print("  correlation  : " + fmt(self.correlation))
        print("  total        : " + fmt(self.total))
        for label, value in (summary or []):
            print(field(label, value))
        return self


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
# Property facade: each assembler reads the reference and correlation blocks by explicit method
# call (no registry, no name-keyed dispatch) and packs the PropertyComponents decomposition.  The
# public ``x.hessian()`` etc. methods on HFwfn / the drivers are thin delegators to these.
# ------------------------------------------------------------------------------------------------


def _wavefunction(obj):
    """The underlying wavefunction of ``obj``: ``obj.wfn`` if ``obj`` is a derivative driver, else
    ``obj`` itself (an ``HFwfn``).  The nuclear terms (molecule / nuclear-repulsion derivatives)
    live on the wavefunction regardless of what was handed in."""
    from .correlatedderivs import CorrelatedDerivs
    return obj.wfn if isinstance(obj, CorrelatedDerivs) else obj


def _is_hf(obj) -> bool:
    """True if ``obj`` is a bare ``HFwfn`` (reference-only: its correlation block is zero)."""
    from .hfwfn import HFwfn
    return isinstance(obj, HFwfn)


def _method_label(obj) -> str:
    """The label for a property report -- ``'SCF'``, ``'MP2'``, or the model of the underlying
    wavefunction (``'CCSD'`` / ``'CCSD(T)'`` / ``'CISD'`` / ...)."""
    from .mpwfn import MPwfn
    if _is_hf(obj):
        return "SCF"
    w = _wavefunction(obj)
    if isinstance(w, MPwfn):
        return "MP2"
    return getattr(w, 'model', type(w).__name__)


def _record(obj, key: str, pc: "PropertyComponents") -> "PropertyComponents":
    """Stash a computed property on the underlying wavefunction so a checkpoint can harvest it
    later (see :func:`~pycc.checkpoint.save_checkpoint`): everything computed on a single point
    accumulates together in ``wfn._property_results`` (a name -> PropertyComponents dict).  Returns
    ``pc`` so an assembler can chain ``return _record(...).report(...)``."""
    w = _wavefunction(obj)
    cache = getattr(w, '_property_results', None)
    if cache is None:
        cache = w._property_results = {}
    cache[key] = pc
    return pc


def dipole(wfn) -> PropertyComponents:
    """Electric-dipole moment as a :class:`PropertyComponents` (``nuclear + reference +
    correlation``, shape ``(3,)`` each) for any supported wavefunction type.  The report gives the
    SCF/correlation/total vectors in a.u. and, for the total, the vector and magnitude in Debye."""
    if _is_hf(wfn):
        reference = np.asarray(wfn._dipole_electronic())
        correlation = np.zeros_like(reference)
    else:
        reference = np.asarray(wfn._reference_hf()._dipole_electronic())
        correlation = np.asarray(wfn._correlation_dipole())
    pc = PropertyComponents(_nuclear_dipole(_wavefunction(wfn).ref.molecule()), reference, correlation)
    total = np.asarray(pc.total)
    mag = float(np.linalg.norm(total))
    debye = np.array2string(total * DEBYE_PER_AU, precision=8, suppress_small=True, separator=", ")
    _record(wfn, 'dipole', pc)
    return pc.report(
        "%s dipole moment" % _method_label(wfn),
        summary=[("total (Debye)", debye),
                 ("|mu|", "%.6f a.u.   %.6f Debye" % (mag, mag * DEBYE_PER_AU))])


def gradient(wfn) -> PropertyComponents:
    """Analytic energy gradient as a :class:`PropertyComponents` (``nuclear + reference +
    correlation``, shape ``(natom, 3)`` each).  The nuclear block is the nuclear-repulsion
    derivative ``dV_NN/dX``."""
    if _is_hf(wfn):
        reference = np.asarray(wfn._gradient_electronic())
        correlation = np.zeros_like(reference)
    else:
        reference = np.asarray(wfn._reference_hf()._gradient_electronic())
        correlation = np.asarray(wfn._correlation_gradient())
    nuclear = np.asarray(_wavefunction(wfn).derivatives.nuclear_repulsion())
    pc = PropertyComponents(nuclear, reference, correlation)
    _record(wfn, 'gradient', pc)
    return pc.report("%s gradient" % _method_label(wfn))


def _omega_to_hartree(omega, units):
    """Interpret a response frequency given in ``units`` and return it in hartree (Eh).

    ``units='Eh'`` (aliases ``'hartree'`` / ``'au'``) leaves ``omega`` unchanged; ``units='nm'``
    treats ``omega`` as a wavelength in nm and converts it (``omega_Eh = NM_PER_EH / lambda_nm``),
    which requires a positive wavelength (there is no static ``omega = 0`` wavelength)."""
    u = str(units).lower()
    if u in ('eh', 'hartree', 'au', 'a.u.'):
        return float(omega)
    if u == 'nm':
        if omega <= 0.0:
            raise ValueError(
                "a wavelength in nm must be positive; there is no static (omega=0) wavelength.  "
                "For the static value use the default units='Eh' with omega=0.")
        return NM_PER_EH / float(omega)
    raise ValueError("unknown frequency units %r; use 'Eh' (hartree/au) or 'nm'." % (units,))


def polarizability(wfn, omega: float = 0.0, relaxed: bool = None,
                   units: str = 'Eh') -> PropertyComponents:
    """Electric-dipole polarizability as a :class:`PropertyComponents`, shape ``(3, 3)`` each.  A
    pure electronic response property: the nuclear block is zero.  The report also prints the field
    frequency (in Eh and nm) and the isotropic invariant ``(1/3) Tr(alpha)`` in a.u. and Angstrom^3.

    Two routes, selected by ``omega`` and ``relaxed``:

    * **relaxed** -- the orbital-relaxed analytic second derivative (2n+1 route), the default at
      ``omega = 0``.  Available for MP2 / CISD / CCSD.  Static only: a relaxed value at ``omega != 0``
      is undefined and raises.
    * **unrelaxed** -- the CC linear-response value
      (:meth:`~pycc.ccderiv.CCderiv._correlation_dynamic_polarizability`), which
      omits the orbital (MO) response.  This is the convention for *dynamic* response properties --
      including the orbital relaxation introduces spurious poles -- so it is the default at
      ``omega != 0``.  **CCSD only** (needs a :class:`~pycc.ccderiv.CCderiv`).  Because there is no
      MO response, there is no Hartree-Fock contribution: the value is correlation-only, so the SCF
      block is zero.

    ``omega`` is the external-field frequency, in hartree by default (``0 = static``).  Set
    ``units='nm'`` to give ``omega`` as a wavelength in nm instead (converted internally; a positive
    wavelength only, since the static value has no wavelength).  ``relaxed`` overrides the route:
    the default (``None``) picks relaxed at ``omega = 0`` and unrelaxed otherwise; ``relaxed=False``
    takes the unrelaxed route at ``omega = 0`` too."""
    omega = _omega_to_hartree(omega, units)
    if relaxed is None:
        relaxed = (omega == 0.0)
    if relaxed and omega != 0.0:
        raise ValueError(
            "a relaxed polarizability at omega != 0 is not defined: dynamic response omits the "
            "orbital relaxation (which introduces spurious poles).  Use the default (relaxed=None) "
            "or relaxed=False for the unrelaxed dynamic value.")

    freq = "%.6f Eh   (static)" % omega if omega == 0.0 else \
           "%.6f Eh   (%.2f nm)" % (omega, NM_PER_EH / omega)

    if relaxed:
        if _is_hf(wfn):
            reference = np.asarray(wfn._polarizability_electronic())
            correlation = np.zeros_like(reference)
        else:
            reference = np.asarray(wfn._reference_hf()._polarizability_electronic())
            correlation = np.asarray(wfn._correlation_polarizability())
        pc = PropertyComponents(np.zeros((3, 3)), reference, correlation)
        route = "relaxed (orbital-relaxed derivative)"
    else:
        if not hasattr(wfn, '_correlation_dynamic_polarizability'):
            raise NotImplementedError(
                "the dynamic (unrelaxed) polarizability is implemented for CCSD only; pass a "
                "pycc.CCderiv for a CCSD wavefunction (got %s)." % type(wfn).__name__)
        # Correlation-only: no MO response, hence no HF/SCF contribution (see the docstring).
        correlation = np.asarray(wfn._correlation_dynamic_polarizability(omega))
        pc = PropertyComponents(np.zeros((3, 3)), np.zeros((3, 3)), correlation)
        route = "unrelaxed (no orbital relaxation; correlation only)"

    iso_au = float(np.trace(np.asarray(pc.total)).real) / 3.0
    _record(wfn, 'polarizability' if omega == 0.0 else 'polarizability(omega=%g)' % omega, pc)
    return pc.report(
        "%s polarizability" % _method_label(wfn),
        params=[("frequency (omega)", freq), ("response", route)],
        summary=[("isotropic (1/3 Tr)", "%.6f a.u.   %.6f Angstrom^3" % (iso_au, iso_au * ANG3_PER_AU))])


def optical_rotation(wfn, omega, units: str = 'Eh') -> PropertyComponents:
    """Optical-rotation (optical-activity) tensor ``G'(omega)`` as a :class:`PropertyComponents`,
    shape ``(3, 3)`` -- the unrelaxed CC linear response of the electric dipole to the magnetic
    dipole (:meth:`~pycc.ccderiv.CCderiv.optical_rotation`).

    **CCSD only** (needs a :class:`~pycc.ccderiv.CCderiv`), and ``omega`` must be nonzero -- there is
    no static optical rotation.  Like the dynamic polarizability it omits the orbital (MO) response,
    so it carries no Hartree-Fock contribution: the SCF block is zero and correlation = total.  The
    report prints the field frequency (Eh and nm), ``Tr(G')`` in a.u., and the specific rotation
    ``[alpha]`` in deg/(dm (g/mL)).

    ``omega`` is in hartree by default; set ``units='nm'`` to give it as a wavelength in nm instead
    (converted internally)."""
    omega = _omega_to_hartree(omega, units)
    if omega == 0.0:
        raise ValueError("optical rotation requires a nonzero field frequency; there is no static "
                         "optical rotation.")
    if not hasattr(wfn, '_correlation_optical_rotation'):
        raise NotImplementedError(
            "optical rotation is implemented for CCSD only; pass a pycc.CCderiv for a CCSD "
            "wavefunction (got %s)." % type(wfn).__name__)
    g = np.asarray(wfn._correlation_optical_rotation(omega))
    pc = PropertyComponents(np.zeros((3, 3)), np.zeros((3, 3)), g)
    trace = float(np.trace(g).real)
    alpha = _specific_rotation(trace, omega, _wavefunction(wfn).ref.molecule())
    _record(wfn, "optical_rotation(omega=%g)" % omega, pc)
    return pc.report(
        "%s optical rotation, G'" % _method_label(wfn),
        params=[("frequency (omega)", "%.6f Eh   (%.2f nm)" % (omega, NM_PER_EH / omega)),
                ("response", "unrelaxed (no orbital relaxation; correlation only)")],
        summary=[("Tr(G')", "%.6f a.u." % trace),
                 ("specific rotation", "%.4f deg/(dm (g/mL))" % alpha)])


def hessian(wfn) -> PropertyComponents:
    """Molecular (nuclear) Hessian as a :class:`PropertyComponents` (``nuclear + reference +
    correlation``, shape ``(3*natom, 3*natom)`` each).  The nuclear block is the nuclear-
    repulsion second derivative ``d^2 V_NN/dX dY``."""
    if _is_hf(wfn):
        reference = np.asarray(wfn._hessian_electronic())
        correlation = np.zeros_like(reference)
    else:
        # The driver computes both blocks in one pass, sharing ao_tei_deriv2 between the reference
        # skeleton and the correlation assembly (computed once, not twice).
        reference, correlation = wfn._hessian_blocks()
        reference, correlation = np.asarray(reference), np.asarray(correlation)
    nuclear = np.asarray(_wavefunction(wfn).derivatives.nuclear_repulsion2())
    pc = PropertyComponents(nuclear, reference, correlation)
    _record(wfn, 'hessian', pc)
    return pc.report("%s Hessian" % _method_label(wfn))


def apt(wfn, gauge='length', route='2n+1-field', orbital_gauge='non-canonical') -> PropertyComponents:
    """Atomic polar tensors (nuclear dipole derivatives) as a :class:`PropertyComponents`
    (``nuclear + reference + correlation``, shape ``(natom, 3, 3)`` each), indexed
    ``[A, beta, alpha]``.  The nuclear block is ``Z_A delta_{alpha,beta}``.

    ``gauge='length'`` (default) is the length-gauge APT (correlation block from the driver's
    :meth:`~pycc.correlatedderivs.CorrelatedDerivs._correlation_dipole_derivatives`);
    ``gauge='velocity'`` is the velocity-gauge APT (correlation block from the driver's
    ``_correlation_velocity_dipole_derivatives``).

    ``route`` (length gauge only) selects the algorithm -- ``'2n+1-field'`` (default -- the
    O(N)-cheaper route, 3 field solves) or ``'2n+1-nuclear'``; both give the same tensor.

    ``orbital_gauge`` (velocity gauge only; **expert only**) selects the redundant momentum
    orbital-rotation gauge of the correlation VG APT (MP2 or CISD), ``'non-canonical'`` (default,
    numerically stable) or ``'canonical'``.  The velocity APT is invariant to this choice, so it
    exists only for verification/debugging.
    Both extra knobs are ignored for an ``HFwfn`` (no correlation)."""
    if gauge == 'length':
        if _is_hf(wfn):
            reference = np.asarray(wfn._dipole_derivatives_electronic())
            correlation = np.zeros_like(reference)
        else:
            reference = np.asarray(wfn._reference_hf()._dipole_derivatives_electronic())
            correlation = np.asarray(wfn._correlation_dipole_derivatives(route=route))
    elif gauge == 'velocity':
        if _is_hf(wfn):
            reference = np.asarray(wfn._velocity_dipole_derivatives_electronic())
            correlation = np.zeros_like(reference)
        else:
            reference = np.asarray(wfn._reference_hf()._velocity_dipole_derivatives_electronic())
            correlation = np.asarray(wfn._correlation_velocity_dipole_derivatives(gauge=orbital_gauge))
    else:
        raise ValueError(f"apt: gauge must be 'length' or 'velocity', got {gauge!r}")
    pc = PropertyComponents(_nuclear_apt(_wavefunction(wfn).ref.molecule()), reference, correlation)
    _record(wfn, 'apt' if gauge == 'length' else 'apt_velocity', pc)
    return pc.report("%s APT (%s gauge)" % (_method_label(wfn), gauge))


def aat(wfn, origin=None, orbital_gauge='non-canonical') -> PropertyComponents:
    """Atomic axial tensors (AATs, for VCD) as a :class:`PropertyComponents`
    (``nuclear + reference + correlation``), shape ``(natom, 3, 3)`` each.  Like the other property
    facades, ``wfn`` is a derivative driver (:class:`~pycc.mpderiv.MPderiv` /
    :class:`~pycc.cideriv.CIderiv`) for a correlated method, or an :class:`~pycc.hfwfn.HFwfn` for the
    SCF reference.

    The correlation block is the genuine correlation contribution (the driver's
    ``_correlation_aat``, which excludes the reference density), the reference block is the
    independent SCF AAT (:meth:`~pycc.hfwfn.HFwfn._aat_electronic`), and the nuclear block is
    ``(Z_A/4) eps R``.  Each block is orbital-gauge invariant, so the decomposition is well defined
    independent of the magnetic oo/vv gauge.

    ``origin`` is the coordinate origin for the nuclear term; ``None`` (default) uses the current
    coordinate origin ``(0, 0, 0)`` in the molecule's input frame (honoring ``no_com`` /
    ``no_reorient``).  The electronic AAT's common gauge origin is psi4's (the input-frame
    origin), which matches the default; a non-default ``origin`` shifts only the nuclear term (the
    matching electronic-gauge shift is a planned follow-up).

    ``orbital_gauge`` (**expert only**) selects the redundant magnetic orbital-rotation gauge of
    the correlation AAT (MP2 or CISD), ``'non-canonical'`` (default, numerically stable) or
    ``'canonical'``.  The AAT is invariant to this choice, so it exists only for
    verification/debugging; it is ignored for an ``HFwfn`` (no correlation)."""
    mol = _wavefunction(wfn).ref.molecule()
    nuclear = _nuclear_aat(mol, origin)
    if _is_hf(wfn):
        reference = np.asarray(wfn._aat_electronic())
        correlation = np.zeros_like(reference)
    else:
        reference = np.asarray(wfn._reference_hf()._aat_electronic())
        correlation = np.asarray(wfn._correlation_aat(gauge=orbital_gauge))
    o = (0.0, 0.0, 0.0) if origin is None else tuple(float(x) for x in origin)
    pc = PropertyComponents(nuclear=nuclear, reference=reference, correlation=correlation, origin=o)
    _record(wfn, 'aat', pc)
    return pc.report("%s AAT" % _method_label(wfn))


def _specific_rotation(trace_G, omega, mol):
    r"""Specific rotation ``[alpha]`` in deg/(dm (g/mL)) from the trace of the optical-activity
    tensor ``G'`` (a.u.), the field frequency ``omega`` (Eh), and the molecule (for its molar mass),
    via the Rosenfeld expression (Barron, *Molecular Light Scattering and Optical Activity*, 2nd ed.,
    p. 143; see the specific-rotation-units derivation)::

        [alpha] = -(3.60e4 / 6 pi) (omega/tau) mu0 N_A (e^2 a0^3 / hbar) Tr(G') / M

    ``omega/tau`` is ``omega`` in rad/s (``tau`` = atomic unit of time = hbar/Eh, so Eh/hbar = 1/tau,
    equivalently ``2 pi c / lambda``); ``M`` is the molar mass in kg/mol; ``Tr(G')`` is the full
    trace (the 1/3 of the isotropic average is folded into the 3.60e4).  Constants from qcelemental."""
    tau = qcel.constants.get('atomic unit of time')
    mu0 = qcel.constants.get('mag. constant')
    n_a = qcel.constants.get('Avogadro constant')
    e = qcel.constants.get('elementary charge')
    a0 = qcel.constants.get('Bohr radius')
    hbar = qcel.constants.get('Planck constant over 2 pi')
    m_kg = sum(mol.mass(atom) for atom in range(mol.natom())) * 1e-3
    return -(3.60e4) / (6.0 * np.pi) * (omega / tau) * mu0 * n_a * (e ** 2 * a0 ** 3 / hbar) / m_kg * trace_G
