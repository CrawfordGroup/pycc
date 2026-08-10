"""
vibanalysis.py: harmonic vibrational analysis from a molecular Hessian.

Steps so far:
  * mass-weight the Cartesian Hessian, project out the rigid-body (translation/rotation)
    degrees of freedom, diagonalize, and return the harmonic frequencies and mass-weighted
    normal modes.

IR/VCD intensities and the frequency report are added in later steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import qcelemental as qcel

#: Conversion of a mass-weighted-Hessian eigenvalue ``k`` (Eh / (a0^2 u)) to a harmonic frequency
#: in cm^-1: ``nu[cm^-1] = sqrt(k) * FREQ_CM_1_PER_SQRT_AU``.  Assembled from qcelemental so it
#: tracks the same CODATA revision as every other pycc constant; machine-identical to psi4's
#: ``sqrt(N_A * Eh_in_J * 1e19) / (2 pi c a0_Angstrom)``.
_NA = qcel.constants.get('Avogadro constant')
_EH_J = qcel.constants.conversion_factor('hartree', 'joule')
_C_M_S = qcel.constants.get('speed of light in vacuum')       # m/s
_A0_ANG = qcel.constants.get('bohr radius') * 1e10            # Angstrom
FREQ_CM_1_PER_SQRT_AU = np.sqrt(_NA * _EH_J * 1e19) / (2.0 * np.pi * _C_M_S * _A0_ANG)

#: Conversion of an IR intensity ``sum_alpha (d mu_alpha / d Q)^2`` (a.u., Q the mass-weighted
#: normal coordinate in u) to km/mol: ``I[km/mol] = (d mu/d Q)^2 * IR_KM_MOL_PER_AU``.  Sourced
#: from qcelemental; identical to psi4's
#: ``N_A pi 1e-3 (m_e in u) alpha^2 a_0 / 3``.
IR_KM_MOL_PER_AU = (_NA * np.pi * 1e-3 * qcel.constants.get('electron mass in u')
                    * qcel.constants.get('fine-structure constant') ** 2
                    * qcel.constants.get('atomic unit of length') / 3.0)

#: Conversion of a normal-coordinate force constant ``mu * nu^2`` (reduced mass in u, frequency in
#: cm^-1) to milli-dyne/Angstrom: ``k[mDyne/A] = mu * nu^2 * FORCE_CONST_MDYNE_A_PER_U_CM2``.
#: Sourced from qcelemental; identical to psi4's ``0.1 (2 pi c)^2 / N_A``.
FORCE_CONST_MDYNE_A_PER_U_CM2 = 0.1 * (2.0 * np.pi * _C_M_S) ** 2 / _NA

#: Conversion of a VCD rotatory strength (the Stephens APT*AAT product, with the normal modes
#: mass-weighted in u as everywhere here) from atomic units to 10^-44 esu^2 cm^2.  Same expression
#: as magpy's ``normal.py`` conv_vcd, but with the atomic mass unit in place of m_e -- pycc
#: mass-weights the modes in u, not m_e, so the a.u. product carries u: ``(e^2 hbar a0)/(u c)
#: (1000 c)^2 10^44``.  Sourced from qcelemental.
_HBAR_J_S = qcel.constants.get('Planck constant') / (2.0 * np.pi)
_E_COULOMB = qcel.constants.get('atomic unit of charge')
_A0_M = qcel.constants.get('bohr radius')
_U_KG = 1.0 / (1000.0 * _NA)                                   # atomic mass unit (kg), = magpy's _u
VCD_ROT_STRENGTH_CGS_PER_AU = ((_E_COULOMB ** 2 * _HBAR_J_S * _A0_M) / (_U_KG * _C_M_S)
                               * (1000.0 * _C_M_S) ** 2 * 1e44)


def _print_report(frequencies: np.ndarray, ir_intensities, rotatory_strengths,
                  force_constants: np.ndarray, reduced_masses: np.ndarray,
                  n_atom: int, n_trans: int, n_rot: int) -> None:
    """Print the harmonic-analysis table: all ``3N`` modes, highest wavenumber to lowest, numbered
    ``1..3N``.  An imaginary frequency (stored as a negative real) is rendered ``|nu|i``; its force
    constant is negative.  The IR column is blank when no APT was supplied; the rotatory-strength
    column appears only when an AAT was supplied (VCD).  Layout follows the pycc property-report
    style (see :meth:`PropertyComponents.report`)."""
    n_mode = frequencies.shape[0]
    n_tr = n_trans + n_rot
    has_rot = rotatory_strengths is not None
    width = 86 if has_rot else 70

    print("\n" + "=" * width)
    print("Harmonic Vibrational Analysis")
    print("=" * width)
    print("  atoms       : %3d" % n_atom)
    print("  modes       : %3d" % n_mode)
    if n_tr:
        print("  projected   : %3d  (%d translation, %d rotation)" % (n_tr, n_trans, n_rot))
    else:
        print("  projected   :   0")
    print("")

    h1 = "   Mode    Frequency   IR Intensity"
    h2 = "            [cm^-1]      [km/mol]   "
    if has_rot:
        h1 += "   Rot. Strength"
        h2 += "   [1e-44 cgs] "
    h1 += "   Force Constant   Reduced Mass"
    h2 += "     [mDyne/A]          [u]"
    print(h1)
    print(h2)
    print("  " + "-" * (width - 4))

    for k in range(n_mode):
        nu = frequencies[k]
        freq_str = ("%9.2fi" % abs(nu)) if nu < 0.0 else ("%9.2f " % nu)
        ir_str = "  %12.4f" % ir_intensities[k] if ir_intensities is not None else "  " + " " * 12
        rot_str = "  %13.4f" % rotatory_strengths[k] if has_rot else ""
        print("   %4d  %s%s%s   %14.4f  %13.4f"
              % (k + 1, freq_str, ir_str, rot_str, force_constants[k], reduced_masses[k]))
    print("=" * width)


def _translation_rotation_projector(masses: np.ndarray, geom: np.ndarray,
                                    project_trans: bool, project_rot: bool,
                                    linear_tol: float):
    r"""Orthogonal projector that removes the rigid-body (translation/rotation) subspace from the
    mass-weighted coordinate space (the analytic Eckart projector).

    Working in mass-weighted coordinates with atoms at the center of mass, the three translation
    vectors ``T_a`` and three rotation vectors ``R_a`` (component of atom ``i`` along Cartesian
    ``alpha``) are::

        T_a[i, alpha] = sqrt(m_i) delta_{a alpha}
        R_a[i, alpha] = sqrt(m_i) (e_a x r_i)_alpha

    Their Gram matrices are exactly the total mass and the moment-of-inertia tensor::

        <T_a|T_b> = M delta_{ab},     <R_a|R_b> = I_{ab}

    so the orthogonal projector onto each subspace is ``|v> G^{-1} <v|`` with the appropriate Gram
    matrix ``G``.  At the center of mass ``T`` and ``R`` are mutually orthogonal, so the only
    possible rank deficiency is a null moment of inertia (a linear molecule), handled by the
    Moore-Penrose pseudo-inverse ``I^+`` (invert the principal moments above ``linear_tol`` times
    the largest, zero the rest -- the null rotation vector vanishes there anyway)::

        P = 1 - (1/M) |T><T| - |R> I^+ <R|

    .. math::

        \begin{aligned}
        \langle T_a|T_b\rangle &= M\,\delta_{ab}, &
        \langle R_a|R_b\rangle &= I_{ab} = \sum_i m_i\,(r_i^2\,\delta_{ab} - r_{ia} r_{ib}) \\
        P &= \mathbb{1} - \tfrac{1}{M}\,|T\rangle\langle T| - |R\rangle\, I^{+}\,\langle R|
        \end{aligned}

    Parameters
    ----------
    masses : np.ndarray
        atomic masses, shape ``(natom,)``, u.
    geom : np.ndarray
        Cartesian coordinates, shape ``(natom, 3)``, bohr (any origin; centered here internally).
    project_trans, project_rot : bool
        whether to remove translations / rotations.
    linear_tol : float
        relative cutoff on the principal moments of inertia: an axis with
        ``lambda_a / lambda_max <= linear_tol`` is treated as a null rotation (drops the molecule
        from 6 to 5 rigid-body modes for a linear geometry).

    Returns
    -------
    P : np.ndarray
        the ``(3N, 3N)`` orthogonal projector (symmetric, idempotent).
    n_tr : int
        the number of rigid-body degrees of freedom removed.
    """
    natom = masses.shape[0]
    total_mass = masses.sum()
    sqrt_m = np.sqrt(masses)

    # Coordinates relative to the center of mass (required for the T/R orthogonality and for I).
    com = (masses[:, None] * geom).sum(axis=0) / total_mass
    r = geom - com

    P = np.identity(3 * natom)
    n_tr = 0

    # Translation projector: (1/M) |T><T|, with T_a[i, alpha] = sqrt(m_i) delta_{a alpha}.
    if project_trans:
        T = np.zeros((3 * natom, 3))
        for i in range(natom):
            T[3 * i:3 * i + 3, :] = sqrt_m[i] * np.identity(3)
        P -= (T @ T.T) / total_mass
        n_tr += 3

    # Rotation projector: |R> I^+ <R|, with R the rigid-rotation vectors and I their Gram matrix.
    if project_rot:
        R = np.zeros((3 * natom, 3))
        inertia = np.zeros((3, 3))
        for i in range(natom):
            x, y, z = r[i]
            # block columns a = x, y, z hold sqrt(m_i)(e_a x r_i)
            R[3 * i:3 * i + 3, :] = sqrt_m[i] * np.array([[0.0, z, -y],
                                                          [-z, 0.0, x],
                                                          [y, -x, 0.0]])
            inertia += masses[i] * (r[i] @ r[i] * np.identity(3) - np.outer(r[i], r[i]))

        moments, axes = np.linalg.eigh(inertia)
        largest = moments.max() if natom > 1 else 0.0
        keep = moments > linear_tol * largest if largest > 0.0 else np.zeros(3, dtype=bool)
        inv_moments = np.where(keep, 1.0 / np.where(keep, moments, 1.0), 0.0)
        inertia_pinv = axes @ np.diag(inv_moments) @ axes.T
        P -= R @ inertia_pinv @ R.T
        n_tr += int(keep.sum())

    return P, n_tr


def harmonic_analysis(source: Any, hessian: np.ndarray = None, apt: np.ndarray = None,
                      aat: np.ndarray = None, project_trans: bool = False,
                      project_rot: bool = False, linear_tol: float = 1e-8) -> dict:
    r"""Harmonic frequencies, normal modes, and (with an APT) IR intensities and (with an APT and
    AAT) VCD rotatory strengths from a Cartesian Hessian.

    Optionally project the rigid-body degrees of freedom out of the mass-weighted Hessian,
    diagonalize, and convert the eigenvalues to frequencies.  Projection is **off by default**:
    the raw analysis exposes a loose geometry (rotations appear as finite or imaginary low modes)
    rather than silently cleaning it up.  Opt in with ``project_trans`` / ``project_rot`` when the
    geometry is knowingly not at a tight stationary point; the residual translation/rotation
    contamination is then removed instead of leaking into the low modes.

    The mass-weighted eigenvectors ``q`` are un-mass-weighted to Cartesian displacements and
    normalized, and the reduced masses follow from their norm::

        wL = M^{-1/2} q          (un-mass-weighted, a0)
        mu = 1 / ||wL||^2        (reduced mass, u)
        xL = sqrt(mu) wL         (normalized displacement, a0)

    With an atomic polar tensor ``apt``, the dipole derivative along each normal coordinate gives
    the IR intensity (``dipder[alpha, 3A+beta] = d mu_alpha / d X_{A beta}``)::

        d mu_alpha / d Q_k = sum_{A beta} dipder[alpha, 3A+beta] wL[3A+beta, k]
        I_k = sum_alpha (d mu_alpha / d Q_k)^2 * IR_KM_MOL_PER_AU

    With both an APT and an atomic axial tensor ``aat`` (magnetic dipole gradient
    ``mdip[beta, 3A+alpha] = d m_beta / d X_{A alpha}``), the VCD rotatory strength follows from
    Stephens' theory as the electric.magnetic transition-moment product (the harmonic frequency
    factors cancel, and the AAT already carries the ``Im`` with its ``i`` stripped)::

        d m_beta / d Q_k = sum_{A alpha} mdip[beta, 3A+alpha] wL[3A+alpha, k]
        R_k = sum_c (d mu_c / d Q_k) (d m_c / d Q_k) * VCD_ROT_STRENGTH_CGS_PER_AU

    .. math::

        R_k = \sum_c \Big(\frac{\partial \mu_c}{\partial Q_k}\Big)
                     \Big(\frac{\partial m_c}{\partial Q_k}\Big)

    With ``M = diag(m_1, m_1, m_1, m_2, ...)`` the atomic-mass matrix over the ``3N`` Cartesian
    coordinates and ``P`` the translation/rotation projector
    (:func:`_translation_rotation_projector`), the mass-weighted Hessian, its projection, its
    eigenpairs ``(k, q)``, and the frequencies are::

        Htilde = M^{-1/2} H M^{-1/2}
        Htilde_proj = P Htilde P
        Htilde_proj q = k q
        nu[cm^-1] = sqrt(k) * FREQ_CM_1_PER_SQRT_AU

    .. math::

        \begin{aligned}
        \tilde{H} &= M^{-1/2} H M^{-1/2}, \qquad \tilde{H}_{\text{proj}} = P\,\tilde{H}\,P \\
        \tilde{H}_{\text{proj}}\, q &= k\, q \\
        \tilde{\nu}\,[\mathrm{cm}^{-1}] &= \sqrt{k}\;\; C, \qquad
            C = \frac{\sqrt{N_A\, E_h\, 10^{19}}}{2\pi c\, a_0}
        \end{aligned}

    A negative eigenvalue ``k`` gives an imaginary frequency, stored as ``-sqrt(-k) C`` (a negative
    real; rendered ``|nu| i`` on output).

    Parameters
    ----------
    source : psi4.core.Molecule | str | Checkpoint
        either the molecule (supplying masses in u, coordinates in bohr, and atom count) with the
        ``hessian`` (and optional ``apt``) passed alongside, **or** a checkpoint -- a ``.npz`` path
        or a loaded :class:`~pycc.checkpoint.Checkpoint` -- from which the molecule, Hessian, and
        APT are drawn (explicit ``hessian`` / ``apt`` arguments still override).
    hessian : np.ndarray, optional
        Cartesian Hessian, shape ``(3N, 3N)``, a.u. (Eh / a0^2), e.g. ``pycc.hessian(d).total``.
        Required unless ``source`` is a checkpoint that carries one.
    apt : np.ndarray, optional
        atomic polar tensor (length-gauge dipole derivatives), shape ``(natom, 3, 3)`` indexed
        ``[A, beta, alpha] = d mu_alpha / d X_{A beta}``, a.u., e.g. ``pycc.apt(d).total``.  When
        given, IR intensities are computed; when omitted, ``ir_intensities`` is ``None``.
    aat : np.ndarray, optional
        atomic axial tensor, shape ``(natom, 3, 3)`` indexed ``[A, alpha, beta]`` with ``alpha``
        the nuclear Cartesian and ``beta`` the magnetic component, a.u., e.g. ``pycc.aat(d).total``.
        When given together with ``apt``, VCD rotatory strengths are computed; otherwise
        ``rotatory_strengths`` is ``None``.  Pulled from a checkpoint ``source`` when present.
    project_trans, project_rot : bool
        remove translations / rotations before diagonalizing (both off by default; opt in for a
        loose geometry).
    linear_tol : float
        relative cutoff on the principal moments of inertia used to detect a linear molecule
        (see :func:`_translation_rotation_projector`).

    Prints the frequency table (all ``3N`` modes, highest wavenumber to lowest) and returns a dict
    of the underlying arrays -- all ordered high-to-low to match the printed numbering -- with keys:
    ``frequencies`` (cm^-1), ``ir_intensities`` (km/mol, or ``None``), ``rotatory_strengths``
    (10^-44 esu^2 cm^2, or ``None`` if no AAT), ``force_constants`` (mDyne/A), ``reduced_masses``
    (u), ``modes`` (mass-weighted, columns), ``cartesian_modes`` (normalized un-mass-weighted
    displacements, columns), and ``n_tr`` (rigid-body modes removed).

    Returns
    -------
    dict
        the arrays above, for programmatic use (e.g. the CFOUR/DALTON-anchored tests).
    """
    # Resolve a checkpoint source (path or Checkpoint) into molecule / hessian / apt; a plain
    # molecule passes through with the hessian/apt given alongside.
    from .checkpoint import Checkpoint, load_checkpoint
    if isinstance(source, str):
        source = load_checkpoint(source)
    if isinstance(source, Checkpoint):
        molecule = source.molecule
        if hessian is None:
            hessian = source.hessian
        if apt is None:
            apt = source.apt
        if aat is None:
            aat = source.aat
    else:
        molecule = source
    if hessian is None:
        raise ValueError(
            "harmonic_analysis needs a Hessian: pass one, or a checkpoint that stores one")

    natom = molecule.natom()
    masses = np.array([molecule.mass(A) for A in range(natom)])
    geom = np.asarray(molecule.geometry())

    if hessian.shape != (3 * natom, 3 * natom):
        raise ValueError(
            "hessian shape %s does not match 3*natom = %d" % (hessian.shape, 3 * natom))

    # Mass-weight with M^{-1/2} (each atom's mass repeated over x, y, z).
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses, 3))
    mw_hessian = inv_sqrt_m[:, None] * hessian * inv_sqrt_m[None, :]

    # Project out the rigid-body subspace, then diagonalize.
    P, n_tr = _translation_rotation_projector(masses, geom, project_trans, project_rot, linear_tol)
    mw_hessian = P @ mw_hessian @ P
    eigenvalues, modes = np.linalg.eigh(mw_hessian)

    # Frequencies: real from k >= 0, imaginary (k < 0) stored as a negative real.
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * FREQ_CM_1_PER_SQRT_AU

    # Un-mass-weight the modes (wL, a0), reduced masses (u), normalized displacements (xL, a0).
    cart_modes = inv_sqrt_m[:, None] * modes
    reduced_masses = 1.0 / np.linalg.norm(cart_modes, axis=0) ** 2
    cartesian_modes = np.sqrt(reduced_masses) * cart_modes

    # Force constants along the normal coordinates (mDyne/A), signed by the eigenvalue so an
    # imaginary mode carries a negative force constant.
    force_constants = (np.sign(eigenvalues) * reduced_masses * frequencies ** 2
                       * FORCE_CONST_MDYNE_A_PER_U_CM2)

    # IR intensities from the APT (electric-dipole gradient along the normal coordinates), and VCD
    # rotatory strengths when an AAT is also supplied (magnetic-dipole gradient dotted with it).
    ir_intensities = None
    rotatory_strengths = None
    if apt is not None:
        dipder = apt.transpose(2, 0, 1).reshape(3, 3 * natom)     # [dipole, 3A+nuclear]
        dmu_dQ = dipder @ cart_modes                              # [dipole, mode] = d mu_c / d Q
        ir_intensities = np.einsum('cm,cm->m', dmu_dQ, dmu_dQ) * IR_KM_MOL_PER_AU
        if aat is not None:
            mdip = aat.transpose(2, 0, 1).reshape(3, 3 * natom)   # [magnetic, 3A+nuclear]
            dm_dQ = mdip @ cart_modes                             # [magnetic, mode] = d m_c / d Q
            rotatory_strengths = np.einsum('cm,cm->m', dmu_dQ, dm_dQ) * VCD_ROT_STRENGTH_CGS_PER_AU

    # Reorder everything highest wavenumber to lowest (matches the printed mode numbering).
    order = np.argsort(frequencies)[::-1]
    frequencies = frequencies[order]
    force_constants = force_constants[order]
    reduced_masses = reduced_masses[order]
    modes = modes[:, order]
    cartesian_modes = cartesian_modes[:, order]
    if ir_intensities is not None:
        ir_intensities = ir_intensities[order]
    if rotatory_strengths is not None:
        rotatory_strengths = rotatory_strengths[order]

    n_trans = 3 if project_trans else 0
    _print_report(frequencies, ir_intensities, rotatory_strengths, force_constants,
                  reduced_masses, natom, n_trans, n_tr - n_trans)

    return {
        'frequencies': frequencies,
        'ir_intensities': ir_intensities,
        'rotatory_strengths': rotatory_strengths,
        'force_constants': force_constants,
        'reduced_masses': reduced_masses,
        'modes': modes,
        'cartesian_modes': cartesian_modes,
        'n_tr': n_tr,
    }


# ------------------------------------------------------------------------------------------------
# one-call spectrum drivers: compute the needed properties (printing their reports), optionally
# archive everything to a checkpoint, then run the harmonic analysis
# ------------------------------------------------------------------------------------------------

def _checkpoint_for_driver(source: Any, driver: str, need_apt: bool = False,
                           need_aat: bool = False):
    """Resolve a checkpoint ``source`` (path or :class:`~pycc.checkpoint.Checkpoint`) for the
    ir/vcd re-analysis path.  A missing Hessian is fatal (no vibrational analysis is possible); a
    missing spectrum component the driver needs (APT for IR, AAT for VCD) is a warning printed to
    the output -- the analysis proceeds with that component as ``None`` (frequencies-only for a
    missing APT).  Returns the loaded :class:`~pycc.checkpoint.Checkpoint`."""
    from .checkpoint import load_checkpoint
    ckpt = load_checkpoint(source) if isinstance(source, str) else source
    label = "%r" % source if isinstance(source, str) else "the checkpoint"
    if ckpt.hessian is None:
        raise ValueError(
            "%s(): %s has no Hessian; a vibrational analysis requires one." % (driver, label))
    if need_apt and ckpt.apt is None:
        print("  ** WARNING (%s): %s has no APT; running a frequencies-only analysis "
              "(IR intensities omitted). **" % (driver, label))
    if need_aat and ckpt.aat is None:
        print("  ** WARNING (%s): %s has no AAT; VCD rotatory strengths are unavailable. **"
              % (driver, label))
    return ckpt


def ir(source: Any, checkpoint: str = None, project_trans: bool = False,
       project_rot: bool = False, linear_tol: float = 1e-8) -> dict:
    """IR driver: the one-call path to a printed IR spectrum, from either a live driver or a
    checkpoint.

    * ``source`` is a **derivative driver** (``CCderiv``/``MPderiv``/``CIderiv``): compute the
      molecular Hessian and the length-gauge APT (each printing its property report and recording
      itself on the wavefunction), optionally archive everything to the ``checkpoint`` path, then
      run and return :func:`harmonic_analysis`::

          pycc.ir(pycc.CCderiv(cc), checkpoint="mol.npz")

    * ``source`` is a **checkpoint** (a ``.npz`` path or a loaded
      :class:`~pycc.checkpoint.Checkpoint`): nothing is computed -- the molecule, Hessian, and APT
      are read back and :func:`harmonic_analysis` is run on them (the ``checkpoint`` output
      argument is ignored)::

          pycc.ir("mol.npz")

    Parameters
    ----------
    source : CCderiv | MPderiv | CIderiv | str | Checkpoint
        the derivative-property driver to compute from, or a checkpoint to re-analyze.
    checkpoint : str, optional
        (driver source only) if given, ``save_checkpoint`` writes every property recorded so far
        here.
    project_trans, project_rot, linear_tol
        passed through to :func:`harmonic_analysis`.

    Returns
    -------
    dict
        the :func:`harmonic_analysis` result.
    """
    from .checkpoint import Checkpoint
    if isinstance(source, (str, Checkpoint)):
        ckpt = _checkpoint_for_driver(source, 'ir', need_apt=True)
        return harmonic_analysis(ckpt, project_trans=project_trans,
                                 project_rot=project_rot, linear_tol=linear_tol)

    from . import properties
    deriv = source
    hess = np.asarray(properties.hessian(deriv).total)
    apt = np.asarray(properties.apt(deriv).total)
    molecule = properties._wfn_of(deriv).ref.molecule()

    if checkpoint is not None:
        from .checkpoint import save_checkpoint
        save_checkpoint(deriv, checkpoint)

    return harmonic_analysis(molecule, hess, apt=apt, project_trans=project_trans,
                             project_rot=project_rot, linear_tol=linear_tol)


def vcd(source: Any, checkpoint: str = None, project_trans: bool = False,
        project_rot: bool = False, linear_tol: float = 1e-8) -> dict:
    """VCD driver: like :func:`ir`, but also computes and archives the atomic axial tensors (AATs)
    needed for VCD rotatory strengths.

    * ``source`` is a **derivative driver**: compute the Hessian, length-gauge APT, and AAT (each
      printing its report and recording itself), optionally archive everything to the
      ``checkpoint`` path, then run and return :func:`harmonic_analysis`.  A checkpoint written by
      ``vcd`` therefore carries everything the VCD analysis needs.
    * ``source`` is a **checkpoint** (path or :class:`~pycc.checkpoint.Checkpoint`): nothing is
      computed; the stored data is re-analyzed via :func:`harmonic_analysis`.

    The rotatory strengths (Stephens' APT*AAT contraction) are validated against DALTON to
    ~5e-4 (10^-44 esu^2 cm^2) for H2O2/STO-3G at the SCF level.  AATs exist for SCF / MP2 / CISD
    (not yet CCSD), so the compute path of ``vcd`` currently requires an HF, MP2, or CISD source.

    Parameters and return are as for :func:`ir`.
    """
    from .checkpoint import Checkpoint
    if isinstance(source, (str, Checkpoint)):
        ckpt = _checkpoint_for_driver(source, 'vcd', need_apt=True, need_aat=True)
        return harmonic_analysis(ckpt, project_trans=project_trans,
                                 project_rot=project_rot, linear_tol=linear_tol)

    from . import properties
    deriv = source
    hess = np.asarray(properties.hessian(deriv).total)
    apt = np.asarray(properties.apt(deriv).total)
    aat = np.asarray(properties.aat(deriv).total)
    molecule = properties._wfn_of(deriv).ref.molecule()

    if checkpoint is not None:
        from .checkpoint import save_checkpoint
        save_checkpoint(deriv, checkpoint)

    return harmonic_analysis(molecule, hess, apt=apt, aat=aat, project_trans=project_trans,
                             project_rot=project_rot, linear_tol=linear_tol)
