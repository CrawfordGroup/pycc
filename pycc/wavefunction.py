"""
wavefunction.py: top-level Wavefunction base class.

Owns the infrastructure shared by every correlated (and SCF-derived) method:
the Psi4 reference, the orbital spaces (full MO basis, with active occupied/virtual
slices that offset past any frozen core), the MO coefficients (optionally with
localized occupied orbitals), the full-MO-basis integrals (Hamiltonian), and the
device/precision manager. Method-specific machinery (amplitudes, denominators,
densities, response, ...) lives in the subclasses (CCwfn, MPwfn, HFwfn, and --
planned -- CIwfn).

Part of the 2026-06 refactor (docs/archive/REFACTOR_PLAN_2026-06.md, Phase 3): the
reference/orbital/integral setup and the DeviceManager were lifted out of
ccwfn.__init__ so PyCC can host more than coupled cluster.
"""

from __future__ import annotations

from typing import Any

import psi4
import numpy as np

from .hamiltonian import Hamiltonian, SpinOrbitalHamiltonian
from .device import DeviceManager
from .derivatives import Derivatives
from .exceptions import InvalidKeywordError, PyCCError


class Wavefunction(object):
    """Reference + orbital + integral + device infrastructure for a wavefunction.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction
        the reference wave function (from Psi4's energy() method)
    eref : float
        reference energy (including nuclear repulsion)
    orbital_basis : str
        ``'spatial'`` (spin-adapted, closed-shell RHF) or ``'spinorbital'``
        (UHF/ROHF). Resolved from the reference (or an explicit override) and read
        by the method subclasses to select their spatial vs spin-orbital kernels.
        See :meth:`_init_spatial` / :meth:`_init_spinorbital` for what each path
        builds.
    nfzc, no, nv, nmo, nact : int
        frozen-core / active-occupied / active-virtual / total-MO / active counts.
        In the spin-orbital path these are spin-orbital counts (``nmo`` is twice the
        number of spatial MOs, etc.).
    o, v : slice
        active occupied and virtual subspaces. Spatial path: offsets into the full
        MO space, ``o = slice(nfzc, nfzc+no)`` (skips the frozen core), ``v`` the
        virtuals. Spin-orbital path: contiguous slices into the active space,
        ``o = slice(0, no)`` and ``v = slice(no, nact)`` (the frozen core is already
        excluded via the ACTIVE orbital subset).
    C : Psi4 Matrix
        (spatial path) full-space MO coefficients (all ``nmo`` columns, global
        energy order; the frozen core occupies columns ``[0:nfzc]``). The active
        occupied block is localized in place if ``localize_occ``.
    Ca, Cb : Psi4 Matrix
        (spin-orbital path) active alpha/beta MO coefficients.
    H : Hamiltonian or SpinOrbitalHamiltonian
        full-MO-basis Fock (F), ERIs, and property integrals, device/precision-seeded.
        The spatial ``Hamiltonian`` additionally carries the spin-adapted ERIs (L);
        the ``SpinOrbitalHamiltonian`` has none (its ERI is the antisymmetrized
        <pq||rs>).
    derivatives : Derivatives
        lazy MO-basis derivative-integral provider (built on first access). Depends
        only on base state (basis set, molecule, ``C``), so derivative property code in
        any subclass -- HF gradients/Hessians/APTs/AATs today, MP2/CI/CC later -- reads
        it from the base rather than constructing its own.
    device_manager : DeviceManager
        owns device/precision, device0/device1, and the contraction backend
    device, precision : str
    device0, device1 : torch.device or None
    contract : ContractionBackend

    Parameters
    ----------
    scf_wfn : Psi4 Wavefunction
        reference computed by Psi4's energy() method
    device : str
        'CPU' or 'GPU' (default 'CPU')
    precision : str
        'SP' or 'DP' (default 'DP')
    localize_occ : bool
        localize the active occupied MOs before building the integrals (default
        False). The localization scheme is ``local_mos``. Subclasses request this
        when they need localized occupied orbitals (e.g. CCwfn for local CC).
    local_mos : str
        'PIPEK_MEZEY' or 'BOYS' (default 'PIPEK_MEZEY'); used only when
        ``localize_occ`` is True.
    orbital_basis : str or None
        force the orbital basis: ``'spatial'`` or ``'spinorbital'``. Default None
        auto-selects from the reference -- closed-shell RHF uses the spin-adapted
        spatial path; any open-shell (UHF/ROHF) reference uses spin orbitals.
        Passing ``'spinorbital'`` on a closed shell is supported (and used to
        validate the spin-orbital path against the spatial one).
    Notes
    -----
    The frozen core is taken from the psi4 reference (``ref.frzcpi()`` / ``ref.nfrzc()``,
    set by psi4's ``freeze_core`` option when the SCF is run) -- there is no pycc-side
    override.  HFwfn is the sole exception: HF properties are all-electron, so it forces
    ``nfzc = 0`` via the class attribute :attr:`_all_electron`.
    """

    # HFwfn sets this True so nfzc is forced to 0 regardless of the reference's frozen-core
    # designation (HF properties are all-electron); correlated subclasses leave it False and
    # take the frozen core straight from psi4.
    _all_electron = False

    def __init__(self, scf_wfn: Any, *, device: str = 'CPU', precision: str = 'DP',
                 localize_occ: bool = False, local_mos: str = 'PIPEK_MEZEY',
                 orbital_basis: Any = None, quiet: bool = False, **kwargs) -> None:
        if 'frozen_core' in kwargs:
            raise TypeError(
                "the 'frozen_core' argument was removed; the frozen core is taken from "
                "the psi4 reference -- set psi4's 'freeze_core' option when running the SCF.")
        # The general kwargs (device/precision/localize_occ/local_mos) are owned
        # here; subclasses pop only their own kwargs and forward the rest, so this
        # is the single place they are parsed. local_mos is validated up front (a
        # bad value fails fast regardless of localize_occ).
        valid_local_mos = ['PIPEK_MEZEY', 'BOYS']
        local_mos = local_mos.upper()
        if local_mos not in valid_local_mos:
            raise InvalidKeywordError('local_mos', local_mos, valid_local_mos)
        self.local_mos = local_mos

        self.ref = scf_wfn
        self.eref = self.ref.energy()

        # Suppress the constructor's MO-summary print -- set for internally-built wavefunctions
        # (e.g. the HFwfn reference the derivative machinery constructs) whose MO table would just
        # duplicate the one the user's own wavefunction already printed.
        self.quiet = quiet

        # Closed-shell RHF takes the spin-adapted spatial path; open-shell
        # (UHF/ROHF) takes the spin-orbital path. An explicit orbital_basis overrides
        # the auto-choice (e.g. forcing 'spinorbital' on a closed shell to validate
        # the spin-orbital code against the spatial one).
        self.orbital_basis = self._resolve_orbital_basis(orbital_basis)

        if self.orbital_basis == 'spatial':
            self._init_spatial(localize_occ)
        else:
            if localize_occ:
                raise PyCCError("Local correlation (localize_occ) is not supported "
                                "in the spin-orbital path.")
            self._init_spinorbital()

        # Device / precision policy: validates the kwargs (fallback warnings for
        # GPU-without-torch / GPU-without-CUDA), resolves device0/device1, and
        # builds the contraction backend.
        self.device_manager = DeviceManager(device=device, precision=precision)
        mgr = self.device_manager
        self.precision = mgr.precision
        self.device = mgr.device
        self.device0 = mgr.device0
        self.device1 = mgr.device1
        self.contract = mgr.contract

        # Seed the integrals: F is compute-resident (device1); the big ERI/L stay
        # CPU-resident (device0). On CPU+DP these are no-ops; on CPU+SP they
        # real-cast to float32; on GPU they become real torch tensors at the
        # matching width. The spin-adapted L exists only in the spatial path.
        self.H.F = mgr.seed_compute(self.H.F)
        self.H.eps = mgr.seed_compute(self.H.eps)
        self.H.ERI = mgr.seed_store(self.H.ERI)
        if self.orbital_basis == 'spatial':
            self.H.L = mgr.seed_store(self.H.L)

        # Derivative-integral provider: built lazily on first access (see the
        # ``derivatives`` property), so methods that never take derivatives pay
        # nothing.
        self._derivatives = None
        # Coupled-perturbed-HF orbital-response solver, likewise lazy (see ``cphf``).
        self._cphf = None

        # The base is the final consumer of forwarded kwargs (each subclass pops
        # its own and passes the remainder through), so anything left over is an
        # unrecognized keyword -- flag it instead of silently ignoring it.
        if kwargs:
            raise PyCCError("Unexpected keyword argument(s): %s" % sorted(kwargs))

        # Everything is now set up (molecule, basis, orbital spaces, device, integrals), so the
        # preamble can report the full computational setup.  Suppressed for internally-built
        # wavefunctions (the HFwfn reference), whose preamble would duplicate the user's own.
        if not self.quiet:
            self._report_preamble()

    def _report_preamble(self) -> None:
        """Print the run preamble: molecule, basis, reference, and computation parameters, then the
        MO-by-energy table.  Every value comes from state set during :meth:`__init__` (the psi4
        reference / molecule / basis, the resolved orbital spaces, and the device manager) plus the
        psi4 SCF options; coordinates and energies are shown to 12 decimals."""
        import psi4
        from . import __version__
        from .utils import field
        mol, bset = self.ref.molecule(), self.ref.basisset()
        method = getattr(self, 'method', type(self).__name__)
        ref_type = psi4.core.get_option('SCF', 'REFERENCE')
        e_conv = float(psi4.core.get_option('SCF', 'E_CONVERGENCE'))
        d_conv = float(psi4.core.get_option('SCF', 'D_CONVERGENCE'))
        puream = 'spherical' if bset.has_puream() else 'cartesian'
        geom = mol.geometry().np                          # computational frame, bohr
        units = mol.units().lower()
        rule = "=" * 68

        print("\n" + rule)
        print("PyCC %s  --  %s wavefunction" % (__version__, method))
        print(rule)
        print("Molecule")
        print(field("charge, multiplicity", "%d, %d" % (mol.molecular_charge(), mol.multiplicity())))
        print(field("point group", mol.point_group().symbol()))
        print("  geometry (%s):" % units)
        for A in range(mol.natom()):
            x, y, z = geom[A]
            print("      %-2s  %18.12f  %18.12f  %18.12f" % (mol.symbol(A), x, y, z))
        print(field("nuclear repulsion", "%.12f Eh" % mol.nuclear_repulsion_energy()))
        print("Basis")
        print(field("set", bset.name()))
        print(field("functions", "%d (%s)" % (bset.nbf(), puream)))
        print("Reference")
        print(field("type", ref_type))
        print(field("SCF energy", "%.12f Eh" % self.eref))
        print(field("frozen core", "%d" % self.nfzc))
        print(field("active occ / vir", "%d / %d   (nmo = %d)" % (self.no, self.nv, self.nmo)))
        print("Computation")
        print(field("orbital basis", self.orbital_basis))
        print(field("device, precision", "%s, %s" % (self.device, self.precision)))
        print(field("SCF E/D convergence", "%.2E / %.2E" % (e_conv, d_conv)))
        print(rule)
        self._print_mo_table()

    def _print_mo_table(self) -> None:
        """The MO-by-energy table (orbital energies to 12 decimals), dispatched to the spatial
        (single-column) or spin-orbital (alpha | beta) layout."""
        ref = self.ref
        nirrep = ref.nirrep()
        irrep_labels = ref.molecule().irrep_labels()
        nmopi = ref.nmopi()
        mo_irreps = np.array([h for h in range(nirrep) for _ in range(nmopi[h])])

        if self.orbital_basis == 'spatial':
            eps_active_so = np.concatenate([np.array(ref.epsilon_a_subset("SO", "ALL").nph[h])
                                            for h in range(nirrep)])
            sort_idx = np.argsort(eps_active_so, kind='stable')
            eps_active = eps_active_so[sort_idx]
            labels = [irrep_labels[mo_irreps[sort_idx][i]] for i in range(len(sort_idx))]
            ndocc = self.nfzc + self.no
            print("\nMOs by energy:")
            print(f"  {'#':>4}  {'Irrep':>6}  {'Energy':>18}")
            print(f"  {'-'*4}  {'-'*6}  {'-'*18}")
            for i, (eps, label) in enumerate(zip(eps_active, labels)):
                if i == ndocc:
                    print(f"  {'.'*4}  {'.'*6}  {'.'*18}")
                idx = i if i < ndocc else i - ndocc
                print(f"  {idx:>4}  {label:>6}  {eps:>18.12f}")
        else:
            def _spin_mos(eps_subset):
                eps = np.concatenate([np.array(eps_subset.nph[h]) for h in range(nirrep)])
                order = np.argsort(eps, kind='stable')
                return eps[order], [irrep_labels[mo_irreps[i]] for i in order]

            a_eps, a_lab = _spin_mos(ref.epsilon_a_subset("SO", "ALL"))
            b_eps, b_lab = _spin_mos(ref.epsilon_b_subset("SO", "ALL"))
            na, nb = ref.nalpha(), ref.nbeta()
            print("\nMOs by energy (alpha | beta):")
            print(f"  {'#':>4}   {'irr':>5} {'o':>1} {'energy':>18}    {'irr':>5} {'o':>1} {'energy':>18}")
            print(f"  {'-'*4}   {'-'*5} {'-'*1} {'-'*18}    {'-'*5} {'-'*1} {'-'*18}")
            for i in range(ref.nmo()):
                aocc = 'o' if i < na else ' '
                bocc = 'o' if i < nb else ' '
                print(f"  {i:>4}   {a_lab[i]:>5} {aocc:>1} {a_eps[i]:>18.12f}"
                      f"    {b_lab[i]:>5} {bocc:>1} {b_eps[i]:>18.12f}")

    def _resolve_orbital_basis(self, orbital_basis: Any) -> str:
        """Resolve the orbital basis from an explicit override or the reference.

        Auto-choice (override is None): a closed-shell RHF reference -- identical
        alpha/beta orbitals and densities -- uses the spin-adapted spatial path;
        any open-shell reference (UHF, ROHF) uses spin orbitals. Spin orbitals are
        always correct; spatial is the closed-shell optimization.
        """
        valid = ['spatial', 'spinorbital']
        if orbital_basis is None:
            closed_shell = (self.ref.same_a_b_orbs() and self.ref.same_a_b_dens())
            return 'spatial' if closed_shell else 'spinorbital'
        ob = str(orbital_basis).lower()
        if ob not in valid:
            raise InvalidKeywordError('orbital_basis', orbital_basis, valid)
        return ob

    def _init_spatial(self, localize_occ: bool) -> None:
        """Build the spatial (spin-adapted, closed-shell RHF) orbital spaces,
        coefficients, and Hamiltonian.

        Always builds the FULL-space Hamiltonian; a frozen core is handled purely as
        a slice OFFSET -- the active occupied slice ``o`` starts past the core rather
        than at 0. ``nfzc`` (how many core orbitals to skip) comes from the psi4
        reference's ``frzcpi()``. This gives every method one MO/integral layout:
        correlated codes work on the active blocks via ``o``/``v``, while all-electron
        properties (HFwfn, via ``_all_electron``) span the whole space (``nfzc = 0``).
        """
        self.nfzc = 0 if self._all_electron else int(sum(self.ref.frzcpi()))
        self.no   = int(sum(self.ref.doccpi())) - self.nfzc
        self.nmo  = self.ref.nmo()
        self.nv   = self.nmo - self.no - self.nfzc
        self.nact = self.no + self.nv

        ndocc = self.nfzc + self.no
        self.o = slice(self.nfzc, ndocc)
        self.v = slice(ndocc, self.nmo)

        # Full MO space, global energy order; the frozen core sits at columns [0:nfzc].
        self.C = self.ref.Ca_subset("AO", "ALL")

        # Localize the occupied MOs if requested (used consistently for the single H
        # build below, so all methods share the same integrals). Only the ACTIVE
        # occupied orbitals are localized; the frozen core is left canonical.
        if localize_occ:
            # Localize the ACTIVE occupied MOs and place them at the active-occupied
            # offset [nfzc:nfzc+no] of the full C; the frozen core stays canonical.
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC")
            LMOS = psi4.core.Localizer.build(self.local_mos, self.ref.basisset(), C_occ)
            LMOS.localize()
            npL = np.asarray(LMOS.L)
            npC = np.asarray(self.C)
            npC[:, self.nfzc:self.nfzc + self.no] = npL
            self.C = psi4.core.Matrix.from_array(npC)

        # MO-basis integrals (built once from the final C).
        self.H = Hamiltonian(self.ref, self.C, self.C, self.C, self.C)

    def _init_spinorbital(self) -> None:
        """Build the spin-orbital (UHF/ROHF) orbital spaces, coefficients, and
        Hamiltonian.

        Built over the FULL spin-orbital MO space (frozen core included), mirroring the
        spatial path (``_init_spatial``): the Hamiltonian integrals span all MOs and the
        active occupied/virtual slices skip the frozen core. Spin orbitals are ordered
        ``[alpha-core, beta-core, alpha-occ, beta-occ, alpha-vir, beta-vir]`` (the frozen
        core first), so ``co``/``o``/``v`` are contiguous; for ``nfzc=0`` this reduces to
        the ``[alpha-occ, beta-occ, alpha-vir, beta-vir]`` ordering. The per-spin-orbital
        ``spin`` (0=alpha, 1=beta) and ``spat`` (index into that spin's full spatial space)
        maps drive the integral build. Keeping the core in the Hamiltonian gives the
        frozen-core gradient its core-virtual/core-active response integrals (the spatial
        path already relies on this).
        """
        ref = self.ref
        self.nfzc = 0 if self._all_electron else ref.nfrzc()
        nao = ref.nalpha() - self.nfzc       # active alpha occupied
        nbo = ref.nbeta() - self.nfzc        # active beta occupied
        nav = ref.nmo() - ref.nalpha()       # active alpha virtual
        nbv = ref.nmo() - ref.nbeta()        # active beta virtual
        self.no   = nao + nbo
        self.nv   = nav + nbv
        self.nmo  = 2 * ref.nmo()
        self.nact = self.no + self.nv

        # Full spin-orbital MO space, ordered [a-core, b-core, a-occ, b-occ, a-vir, b-vir]
        # with the frozen core first (mirrors _init_spatial); the active slices skip it.
        # For nfzc=0 the core blocks are empty and this is the previous active ordering.
        nc = self.nfzc                       # frozen core per spin (alpha = beta = nfzc)
        self.co = slice(0, 2 * nc)
        self.o = slice(2 * nc, 2 * nc + self.no)
        self.v = slice(2 * nc + self.no, self.nmo)

        aco = slice(0, nc)
        bco = slice(nc, 2 * nc)
        ao = slice(2 * nc, 2 * nc + nao)
        bo = slice(2 * nc + nao, 2 * nc + self.no)
        av = slice(2 * nc + self.no, 2 * nc + self.no + nav)
        bv = slice(2 * nc + self.no + nav, self.nmo)
        spin = np.zeros(self.nmo, dtype=int)
        spin[bco] = 1
        spin[bo] = 1
        spin[bv] = 1
        spat = np.zeros(self.nmo, dtype=int)
        spat[aco] = np.arange(nc)                       # alpha core: spatial 0..nc-1
        spat[bco] = np.arange(nc)                       # beta core
        spat[ao] = np.arange(nc, nc + nao)              # alpha active occ: spatial nc..na-1
        spat[bo] = np.arange(nc, nc + nbo)
        spat[av] = np.arange(nc + nao, nc + nao + nav)  # alpha virtual
        spat[bv] = np.arange(nc + nbo, nc + nbo + nbv)
        self.spin = spin
        self.spat = spat

        self.Ca = ref.Ca_subset("AO", "ALL")
        self.Cb = ref.Cb_subset("AO", "ALL")
        # Optional static external field (finite-field CC properties); only CCwfn sets
        # these attributes, so default to no field for the other method classes.
        field_strength, field_axis = 0.0, 2
        if getattr(self, 'field', False):
            field_strength = getattr(self, 'field_strength', 0.0)
            field_axis = {'X': 0, 'Y': 1, 'Z': 2}[str(getattr(self, 'field_axis', 'Z')).upper()]
        # The Hamiltonian splits each spin's full MO set into occ/vir for the semicanonical
        # rotation; pass the full per-spin occupied counts (core + active = nalpha/nbeta).
        # The rotation is a no-op for canonical RHF/UHF, so the frozen core is not mixed
        # into the active occupied (ROHF frozen-core response is deferred regardless).
        self.H = SpinOrbitalHamiltonian(ref, self.Ca, self.Cb, spin, spat,
                                        nc + nao, nc + nbo,
                                        field_strength=field_strength, field_axis=field_axis)

    @property
    def derivatives(self) -> Derivatives:
        """Lazy MO-basis derivative-integral provider, built on first access and
        cached. Lives on the base (it depends only on base state) so any subclass's
        derivative property code reaches it through ``self.derivatives``."""
        if self._derivatives is None:
            self._derivatives = Derivatives(self)
        return self._derivatives

    @property
    def cphf(self):
        """Lazy coupled-perturbed-Hartree-Fock orbital-response solver, built on first
        access and cached. Lives on the base (it depends only on base state -- the
        orbital energies and the orbital Hessian's two-electron integrals) so HF, MP2,
        and CC orbital-response code all reach it through ``self.cphf``. Basis-aware: the
        spatial path uses the spin-adapted ``H.L``, the spin-orbital path ``H.ERI``."""
        if self._cphf is None:
            from .cphf import CPHF
            self._cphf = CPHF(self)
        return self._cphf

