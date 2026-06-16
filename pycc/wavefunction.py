"""
wavefunction.py: top-level Wavefunction base class.

Owns the infrastructure shared by every correlated (and SCF-derived) method:
the Psi4 reference, the orbital spaces (full MO basis, with active occupied/virtual
slices that offset past any frozen core), the MO coefficients (optionally with
localized occupied orbitals), the full-MO-basis integrals (Hamiltonian), and the
device/precision manager. Method-specific machinery (amplitudes, denominators,
densities, response, ...) lives in the subclasses (CCwfn, MPwfn, HFwfn, and --
planned -- CIwfn).

Part of the 2026-06 refactor (docs/REFACTOR_PLAN_2026-06.md, Phase 3): the
reference/orbital/integral setup and the DeviceManager were lifted out of
ccwfn.__init__ so PyCC can host more than coupled cluster.
"""

from __future__ import annotations

from typing import Any

import psi4
import numpy as np

from .hamiltonian import Hamiltonian
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
    nfzc, no, nv, nmo, nact : int
        frozen-core / active-occupied / active-virtual / total-MO / active counts
    o, v : slice
        active occupied and virtual subspaces, as offsets into the full MO space:
        ``o = slice(nfzc, nfzc+no)`` skips the frozen core and ``v`` is the
        virtuals. With no frozen core (``nfzc == 0``) ``o`` starts at 0 as usual.
    C : Psi4 Matrix
        full-space MO coefficients (all ``nmo`` columns, global energy order; the
        frozen core occupies columns ``[0:nfzc]``). The active occupied block is
        localized in place if ``localize_occ``.
    H : Hamiltonian
        full-MO-basis Fock (F), ERIs, spin-adapted ERIs (L), and property integrals,
        device/precision-seeded
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
    frozen_core : bool
        whether to honor the reference's frozen core (default True, for correlated
        methods). The Hamiltonian is ALWAYS built over the full MO space; this flag
        only sets ``nfzc`` -- i.e. how far the active occupied slice ``o`` is offset
        past the core. False means no core (``nfzc = 0``); HFwfn passes this, since
        HF properties are all-electron.
    """

    def __init__(self, scf_wfn: Any, *, device: str = 'CPU', precision: str = 'DP',
                 localize_occ: bool = False, local_mos: str = 'PIPEK_MEZEY',
                 frozen_core: bool = True, **kwargs) -> None:
        # A subclass may set its own attributes before calling super().__init__;
        # snapshot them so _base_attrs (recorded at the end) captures only what the
        # base itself adds -- which _from_shared_base then replicates.
        _preexisting = set(vars(self))

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

        # Always build the FULL-space Hamiltonian; a frozen core is handled purely as
        # a slice OFFSET -- the active occupied slice ``o`` starts past the core
        # rather than at 0. ``frozen_core`` only sets how many core orbitals to skip.
        # This gives every method one MO/integral layout: correlated codes work on
        # the active blocks via ``o``/``v``, while all-electron properties (HFwfn)
        # pass frozen_core=False so the slices span the whole space.
        self.nfzc = int(sum(self.ref.frzcpi())) if frozen_core else 0
        self.no   = int(sum(self.ref.doccpi())) - self.nfzc
        self.nmo  = self.ref.nmo()
        self.nv   = self.nmo - self.no - self.nfzc
        self.nact = self.no + self.nv

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (self.nmo, self.nact, self.no, self.nv))

        ndocc = self.nfzc + self.no
        self.o = slice(self.nfzc, ndocc)
        self.v = slice(ndocc, self.nmo)

        # Full MO space, global energy order; the frozen core sits at columns [0:nfzc].
        self.C = self.ref.Ca_subset("AO", "ALL")

        eps_so_blocked  = self.ref.epsilon_a_subset("SO", "ALL")
        eps_active_so   = np.concatenate([np.array(eps_so_blocked.nph[h])
                                          for h in range(self.ref.nirrep())])
        sort_idx        = np.argsort(eps_active_so, kind='stable')

        irrep_labels    = self.ref.molecule().irrep_labels()
        nirrep          = self.ref.nirrep()
        nmopi           = self.ref.nmopi()
        mo_irreps       = np.array([h for h in range(nirrep) for _ in range(nmopi[h])])
        mo_irreps       = mo_irreps[sort_idx]
        mo_irrep_labels = [irrep_labels[h] for h in mo_irreps]
        eps_active      = eps_active_so[sort_idx]

        # Print MO summary
        print("\nMOs by energy:")
        print(f"  {'#':>4}  {'Irrep':>6}  {'Energy':>16}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*16}")
        for i, (eps, label) in enumerate(zip(eps_active, mo_irrep_labels)):
            if i == ndocc:
                print(f"  {'.'*4}  {'.'*6}  {'.'*16}")
            idx = i if i < ndocc else i - ndocc
            print(f"  {idx:>4}  {label:>6}  {eps:>16.10f}")

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
        # matching width.
        self.H.F = mgr.seed_compute(self.H.F)
        self.H.ERI = mgr.seed_store(self.H.ERI)
        self.H.L = mgr.seed_store(self.H.L)

        # Derivative-integral provider: built lazily on first access (see the
        # ``derivatives`` property), so methods that never take derivatives pay
        # nothing. Recorded here as part of the base so _from_shared_base carries it.
        self._derivatives = None

        # The base is the final consumer of forwarded kwargs (each subclass pops
        # its own and passes the remainder through), so anything left over is an
        # unrecognized keyword -- flag it instead of silently ignoring it.
        if kwargs:
            raise PyCCError("Unexpected keyword argument(s): %s" % sorted(kwargs))

        # Record exactly which attributes the base set (excluding anything the
        # subclass set first), so _from_shared_base can replicate this base onto a
        # new instance without re-running __init__ (and re-transforming integrals).
        self._base_attrs = tuple(k for k in vars(self) if k not in _preexisting)

    @property
    def derivatives(self) -> Derivatives:
        """Lazy MO-basis derivative-integral provider, built on first access and
        cached. Lives on the base (it depends only on base state) so any subclass's
        derivative property code reaches it through ``self.derivatives``."""
        if self._derivatives is None:
            self._derivatives = Derivatives(self)
        return self._derivatives

    @classmethod
    def _from_shared_base(cls, source: "Wavefunction") -> "Wavefunction":
        """Create a ``cls`` instance that REUSES ``source``'s already-built base --
        reference, orbital spaces, seeded integrals, device manager -- by bypassing
        ``__init__`` so the Hamiltonian is not transformed a second time. The caller
        adds its own method-specific state afterward.

        Lets one wavefunction compose another over the same base (e.g. ccwfn holds an
        MPwfn for its MP2 guess/denominators) with a single integral build.
        """
        obj = cls.__new__(cls)
        for name in source._base_attrs:
            setattr(obj, name, getattr(source, name))
        obj._base_attrs = source._base_attrs
        return obj
