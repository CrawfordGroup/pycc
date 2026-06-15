"""
wavefunction.py: top-level Wavefunction base class.

Owns the infrastructure shared by every correlated (and SCF-derived) method:
the Psi4 reference, the active-orbital spaces, the MO coefficients (optionally
with localized occupied orbitals), the MO-basis integrals (Hamiltonian), and the
device/precision manager. Method-specific machinery (amplitudes, denominators,
densities, response, ...) lives in the subclasses (CCwfn, and -- planned -- MPwfn,
CIwfn, HFwfn).

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
        occupied and virtual active-orbital subspaces
    C : Psi4 Matrix
        active MO coefficients (occupied block localized if ``localize_occ``)
    H : Hamiltonian
        MO-basis Fock (F), ERIs, spin-adapted ERIs (L), and property integrals,
        device/precision-seeded
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
        honor the reference's frozen core, i.e. use the active MO space (default
        True, for correlated methods). False uses the full, all-electron MO space
        (nfzc = 0) -- HFwfn passes this, since HF properties are all-electron.
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

        # Orbital counts and the MO subset. Correlated methods honor the reference's
        # frozen core (frozen_core=True -> the active space). HF properties are
        # all-electron, so HFwfn passes frozen_core=False to use the full MO space
        # (nfzc=0). frzcpi/doccpi are per-irrep Dimension objects; sum over irreps to
        # support both C1 and higher-symmetry references.
        subset = "ACTIVE" if frozen_core else "ALL"
        self.nfzc = int(sum(self.ref.frzcpi())) if frozen_core else 0
        self.no   = int(sum(self.ref.doccpi())) - self.nfzc
        self.nmo  = self.ref.nmo()
        self.nv   = self.nmo - self.no - self.nfzc
        self.nact = self.no + self.nv

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (self.nmo, self.nact, self.no, self.nv))

        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)

        # Ca_subset("AO", subset) returns columns in global energy order;
        # Ca_subset("SO", subset) in irrep-block order (used only for irrep labels).
        self.C = self.ref.Ca_subset("AO", subset)

        eps_so_blocked  = self.ref.epsilon_a_subset("SO", subset)
        eps_active_so   = np.concatenate([np.array(eps_so_blocked.nph[h])
                                          for h in range(self.ref.nirrep())])
        sort_idx        = np.argsort(eps_active_so, kind='stable')

        irrep_labels    = self.ref.molecule().irrep_labels()
        nirrep          = self.ref.nirrep()
        nmopi           = self.ref.nmopi()
        if frozen_core:
            frzcpi, frzvpi = self.ref.frzcpi(), self.ref.frzvpi()
            mopi = [nmopi[h] - frzcpi[h] - frzvpi[h] for h in range(nirrep)]
        else:
            mopi = [nmopi[h] for h in range(nirrep)]
        mo_irreps       = np.array([h for h in range(nirrep) for _ in range(mopi[h])])
        mo_irreps       = mo_irreps[sort_idx]
        mo_irrep_labels = [irrep_labels[h] for h in mo_irreps]
        eps_active      = eps_active_so[sort_idx]

        # Print MO summary
        print("\nActive MOs by energy:")
        print(f"  {'#':>4}  {'Irrep':>6}  {'Energy':>16}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*16}")
        for i, (eps, label) in enumerate(zip(eps_active, mo_irrep_labels)):
            if i == self.no:
                print(f"  {'.'*4}  {'.'*6}  {'.'*16}")
            idx = i if i < self.no else i - self.no
            print(f"  {idx:>4}  {label:>6}  {eps:>16.10f}")

        # Localize the occupied MOs if requested (used consistently for the single H
        # build below, so all methods share the same integrals). The occupied subset
        # tracks frozen_core: active occupied for correlated methods, all occupied
        # for the full (HF) space.
        if localize_occ:
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC" if frozen_core else "OCC")
            LMOS = psi4.core.Localizer.build(self.local_mos, self.ref.basisset(), C_occ)
            LMOS.localize()
            npL = np.asarray(LMOS.L)
            npC = np.asarray(self.C)
            npC[:, :self.no] = npL
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

        # The base is the final consumer of forwarded kwargs (each subclass pops
        # its own and passes the remainder through), so anything left over is an
        # unrecognized keyword -- flag it instead of silently ignoring it.
        if kwargs:
            raise PyCCError("Unexpected keyword argument(s): %s" % sorted(kwargs))

        # Record exactly which attributes the base set (excluding anything the
        # subclass set first), so _from_shared_base can replicate this base onto a
        # new instance without re-running __init__ (and re-transforming integrals).
        self._base_attrs = tuple(k for k in vars(self) if k not in _preexisting)

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
