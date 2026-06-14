"""Device / precision manager for PyCC wavefunctions.

A single object that owns the ``device`` ('CPU'/'GPU') and ``precision``
('SP'/'DP') policy: it validates the kwargs, resolves the storage (``device0``)
and compute (``device1``) handles, builds the contraction backend, and applies
the dtype/placement cast policy through :meth:`seed_compute` / :meth:`seed_store`.

This centralizes logic that was previously duplicated across ``ccwfn.__init__``
(the SP real-casts and the GPU torch casts) and the contraction backend's
``__init__`` (a third copy of the device resolution). In the 2026-06 refactor the
manager is created in ``ccwfn.__init__``; it moves to the ``Wavefunction`` base in
Phase 3.

Notes
-----
GPU tensors are currently always *complex* (the historical behavior, shared with
RT-CC). The ``needs_complex`` seam exists so a future phase can give ground-state
methods a real GPU dtype; it defaults to True, so present behavior is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import opt_einsum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .exceptions import InvalidKeywordError, PyCCWarning


class ContractionBackend(object):
    """A device-aware wrapper around ``opt_einsum.contract``.

    On CPU it calls ``opt_einsum.contract`` directly; on GPU it first moves any
    operand not already resident on the compute device (``device1``) there, so
    CPU-stored integrals (``H.ERI``/``H.L``) can participate in GPU contractions.
    Created and owned by :class:`DeviceManager`, which injects the resolved
    ``device1``; also passed by value into the module-level (T)/triples kernels.
    """

    def __init__(self, device='CPU', device1=None):
        """
        Parameters
        ----------
        device : str
            'CPU' or 'GPU' (default 'CPU').
        device1 : torch.device, optional
            the compute device, normally injected by DeviceManager (the single
            resolver). Falls back to local resolution for standalone use.
        """
        self.device = device
        if self.device == 'GPU':
            # torch.device is the device on which a torch.Tensor is/will be allocated.
            self.device1 = device1 if device1 is not None else torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, subscripts, *operands):
        """Contract ``operands`` per ``subscripts`` (numpy.einsum format),
        returning an ndarray (CPU) or torch.Tensor (GPU)."""
        if self.device == 'CPU':
            return opt_einsum.contract(subscripts, *operands)
        elif self.device == 'GPU':
            # Move any CPU-resident operand (e.g. ERI) onto the compute device.
            input_list = list(operands)
            for i in range(len(input_list)):
                if not input_list[i].is_cuda:
                    input_list[i] = input_list[i].to(self.device1)
            output = opt_einsum.contract(subscripts, *input_list)
            del input_list
            return output


class DeviceManager(object):
    """Owns the device/precision policy and the contraction backend.

    Parameters
    ----------
    device : str
        'CPU' or 'GPU'. 'GPU' falls back to 'CPU' (with a ``PyCCWarning``) when
        PyTorch is unavailable; with PyTorch but no CUDA device it stays 'GPU'
        but tensors run on CPU.
    precision : str
        'SP' (single) or 'DP' (double).
    needs_complex : bool
        Whether GPU tensors must be complex (RT-CC). Defaults True, reproducing
        the historical always-complex-on-GPU behavior; Phase 2b will set it False
        for real ground-state methods.

    Attributes
    ----------
    device, precision : str
        The resolved (post-fallback) device and precision.
    device0, device1 : torch.device or None
        Storage (CPU-resident integrals) and compute handles; None on CPU.
    contract : ContractionBackend
        The contraction callable, owning the single resolved ``device1``.
    real_dtype : numpy dtype
        float32 for SP, float64 for DP.
    """

    VALID_DEVICE = ['CPU', 'GPU']
    VALID_PRECISION = ['SP', 'DP']

    def __init__(self, device: str = 'CPU', precision: str = 'DP',
                 needs_complex: bool = True) -> None:
        # --- precision ---
        if precision.upper() not in self.VALID_PRECISION:
            raise InvalidKeywordError('precision', precision, self.VALID_PRECISION)
        self.precision = precision.upper()

        # --- device (with graceful fallback) ---
        if device.upper() not in self.VALID_DEVICE:
            raise InvalidKeywordError('device', device, self.VALID_DEVICE)
        self.device = device.upper()
        if self.device == 'GPU' and not HAS_TORCH:
            warnings.warn("GPU requested, but PyTorch is not available; "
                          "falling back to CPU.", PyCCWarning, stacklevel=3)
            self.device = 'CPU'
        elif self.device == 'GPU' and not torch.cuda.is_available():
            warnings.warn("GPU requested, but no CUDA device is available; "
                          "PyTorch tensors will run on CPU.", PyCCWarning,
                          stacklevel=3)

        self.needs_complex = needs_complex

        # --- device handles ---
        # device0: storage/CPU-resident (the big integrals); device1: compute.
        # Both None on CPU so clone(x, device=device1) is a plain copy.
        self.device0 = None
        self.device1 = None
        if self.device == 'GPU':
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # --- dtypes ---
        # The numpy real dtype rides precision; the torch dtype for GPU-resident
        # tensors is complex (under needs_complex) at the matching width.
        self.real_dtype = np.float32 if self.precision == 'SP' else np.float64
        self._torch_dtype = None
        if HAS_TORCH:
            if self.needs_complex:
                self._torch_dtype = (torch.complex64 if self.precision == 'SP'
                                     else torch.complex128)
            else:
                self._torch_dtype = (torch.float32 if self.precision == 'SP'
                                     else torch.float64)

        # --- contraction backend (single owner of the resolved device1) ---
        self.contract = ContractionBackend(device=self.device, device1=self.device1)

    def _real_cast(self, a):
        # SP -> float32; DP -> unchanged (matches the historical behavior, which
        # only re-cast arrays on the SP path).
        return np.float32(a) if self.precision == 'SP' else a

    def seed_compute(self, a):
        """Cast + place a compute-resident array (amplitudes/Fock; rtcc mu/m)."""
        a = self._real_cast(a)
        if self.device == 'GPU':
            return torch.tensor(a, dtype=self._torch_dtype, device=self.device1)
        return a

    def seed_store(self, a):
        """Cast + place a storage-resident array (the big integrals, ERI/L)."""
        a = self._real_cast(a)
        if self.device == 'GPU':
            return torch.tensor(a, dtype=self._torch_dtype, device=self.device0)
        return a
