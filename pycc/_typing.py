"""Shared type aliases for PyCC public-API annotations.

These describe the dense tensors and orbital-space selectors passed throughout
the coupled-cluster machinery.

``Tensor`` covers both the default NumPy backend and the optional PyTorch
(GPU / mixed-precision) backend. The ``torch.Tensor`` arm is written as a
forward reference so the alias stays importable even when PyTorch is not
installed; the annotations that use it are themselves lazy (every annotated
module enables ``from __future__ import annotations``), so nothing here is
evaluated at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np

#: A dense amplitude / integral tensor in either supported backend.
if TYPE_CHECKING:
    # Type-checkers always see the full union, regardless of the runtime env.
    import torch
    Tensor = Union[np.ndarray, "torch.Tensor"]
else:
    # At run time, only reference torch when it is actually importable so that
    # ``typing.get_type_hints`` can resolve the alias even without PyTorch.
    try:
        import torch as _torch
        Tensor = Union[np.ndarray, _torch.Tensor]
    except ModuleNotFoundError:
        Tensor = np.ndarray

#: An orbital-subspace selector: the occupied/virtual ``slice`` objects
#: (``o``/``v``), a single orbital index, or a fancy-index array.
Slice = Union[slice, int, np.ndarray]

#: A tensor-contraction callable (e.g. ``opt_einsum.contract`` or the einsums
#: backend), invoked as ``contract(subscripts, *operands)``.
Contract = Callable[..., Tensor]
