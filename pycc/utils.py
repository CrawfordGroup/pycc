import numpy as np
from pycc.ccwfn import HAS_TORCH
if HAS_TORCH:
    import torch
import opt_einsum


def zeros_like(a):
    """Backend-aware ``zeros_like``: ``torch.zeros_like`` for a torch tensor,
    else ``np.zeros_like``.

    Collapses the ``if HAS_TORCH and isinstance(a, torch.Tensor): ... else: ...``
    allocation branch that recurs throughout the CC modules into a single call.
    """
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.zeros_like(a)
    return np.zeros_like(a)


def zeros(shape, like):
    """Allocate a zero tensor of ``shape`` matching ``like``'s backend/dtype/device.

    Dispatches on the runtime type of ``like`` (NumPy vs torch) and inherits its
    dtype — and, for torch, its device — so single/mixed precision and GPU
    placement ride along automatically. Use this in place of the
    ``zeros_like(x)`` + ``np.pad``/``torch.nn.functional.pad`` idiom when the
    target shape mixes occupied and virtual dimensions and so is not
    ``zeros_like`` of any single amplitude array.
    """
    if HAS_TORCH and isinstance(like, torch.Tensor):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=like.dtype)


def diag(a):
    """Backend-aware ``diag``: ``torch.diag`` for a torch tensor, else ``np.diag``.

    Collapses the ``if HAS_TORCH and isinstance(a, torch.Tensor): torch.diag(a)
    else: np.diag(a)`` branch (paired with ``zeros_like`` in the triples-denominator
    construction throughout cctriples).
    """
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.diag(a)
    return np.diag(a)


def clone(a, device=None):
    """Backend-aware deep copy: ``a.clone()`` for a torch tensor (optionally moved
    to ``device``), else ``a.copy()``.

    Collapses the ``if HAS_TORCH and isinstance(a, torch.Tensor): a.clone()
    [.to(self.device1)] else: a.copy()`` branch that recurs throughout the CC
    modules. ``device`` is ignored for NumPy arrays (always CPU).
    """
    if HAS_TORCH and isinstance(a, torch.Tensor):
        a = a.clone()
        return a.to(device) if device is not None else a
    return a.copy()


def real_zeros(shape, like):
    """Allocate a real-dtype zero tensor of ``shape`` on ``like``'s backend/device.

    Uses the *real-component* dtype of ``like`` (``like.real.dtype``), so a complex
    amplitude yields a real buffer of the matching precision (complex128->float64,
    complex64->float32) while a real amplitude is unchanged. Used for the DIIS B
    matrix / residual vector, which must stay real even when the amplitudes are
    complex (e.g. frequency-dependent response in ``ccresponse``); precision and,
    for torch, device placement ride along from the data.
    """
    if HAS_TORCH and isinstance(like, torch.Tensor):
        return torch.zeros(shape, dtype=like.real.dtype, device=like.device)
    return np.zeros(shape, dtype=like.real.dtype)


def dot(a, b):
    """Backend-aware 1-D dot product: ``torch.dot`` for torch tensors, else ``np.dot``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.dot(a, b)
    return np.dot(a, b)


def absolute(a):
    """Backend-aware elementwise absolute value: ``torch.abs`` for a torch tensor,
    else ``np.abs``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.abs(a)
    return np.abs(a)


def conj(a):
    """Backend-aware complex conjugate: ``torch.conj`` for a torch tensor, else ``np.conj``."""
    if HAS_TORCH and isinstance(a, torch.Tensor):
        return torch.conj(a)
    return np.conj(a)


def solve(A, b):
    """Backend-aware dense linear solve: ``torch.linalg.solve`` for a torch tensor,
    else ``np.linalg.solve``."""
    if HAS_TORCH and isinstance(A, torch.Tensor):
        return torch.linalg.solve(A, b)
    return np.linalg.solve(A, b)


class helper_diis(object):
    def __init__(self, t1, t2, max_diis, precision='DP'):
        self.oldt1 = clone(t1)
        self.oldt2 = clone(t2)
        self.diis_vals_t1 = [clone(t1)]
        self.diis_vals_t2 = [clone(t2)]

        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis
        self.precision = precision

    def add_error_vector(self, t1, t2):
        # Add DIIS vectors
        self.diis_vals_t1.append(clone(t1))
        self.diis_vals_t2.append(clone(t2))
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        if HAS_TORCH and isinstance(t1, torch.Tensor):
            self.diis_errors.append(torch.cat((error_t1, error_t2)))
        else:
            self.diis_errors.append(np.concatenate((error_t1, error_t2)))
        self.oldt1 = clone(t1)
        self.oldt2 = clone(t2)

    def extrapolate(self, t1, t2):
        
        if (self.max_diis == 0):
            return t1, t2

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build error matrix B. B/resid are real even when the amplitudes are complex
        # (e.g. ccresponse), so they are seeded from real_zeros (real-component dtype of
        # oldt1), which also carries the precision and, for torch, the device.
        B = real_zeros((self.diis_size + 1, self.diis_size + 1), self.oldt1) - 1.0
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = dot(e1, e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 >= n2:
                    continue
                B[n1, n2] = dot(e1, e2)
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= absolute(B[:-1, :-1]).max()

        # Build residual vector
        resid = real_zeros((self.diis_size + 1,), self.oldt1)
        resid[-1] = -1

        # Solve pulay equations
        ci = solve(B, resid)

        # Calculate new amplitudes
        t1 = zeros_like(self.oldt1)
        t2 = zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += ci[num] * self.diis_vals_t1[num + 1]
            t2 += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = clone(t1)
        self.oldt2 = clone(t2)

        return t1, t2

class cc_contract(object):
    """
    A wrapper for opt_einsum.contract with tensors stored on CPU and/or GPU.
    """
    def __init__(self, device='CPU'):
        """
        Parameters
        ----------
        device: string
            initiated in ccwfn object, default: 'CPU'
        
        Returns
        -------
        None
        """
        self.device = device
        if self.device == 'GPU':
            # torch.device is an object representing the device on which torch.Tensor is or will be allocated.
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def __call__(self, subscripts, *operands): 
        """
        Parameters
        ----------
        subscripts: string
            specify the subscripts for summation (same format as numpy.einsum)
        *operands: list of array_like
            the arrays/tensors for the operation
   
        Returns
        -------
        An ndarray/torch.Tensor that is calculated based on Einstein summation convention.   
        """       
        if self.device == 'CPU':
            return opt_einsum.contract(subscripts, *operands)
        elif self.device == 'GPU':
            # Check the type and allocation of the tensors 
            # Transfer the copy from CPU to GPU if needed (for ERI)
            input_list = list(operands)
            for i in range(len(input_list)):
                if (not input_list[i].is_cuda):
                    input_list[i] = input_list[i].to(self.device1)               
            #print(len(input_list), type(input_list[0]), type(input_list[1]))    
            output = opt_einsum.contract(subscripts, *input_list)
            del input_list
            return output

