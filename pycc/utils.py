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


class helper_diis(object):
    def __init__(self, t1, t2, max_diis, precision='DP'):
        if HAS_TORCH and isinstance(t1, torch.Tensor):
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

        if HAS_TORCH and isinstance(t1, torch.Tensor):
            # Build error matrix B
            if self.precision == 'DP':
                B = torch.ones((self.diis_size + 1, self.diis_size + 1), dtype=torch.float64, device=self.device1) * -1
            elif self.precision == 'SP':
                B = torch.ones((self.diis_size + 1, self.diis_size + 1), dtype=torch.float32, device=self.device1) * -1
            B[-1, -1] = 0

            for n1, e1 in enumerate(self.diis_errors):
                B[n1, n1] = torch.dot(e1, e1)
                for n2, e2 in enumerate(self.diis_errors):
                    if n1 >= n2:
                        continue
                    B[n1, n2] = torch.dot(e1, e2)
                    B[n2, n1] = B[n1, n2]

            B[:-1, :-1] /= torch.abs(B[:-1, :-1]).max()

            # Build residual vector
            if self.precision == 'DP':
                resid = torch.zeros((self.diis_size + 1), dtype=torch.float64, device=self.device1)
            elif self.precision == 'SP':
                resid = torch.zeros((self.diis_size + 1), dtype=torch.float32, device=self.device1)
            resid[-1] = -1

            # Solve pulay equations
            ci = torch.linalg.solve(B, resid)

            # Calculate new amplitudes
            t1 = torch.zeros_like(self.oldt1)
            t2 = torch.zeros_like(self.oldt2)
            for num in range(self.diis_size):
                t1 += torch.real(ci[num] * self.diis_vals_t1[num + 1])
                t2 += torch.real(ci[num] * self.diis_vals_t2[num + 1])

        else:
            # Build error matrix B
            if self.precision == 'DP':
                B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
            elif self.precision == 'SP':
                B = np.ones((self.diis_size + 1, self.diis_size + 1), dtype=np.float32) * -1
            B[-1, -1] = 0

            for n1, e1 in enumerate(self.diis_errors):
                B[n1, n1] = np.dot(e1, e1)
                for n2, e2 in enumerate(self.diis_errors):
                    if n1 >= n2:
                        continue
                    B[n1, n2] = np.dot(e1, e2)
                    B[n2, n1] = B[n1, n2]

            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

            # Build residual vector
            if self.precision == 'DP':
                resid = np.zeros(self.diis_size + 1)
            elif self.precision == 'SP':
                resid = np.zeros((self.diis_size + 1), dtype=np.float32)
            resid[-1] = -1

            # Solve pulay equations
            ci = np.linalg.solve(B, resid)

            # Calculate new amplitudes
            t1 = np.zeros_like(self.oldt1)
            t2 = np.zeros_like(self.oldt2)
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

