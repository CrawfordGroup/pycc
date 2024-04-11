import numpy as np
import torch
import opt_einsum


class DIIS(object):
    def __init__(self, T, max_diis, precision='DP'):
        if isinstance(T, torch.Tensor):
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.diis_T = [T.clone()]
        else:
            self.diis_T = [T.copy()]

        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis
        self.precision = precision

    def add_error_vector(self, T, e):
        if isinstance(T, torch.Tensor):
            self.diis_T.append(T.clone())
            self.diis_errors.append(e.ravel().clone())
        else:
            self.diis_T.append(T.copy())
            self.diis_errors.append(e.ravel())

    def extrapolate(self, T):
        
        if (self.max_diis == 0):
            return T

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_T[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        if isinstance(T, torch.Tensor):
            # Build error matrix B
            if self.precision == 'DP':
                B = -1 * torch.ones((self.diis_size + 1, self.diis_size + 1), dtype=torch.float64, device=self.device1)
            elif self.precision == 'SP':
                B = -1 * torch.ones((self.diis_size + 1, self.diis_size + 1), dtype=torch.float32, device=self.device1)

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
            T *= 0
            for num in range(self.diis_size):
                T += torch.real(ci[num] * self.diis_T[num + 1])

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
            T *= 0
            for num in range(self.diis_size):
                T += ci[num] * self.diis_T[num + 1]

        return T

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

