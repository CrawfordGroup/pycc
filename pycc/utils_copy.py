import numpy as np
import torch
import opt_einsum


class helper_diis(object):
    def __init__(self, t1, t2, max_diis, precision='DP'):
        if isinstance(t1, torch.Tensor):
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.oldt1 = t1.clone()
            self.oldt2 = t2.clone()
            self.diis_vals_t1 = [t1.clone()]
            self.diis_vals_t2 = [t2.clone()]
        else:
            self.oldt1 = t1.copy()
            self.oldt2 = t2.copy()
            print(self.oldt1.shape)
            print(self.oldt1)
            print(self.oldt2.shape)
            self.diis_vals_t1 = [t1.copy()]
            self.diis_vals_t2 = [t2.copy()]

        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis
        self.precision = precision

    def add_error_vector(self, t1, t2):
        if isinstance(t1, torch.Tensor):
            # Add DIIS vectors
            self.diis_vals_t1.append(t1.clone())
            self.diis_vals_t2.append(t2.clone())
            # Add new error vectors
            error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
            error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
            self.diis_errors.append(torch.cat((error_t1, error_t2)))
            self.oldt1 = t1.clone()
            self.oldt2 = t2.clone()
        else:
            # Add DIIS vectors
            self.diis_vals_t1.append(t1.copy())
            self.diis_vals_t2.append(t2.copy())
            
            # Add new error vectors
            error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
            error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
            self.diis_errors.append(np.concatenate((error_t1, error_t2)))
            self.oldt1 = t1.copy()
            self.oldt2 = t2.copy()

    def extrapolate(self, t1, t2):
        
        if (self.max_diis == 0):
            return t1, t2

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        if isinstance(t1, torch.Tensor):
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

            # Save extrapolated amplitudes to old_t amplitudes
            self.oldt1 = t1.clone()
            self.oldt2 = t2.clone()

        else:
            # Build error matrix B
            if self.precision == 'DP':
                B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
            elif self.precision == 'SP':
                B = np.ones((self.diis_size + 1, self.diis_size + 1), dtype=np.float32) * -1
            B[-1, -1] = 0

            for n1, e1 in enumerate(self.diis_errors):
                B[n1, n1] = np.dot(e1, e1)
                print("diis_errors",self.diis_errors[n1])
                print("B", B.shape)
                print("n1","e1",n1,e1.shape)      
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
            self.oldt1 = t1.copy()
            self.oldt2 = t2.copy()

        return t1, t2

class helper_ldiis(object):
    def __init__(self, no,t1_ii, t2_ij, max_diis):
        oldt1 = []
        oldt2 = []
        for i in range(no): 
             
            oldt1.append(np.array(t1_ii[i],dtype="object"))
            for j in range(no):
                ij = i*no + j
                
                oldt2.append(np.array(t2_ij[ij],dtype="object"))

        self.oldt1 = oldt1
        self.oldt2 = oldt2
        print(self.oldt1)
        self.diis_vals_t1 = [np.array(self.oldt1,dtype="object")]
        self.diis_vals_t2 = [np.array(self.oldt2,dtype="object")]
        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis

    def add_error_vector(self, no,t1_ii, t2_ij):
        # Add DIIS vectors
        self.addt1 = []
        self.addt2 = []
        for i in range(no):
            ii = i*no + i 
 
            self.addt1.append(np.array(t1_ii[i],dtype="object"))
            for j in range(no):
                ij = i*no + j 
    
                self.addt2.append(np.array(t2_ij[ij],dtype="object"))

        self.diis_vals_t1.append(np.array(self.addt1,dtype="object"))
        self.diis_vals_t2.append(np.array(self.addt2,dtype="object"))
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        self.diis_errors.append(np.concatenate((error_t1, error_t2)))
        self.oldt1 = self.addt1.copy()
        self.oldt2 = self.addt2.copy()

    def extrapolate(self, t1_ii, t2_ij):

        if (self.max_diis == 0):
            return t1_ii, t2_ij

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)
     
        B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
             print("B",B.shape )
             print("n1","e1", n1, e1.shape)
             print("diis_errors",self.diis_errors[n1])
             B[n1, n1] = np.dot(e1, e1)
             for n2, e2 in enumerate(self.diis_errors):
                 if n1 >= n2:
                     continue
                 B[n1, n2] = np.dot(e1, e2)
                 B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector
        resid = np.zeros(self.diis_size + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1_ii = np.zeros_like(self.oldt1)
        t2_ij = np.zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1_ii += ci[num] * self.diis_vals_t1[num + 1]
            t2_ij += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = t1_ii.copy()
        self.oldt2 = t2_ij.copy()

        t1_ii = list(np.float_(t1_ii))#float(t1_ii.tolist())
        t2_ij = list(np.float_(t2_ij))#float(t2_ij.tolist())

        return t1_ii, t2_ij

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

