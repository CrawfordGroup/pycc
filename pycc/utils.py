import numpy as np


class helper_diis(object):
    def __init__(self, t1, t2, max_diis):

        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()
        self.diis_vals_t1 = [t1.copy()]
        self.diis_vals_t2 = [t2.copy()]
        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis

    def add_error_vector(self, t1, t2):

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

        # Build error matrix B
        B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
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
        resid = np.zeros(self.diis_size + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1 = np.zeros_like(self.oldt1)
        t2 = np.zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += np.real(ci[num] * self.diis_vals_t1[num + 1])
            t2 += np.real(ci[num] * self.diis_vals_t2[num + 1])

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

        return t1, t2
