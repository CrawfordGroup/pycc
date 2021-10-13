"""
integrators.py: various ordinary differential equation solvers for time-domain propagation
"""

import numpy as np

class rk2(object):
    """
    Integrator object for 2nd-order Runge-Kutta ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        k2 = f(t + 2 / 3 * self.h, y + self.h * 2 / 3 * k1)

        # time step i + h
        y_new = y + self.h * (0.25 * k1 + 0.75 * k2)

        return y_new

class rk3(object):
    """
    Integrator object for 3rd-order Runge-Kutta ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        k2 = f(t + 0.5 * self.h, y + 0.5 * self.h * k1)
        k3 = f(t + self.h, y + self.h * (-1.0 * k1 + 2.0 * k2))

        # time step i + h
        y_new = y + self.h * (k1 + 4 * k2 + k3) / 6.0

        return y_new

class rk38(object):
    """
    Integrator object for "corrected" 3rd-order Runge-Kutta ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # For time step i
        # time step i
        k1 = f(t, y)
        k2 = f(t + 1 / 3 * self.h, y + 1 / 3 * self.h * k1)
        k3 = f(t + 2 / 3 * self.h, y + self.h * (-1 / 3 * k1 + k2))
        k4 = f(t + self.h, y + self.h * (k1 - k2 + k3))

        # time step i + h
        y_new = y + self.h * (k1 + 3 * k2 + 3 * k3 + k4) / 8.0
    
        return y_new

class rk4(object):
    """
    Integrator object for 4th-order Runge-Kutta ODE propagation.
    """
    def __init__(self,h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        k2 = f(t + 0.5 * self.h, y + 0.5 * self.h * k1)
        k3 = f(t + 0.5 * self.h, y + 0.5 * self.h * k2)
        k4 = f(t + self.h, y + self.h * k3)

        # time step i + h
        y_new = y + self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        return y_new

class gl6(object):
    """
    Integrator object for 6th-order Gauss-Legendre ODE propagation.
    """
    def __init__(self, h, Z_conv=1e-7):
        self.h = float(h)
        self.Z_conv = float(Z_conv)

    def __call__(self, f, t, y):
        # Order 2s = 6
        s = 3
    
        # input coefficients
        A = np.array([[5 / 36, 2 / 9 - np.sqrt(15) / 15, 5 / 36 - np.sqrt(15) / 30],
                      [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
                      [5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15) / 15, 5 / 36]])
        B = np.array([5 / 18, 4 / 9, 5 / 18])
        C = np.array([1 / 2 - np.sqrt(15) / 10, 1 / 2, 1 / 2 + np.sqrt(15) / 10])
    
        # check the coeffecients
        """
        a = [0, 0, 0]
        b = 0
        for i in range(3):
            aa = 0
            for j in range(3):
                aa += A[i][j]
            a[i] = aa
            b += B[i]
        for i in range(3):
            if a[i] - C[i] < 1E-10:
                pass
                # print("coefficients a%d, c%d pass!"% (i, i))
            else:
                print("check a, c again!")
        if b == 1:
            pass
            # print("coefficients b pass!")
        else:
            print("check c again!")
        """

        # gl6
        # Calculate Z
        # Maxiter for Z
        nk = 10
        # Initial guess of Z
        F = np.array([f(t + C[0] * self.h, y), f(t + C[1] * self.h, y), f(t + C[2] * self.h, y)])
        Z = self.h * (C.T * F.T).T * 0.0
        # print(Z.shape)
        # Z = np.zeros((3, 18240)) + 0.0j
        Z_new = Z.copy()
        k = 0
        for k in range(nk):
            F = np.array([f(t + C[0] * self.h, y + Z[0]), f(t + C[1] * self.h, y + Z[1]),
                          f(t + C[2] * self.h, y + Z[2])])
            for m in range(s):
                Z_new[m] = self.h * np.dot(A[m], F)
            if np.linalg.norm(Z_new - Z) < self.Z_conv:
                Z = Z_new
                F = np.array([f(t + C[0] * self.h, y + Z[0]), f(t + C[1] * self.h, y + Z[1]),
                              f(t + C[2] * self.h, y + Z[2])])
#                print("Z has converged in %d iterations." % (k + 1))
                break
            else:
                Z = Z_new
    
        if k == nk - 1:
            print("Z has not convered in %d iterations, please choose a larger nk." % nk)

        # time step i + h
        y_new = y + self.h * np.dot(B, F)
    
        return y_new
