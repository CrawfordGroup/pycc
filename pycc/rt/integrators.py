"""
integrators.py: various ordinary differential equation solvers for time-domain propagation
"""

__all__ = ['Euler', 'Midpoint', 'Heun', 'Ralston', 'rk3', 'rk38', 'rk4', 'HE', 'Fehlberg', 'BS', 'CK', 'dopri5',
           'Euler_I', 'Midpoint_I', 'Radau_IA3', 'Radau_IIA3', 'Radau_IA5', 'Radau_IIA5', 'SDIRK5', 'gl4', 'gl6']

import numpy as np

"""
Runge-Kutta family of integrators for ODE propagaton.
"""

"""
1. Explicit integrtors:
   1st-order: Euler;
   2nd-order: Midpoint, Heun, Ralston;
   3rd-order: RK3;
   4th-order: RK4, RK38;

2. Adaptive (embeded) integrators:
   1st-order: Heun-Euler;
   2nd-order: Fehlberg;
   3rd-order: Bogacki-Shampine;
   4th-order: Cash-Karp;
   5th-order: Dormand-Prince;

3. implicit integrators:
   1st-order: Euler_I (diagoally implicit method);
   2nd-order: Midpoint_I (diagonally implicit method);
   3rd-order: Radau_IA3, Radau_IIA3;
   4th-order: SDIRK5 (singly diagonally implicit method),
              Gauss-legendre 4th-order method;
   5th-order: Radau_IA5, Radau_IIA5;
   6th-order: Gauss-legendre 6th-order method.     
"""

"""
Explicit integrators
"""
class Euler(object):
    """
    Integrator object for Euler ODE propagaton.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        
        # time step i+1
        y_new = y + self.h * k1

        return y_new

class Midpoint(object):
    """
    Integrator object for Midpoint ODE propagation
    """
    def __init__(self, h):
        self.h = float(h)
    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        k2 = f(t + 0.5 * self.h, y + 0.5 * self.h * k1)
        
        # time step i+1
        y_new = y + self.h * k2
        
        return y_new

class Heun(object):
    """
    Integrator object for Heun ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)
    
    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        k2 = f(t + self.h, y + self.h * k1)

        # time step i + h
        y_new = y + self.h * (k1 + k2) / 2

        return y_new

class Ralston(object):
    """
    Integrator object for Ralston ODE propagation.
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
        k3 = f(t + self.h, y + self.h * ( -k1 + 2 * k2))

        # time step i + h
        y_new = y + self.h * (k1 + 4 * k2 + k3) / 6

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

class rk38(object):
    """
    Integrator object for "corrected" 4th-order Runge-Kutta ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # time step i
        k1 = f(t, y)
        k2 = f(t + 1 / 3 * self.h, y + 1 / 3 * self.h * k1)
        k3 = f(t + 2 / 3 * self.h, y + self.h * (-1 / 3 * k1 + k2))
        k4 = f(t + self.h, y + self.h * (k1 - k2 + k3))

        # time step i + h
        y_new = y + self.h * (k1 + 3 * k2 + 3 * k3 + k4) / 8.0
    
        return y_new

"""
Adaptive integrators
"""
class HE(object):
    """
    Integrator object for Heun-Ruler ODE propagation.
    1st-order Runge-Kutta method with 2 stages.
    """
    def __init__(self, h):
        self.h = float(h)
    
    def __call__(self, f, t, y):
        k1 = f(t, y)
        # 1st-order solution
        y_new1 = y + self.h * k1

        k2 = f(t + self.h, y_new1)
        # 2nd-order solution for adjusting the step-size
        y_new2 = y + self.h * (k1 + k2) / 2

        pass

class Fehlberg(object):
    """
    Integrator object for Fehlberg ODE propagation.
    2nd-order integrators with 3 stages.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        k1 = f(t, y)
        k2 = f(t + self.h * 0.5, y + self.h * 0.5 * k1)
        
        # 2nd-order solution 
        y_new1 = y + self.h * (k1 + 255 * k2) / 256
        
        k3 = f(t + self.h, y_new1)
        # 3rd-order solution for adjusting the step-size
        y_new2 = y + self.h * (k1 + 510 * k2 + k3) / 512
        
        pass

class BS(object):
    """
    Integrator object for Bogacki-Shampine ODE propagation.
    3rd-order Runge-Kutta integrator with 4 stages.
    """
    def __init__(self, h):
        self.h = float(h)
    
    def __call__(self, f, t, y):
        k1 = f(t, y)
        k2 = f(t + 0.5 * self.h. y + 0.5 * self.h * k1)
        k3 = f(t + 0.75 * self.h, y + 0.75 * self.h * k2)
        # 3rd-order solution
        y_new1 = y + self.h * (2 * k1 + 3 * k2 + 4 * k3) / 9.0 

        k4 = f(t + self.h, y_new)
        # 4th-order solution for adjusting the step-size
        y_new2 = y + self.h * (7 * k1 + 6 * k2 + 8 * k3 + 3 * k4) / 24.0
     
        pass

class CK(object):
    """
    Integrator oject for Cash-Karp ODE propagation
    4th-order Runge-Kutta integrator with 6 stages
    """
    def __init__(self, maxiter=10, yconv=1e-7):
        self.maxiter = int(maxiter)
        self.yconv = float(yconv)

    def __call__(self, f, t, y, h0):
        h = float(h0)
        k1 = f(t, y)
        for i in range(maxiter):
            k2 = f(t + 0.2 * h, y + h * 0.2 * k1)
            k3 = f(t + 0.3 * h, y + h * (3 * k1 + 9 * k2) / 40)
            k4 = f(t + 0.6 * h, y + h * (3 * k1 - 9 * k2 + 12 * k3) / 10)
            k5 = f(t + h, y + h * (-11 / 54 * k1 + 2.5 * k2 - 70 / 27 * k3 + 35 / 27 * k4))
            k6 = f(t + 0.875 * h, y + h * (1631 / 55296 * k1 + 175 / 512 * k2 + 575 / 13824 * k3 + 44275 / 110592 * k4 + 253 / 4096 * k5))

            # Fourth-order solution
            y_new1 = y + h * (37 / 378 * k1 + 250 / 621 * k3 + 125 / 594 * k4 + 512 / 1771 * k6)
            # Fifth-order solution
            y_new2 = y + h * (2825 / 27648 * k1 + 18575 / 48384 * k3 + 13525 / 55296 * k4 + 277 / 14336 * k5 + k6 / 4)
         
            # For the adaptive time step, the difference b/t the fourth- and fifth-order solutions will be needed
            err = np.linalg.norm(y_new1 - y_new2)
            h_new = 0.84 * h * pow((yconv / err), 0.25)
            if (err < conv):
                return (y_new1, h, h_new)
            if (i == maxiter - 1):
                print("y did not converge with in %d iterations \n" % maxiter)
                return (y_new1, h, h_new)
            h = h_new

class dopri5(object):
    """
    Integrator object for Dormand-Prince ODE propagation
    5th-order Runge-kutta integrator with 7 stages
    """
    def __init__(self, maxiter, y_conv):
        self.maxiter = int(maxiter)
        self.y_conv = float(y_conv)

    def __call__(self, f, t, y, h0):
        h = float(h0)
        k1 = f(t, y)
        for i in range(maxiter):            
            k2 = f(t + 0.2 * h, y + h * 0.2 * k1)
            k3 = f(t + 0.3 * h, y + h * (3 * k1 + 9 * k2) / 40)
            k4 = f(t + 0.8 * h, y + h * (44 * k1 - 168 * k2 + 160 * k3) / 45)
            k5 = f(t + 8 / 9 * h, y + h * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4))
            k6 = f(t + h, y + h * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 + 11 / 84 * k5))
            # Fifth-order solution
            y_new1 = y + h * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)

            # Same as k1 in the next step
            k7 = f(t + h, y_new1)
            # Sixth-order solution
            y_new2 = y + h * (5179 / 57600 * k1 + 7571 / 16695 * k3 + 393 / 640 * k4 - 92097 / 339200 * k5 + 187 / 2100 * k6 + k7 / 40)

            err = np.linalg.norm(y_new1 - y_new2)       
            h_new = 0.84 * pow((yconv / err), 0.25) * h
            if (err < yconv):      
                return (y_new1, h_new)
            if (i == maxiter - 1):
                print("y did not converge with in %d iterations \n" % maxiter) 
                return (y_new1, h_new)
            h = h_new

"""
Implicit integrators
"""     
class Euler_I(object):
    """
    Integrator object for Euler implicit ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)
  
    def __call__(self, f, t, h):
        # order 1
        A = np.asarray([1])
        b = [1]
        c = [1] 
        pass

class Midpoint_I(object):
    """
    Integrator object for Midpoint implicit ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)
    
    def __init__(self, f, t, y):
        # order 2
        A = np.asarray([0.5])
        b = [1]
        c = [0.5]
        pass

class SDIRK5(object):
    """
    Integrator object for singly diagonally implicit Runge-Kutta 5th-order ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)
   
    def __call__(self, f, t, y):
        A = np.asarray([
            [1/4,0,0,0,0],
            [1/2,1/4,0,0,0],
            [17/50,-1/25,1/4,0,0],
            [371/1630,-137/2720,15/544,1/4,0],
            [25/24,-49/48,125/16,-85/12,1/4]])
        b = [25/24,-49/48,125/16,-85/12,1/4]
        c = [1/4,3/4,11/20,1/2,1] 
        pass  

class Radau_IA3(object):
    """
    Integrator object for Radau IA 3rd-order ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # order 3 
        A = np.asarray([[1/4,-1/4],[1/4,5/12]])
        b = [1/4,3/4]
        c = [0,2/3]
        pass

class Radau_IIA3(object):
    """
    Integrator object for Raudau IIA 3rd-order ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # order 3
        A = np.asarray([[5/12,-1/12],[3/4,1/4]])
        b = [3/4,1/4]
        c = [1/3,1]
        pass

class Radau_IA5(object):
    """
    Integrator object for Raudau IA 5th-order ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # order 5
        a = np.sqrt(6)
        A = np.asarray([
            [1/9,(-1-a)/18,(-1+a)/18],
            [1/9,11/45+7*a/360,11/45-43*a/360],
            [1/9,11/45+43*a/360,11/45-7*a/360]])
        b = [1/9,4/9+a/36,4/9-a/36]
        c = [0,3/5-a/10,3/5+a/10]
        pass

class Radau_IIA5(object):
    """
    Integrator object for Raudau IIA 5th-order ODE propagation.
    """
    def __init__(self, h):
        self.h = float(h)

    def __call__(self, f, t, y):
        # order 5
        a = np.sqrt(6)
        A = np.asarray([
            [11/45-7*a/360,37/225-169*a/1800,-2/225+a/75],
            [37/225+169*a/1800,11/45+7*a/360,-2/225-a/75],
            [4/9-a/36,4/9+a/36,1/9]])
        b = [4/9-a/36,4/9+a/36,1/9]
        c = [2/5-a/10,2/5+a/10,1]
        pass

class gl4(object):
    """
    """
    def __init__(self, h, Z_conv=1e-7):
        self.h = float(h)
        self.Z_conv = float(Z_conv)
    
    def __call__(self, f, t, y):
        # Order 2s = 4
        s = 2

        # input coefficients
        A = np.array([[1 / 4 , 1 / 4 + np.sqrt(3) / 6], [1 / 4, 1 / 4 - np.sqrt(3) / 6]])
        B = np.array([1 / 2, 1 / 2])
        C = np.array([1 / 2 - np.sqrt(3) / 6, 1 / 2 + np.sqrt(3) / 6])
        # check the coeffecients

        """
        a = [0, 0, 0]
        b = 0
        for i in range(2):
            aa = 0
            for j in range(2):
                aa += A[i][j]
            a[i] = aa
            b += B[i]
        for i in range(2):
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
        # gl4
        # Calculate Z
        # Maxiter for Z
        nk = 10
        # Initial guess of Z
        F = np.array([self.f(t + C[0] * self.h, y), self.f(t + C[1] * self.h, y), self.f(t + C[2] * self.h, y)])
        Z = self.h * (C.T * F.T).T * 0.0
        #print(Z.shape)
        #Z = np.zeros((3, 18240)) + 0.0j
        Z_new = Z.copy()
        k = 0
        for k in range(nk):
            F = np.array([self.f(t + C[0] * self.h, y + Z[0]), self.f(t + C[1] * self.h, y + Z[1]),
                          self.f(t + C[2] * self.h, y + Z[2])])
            for m in range(s):
                Z_new[m] = h * np.dot(A[m], F)
            if np.linalg.norm(Z_new - Z) < Z_conv:
                Z = Z_new
                F = np.array([self.f(t + C[0] * self.h, y + Z[0]), self.f(t + C[1] * self.h, y + Z[1]),
                           self.f(t + C[2] * self.h, y + Z[2])])
                print("Z has converged in %d iterations." % (k + 1))
                break
            else:
                Z = Z_new

        if k == nk - 1:
            print("Z has not convered in %d iterations, please choose a larger nk." % nk)
        y_new = y + h * np.dot(B, F)

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


