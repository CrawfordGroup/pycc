import numpy as np
from scipy import special

# Checks for type (explicit, implicit, ..), consistency, symplecticity, ... (useful to spot typos in Tableaux)
def check(method):
    # Check if method is explicit
    if np.allclose(method.A,np.tril(method.A)):
        if np.trace(method.A)==0:
            print('The %s method is explicit' %method.name)
    # Check if method is singly diagonally implicit
        elif np.allclose(np.eye(len(method.b)),np.diag(np.diag(method.A)/method.A[0,0])):
            print('The %s method is singly diagonally implicit' %method.name)
    # Check if method is diagonally implicit
        else:
            print('The %s method is diagonally implicit' %method.name)
    # Check if method is implicit
    else:
        print('The %s method is implicit' %method.name)

    # Check consistency
    if not np.allclose(method.A.sum(axis=1),method.c):
        print('Warning: There is something wrong with the Butcher Tableau for the %s method' %method.name)
    else:
        print('The %s method is consistent' %method.name)
        if sum(method.b)-1>1e-12:
            print('Warning: The %s method may lead to more than one ODE solution' %method.name)

    # Check symplecticity
    symplectic = True
    s = len(method.b)
    for i in range(s):
        for j in range(s):
            if abs(method.b[i] * method.A[i,j] + method.b[j] * method.A[j,i] - method.b[i] * method.b[j]) > 1e-12:
                symplectic = False
                break
    if symplectic:
        print('The %s method is symplectic' %method.name)
    else:
        print('The %s method is not symplectic' %method.name)

# Main Propagator class
class Propagators(object):
    pass

# Subclasses: Explicit, Embedded (adaptive), Implicit
class Explicit(Propagators):
    flag = 'Explicit'
class Embedded(Propagators):
    flag = 'Embedded'
class Implicit(Propagators):
    flag = 'Implicit'

# Explicit methods
class Euler(Explicit):        # order 1
    name = 'Euler'
    order = 1
    A = np.asarray([[0]])
    b = [1]
    c = [0]

class Midpoint(Explicit):     # order 2
    name = 'Midpoint'
    order = 2
    A = np.asarray([[0,0],[1/2,0]])
    b = [0,1]
    c = [0,1/2]

class Heun(Explicit):         # order 2
    name = 'Heun'
    order = 2
    A = np.asarray([[0,0],[1,0]])
    b = [1/2,1/2]
    c = [0,1]

class Ralston(Explicit):      # order 2
    name = 'Ralston'
    order = 2
    A = np.asarray([[0,0],[2/3,0]])
    b = [1/4,3/4]
    c = [0,2/3]

class Kutta3(Explicit):       # order 3
    name = '3rd order Kutta'
    order = 3
    A = np.asarray([[0,0,0],[1/2,0,0],[-1,2,0]])
    b = [1/6,2/3,1/6]
    c = [0,1/2,1]

class RK4(Explicit):          # order 4
    name = '4th order Runge Kutta'
    order = 4
    # Butcher Tableau for RK4
    A = np.asarray([[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]])
    b = [1/6,1/3,1/3,1/6]
    c = [0,1/2,1/2,1]

class RK3_8(Explicit):        # order 4
    name = '3/8 rule Runge Kutta'
    order = 4
    # Butcher Tableau for 3/8 rule
    A = np.asarray([[0,0,0,0],[1/3,0,0,0],[-1/3,1,0,0],[1,-1,1,0]])
    b = [1/8,3/8,3/8,1/8]
    c = [0,1/3,2/3,1]

# Adaptive (Embedded) methods
class HE(Embedded):           # order 1(2)
    name = 'Heun-Euler'
    order = 2
    # Heun-Euler Method
    A = np.asarray([[0,0],[1,0]])
    b = [1,0]
    b2= [1/2,1/2]
    c = [0,1]

class RK12(Embedded):         # order 1(2)
    name = 'Fehlberg'
    order = 2
    # Fehlberg method
    A = np.asarray([[0,0,0],[1/2,0,0],[1/256,255/256,0]])
    b = [1/256,255/256,0]
    b2= [1/512,255/256,1/512]
    c = [0,1/2,1]

class BS(Embedded):           # order 2(3)
    name = 'Bogacki-Shampine'
    order = 3
    # Bogacki-Shampine method
    A = np.asarray([[0,0,0,0],[1/2,0,0,0],[0,3/4,0,0],[2/9,1/3,4/9,0]])
    b = [7/24,1/4,1/3,1/8]
    b2= [2/9,1/3,4/9,0]
    c = [0,1/2,3/4,1]

class RKF(Embedded):          # order 4(5)
    name = 'Runge Kutta Fehlberg'
    order = 5
    # Runge Kutta Fehlberg method
    A = np.asarray([[0,0,0,0,0,0],[1/4,0,0,0,0,0],[3/32,9/32,0,0,0,0],[1932/2197,-7200/2197,7296/2197,0,0,0],[439/216,-8,3680/513,-845/4104,0,0],[-8/27,2,-3544/2565,1859/4104,-11/40,0]])
    b = [25/216,0,1408/2565,2197/4104,-1/5,0]
    b2= [16/135,0,6656/12825,28561/56430,-9/50,2/55]
    c = [0,1/4,3/8,12/13,1,1/2]

class CK(Embedded):           # order 4(5)
    name = 'Cash-Karp'
    order = 5
    # Cash-Karp method
    A = np.asarray([[0,0,0,0,0,0],[1/5,0,0,0,0,0],[3/40,9/40,0,0,0,0],[3/10,-9/10,6/5,0,0,0],[-11/54,5/2,-70/27,35/27,0,0],[1631/55296,175/512,575/13824,44275/110592,253/4096,0]])
    b = [2825/27648,0,18575/48384,13525/55296,277/14336,1/4]
    b2= [37/378,0,250/621,125/594,0,512/1771]
    c = [0,1/5,3/10,3/5,1,7/8]

class DP(Embedded):           # order 4(5)
    name = 'Dormand-Prince'
    order = 5
    # Dormand-Prince method
    A = np.asarray([[0,0,0,0,0,0,0],[1/5,0,0,0,0,0,0],[3/40,9/40,0,0,0,0,0],[44/45,-56/15,32/9,0,0,0,0],[19372/6561,-25360/2187,64448/6561,-212/729,0,0,0],[9017/3168,-355/33,46732/5247,49/176,-5103/18656,0,0],
        [35/384,0,500/1113,125/192,-2187/6784,11/84,0]])
    b = [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40]
    b2= [35/384,0,500/1113,125/192,-2187/6784,11/84,0]
    c = [0,1/5,3/10,4/5,8/9,1,1]



# Singly Diagonally Implicit methods
class SDIRK5(Implicit):       # order 4
    name = '4th order SDIRK'
    order = 4
    A = np.asarray([
        [1/4,0,0,0,0],
        [1/2,1/4,0,0,0],
        [17/50,-1/25,1/4,0,0],
        [371/1360,-137/2720,15/544,1/4,0],
        [25/24,-49/48,125/16,-85/12,1/4]])
    b = [25/24,-49/48,125/16,-85/12,1/4]
    c = [1/4,3/4,11/20,1/2,1]
    
# Diagonally Implicit methods
class Euler_I(Implicit):      # order 1
    name = 'Backwards Euler'
    flag = 'DIRK'
    order = 1
    # Backward Euler method
    A = np.asarray([[1]])
    b = [1]
    c = [1]

class Midpoint_I(Implicit):   # order 2
    name = 'Implicit Midpoint'
    flag = 'DIRK'
    order = 2
    # Implicit midpoint method
    A = np.asarray([[1/2]])
    b = [1]
    c = [1/2]

# Implicit methods
class RadauIA3(Implicit):     # order 3
    name = 'Radau IA 3rd order'
    order = 3
    # Radau IA method
    A = np.asarray([[1/4,-1/4],[1/4,5/12]])
    b = [1/4,3/4]
    c = [0,2/3]

class RadauIIA3(Implicit):    # order 3
    name = 'Radau IIA 3rd order'
    order = 3
    # Radau IIA method
    A = np.asarray([[5/12,-1/12],[3/4,1/4]])
    b = [3/4,1/4]
    c = [1/3,1]

class RadauIA5(Implicit):     # order 5
    name = 'Radau IA 5th order'
    order = 5
    # Radau IA method
    a = np.sqrt(6)
    A = np.asarray([
        [1/9,(-1-a)/18,(-1+a)/18],
        [1/9,11/45+7*a/360,11/45-43*a/360],
        [1/9,11/45+43*a/360,11/45-7*a/360]])
    b = [1/9,4/9+a/36,4/9-a/36]
    c = [0,3/5-a/10,3/5+a/10]

class RadauIIA5(Implicit):    # order 5
    name = 'Radau IIA 5th order'
    order = 5
    # Radau IIA method
    a = np.sqrt(6)
    A = np.asarray([
        [11/45-7*a/360,37/225-169*a/1800,-2/225+a/75],
        [37/225+169*a/1800,11/45+7*a/360,-2/225-a/75],
        [4/9-a/36,4/9+a/36,1/9]])
    b = [4/9-a/36,4/9+a/36,1/9]
    c = [2/5-a/10,2/5+a/10,1]

class GL4(Implicit):          # order 4
    name = 'Gauss Legendre 4th order'
    order = 4
    # Gauss-Legendre method
    a_p = 1/4 + np.sqrt(3)/6
    a_m = 1/4 - np.sqrt(3)/6
    A = np.asarray([[1/4,a_m],[a_p,1/4]])
    b = [1/2,1/2]
    c = [1/4+a_m,1/4+a_p]

class GL6(Implicit):          # order 6
    name = 'Gauss Legendre 6th order'
    order = 6
    # Gauss-Legendre method
    a_p = 2/9 + np.sqrt(15)/15
    a_m = 2/9 - np.sqrt(15)/15
    a1_p = 5/36 + np.sqrt(15)/24
    a1_m = 5/36 - np.sqrt(15)/24
    a2_p = 5/36 + np.sqrt(15)/30
    a2_m = 5/36 - np.sqrt(15)/30
    A = np.asarray([[5/36,a_m,a2_m],[a1_p,2/9,a1_m],[a2_p,a_p,5/36]])
    b = [5/18,4/9,5/18]
    c = [1/2-np.sqrt(15)/10,1/2,1/2+np.sqrt(15)/10]


class RK(object):
    def __init__(self,f,y0,t0,tf,h,method):
        self.f  = f
        self.y0 = y0
        self.t0 = t0
        self.h  = h
        self.N  = int((tf-t0)/h)+1
        self.grid = np.linspace(t0,tf,self.N)

        self.A = method.A
        self.b = method.b
        self.c = method.c

        self.counter = 0

    def stages(self,t_n,y_n):
        s = len(self.c)
        k = np.zeros((s,len(y_n)))
        for i in range(s):
            y = np.zeros_like(y_n)
            for j in range(s):
                y = y + self.A[i,j] * k[j]
                k[i] = self.f(t_n + self.c[i]*self.h, y_n + self.h*y)
                self.counter += 1
        return k

    def step(self):
        t_n,y_n = self.grid[0],self.y0        # initial value
        #print('{:<6s}{:5d}'.format('Step #',0)+'{:>15s}{:6.2f}'.format('time = ',t_n)+'{:>13s}{:10.6f}'.format('y = ',y_n))
        yield t_n,y_n

        for n in range(1,self.N):
            k_n = self.stages(t_n,y_n)
            t_n += self.h
            for i in range(len(self.b)):
                y_n = y_n + self.h*self.b[i]*k_n[i]
            #print('{:<6s}{:5d}'.format('Step #',n)+'{:>15s}{:6.2f}'.format('time = ',t_n)+'{:>13s}{:10.6f}'.format('y = ',y_n))
            #print('{:>15s}{:10.6f}'.format('k1 = ',k_n[0]))
            #print('{:>15s}{:10.6f}'.format('k2 = ',k_n[1]))
            #print('{:>15s}{:10.6f}'.format('k3 = ',k_n[2]))
            #print('{:>15s}{:10.6f}'.format('k4 = ',k_n[3]))
            #print()
            yield t_n,y_n

    def solve(self):
        self.solution = list(self.step())

class RK_adaptive(object):
    def __init__(self,f,y0,t0,tf,h,method,tolerance,order=''):
        self.f  = f
        self.y0 = y0
        self.t0 = t0
        self.tf = tf
        self.h  = h
        self.N  = int((tf-t0)/h)+1
        self.tol= tolerance
        self.order = order

        self.A = method.A
        self.b = method.b
        self.b2= method.b2
        self.c = method.c

        self.counter = 0
        self.stepsize = []

    def stages(self,ti,yi):
        s = len(self.c)
        k = np.zeros((s))
        for n in range(s):
            k[n] = self.h*self.f(ti + self.c[n]*self.h , yi + np.dot(self.A[n,:],k))
            self.counter += 1
        return k

    def step(self):
        ti,yi = self.t0,self.y0        # initial value
        yield ti,yi

        while ti<self.tf:
            self.h = min(self.h,self.tf-ti)

            ki = self.stages(ti,yi)
            y1 = yi + np.dot(self.b,ki)
            y2 = yi + np.dot(self.b2,ki)

            R = abs(y1-y2)/self.h
            delta = (self.tol/(2*R))**(1/4)

            if R<=self.tol:
                ti += self.h
                if self.order == 'low':
                    yi = y1             # y1 is nth order, y2 is (n+1)th order
                else:
                    yi = y2             # default, solution is advanced with the higher order estimate
            #else:
            self.h *= delta

            self.stepsize.append(self.h)

            yield ti,yi

    def solve(self):
        self.solution = list(self.step())

class DIRK(object):
    def __init__(self,f,y0,t0,tf,h,method):
        self.f  = f
        self.y0 = y0
        self.t0 = t0
        self.h  = h
        self.N  = int((tf-t0)/h)+1
        self.grid = np.linspace(t0,tf,self.N)

        self.A = method.A
        self.b = method.b
        self.c = method.c

        # Check if Runge-Kutta is Diagonally implicit
        if not np.allclose(self.A,np.tril(self.A)):
            print('Warning: The method supplied is not diagonally implicit, use a more general solver')
            self.solution = []
        # Check balance
        elif not np.allclose(self.A.sum(axis=1),self.c):
            print('Warning: There is something wrong with the Butcher Tableau')
            self.solution = []
        else:
            # Check if method is explicit
            if np.trace(self.A) == 0:
                print('Method is explicit, consider using the explicit solver')
            # Check consistency
            if sum(self.b)-1>1e-12:
                print('Warning: The ODE solution may have more than one solution')
            self.counter = 0


    def stages(self,ti,yi):
        skip = False
        s = len(self.c)
        k = np.zeros((s))
        for n in range(s):
            k_converged = False
            while not k_converged:
                tmp = self.h*self.f(ti + self.c[n]*self.h , yi + np.dot(self.A[n,:],k))
                if tmp == 0:
                    print('Warning')
                    skip = True
                    return k,skip
                self.counter += 1
                if abs(k[n]-tmp)<1e-6:
                    k_converged = True
                k[n] = tmp
        return k,skip

    def step(self):
        ti,yi = self.grid[0],self.y0        # initial value
        yield ti,yi

        for i in range(1,self.N):
            ki,skip = self.stages(ti,yi)
            if not skip:
                ti += self.h
                yi += np.dot(self.b,ki)
            else:
                ti += self.h
                ki,skip = self.stages(ti,yi)
                ti += self.h
                yi += np.dot(self.b,ki)
            yield ti,yi

    def solve(self):
        self.solution = list(self.step())

class IRK(object):
    def __init__(self,f,y0,t0,tf,h,method,guess=0,newton=0):
        self.f  = f
        self.y0 = y0
        self.t0 = t0
        self.h  = h
        self.N  = int((tf-t0)/h)+1
        self.grid = np.linspace(t0,tf,self.N)

        self.A = method.A
        self.b = method.b
        self.c = method.c

        self.Z = []

        self.counter = 0
        self.iterations = 0
        self.guess = guess
        self.newton = newton


    def stages(self,t_n,y_n):
        s = len(self.c)

        # Default guess, initial Z vector = 0                                                           (bad, plateaus at 3s f-evaluations)
        Z = np.zeros((s))
        #print('{:<6s}{:5d}'.format('Step #',int(t_n/self.h)+1))

        # Initial guess
        if t_n != self.t0:

            if self.guess == 1:         # Use Z vector from previous time step                          (good, plateaus at 2s f-evaluations)
                Z = self.Z[-1].copy()

            elif self.guess == 2:       # Use ode at (y_n,t_n)              1 extra f evaluation        (decent, plateaus at 2s+1 f-evaluations)
                F = self.f(t_n,y_n)
                self.counter += 1
                for i in range(s):
                    Z[i] = self.h*self.c[i]*F

            elif self.guess == 3:       # Use ode at (y_n,t_n + c[i]*h)     s extra f evaluations       (bad, plateaus at 3s f-evaluations)
                for i in range(s):
                    Z[i] = self.h*self.c[i]*self.f(t_n + self.c[i]*self.h, y_n)
                    self.counter += 1

            elif self.guess == 4:       # Use F from previous step                                      (best, plateaus at s f-evaluations)
                # Solve for beta coefficients using the linear Vandermonde equation
                V = np.vander(self.c,s,increasing=True)
                V2 = np.vander(np.add(self.c,1),s+1,increasing=True)[:,1:]
                for k in range(2,s+1):
                    V2[:,k-1] = V2[:,k-1]/k
                beta = np.linalg.solve(V.transpose(),V2.transpose()).transpose()

                Z = np.add(self.previous['y']-y_n,self.h*np.dot(beta, self.previous['F']))

            elif self.guess == 5:       # Use backwards differences with previous Z vectors             (good, plateaus under2s-evaluations)
                if self.newton == 0:
                    n = min(3,len(self.Z)-1)
                else:
                    n = min(self.newton,len(self.Z)-1)
                Z = self.Z[-1].copy()
                for k in range(1,n+1):
                    binom = [special.binom(i,k) for i in range(k,n+1)]
                    Z += self.Z[-1]
                    Z += (-1)**k * sum(binom) * self.Z[-k-1]


        #print('{:>12s}{:10f}{:10f}'.format('Guess: ',Z[0],Z[1]))
        #print()

        # Fixed point iterations
        converged = False
        while not converged:
            self.iterations += 1
            # Build F
            F = np.zeros((s))
            for i in range(s):
                F[i] = self.f(t_n + self.c[i]*self.h, y_n + Z[i])
                self.counter += 1
            # Build new Z vector
            Z_new = self.h * np.dot(self.A,F)
            #print('{:>12s}{:10f}{:10f}{:>10s}{:10f}'.format('Iter: ',Z_new[0],Z_new[1],'Error:',np.linalg.norm(Z_new-Z)))
            # Check convergence
            if np.allclose(Z,Z_new):
                converged = True
                self.Z.append(Z)
                #print()
            Z = Z_new
        return F

    def step(self):
        t_n,y_n = self.grid[0],self.y0        # initial value
        #print('{:<6s}{:5d}'.format('Step #',0)+'{:>15s}{:6.2f}'.format('time = ',t_n)+'{:>13s}{:10.6f}'.format('y = ',y_n))
        yield t_n,y_n

        for n in range(1,self.N):
            F_n = self.stages(t_n,y_n)
            self.previous = {'y':y_n,'F':F_n}
            t_n += self.h
            y_n += self.h*np.dot(self.b,F_n)
            #print('{:<6s}{:5d}'.format('Step #',n)+'{:>15s}{:6.2f}'.format('time = ',t_n)+'{:>13s}{:10.6f}'.format('y = ',y_n))
            #print()
            yield t_n,y_n

    def solve(self):
        self.solution = list(self.step())


class solve(object):
    def __init__(self,f,y0,t0,tf,h,method,tol=0.001,guess=0,newton=0):

        if method.flag == 'Explicit':
            S = RK(f,y0,t0,tf,h,method)
            self.iterations = 0
        elif method.flag == 'Embedded':
            S = RK_adaptive(f,y0,t0,tf,h,method,tol)
            self.stepsize = np.mean(S.stepsize)
        elif method.flag == 'DIRK':
            S = DIRK(f,y0,t0,tf,h,method)
        elif method.flag == 'Implicit':
            S = IRK(f,y0,t0,tf,h,method,guess,newton)
            self.iterations = S.iterations
        self.solution = S.solution
        self.counter = S.counter
