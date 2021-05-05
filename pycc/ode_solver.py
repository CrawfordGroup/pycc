import numpy as np
from scipy import special


class RK(object):
    def __init__(self,f,y0,t0,tf,h,method,dtype='complex128'):
        self.f  = f
        self.y0 = y0
        self.t0 = t0
        self.h  = h
        self.N  = int((tf-t0)/h)+1
        self.grid = np.linspace(t0,tf,self.N)
        self.dtype = dtype

        self.A = method.A
        self.b = method.b
        self.c = method.c

        self.counter = 0

    def stages(self,t_n,y_n):
        s = len(self.c)
        k = np.zeros((s,len(y_n)),dtype=self.dtype)
        for i in range(s):
            y = np.zeros_like(y_n)
            for j in range(s):
                y = y + self.A[i,j] * k[j]
                k[i] = self.f(t_n + self.c[i]*self.h, y_n + self.h*y)
                self.counter += 1
        return k

    def step(self):
        t_n,y_n = self.grid[0],self.y0        # initial value
        yield t_n,y_n

        for n in range(1,self.N):
            k_n = self.stages(t_n,y_n)
            t_n += self.h
            for i in range(len(self.b)):
                y_n = y_n + self.h*self.b[i]*k_n[i]
            yield t_n,y_n

    def solve(self):
        self.solution = list(self.step())

class RK_adaptive(object):
    def __init__(self,f,y0,t0,tf,h,method,tolerance,order='',dtype='complex128'):
        self.f  = f
        self.y0 = y0
        self.t0 = t0
        self.tf = tf
        self.h  = h
        self.N  = int((tf-t0)/h)+1
        self.tol= tolerance
        self.order = order
        self.dtype = dtype

        self.A = method.A
        self.b = method.b
        self.b2= method.b2
        self.c = method.c

        self.counter = 0
        self.stepsize = []

    def stages(self,ti,yi):
        s = len(self.c)
        k = np.zeros((s,len(yi)),dtype=self.dtype)
        for n in range(s):
            y = np.zeros_like(yi)
            for m in range(s):
                y = y + self.A[n, m] * k[m]
                k[n] = self.f(ti + self.c[n]*self.h , yi + self.h*)
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
