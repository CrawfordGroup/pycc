import numpy as np
from scipy.linalg import solve,toeplitz
from scipy.fftpack import fft,fftfreq

def FT(data,dt=1,norm=False,n=None):
    """
    Quick Fourier transform through scipy's FFTPACK
    Given data, timestep
    Returns positive region of symmetric FFT
    Optional normalization (base peak of real and imag to +/-1)
    Optional padding or cutting of resulting FFT
    """
    if not n:
        n = len(data)

    FT = fft(data,n=n)[1:n//2]
    freq = fftfreq(n)[1:n//2]*2*np.pi/dt

    if norm: # normalize real and imaginary components separately
        r = np.real(FT) / np.abs(np.real(FT)).max()
        i = np.imag(FT) / np.abs(np.imag(FT)).max()
        FT = r + i*1j

    return freq,FT

class Pade():
    """
    A container for Padé approximants
    Based on Bruner et al (10.1021/acs.jctc.6b00511)
    """

    def __init__(self,data,dt=1):
#        # len(data) == M+1 and we require type(N) == type(M/2) == int
#        # therefore len(data) needs to be odd
#        if (len(data) % 2 == 0):
#            print("Odd number required - removing last data point.")
#            self.data = data[:-1]
#        else:
#            self.data = data
#        self.M = len(self.data) - 1
        # actually, len(data) == M
        if (len(data) % 2 != 0):
            print("Even number of data points required - removing last data point.")
            self.data = data[:-1]
        else:
            self.data = data
        self.M = len(self.data)
        self.N = int(self.M / 2)
        self.dt = dt

    def build(self,toeplitz_solver=True):
        """
        form c, d, and G
        solve Eq34 for b and Eq35 for a
        """
        M = self.M
        N = self.N

        c = self.data # M data points
        d = -1 * c[N:] 

        # solve eq 34
        b = np.ones(N+1)
        if toeplitz_solver:
            G = (c[N:2*N], np.flip(c[:N+1])[:-1])
            b[1:] = solve(toeplitz(*G), d, overwrite_a=True, overwrite_b=True)
        else:
            G = np.zeros((N,N))
            for k in range(0,N):
                for m in range(0,N):
                    G[k][m] = c[N-m+k]
            b[1:] = solve(G,d)

        # solve eq 35
        a = np.zeros(N+1)
#        a[0] = c[0]
        for k in range(0,N):
            for m in range(0,k+1):
                a[k] += b[m]*c[k-m]

        self.a = a
        self.b = b

    def approx(self,o,norm=False):
        """
        approximate spectrum in range o (Eq29)
        """
        try:
            a = self.a
            b = self.b
        except AttributeError:
            raise AttributeError("Please `build()` Padé object.")

        O = np.exp(-1j*o*self.dt)
        p = np.zeros(len(o), dtype='complex128') 
        q = np.zeros(len(o), dtype='complex128') 
        zk = 1
        for k in range(0,self.N+1):
            zk *= O
            p += a[k]*zk
            q += b[k]*zk
        F = p/q

        if norm:
            r = np.real(F) / np.abs(np.real(F)).max()
            i = np.imag(F) / np.abs(np.imag(F)).max()
            F = r + i*1j

        return F
