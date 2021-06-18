import numpy as np
from scipy.linalg import solve,toeplitz
from scipy.fftpack import fft,fftfreq

def FT(data,dt=1,norm=False,n=None):
    """
    Quick Fourier transform through scipy's FFTPACK
    Given data, timestep
    Returns positive region of symmetric FFT
    Optional normalization (base peak to 1)
    Optional padding or cutting of resulting FFT
    """
    if not n:
        n = len(data)

    FT = fft(data,n=n)
    freq = fftfreq(FT.size)*2*np.pi/dt

    if norm:
        FT /= FT.max()

    return freq[1:len(freq)//2],FT[1:len(freq)//2]

class Pade():
    """
    A container for Padé approximants
    Based on Bruner et al (10.1021/acs.jctc.6b00511)
    """

    def __init__(self,data,dt=1):
        self.data = data
        self.M = len(data)
        self.N = M//2
        self.dt = dt

    def build(self,toeplitz_solver=True):
        """
        form c, d, and G
        solve Eq34 for b and Eq35 for a
        """
        M = self.M
        N = self.N

        c = self.data[:M+1]
        d = -1 * c[N+1:]

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
        a[0] = c[0]
        for k in range(1,N+1):
            for m in range(k+1):
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
        p = np.zeros(len(o), dtype='complex128') + a[0]
        q = np.zeros(len(o), dtype='complex128') + b[0]
        z = 1
        for k in range(1, len(a)):
            z *= O
            p += a[k]*z
            q += b[k]*z
        F = p/q

        if norm:
            F /= F.max()

        return F
