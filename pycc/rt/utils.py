import numpy as np
from scipy.linalg import solve,toeplitz
from scipy.fftpack import fft,fftfreq
from scipy.signal import chirp, find_peaks, peak_widths

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

def denoise(f, filter_level, timestep):
    """
    Denoise the signal in the time domain using FFT
    """
    # Use PS to filter out the noise
    length = len(f) #Number of data points
    x_array = np.arange(0, length)
    fhat = np.fft.fft(f, length) #FFT
    PS = fhat * np.conj(fhat)/length #Power spectrum
    freq = np.fft.fftfreq(length)*2*np.pi/timestep
    L = np.arange(1, np.floor(length/2), dtype = 'int')
    
    indices = PS > filter_level #Filter out the significant frequencies
    PSClean = PS * indices #Zero out all other values
    fhat = indices * fhat  #Zero out small FT coeff.
    fifft = np.fft.ifft(fhat) #Inverse FFT
    return np.real(fifft)
    
def damp(f, timestep, Tau):
    """
    Dampen the signal in the time domain
    """
    t = np.arange(0, len(f))*timestep
    damped_sig = f*np.exp(-t/T)
    x_array = np.arange(0, len(damped_sig))
    return damped_sig
    
def FWHM(freq_f, timestep):
    """
    Find the FWHM of the signal in the frequency domain
    """

    if len(freq_f)%2 == 1:
        length = 2*(len(freq_f)+1)
    else:
        length = 2*len(freq_f)
    freq = np.fft.fftfreq(length)*2*np.pi/timestep
    L = np.arange(1, np.floor(length/2), dtype = 'int')
    
    peaks, _ = find_peaks(freq_f)
    sf = abs(freq[L][0]-freq[L][1])
    results_half = peak_widths(freq_f, peaks, rel_height=0.5)
    FWHM = results_half[0][np.where(results_half[1] == max(results_half[1]))]*sf
    max_height = max(results_half[1])*2
    return FWHM[0]    

class Pade():
    """
    A container for Padé approximants
    Based on Bruner et al (10.1021/acs.jctc.6b00511)
    """

    def __init__(self,data,dt=1):
        if (len(data) % 2 == 0):
            print("Odd number required - removing last data point.")
            self.data = data[:-1]
        else:
            self.data = data
        self.M = len(data)-1
        self.N = self.M//2
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
