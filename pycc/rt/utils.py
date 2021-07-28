import numpy as np
from scipy.linalg import solve,toeplitz
from scipy.fft import fft,fftfreq,ifft
import copy
from scipy.signal import find_peaks, peak_widths


def FT(data,dt=1,norm=False,n=None):
    """
    Fast discrete Fourier transform through scipy's FFTPACK

    Parameters
    ----------
    data : array
        one-dimensional time-domain data

    Optionals
    ---------
    dt : float
        time step for calculating frequency, default=1
    norm : bool
        return component-normalized signal, default=False
    n : int
        number of points desired for the FFT, truncates or zero-pads
        default=None [len(data) is used]

    Returns
    -------
    freq : np.ndarray
        frequencies of the resulting FFT
    FT : np.ndarray
        the resulting frequencies

    Examples
    --------
    >>> w,i = FT(dipole_array,dt=0.01,norm=True)
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
    Denoise a given signal in the time domain using fast fourier transform

    Parameters
    ----------
    f : np.ndarray
        one-dimensional time-domain data
    filter_level : float/int
                   level below which the values can be zeroed out
                    in the frequency domain
    timestep : float/int
               incremental change in the independent variable
    Returns
    -------
    fifft : np.ndarray
            denoised signal in the time domain
    """

    length = len(f)
    fhat = fft(f, length)
    PS = fhat * np.conj(fhat)/length

    indices = PS > filter_level
    fhat = indices * fhat
    fifft = ifft(fhat)
    fifft = np.real(fifft)
    return fifft


def damp(f, timestep, Tau):
    """
    Dampen a given signal in the time domain using the
    equation: f*e^(-t*Tau), where t is the time domain.

    Parameters
    ----------
    f : np.ndarray
        one-dimensional time-domain data

    timestep : float/int
               incremental change in the independent variable

    Tau : float/int
          Damping factor

    Returns
    -------
    damped_sig : np.ndarray
                 dampened signal in the time domain using the
                 equation: f*e^(-t*Tau), where t is the time
                 domain.
    """
    t = np.arange(0, len(f))*timestep
    damped_sig = f*np.exp(-t/Tau)
    return damped_sig


def FWHM(freq_f, timestep):
    """
    Find the Full Width Half Max of a function by returning the
    width at the halfway point of the highest peak in the frequency
    domain.

    Parameters
    ----------
    freq_f : np.ndarray
             one-dimensional frequency-domain data

    timestep : float/int
               incremental change in the independent variable

    Returns
    -------
    FWHM : float/int
           the full width half max of the signal in the frequency domain
    """

    length = len(freq_f)

    PS = np.real(freq_f * np.conj(freq_f)/length)
    freq = np.real(fftfreq(length)*2*np.pi/timestep)
    L = np.arange(1, np.floor(length/2), dtype = 'int')

    peaks, _ = find_peaks(PS[L])
    sf = abs(freq[L][0]-freq[L][1])

    results_half = peak_widths(PS[L], peaks, rel_height=0.5)
    FWHM = results_half[0][np.where(results_half[1] == max(results_half[1]))]*sf
    FWHM = FWHM[0]
    return FWHM


class Pade():
    """
    A container for Padé approximants
    Based on Bruner et al (10.1021/acs.jctc.6b00511)

    Methods
    -------
    build()
        solves the system of equations for the a and b Padé coefficients
    approx()
        approximates the discrete Fourier transform for a given frequency range

    Examples
    --------
    >>> p = Pade(data,dt=0.01)
    >>> p.build()
    >>> F = p.approx(np.linspace(0,10,101),norm=True)
    """

    def __init__(self,data,dt=1):
        """
        Parameters
        ----------
        data : array
            one-dimensional time-domain data

        Optionals
        ---------
        dt : float
            time step for calculating frequency, default=1
        """
        # len(data) == M+1 and we require type(N) == type(M/2) == int
        # therefore len(data) needs to be odd
        if (len(data) % 2 == 0):
            print("Odd number required - removing last data point.")
            self.data = copy.deepcopy(data[:-1])
        else:
            self.data = copy.deepcopy(data)
        self.M = len(self.data) - 1
        self.N = int(self.M / 2)
        self.dt = dt


    def build(self,toeplitz_solver=True):
        """
        forms c, d, and G
        solve Eq34 for b and Eq35 for a

        Optionals
        ---------
        toeplitz_solver : bool
            solve the b equations by recognizing G as a toeplitz matrix
            default = True
        """
        N = self.N

        c = self.data # M+1 data points
        d = -1 * c[N+1:]
        if len(d) != N:
            raise ValueError("Why is your d vector {} elements long?".format(len(d)))
        self.d = d

        # solve eq 34
        # toeplitz solve courtesy of Eirill Strand Hauge
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
        # toeplitz courtesy of Joshua Goings
        if toeplitz_solver:
            a = np.dot(np.tril(toeplitz(c[0:N+1])),b)
        else:
            a = np.zeros(N+1)
            a[0] = c[0]
            for k in range(1,N+1):
                for m in range(0,k+1):
                    a[k] += b[m]*c[k-m]
        self.a = np.asarray(a)
        self.b = np.asarray(b)


    def approx(self,o,norm=False):
        """
        approximate spectrum in range o (Eq29)

        Parameters
        ----------
        o : array
            array of frequencies to evaluate FFT[data]

        Optionals
        ---------
        norm : bool
            return component-normalized signal, default=False

        Returns
        -------
        F : np.ndarray
            Complex Padé approximant to the Fourier transform
        """
        try:
            a = self.a
            b = self.b
        except AttributeError:
            raise AttributeError("Please `build()` Padé object.")

        # poly1d trick courtesy of Joshua Goings
        O = np.exp(-1j*o*self.dt)
        p = np.poly1d(np.flip(a))
        q = np.poly1d(np.flip(b))

        F = p(O)/q(O)

        if norm:
            r = np.real(F) / np.abs(np.real(F)).max()
            i = np.imag(F) / np.abs(np.imag(F)).max()
            F = r + i*1j

        return np.asarray(F)
