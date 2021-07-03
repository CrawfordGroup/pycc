"""
FFT test for the real-time submodule.
"""

# Import package, test suite, and other packages as needed
from pycc.rt.utils import FT
from scipy.fftpack import fft,fftfreq
import pytest
import numpy as np

def test_ft():
    np.random.seed(7)
    y = np.random.rand(500)

    w,i = FT(y)

    # rt.FT returns only the symmetric part
    i_ref = fft(y)[1:500//2]
    w_ref = fftfreq(500)[1:500//2] * 2*np.pi

    assert np.allclose(w, w_ref)
    assert np.allclose(i, i_ref)

def test_ft_norm():
    np.random.seed(10)
    y = np.random.rand(500)

    w,i = FT(y,norm=True)

    w_ref = fftfreq(500)[1:500//2] * 2*np.pi
    i_ref = fft(y)[1:500//2]

    # norm handles real and imaginary normalization separately
    re = np.real(i_ref) / np.abs(np.real(i_ref)).max()
    im = np.imag(i_ref) / np.abs(np.imag(i_ref)).max()
    i_ref = (re + im*1j)

    assert np.abs((np.abs(np.real(i)).max() - 1)) < 1E-8
    assert np.abs((np.abs(np.imag(i)).max() - 1)) < 1E-8
    assert np.allclose(w, w_ref)
    assert np.allclose(i, i_ref)

def test_ft_sanity():
    '''given enough points, FFT[sin(t) + cos(3t)] should have one real and imaginary
    peak at 1 and 3, respectively, but numerically there will be slight errors'''
    t = np.linspace(0,2*np.pi*100,10000)
    y = np.sin(t) + np.cos(3*t)
    dt = t[-1] - t[-2]

    w, i = FT(y,dt=dt,norm=True)

    re_max = np.argmax(np.abs(np.real(i)))
    im_max = np.argmax(np.abs(np.imag(i)))

    assert (w[re_max] - 3) < 1E-4 
    assert (w[im_max] - 1) < 1E-4 
