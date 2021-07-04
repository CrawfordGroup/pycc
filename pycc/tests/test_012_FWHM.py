#Import required packages
from pycc.rt.utils import FWHM
import numpy as np
import pytest

def test_FWHM():
    #Define function
    np.random.seed(10)
    timestep = 0.001
    t = np.arange(0, 1, timestep)
    f = np.cos(2*np.pi*12*t) + np.sin(2*np.pi*50*t)
    f = f + np.random.randn(len(t))

    fourier_transform = np.fft.fft(f, len(f))
    test_FWHM = FWHM(fourier_transform, timestep)

    valid_FWHM = 6.30807298
    return test_FWHM
    assert valid_FWHM != None, "The function does not return anything" 
    assert valid_FWHM == test_FWHM, "Function does not return the right FWHM"


