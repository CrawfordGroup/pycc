"""
Test real-time module Padé approximant to a Fourier series.
"""

# Import package, test suite, and other packages as needed
from pycc.rt.utils import Pade,damp
import pytest
import numpy as np

def test_pade_sanity():
    """Padé should find correct peaks for sin(2t) + cos(4t)
    iff damped"""
    t = np.linspace(0,2*np.pi*20,1000)
    y = np.sin(2*t) + np.cos(4*t)
    dt = t[-1] - t[-2]

    p = Pade(damp(y,dt,50),dt=dt)
    p.build()

    w = np.linspace(0,10,101)
    i = p.approx(w,norm=True)

    re_max = np.argmax(np.abs(np.real(i)))
    im_max = np.argmax(np.abs(np.imag(i)))

    assert np.real(i)[re_max] == 1
    assert np.imag(i)[im_max] == -1
    assert (w[re_max] - 4) < 1E-4
    assert (w[im_max] - 2) < 1E-4

@pytest.mark.xfail
def test_pade_sanity_xfail():
    """Padé should find correct peaks for sin(2t) + cos(4t)
    iff damped"""
    t = np.linspace(0,2*np.pi*20,1000)
    y = np.sin(2*t) + np.cos(4*t)
    dt = t[-1] - t[-2]

    p = Pade(y,dt=dt)
    p.build()

    w = np.linspace(0,10,101)
    i = p.approx(w,norm=True)

    re_max = np.argmax(np.abs(np.real(i)))
    im_max = np.argmax(np.abs(np.imag(i)))

    assert (w[re_max] - 4) < 1E-4
    assert (w[im_max] - 2) < 1E-4
