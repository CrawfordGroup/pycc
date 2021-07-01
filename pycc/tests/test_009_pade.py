"""
Test real-time module Padé approximant to a Fourier series.
"""

# Import package, test suite, and other packages as needed
from pycc.rt.utils import Pade
import pytest
import numpy as np

def test_pade():
    """Ref data from sin(t) + cos(3t)"""
    t = np.linspace(-40,40,800) # 0.1 timestep
    y = np.sin(t) + np.cos(3*t)

    p = Pade(y,0.1)
    p.build()

    w = np.linspace(0,5,100)
    i = p.approx(w)

    pass 
