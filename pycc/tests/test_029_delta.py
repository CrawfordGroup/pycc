"""
Test delta_pulse_laser function.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
import pycc
import pytest
from pycc.rt.lasers import delta_pulse_laser

def test_delta_pulse():
    F_str = 1
    center = 5
    V = delta_pulse_laser(F_str, center)
    V_test = np.zeros(20) 
    V_ref = np.zeros(20)
    V_ref[center - 1] = 1
    for i in range(20):
        V_test[i] = V(i)

    assert V_test.all() == V_ref.all()
    




