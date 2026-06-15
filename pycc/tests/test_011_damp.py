#Import required packages
import numpy as np
import pytest
from pycc.rt.utils import damp

def test_damp(datadir):
    #Define function
    np.random.seed(10)
    timestep = 0.001
    t = np.arange(0, 1, timestep)
    f = np.cos(2*np.pi*12*t) + np.sin(2*np.pi*50*t)
    f = f + np.random.randn(len(t))

    ref_file = datadir.join(f"ref_011.npy")
    ref_array = np.load(str(ref_file))

    test_array = damp(f, timestep, 150)

    assert len(test_array) != 0, "The test array was empty"
    assert len(test_array) == len(ref_array), "The length of the test and the valid array are not equal"
    assert np.allclose(ref_array, test_array), "The test and valid array are not equal"
