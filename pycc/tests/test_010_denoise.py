# Import required packages
import numpy as np
import pytest
from pycc.rt.utils import denoise

def test_denoise(datadir):
    # Define function
    np.random.seed(10)
    timestep = 0.001
    t = np.arange(0, 1, timestep)
    f = np.cos(2*np.pi*12*t) + np.sin(2*np.pi*50*t)
    f = f + np.random.randn(len(t))

    ref_file = datadir.join(f"ref_010.npy")
    ref_array = np.load(str(ref_file))

    test_array = denoise(f, 100, timestep)

    assert len(test_array) != 0, "The test array is empty"
    assert len(test_array) == len(ref_array), "The test array is of a different length than the valid array"
    assert np.allclose(ref_array, test_array), "The test and valid array are not equal"
