# Import required packages
import numpy as np
import pytest
import os
from pycc.rt.utils import denoise
from pytest import fixture
from distutils import dir_util

@fixture
def datadir(tmpdir, request):
    '''
    from: https://stackoverflow.com/a/29631801
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    else:
        raise FileNotFoundError("Test folder not found.")

    return tmpdir

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
