"""
Unit and regression test for the pycc package.
"""

# Import package, test suite, and other packages as needed
import pycc
import pytest
import sys

def test_pycc_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pycc" in sys.modules
