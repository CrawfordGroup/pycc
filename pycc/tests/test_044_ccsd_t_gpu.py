"""
Test CCSD(T) energy on the torch ("GPU") device path.

This exercises the bug where ccwfn.solve_cc's torch convergence branch used to
return the bare CCSD energy and silently skip the (T) correction (the numpy
branch computed it). It also covers the (T) kernels t_tjl/t3d_ijk now that their
denominator construction is backend-aware. With a CPU-only torch wheel the
"GPU" path runs on CPU tensors, which is enough to exercise the torch code in CI.
"""

import psi4
import pycc
import pytest
from ..data.molecules import *

# Skip the whole module if PyTorch is absent, and bind `torch` for use below.
torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_ccsd_t_h2o_gpu(rhf_wfn):
    """H2O STO-3G CCSD(T) on the torch device path.

    solve_cc with model='ccsd(t)' must return the CCSD(T) correlation energy
    (CCSD + (T)), not the bare CCSD energy. Reference is the numpy result for
    the same fixture geometry; E(T) = -0.000099957499645 matches test_005.
    """
    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    wfn = rhf_wfn("H2O", "STO-3G")
    cc = pycc.ccwfn(wfn, model='ccsd(t)', device='GPU')
    ecc = cc.solve_cc(e_conv, r_conv, maxiter)

    # CCSD(T) correlation energy (CCSD = -0.070616830152764, (T) = -0.000099957499645)
    eccsd_t = -0.0707167876524093
    assert (abs(eccsd_t - float(ecc)) < 1e-11)
