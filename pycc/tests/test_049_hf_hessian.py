"""
Test the RHF nuclear (molecular) Hessian -- the force-constant matrix -- against
Psi4's analytic SCF Hessian.

The Hessian is an energy second derivative, so it is the first property to exercise
the full nuclear CPHF response *assembly* (the cached U^X plus the first-derivative
cross terms), not just the response in isolation. It also confirms the shared nuclear-
response cache: the 3*natom solves done for the Hessian are reused by the dipole
derivatives / APTs with no rebuild of the heavy derivative integrals.
"""

import psi4
import pycc
import numpy as np


def test_hf_hessian_h2o(rhf_wfn):
    """H2O STO-3G molecular Hessian vs psi4.hessian('scf') (frame locked)."""
    wfn = rhf_wfn("H2O", "STO-3G", geom_extra="\nsymmetry c1\nnoreorient\nnocom",
                  e_convergence=1e-11, d_convergence=1e-11)
    analytic = pycc.HFwfn(wfn).hessian()        # (3*natom, 3*natom)

    ref = np.asarray(psi4.hessian('scf'))       # same molecule/options still set
    assert analytic.shape == ref.shape
    assert np.allclose(analytic, analytic.T, atol=1e-10)  # Hessian is symmetric
    assert np.allclose(analytic, ref, atol=1e-9)


def test_hf_hessian_h2o_ccpvdz(rhf_wfn):
    """Larger basis (cc-pVDZ): the analytic RHF Hessian reproduces psi4.hessian('scf') for a real
    virtual space -- polarization functions, several virtuals per irrep, A2-symmetry MOs -- that
    STO-3G/H2O lacks (frame locked)."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", geom_extra="\nsymmetry c1\nnoreorient\nnocom",
                  e_convergence=1e-11, d_convergence=1e-11)
    analytic = pycc.HFwfn(wfn).hessian()
    ref = np.asarray(psi4.hessian('scf'))       # cheap SCF oracle, same molecule/options still set
    assert analytic.shape == ref.shape
    assert np.allclose(analytic, analytic.T, atol=1e-10)
    assert np.allclose(analytic, ref, atol=1e-8)


def test_hf_hessian_shares_nuclear_cache(rhf_wfn):
    """Computing the Hessian then the APTs solves the nuclear CPHF response once:
    the APT call must not rebuild any per-atom derivative integrals."""
    wfn = rhf_wfn("H2O", "STO-3G", geom_extra="\nsymmetry c1\nnoreorient\nnocom",
                  e_convergence=1e-11, d_convergence=1e-11)
    hf = pycc.HFwfn(wfn)
    cphf = hf.cphf

    builds = {"n": 0}
    original = cphf._build_nuclear

    def counted(atom):
        builds["n"] += 1
        return original(atom)

    cphf._build_nuclear = counted
    hf.hessian()
    after_hessian = builds["n"]
    hf.dipole_derivatives()
    after_apt = builds["n"]

    assert after_hessian == wfn.molecule().natom()   # one heavy build per atom
    assert after_apt == after_hessian                # APTs add none -- cache shared
