"""
Frozen-core CC one-particle density and the RT-CC observables built on it.

Since the spatial Wavefunction builds the full-MO Hamiltonian and treats a frozen core as an
offset of the active slices (``o = slice(nfzc, nfzc+no)``), ``ccdensity.compute_onepdm`` must
return an ``nmo x nmo`` matrix (core rows/columns zero) to match the property integrals, and the
CC3 T1-transformed integrals in ``rtcc.dipole`` must use the ``o``/``v`` slices rather than
``[:no]``/``[-nv:]``.  All other RT tests run all-electron, so neither path was exercised.
"""

import numpy as np
import psi4
import pycc
from pycc.rt.lasers import sine_square_laser

# H2O, frame locked (as in test_034).
GEOM = """
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
no_com
no_reorient
"""

# Frozen Fock-perturbation FD oracle for the frozen-core CCSD unrelaxed correlation dipole mu_z
# (H2O/cc-pVDZ, GEOM, freeze_core): 5-point FD of E_corr[CCSD] w.r.t. V = -s*mu_z with fixed
# orbitals (h = 1e-4; see _findiff_dipole_z).  The analytic value agreed to 2.4e-13.
FC_CCSD_DIPOLE_Z_REF = 0.078423760470579


def _findiff_dipole_z(wfn, model='CCSD', h=1e-4):
    """Regeneration recipe for FC_CCSD_DIPOLE_Z_REF (not run in the tests)."""
    def E(s):
        c = pycc.CCwfn(wfn, model=model)
        c.H.F = np.asarray(c.H.F) - s * np.asarray(c.H.mu[2])
        return c.solve_cc(1e-12, 1e-12, 200)
    return -(-E(2 * h) + 8 * E(h) - 8 * E(-h) + E(-2 * h)) / (12 * h)


def _solve(wfn, model):
    cc = pycc.CCwfn(wfn, model=model)
    ecc = cc.solve_cc(1e-12, 1e-12, 200)
    lam = pycc.cclambda(cc, pycc.cchbar(cc))
    lam.solve_lambda(1e-12, 1e-12, 200)
    return cc, ecc, lam, pycc.ccdensity(cc, lam)


def test_frozen_core_ccsd_onepdm_and_rtcc(rhf_wfn):
    """Frozen-core CCSD: full-MO density with zero core rows/columns; its dipole matches the FD
    oracle; rtcc.dipole and rtcc.lagrangian (t = 0, zero field) reproduce it and E_CCSD."""
    wfn = rhf_wfn(GEOM, "cc-pVDZ", freeze_core="true")
    cc, ecc, lam, dens = _solve(wfn, 'CCSD')
    assert cc.nfzc == 1

    D = dens.compute_onepdm(cc.t1, cc.t2, lam.l1, lam.l2)
    assert D.shape == (cc.nmo, cc.nmo)
    assert np.abs(D[:cc.nfzc]).max() == 0.0 and np.abs(D[:, :cc.nfzc]).max() == 0.0
    mu_z = cc.contract('pq,pq->', cc.H.mu[2], D)
    assert abs(mu_z - FC_CCSD_DIPOLE_Z_REF) < 1e-10, mu_z

    rt = pycc.rtcc(cc, lam, dens, sine_square_laser(1.0, 2.87, 5.0))   # V(0) = 0
    y0 = rt.collect_amps(cc.t1, cc.t2, lam.l1, lam.l2, 0).astype('complex128')
    t1, t2, l1, l2, _ = rt.extract_amps(y0)
    assert abs(rt.dipole(t1, t2, l1, l2)[2] - FC_CCSD_DIPOLE_Z_REF) < 1e-10
    elag = rt.lagrangian(0.0, t1, t2, l1, l2) - rt._eref(np.asarray(cc.H.F))
    assert abs(elag - ecc) < 1e-10, (elag, ecc)

    psi4.core.clean()


def test_frozen_core_cc3_rtcc_dipole(rhf_wfn):
    """Frozen-core CC3 rtcc.dipole equals the same contraction carried out on the active-space
    blocks alone (the pre-offset layout, occupied at [0:no] and virtual at [no:]), i.e. the
    T1-transformed integrals are taken from the active o/v blocks, not the core rows."""
    wfn = rhf_wfn(GEOM, "cc-pVDZ", freeze_core="true")
    cc, _, lam, dens = _solve(wfn, 'CC3')
    nfzc, no = cc.nfzc, cc.no
    assert nfzc == 1

    rt = pycc.rtcc(cc, lam, dens, sine_square_laser(1.0, 2.87, 5.0))
    y0 = rt.collect_amps(cc.t1, cc.t2, lam.l1, lam.l2, 0).astype('complex128')
    t1, t2, l1, l2, _ = rt.extract_amps(y0)
    mu = rt.dipole(t1, t2, l1, l2)

    D, D3 = dens.compute_onepdm(t1, t2, l1, l2)
    assert D.shape == D3.shape == (cc.nmo, cc.nmo)
    act = slice(nfzc, cc.nmo)
    D, D3 = D[act, act], D3[act, act]
    for ax in range(3):
        m = np.asarray(cc.H.mu[ax])[act, act]
        Mt = np.zeros_like(m, dtype=complex)
        Mt[:no, :no] = m[:no, :no] + m[:no, no:] @ t1.T
        Mt[no:, no:] = m[no:, no:] - t1.T @ m[:no, no:]
        ref = np.sum(m * D) + np.sum(Mt * D3)
        assert abs(mu[ax] - ref) < 1e-12, (ax, mu[ax], ref)

    psi4.core.clean()
