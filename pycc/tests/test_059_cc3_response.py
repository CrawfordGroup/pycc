"""
Spin-orbital CC3 dynamic response (dipole polarizability + optical rotation) via the
symmetric linear-response function (right-hand perturbed amplitudes only);
docs/ENHANCEMENT_PLAN_2026-06.md.

CC3 response requires the full connected triples T3/L3/X3. The store_triples=True
path forms and stores these via whole-array kernels; store_triples=False uses the
batched (per-ijk) kernels and materializes the full arrays where the response
needs them. Both must give the same response.

Validation:
  * CC3 polarizability alpha_zz (omega=0.1, H2O/STO-3G) vs an independent Dalton
    reference (matches socc test_008) -- fast, non-slow.
  * [slow] CC3 polarizability tensor vs Dalton (full-array path) and
    store_triples=True == store_triples=False (the two triples algorithms agree).
  * [slow] CC3 optical rotation G'_zz (omega=0.077357, length gauge, chiral
    H2O2/STO-3G) vs an independent Dalton reference (matches socc test_009); the
    first CC3-response use of the spin-orbital magnetic-dipole integrals.
"""

import psi4
import pycc
import numpy as np
import pytest

# socc test_008 / Dalton-reference geometry (Cartesian, bohr).
H2O = """
0 1
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""

# Dalton CC3 dipole polarizability, omega = 0.1 a.u. (from socc test_008).
DALTON_POLAR = np.array([[0.061593757, 0.0, 0.0],
                         [0.0, 7.0661684, 0.0],
                         [0.0, 0.0, 3.0604929]])

# Chiral H2O2 (Dalton geometry; socc test_009) for the optical-rotation reference.
H2O2 = """
O   1.3133596569   0.0000000000  -0.0932359644
O  -1.3133596569  -0.0000000000  -0.0932359644
H   1.6917745981   0.7334825768   1.4797224976
H  -1.6917745981  -0.7334825768   1.4797224976
symmetry c1
units bohr
"""

# Dalton CC3 optical rotation (length gauge, omega=0.077357), zz component, from
# socc test_009's diagonal reference [-0.012797, 0.124157, -0.113186].
DALTON_OPTROT_ZZ = -0.113186


def test_cc3_polarizability_zz():
    """Fast (non-slow) CC3 coverage: a single polarizability component, alpha_zz,
    via two perturbed-wave-function solves (X(mu_z, -+omega)) + the symmetric
    response, vs the Dalton reference. Uses the full-array store_triples=True path
    (cheaper than the batched per-ijk kernels at this size). The full tensor and
    the store_triples True/False equivalence are checked in the slow test below."""
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'mp2_type': 'conv',
                      'freeze_core': 'false', 'reference': 'rhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)

    cc = pycc.CCwfn(wfn, frozen_core=False, model='CC3',
                    orbital_basis='spinorbital', store_triples=True)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-12)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-12)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    resp = pycc.ccresponse(dens)

    # alpha_zz = -<<mu_z; mu_z>>_omega, from X(mu_z, -omega) and X(mu_z, +omega).
    A = resp.pertbar["MU_Z"]
    Xm, _ = resp.solve_right(A, -0.1)
    Xp, _ = resp.solve_right(A, 0.1)
    alpha_zz = -resp.linresp_sym(A, Xm, A, Xp)
    assert abs(alpha_zz - DALTON_POLAR[2, 2]) < 1e-6


@pytest.mark.slow
def test_cc3_polarizability():
    """Spin-orbital CC3 dynamic polarizability (omega=0.1).

    Full-array (store_triples=True) path: the complete tensor vs the Dalton
    reference (plus the CC3 energy and Lambda pseudoenergy). Batched
    (store_triples=False) path: a single element, alpha_zz, must reproduce the
    full-array value -- the batched kernels are identical across the three
    Cartesian axes, so one element exercises the whole batched solve+response
    chain. (The batched path is ~100x slower per element here, so checking one
    element rather than the full tensor keeps this test from ballooning.)"""
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.be_quiet()
    psi4.geometry(H2O)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'mp2_type': 'conv',
                      'freeze_core': 'false', 'reference': 'rhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)

    def setup(store_triples):
        cc = pycc.CCwfn(wfn, frozen_core=False, model='CC3',
                        orbital_basis='spinorbital', store_triples=store_triples)
        ecc = cc.solve_cc(e_conv=1e-12, r_conv=1e-12)
        hbar = pycc.cchbar(cc)
        lam = pycc.cclambda(cc, hbar)
        lcc = lam.solve_lambda(e_conv=1e-12, r_conv=1e-12)
        dens = pycc.ccdensity(cc, lam, onlyone=True)
        return ecc, lcc, pycc.ccresponse(dens)

    def alpha_zz(resp):
        # alpha_zz = -<<mu_z; mu_z>>_omega, from X(mu_z, -+omega).
        A = resp.pertbar["MU_Z"]
        Xm, _ = resp.solve_right(A, -0.1)
        Xp, _ = resp.solve_right(A, 0.1)
        return -resp.linresp_sym(A, Xm, A, Xp)

    # Full-array path: full tensor vs Dalton, plus the energy / Lambda references.
    e_full, l_full, resp_full = setup(True)
    assert abs(e_full - (-0.070778085758433)) < 1e-11
    assert abs(l_full - (-0.068979529552146)) < 1e-11
    P_full = resp_full.polarizability(0.1)
    assert np.allclose(P_full, DALTON_POLAR, atol=1e-6)

    # Batched path: one element must reproduce the full-array result.
    _, _, resp_batched = setup(False)
    assert abs(alpha_zz(resp_batched) - P_full[2, 2]) < 1e-10


@pytest.mark.slow
def test_cc3_optical_rotation_zz():
    """CC3 optical rotation, single element G'_zz (chiral H2O2/STO-3G, length
    gauge, omega=0.077357) vs the Dalton reference (socc test_009).

    G'_zz = -<<mu_z; m_z>>_omega assembled from four perturbed wave functions --
    X(mu_z, -+omega), X(m_z, +omega), X(m*_z, -omega) -- this is the first
    CC3-response use of the spin-orbital magnetic-dipole integrals. Uses the
    full-array store_triples=True path; a single diagonal element keeps the cost
    to four solves."""
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.be_quiet()
    psi4.geometry(H2O2)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'mp2_type': 'conv',
                      'freeze_core': 'false', 'reference': 'rhf',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12,
                      'r_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)

    cc = pycc.CCwfn(wfn, frozen_core=False, model='CC3',
                    orbital_basis='spinorbital', store_triples=True)
    ecc = cc.solve_cc(e_conv=1e-12, r_conv=1e-12)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lcc = lam.solve_lambda(e_conv=1e-12, r_conv=1e-12)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    resp = pycc.ccresponse(dens)

    # socc/Psi4 reference CC3 energy and Lambda pseudoenergy for this geometry.
    assert abs(ecc - (-0.097663033394620)) < 1e-11
    assert abs(lcc - (-0.095210691090877)) < 1e-11

    w = 0.077357
    A = resp.pertbar["MU_Z"]
    Bm = resp.pertbar["M_Z"]
    Bs = resp.pertbar["M*_Z"]
    Xmu_m, _ = resp.solve_right(A, -w)
    Xmu_p, _ = resp.solve_right(A, w)
    Xm_p, _ = resp.solve_right(Bm, w)
    Xms_m, _ = resp.solve_right(Bs, -w)
    g_zz = 0.5 * resp.linresp_sym(A, Xmu_m, Bm, Xm_p) \
        + 0.5 * resp.linresp_sym(A, Xmu_p, Bs, Xms_m)
    assert abs(np.real(g_zz) - DALTON_OPTROT_ZZ) < 1e-4
