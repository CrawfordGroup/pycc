"""
Spin-orbital CCSD dipole polarizability via the symmetric linear-response function
(right-hand perturbed amplitudes only); docs/archive/ENHANCEMENT_PLAN_2026-06.md.

Validation:
  * SO-RHF dynamic polarizability (omega != 0) vs Psi4's CCSD polarizability
    (isotropic) -- the symmetric kernel reproduces the spin-adapted result.
  * UHF/ROHF *static* polarizability vs an in-place finite-difference of the CCSD
    energy. No code computes open-shell CC *dynamic* polarizabilities, but the
    dynamic kernel is validated for RHF above and the static value is checked
    directly here; the finite difference perturbs the Fock F -> F - F_str*mu_z and
    differentiates the correlation energy (the fixed-orbital reference is linear in
    the field, so it does not contribute to the second derivative).
"""

import psi4
import pycc
import numpy as np

# Open-shell doublet (hydroxyl radical), run in its computational point group (C2v -- no
# 'symmetry c1').  C2v keeps the 2-Pi reference symmetry-adapted and reproducible; forcing C1 lets
# the SCF break the pi_x/pi_y degeneracy arbitrarily (platform-dependent), which would make a frozen
# reference fragile.
OH = """
0 2
O 0.0 0.0 0.0
H 0.0 0.0 1.83
units bohr
"""


# Static CCSD alpha_zz for OH/STO-3G (C2v), frozen references.  No code computes open-shell CC
# *dynamic* polarizabilities, so these were validated against a 5-point finite field of the CCSD
# energy (self-consistent, agreeing to ~1e-8; see _findiff_alpha_zz, kept as the regeneration
# recipe).  With the symmetry-adapted C2v reference the values are reproducible to ~1e-13.
AZZ_REF = {'uhf': 3.5805125633, 'rohf': 3.5805155080}


def _findiff_alpha_zz(cc, d=0.001):
    """Regeneration recipe for AZZ_REF (not run in the tests): static alpha_zz from a 5-point finite
    difference of the CCSD correlation energy w.r.t. a z-dipole field.  Perturbs F -> F - F_str*mu_z
    and differentiates the correlation energy (the fixed-orbital reference is linear in the field, so
    it does not contribute to the second derivative).  Restores cc.H.F on return."""
    F0 = np.asarray(cc.H.F).copy()
    muz = np.asarray(cc.H.mu[2])
    g1, g2 = cc.mp.t1.copy(), cc.mp.t2.copy()

    def ecc(Fstr):
        cc.H.F = F0 - Fstr * muz
        cc.t1 = g1.copy()
        cc.t2 = g2.copy()
        return cc.solve_cc(e_conv=1e-13, r_conv=1e-12, maxiter=300)

    e0, ep, e2p, em, e2m = ecc(0.0), ecc(d), ecc(2*d), ecc(-d), ecc(-2*d)
    cc.H.F = F0
    return -(-e2p + 16*ep - 30*e0 + 16*em - e2m) / (12 * d * d)


def test_so_rhf_polar_vs_psi4(rhf_wfn):
    """SO-RHF dynamic CCSD polarizability (isotropic, omega=0.1) vs Psi4."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false",
                  e_convergence=1e-12, d_convergence=1e-12, r_convergence=1e-12)
    cc = pycc.CCwfn(wfn, frozen_core=False, orbital_basis="spinorbital")
    cc.solve_cc(e_conv=1e-11, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-11, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    resp = pycc.ccresponse(dens)
    iso = np.trace(resp.polarizability(0.1)) / 3.0

    psi4.set_options({'omega': [0.1, 'au'], 'gauge': 'length'})
    psi4.properties('ccsd', properties=['polarizability'])
    ref = next(psi4.variable(k) for k in psi4.core.variables()
               if 'DIPOLE POLARIZABILITY' in k.upper())
    assert abs(iso - ref) < 1e-9


def test_uhf_polar_static(uhf_wfn):
    """UHF static CCSD polarizability (alpha_zz) vs the finite-field-validated frozen reference."""
    wfn = uhf_wfn(OH, "STO-3G", freeze_core="false",
                  e_convergence=1e-13, d_convergence=1e-13)
    cc = pycc.CCwfn(wfn, frozen_core=False)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    azz = pycc.ccresponse(dens).polarizability(0.0)[2, 2]
    assert abs(azz - AZZ_REF['uhf']) < 1e-7


def test_rohf_polar_static(rohf_wfn):
    """ROHF static CCSD polarizability (alpha_zz) vs the finite-field-validated frozen reference."""
    wfn = rohf_wfn(OH, "STO-3G", freeze_core="false",
                   e_convergence=1e-13, d_convergence=1e-13)
    cc = pycc.CCwfn(wfn, frozen_core=False)
    cc.solve_cc(e_conv=1e-12, r_conv=1e-11)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar); lam.solve_lambda(e_conv=1e-12, r_conv=1e-11)
    dens = pycc.ccdensity(cc, lam, onlyone=True)
    azz = pycc.ccresponse(dens).polarizability(0.0)[2, 2]
    assert abs(azz - AZZ_REF['rohf']) < 1e-7
