"""
HF velocity-gauge (VG) atomic polar tensors -- HFwfn.velocity_dipole_derivatives().

The momentum-form APT P^lambda_{beta,alpha} = d(mu_alpha)/dX_{A,beta}, formulated (Amos,
Jalkanen & Stephens, J. Phys. Chem. 92, 5571 (1988), Eq. 14; Shumberger, Cheeseman, Caricato
& Crawford, LG(OI) VCD, Eq. 15) as an overlap of wave-function derivatives,

    [P^A_{beta,alpha}]^VG = -4 sum_ia (U^R_{ia,beta} + <phi^R_i|phi_a>) U^A_{ia,alpha}
                            + Z_A delta_{alpha,beta}   (closed-shell; -2 spin-orbital),

with the linear-momentum (magnetic-vector-potential) response U^A (CPHF.solve_momentum, dPsi/dA)
in place of the AAT's magnetic response U^B. Unlike the length-gauge APT (dipole_derivatives),
the VG APT differs from it in a finite basis (converging only toward the basis-set limit) but is
likewise origin-independent.

Oracle: the Amos et al. NH3 P(pi) values (their Table I), reproduced to the paper's 3-digit
precision. Geometry is their Table I geometry (bohr, no_com/no_reorient). Also: the SO == spatial
keystone, and origin invariance (an FD-free physics check).

TODO(precision): the Amos et al. numbers are only 3 significant figures, so the oracle tolerance
here is a loose ~2e-3. The PI will provide higher-precision reference VG APTs; tighten
`test_hf_vgapt_vs_amos` accordingly once available. (The SO==spatial keystone and origin-
invariance checks are already at ~1e-11/1e-9 and are the tight internal anchors in the meantime.)
"""

import psi4
import pycc
import numpy as np


# Amos et al., Table I geometry (atomic units); atom 0 = N, atom 3 = H3.
NH3 = """
units bohr
symmetry c1
no_com
no_reorient
N  0.0000  0.0000  0.1278
H -0.8855  1.5337 -0.5920
H -0.8855 -1.5337 -0.5920
H  1.7710  0.0000 -0.5920
"""

# Amos Table I P(pi) (velocity gauge), keyed [atom, nuc_cart, dip_dir].
AMOS_PI = {
    '6-31G*':  {(0, 0, 0): 2.791, (0, 2, 2): 3.355, (3, 0, 0): 0.127, (3, 0, 2): 0.284,
                (3, 1, 1): 0.618, (3, 2, 0): 0.356, (3, 2, 2): 0.582},
    '6-31G**': {(0, 0, 0): 2.484, (0, 2, 2): 3.005, (3, 0, 0): 0.065, (3, 0, 2): 0.235,
                (3, 1, 1): 0.451, (3, 2, 0): 0.307, (3, 2, 2): 0.424},
}


def _hfwfn(basis, orbital_basis='spatial', geom=NH3):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(geom)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return pycc.HFwfn(wfn, orbital_basis=orbital_basis)


def test_hf_vgapt_vs_amos():
    """Spatial VG APT reproduces the Amos et al. NH3 P(pi) values (Table I) to ~3 digits,
    for both 6-31G* and 6-31G**."""
    for basis, ref in AMOS_PI.items():
        P = np.asarray(_hfwfn(basis).velocity_dipole_derivatives())
        for (A, nc, dp), val in ref.items():
            assert abs(P[A, nc, dp] - val) < 2e-3, (basis, A, nc, dp, P[A, nc, dp], val)


def test_hf_vgapt_so_vs_spatial():
    """Keystone: closed-shell spin-orbital == spin-adapted VG APT (both bases)."""
    for basis in ('6-31G*', '6-31G**'):
        Psa = np.asarray(_hfwfn(basis, 'spatial').velocity_dipole_derivatives())
        Pso = np.asarray(_hfwfn(basis, 'spinorbital').velocity_dipole_derivatives())
        assert np.max(np.abs(Pso - Psa)) < 1e-11


def test_hf_vgapt_origin_invariant():
    """The VG APT is origin-independent (Amos et al.): a rigid translation of the molecule (and
    thus the gauge origin) leaves it unchanged -- an FD-free physics check, and the whole point
    of the velocity gauge for VCD."""
    shifted = NH3.replace('N  0.0000  0.0000  0.1278',
                          'N  5.0000 -3.0000  4.1278').replace(
                          'H -0.8855  1.5337 -0.5920', 'H  4.1145 -1.4663  3.4080').replace(
                          'H -0.8855 -1.5337 -0.5920', 'H  4.1145 -4.5337  3.4080').replace(
                          'H  1.7710  0.0000 -0.5920', 'H  6.7710 -3.0000  3.4080')
    P0 = np.asarray(_hfwfn('6-31G*').velocity_dipole_derivatives())
    P1 = np.asarray(_hfwfn('6-31G*', geom=shifted).velocity_dipole_derivatives())
    assert np.max(np.abs(P0 - P1)) < 1e-9


def test_hf_vgapt_differs_from_lg_finite_basis():
    """Sanity: VG and LG APTs are genuinely different in a finite basis (they agree only toward
    the basis-set limit), and they share the same Z_A delta nuclear term, so their difference is
    purely electronic and nonzero here."""
    hf = _hfwfn('6-31G*')
    Pvg = np.asarray(hf.velocity_dipole_derivatives())
    Plg = np.asarray(hf.dipole_derivatives())
    assert np.max(np.abs(Pvg - Plg)) > 1.0        # N diagonal differs by ~3 in 6-31G*
