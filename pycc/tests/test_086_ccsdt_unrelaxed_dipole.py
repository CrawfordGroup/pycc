"""CCSD(T) unrelaxed one-electron property -- guards the (T) occ-virt (ov) 1-PDM.

The analytic CCSD(T) *gradient* (test_083) is completely blind to the ov block of the one-particle
density: the Z-vector / orbital-relaxation absorbs the ov 1-PDM exactly (Handy-Schaefer
stationarity), so zeroing the entire (T) ov density moves the gradient by ~1e-17.  Any relaxed
first-order property (gradient, relaxed dipole) is likewise blind.  A latent factor error in the
(T) ov density therefore hides completely behind test_083.

The property that *does* exercise the ov 1-PDM is an UNRELAXED one-electron expectation value.  This
test finite-differences the CCSD(T) correlation energy in a static electric-dipole field with the
orbitals held frozen (the field is added to the spin-orbital Fock *after* SCF, so the HF orbitals do
not relax): mu_a = -dE_corr/d(field_a) = Tr(D . mu_a) with D the unrelaxed correlation 1-PDM.  This
is the exact oracle that pinned the spin-orbital (T) ov density to the 1/4 (T2-dagger normalization)
of D_ia = 1/4 sum_{lm,ef} t3c^{aef}_{ilm} t^{ef}_{lm}; a missing 1/4 in so_t3_density shows up here
at ~2e-3.  It is also the class of property (unrelaxed / perturbed density) that the (T)
polarizability depends on, so this guards that path too.

H2O/6-31G in a fixed, fully asymmetric (C1) frame so all three dipole components are nonzero.
"""

import contextlib
import os

import numpy as np
import psi4
import pycc

ATOMS = ['O', 'H', 'H']
# Asymmetric frame (one H pushed off the yz-plane) so mu_x, mu_y, mu_z are all nonzero and the ov
# density is exercised in every direction.  Frame locked (no_com/no_reorient) for reproducibility.
GEOM = np.array([
    [0.0,          0.000000000000,  0.143225857167],
    [0.1,         -1.638037301628, -1.136549142277],
    [0.0,          1.638037301628, -1.136549142277],
])
AXIS_LABEL = ['X', 'Y', 'Z']


def _scf_wfn():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.be_quiet()
    body = "\n".join(f"{s} {c[0]:.15f} {c[1]:.15f} {c[2]:.15f}" for s, c in zip(ATOMS, GEOM))
    psi4.geometry(body + "\nsymmetry c1\nunits bohr\nno_com\nno_reorient\n")
    psi4.set_options({'basis': '6-31G', 'scf_type': 'pk', 'freeze_core': 'false',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    _, wfn = psi4.energy('scf', return_wfn=True)
    return wfn


def _ccwfn(wfn, orbital_basis):
    cc = pycc.ccwfn(wfn, model='ccsd(t)', make_t3_density=True, orbital_basis=orbital_basis)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
        cc.solve_cc(1e-12, 1e-12, 200)
    return cc


def _unrelaxed_dipole(cc):
    """Unrelaxed correlation dipole Tr(D . mu) for all three axes, from the full unrelaxed 1-PDM
    (gradient_densities places the Doo/Dov/Dvo/Dvv blocks, including the (T) ov increment)."""
    D = np.asarray(pycc.CCderiv(cc)._density().gradient_densities()[0])
    return np.array([np.einsum('pq,pq->', D, np.asarray(cc.H.mu[a])) for a in range(3)])


def _findiff_dipole_so(wfn, h=1e-3):
    """Independent oracle: mu_a = -dE_corr/d(field_a) of the spin-orbital CCSD(T) correlation energy,
    5-point O(h^4) central difference.  The static field V = -field * mu[a] is added to the SO Fock
    (orbitals frozen), so this is the *unrelaxed* dipole.  (The field is implemented only in the
    spin-orbital Hamiltonian.)"""
    def E(s, a):
        cc = pycc.ccwfn(wfn, model='ccsd(t)', orbital_basis='spinorbital',
                        field=(s != 0.0), field_strength=s, field_axis=AXIS_LABEL[a])
        with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
            return cc.solve_cc(1e-12, 1e-12, 200)

    mu = np.zeros(3)
    for a in range(3):
        mu[a] = -(-E(2*h, a) + 8*E(h, a) - 8*E(-h, a) + E(-2*h, a)) / (12*h)
    return mu


def test_so_ccsdt_unrelaxed_dipole_vs_findiff():
    """The spin-orbital analytic unrelaxed CCSD(T) dipole Tr(D . mu) matches a finite difference of
    the CCSD(T) correlation energy in a frozen-orbital field.  This is the check the gradient cannot
    make: it directly exercises the (T) ov 1-PDM.  Guard 1e-7 (5-point FD is good to ~1e-10 here);
    a missing 1/4 on the (T) ov density fails by ~2e-3."""
    wfn = _scf_wfn()
    mu_an = _unrelaxed_dipole(_ccwfn(wfn, 'spinorbital'))
    mu_fd = _findiff_dipole_so(wfn)
    assert np.max(np.abs(mu_an - mu_fd)) < 1e-7, (mu_an, mu_fd)


def test_ccsdt_unrelaxed_dipole_so_equals_spatial():
    """SO == spatial keystone for the unrelaxed CCSD(T) dipole.  Because every relaxed property is
    blind to the ov 1-PDM, this unrelaxed contraction is the keystone that actually tests the ov
    block: it fails if the spin-orbital (1/4 T2-dagger) and spatial (spin-adapted) (T) ov densities
    disagree."""
    wfn = _scf_wfn()
    mu_so = _unrelaxed_dipole(_ccwfn(wfn, 'spinorbital'))
    mu_sp = _unrelaxed_dipole(_ccwfn(wfn, 'spatial'))
    assert np.max(np.abs(mu_so - mu_sp)) < 1e-9, (mu_so, mu_sp)
