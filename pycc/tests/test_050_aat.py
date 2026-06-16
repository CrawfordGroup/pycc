"""
Test the RHF atomic axial tensors (AATs) -- the electronic magnetic-dipole
vibrational transition moment, the magnetic analogue of the APTs and the last piece
of the SCF VCD machinery.

The AAT (Eq. 16 of the AAT note) is the overlap of the nuclear- and magnetic-field
wavefunction derivatives. It reuses the cached nuclear CPHF response, adds a magnetic-
field response (the antisymmetric ``kind='magnetic'`` orbital Hessian), and the nuclear
half-derivative overlaps. The reference is DALTON's analytic SCF AATs for hydrogen
peroxide in DALTON's STO-3G, as tabulated in the psi4numpy SCF-VCD example.
"""

import psi4
import pycc
import numpy as np


# DALTON's STO-3G (spherical) -- the exact contraction the reference AATs were computed
# with; the built-in STO-3G differs, so the hard reference numbers require this basis.
def _dalton_sto3g(mol, role):
    mol.set_basis_all_atoms("sto-3g", role=role)
    basis = """
spherical
****
H 0
S 3 1.00
      3.4252509 0.15432897
      0.6239137 0.53532814
      0.1688554 0.44463454
****
O 0
S 3 1.00
    130.7093200 0.15432897
     23.8088610 0.53532814
      6.4436083 0.44463454
SP 3 1.00
      5.0331513 -0.09996723 0.15591627
      1.1695961 0.39951283 0.60768372
      0.3803890 0.70011547 0.39195739
****
"""
    return {'sto-3g': basis}


# DALTON total AATs (electronic + nuclear), [3*natom, 3] = [nuclear coord, B component].
AAT_DALTON = np.array([
    [-0.16438927, -0.10446393, -2.00901728],
    [ 0.09439940,  0.00574249,  0.11809387],
    [ 2.19819309, -0.11635324,  0.15601962],
    [-0.16438927, -0.10446393,  2.00901728],
    [ 0.09439940,  0.00574249, -0.11809387],
    [-2.19819309,  0.11635324,  0.15601962],
    [-0.11803349,  0.11314253, -0.10726430],
    [-0.19342146,  0.00350944,  0.36522207],
    [ 0.25524026, -0.20123957,  0.12058354],
    [-0.11803349,  0.11314253,  0.10726430],
    [-0.19342146,  0.00350944, -0.36522207],
    [-0.25524026,  0.20123957,  0.12058354]])


def test_aat_h2o2_vs_dalton():
    """H2O2 electronic AATs (+ nuclear term) vs DALTON's analytic SCF AATs."""
    mol = psi4.geometry("""
        O   0.0000000000   1.3192641900  -0.0952542913
        O  -0.0000000000  -1.3192641900  -0.0952542913
        H   1.6464858700   1.6841036400   0.7620343300
        H  -1.6464858700  -1.6841036400   0.7620343300
        symmetry c1
        units bohr
        noreorient
        no_com
    """)
    psi4.qcdb.libmintsbasisset.basishorde['STO-3G_DALTON'] = _dalton_sto3g
    psi4.core.set_global_option("BASIS", "STO-3G_dalton")
    psi4.set_options({'scf_type': 'pk', 'e_convergence': 1e-12, 'd_convergence': 1e-12})

    _, wfn = psi4.energy('SCF', return_wfn=True)
    natom = mol.natom()

    # Electronic AAT (Eq. 16), reshaped [lambda, alpha, beta] -> [3*natom, 3].
    aat_elec = pycc.HFwfn(wfn).atomic_axial_tensors().reshape(3 * natom, 3)

    # Nuclear contribution: (Z_lambda / 4) eps_{alpha,beta,gamma} R_{lambda,gamma}.
    geom = np.asarray(mol.geometry())
    eps_lc = np.zeros((3, 3, 3))
    eps_lc[0, 1, 2] = eps_lc[1, 2, 0] = eps_lc[2, 0, 1] = 1.0
    eps_lc[0, 2, 1] = eps_lc[1, 0, 2] = eps_lc[2, 1, 0] = -1.0
    aat_nuc = np.zeros((3 * natom, 3))
    for A in range(natom):
        for a in range(3):
            for b in range(3):
                aat_nuc[3 * A + a, b] = (mol.Z(A) / 4.0) * np.einsum(
                    'g,g->', eps_lc[a, b], geom[A])

    assert np.allclose(aat_elec + aat_nuc, AAT_DALTON, atol=1e-6)
