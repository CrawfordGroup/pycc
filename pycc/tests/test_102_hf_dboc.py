"""
HF-SCF diagonal Born-Oppenheimer correction (HFwfn.dboc) against
CFOUR v2.1 (CALC=SCF, DBOC=ON; outputs in tools/cfour_runs), cc-pVDZ,
at the exact HEAT (Ref. 14) geometries of Gauss, Tajti, Kallay,
Stanton, and Szalay, J. Chem. Phys. 125, 144111 (2006); consistent
with the paper's Table I(a) SCF column.  Nuclear (bare) masses; the Q part
of the contact term via the kinetic sum rule
(Derivatives.overlap_dd_sum).
"""

import psi4
import pycc

HARTREE_CM = 219474.6313632

# exact ae-CCSD(T)/cc-pVQZ structures of the paper's Ref. 14 (HEAT:
# Tajti et al., J. Chem. Phys. 121, 11599 (2004), footnote)
GEOMS = {
    'h2': """
0 1
H
H 1 0.74186
symmetry c1
""",
    'hf': """
0 1
F
H 1 0.91516
symmetry c1
""",
    'h2o': """
0 1
O
H 1 0.95623
H 1 0.95623 2 104.25
symmetry c1
""",
}


def _hf_dboc_cm(geom):
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    e, wfn = psi4.energy('scf', return_wfn=True)
    return pycc.HFwfn(wfn).dboc() * HARTREE_CM


def test_hf_dboc_h2():
    assert abs(_hf_dboc_cm(GEOMS['h2']) - 99.376297) < 0.001


def test_hf_dboc_hf():
    assert abs(_hf_dboc_cm(GEOMS['hf']) - 604.556499) < 0.001


def test_hf_dboc_h2o():
    assert abs(_hf_dboc_cm(GEOMS['h2o']) - 600.367450) < 0.001
