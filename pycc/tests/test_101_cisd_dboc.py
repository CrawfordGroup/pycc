"""
CISD diagonal Born-Oppenheimer correction (CIderiv.dboc) against the
analytic values of Gauss, Tajti, Kallay, Stanton, and Szalay,
J. Chem. Phys. 125, 144111 (2006), Table I(a), cc-pVDZ.

Geometries are the exact all-electron CCSD(T)/cc-pVQZ structures of the
paper's Ref. 14 (HEAT: Tajti et al., JCP 121, 11599 (2004), footnote);
with them the CISD values match the paper to its printed precision.
Nuclear (bare) masses are used inside CIderiv.dboc,
matching the paper's convention.
"""

import psi4
import pycc
from ..cideriv import CIderiv

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


def _cisd_dboc_cm(geom):
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pvdz', 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    e, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD')
    ci.solve_ci(e_conv=1e-11, r_conv=1e-11, maxiter=200)
    return CIderiv(ci).dboc() * HARTREE_CM


def test_cisd_dboc_h2():
    assert abs(_cisd_dboc_cm(GEOMS['h2']) - 111.91) < 0.01


def test_cisd_dboc_hf():
    assert abs(_cisd_dboc_cm(GEOMS['hf']) - 616.38) < 0.01


def test_cisd_dboc_h2o():
    assert abs(_cisd_dboc_cm(GEOMS['h2o']) - 615.69) < 0.01


def _cisd_wfn(geom, basis='sto-3g'):
    """Converged CISD wavefunction only (the reference checks above go through _cisd_dboc_cm,
    which returns the DBOC directly and so cannot be reused for a two-driver comparison)."""
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': basis, 'scf_type': 'pk',
                      'e_convergence': 1e-12, 'd_convergence': 1e-12})
    psi4.geometry(geom)
    _e, wfn = psi4.energy('scf', return_wfn=True)
    ci = pycc.CIwfn(wfn, model='CISD')
    ci.solve_ci(e_conv=1e-11, r_conv=1e-11, maxiter=200)
    return ci


def test_cisd_dboc_perturbed_mo_gauge_invariance():
    """The CISD DBOC is invariant to the perturbed-MO gauge, the canonical vs non-canonical choice
    governing the *nuclear* perturbations it is built from.

    That choice is real, but it is NOT what the former ``dboc(gauge=...)`` argument selected: that
    one was the *imaginary*-perturbation gauge, which the DBOC's nuclear-only perturbations never
    consume, so it was bitwise inert and the cross-check it invited could not fail.  The genuine
    choice reaches the CPCI solve from the driver, and this guards it."""
    for geom in (GEOMS['h2'], GEOMS['h2o']):
        ci = _cisd_wfn(geom)
        nc = CIderiv(ci).dboc()
        cd = CIderiv(ci)
        cd._gauge_override = 'canonical'
        ca = cd.dboc()
        assert abs(nc - ca) < 1e-9 * max(1.0, abs(nc)), (nc, ca)
