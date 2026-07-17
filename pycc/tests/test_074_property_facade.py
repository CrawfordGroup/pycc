"""
Property facade -- pycc.aat() and pycc.PropertyComponents.

The facade returns the additive physical decomposition ``total = nuclear + reference +
correlation`` for any supported wavefunction type, with the pieces genuinely computed apart
(the correlation excludes the SCF reference density; the reference is the independent SCF AAT).
"""

import psi4
import pycc
import numpy as np


H2O2 = """
H -1.780954530308296   1.411647335546379   0.872055376436941
H  1.780954530308296  -1.411647335546379   0.872055376436941
O -1.371214332646589  -0.115525249760340  -0.054947416764017
O  1.371214332646589   0.115525249760340  -0.054947416764017
no_com
no_reorient
symmetry c1
units bohr
"""


def _wfns(freeze_core='false'):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.geometry(H2O2)
    psi4.set_options({'basis': 'STO-3G', 'scf_type': 'pk', 'freeze_core': freeze_core,
                      'e_convergence': 1e-11, 'd_convergence': 1e-11})
    _, wfn = psi4.energy('scf', return_wfn=True)
    mp = pycc.MPwfn(wfn)
    mp.compute_energy()
    return pycc.HFwfn(wfn), mp


def test_aat_decomposition_identities():
    """total == nuclear + reference + correlation; electronic == reference + correlation; the
    derived accessors and the scf/hf aliases are self-consistent."""
    _, mp = _wfns()
    r = pycc.aat(mp)
    assert np.max(np.abs(r.total - (r.nuclear + r.reference + r.correlation))) < 1e-14
    assert np.max(np.abs(r.electronic - (r.reference + r.correlation))) < 1e-14
    assert np.array_equal(r.scf, r.reference) and np.array_equal(r.hf, r.reference)
    assert r.origin == (0.0, 0.0, 0.0)


def test_aat_genuine_separation():
    """The reference block is exactly the independent SCF AAT, and the correlation block is a
    real, nonzero contribution computed apart from it."""
    hf, mp = _wfns()
    r = pycc.aat(mp)
    assert np.max(np.abs(r.reference - np.asarray(hf.atomic_axial_tensors()))) < 1e-12
    assert np.max(np.abs(r.correlation - np.asarray(mp.atomic_axial_tensors()))) < 1e-12
    assert np.max(np.abs(r.correlation)) > 1e-4      # correlation really present


def test_aat_nuclear_term():
    """The nuclear block is (Z_A/4) eps_{alpha,beta,gamma} R_{A,gamma}."""
    _, mp = _wfns()
    r = pycc.aat(mp)
    mol = mp.ref.molecule()
    R = mol.geometry().np
    for A in range(mol.natom()):
        Z, (x, y, z) = mol.Z(A), R[A]
        expect = 0.25 * Z * np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]])
        assert np.max(np.abs(r.nuclear[A] - expect)) < 1e-12, A


def test_aat_hf_wavefunction():
    """pycc.aat(HFwfn): correlation is an all-zeros array (same shape/type as for MP2), and the
    reference is the SCF AAT."""
    hf, _ = _wfns()
    r = pycc.aat(hf)
    assert r.correlation.shape == r.reference.shape
    assert np.all(r.correlation == 0.0)
    assert np.max(np.abs(r.reference - np.asarray(hf.atomic_axial_tensors()))) < 1e-12
    assert np.max(np.abs(r.electronic - r.reference)) < 1e-14


def test_aat_origin_argument():
    """A non-default origin shifts only the nuclear block (electronic terms unchanged), by the
    expected -(Z_A/4) eps O."""
    _, mp = _wfns()
    r0 = pycc.aat(mp)
    O = (0.5, -0.3, 0.1)
    rO = pycc.aat(mp, origin=O)
    assert np.max(np.abs(rO.electronic - r0.electronic)) < 1e-14
    mol = mp.ref.molecule()
    ox, oy, oz = O
    shift = -0.25 * np.array([[0, oz, -oy], [-oz, 0, ox], [oy, -ox, 0]])
    for A in range(mol.natom()):
        assert np.max(np.abs((rO.nuclear[A] - r0.nuclear[A]) - mol.Z(A) * shift)) < 1e-12, A
    assert rO.origin == O


def test_aat_frozen_core_total():
    """Frozen core: total = all-electron SCF reference + frozen-core correlation + nuclear; the
    reference is unaffected by freezing (it is the all-electron SCF AAT)."""
    _, mp_ae = _wfns('false')
    _, mp_fc = _wfns('true')
    r_ae, r_fc = pycc.aat(mp_ae), pycc.aat(mp_fc)
    assert np.max(np.abs(r_fc.reference - r_ae.reference)) < 1e-12
    assert np.max(np.abs(r_fc.nuclear - r_ae.nuclear)) < 1e-14
    # freezing changes the correlation (and hence the total) a little, but not wildly
    assert 1e-6 < np.max(np.abs(r_fc.correlation - r_ae.correlation)) < 1e-2


# ---- the other properties under the same umbrella ----

def _facade_and_pieces(hf, mp):
    """Each facade function paired with (HF public method, MP2 correlation method) that
    reconstruct the physical total, for the composition check."""
    return [
        ("dipole",         pycc.dipole(mp),               hf.dipole(),                     mp.relaxed_dipole()),
        ("gradient",       pycc.gradient(mp),             hf.gradient(),                   mp.gradient()),
        ("polarizability", pycc.polarizability(mp),       hf.polarizability(),             mp.polarizability()),
        ("hessian",        pycc.hessian(mp),              hf.hessian(),                    mp.hessian()),
        ("apt-length",     pycc.apt(mp, 'length'),        hf.dipole_derivatives(),         mp.dipole_derivatives()),
        ("apt-velocity",   pycc.apt(mp, 'velocity'),      hf.velocity_dipole_derivatives(), mp.velocity_dipole_derivatives()),
    ]


def test_facade_decomposition_and_total():
    """For every property: PropertyComponents.total == nuclear + reference + correlation, and it
    equals the physical total (SCF-reference public method + MP2 correlation method)."""
    hf, mp = _wfns()
    for name, comp, hf_pub, mp_corr in _facade_and_pieces(hf, mp):
        assert np.max(np.abs(comp.total - (comp.nuclear + comp.reference + comp.correlation))) < 1e-12, name
        assert np.max(np.abs(comp.total - (np.asarray(hf_pub) + np.asarray(mp_corr)))) < 1e-10, name


def test_facade_hf_wavefunction():
    """pycc.<property>(HFwfn): correlation is an all-zeros block and the total equals the SCF
    public method (same shape/type as the MP2 result)."""
    hf, _ = _wfns()
    for name, fn, pub in [
            ("dipole", pycc.dipole, hf.dipole()),
            ("gradient", pycc.gradient, hf.gradient()),
            ("polarizability", pycc.polarizability, hf.polarizability()),
            ("hessian", pycc.hessian, hf.hessian()),
            ("apt-length", lambda w: pycc.apt(w, 'length'), hf.dipole_derivatives()),
            ("apt-velocity", lambda w: pycc.apt(w, 'velocity'), hf.velocity_dipole_derivatives())]:
        r = fn(hf)
        assert np.all(r.correlation == 0.0), name
        assert np.max(np.abs(r.total - np.asarray(pub))) < 1e-12, name


def test_facade_polarizability_no_nuclear():
    """The polarizability has no nuclear contribution (pure electronic response)."""
    _, mp = _wfns()
    assert np.all(pycc.polarizability(mp).nuclear == 0.0)


def test_apt_bad_gauge():
    """pycc.apt rejects an unknown gauge."""
    _, mp = _wfns()
    try:
        pycc.apt(mp, gauge='nonsense')
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_facade_route_option():
    """route (MP2 correlation algorithm) is exposed and passed through; the two APT 2n+1 routes
    agree, and route is ignored for HF."""
    hf, mp = _wfns()
    assert np.max(np.abs(pycc.polarizability(mp, route='2n+1').total
                         - pycc.polarizability(mp).total)) < 1e-12
    assert np.max(np.abs(pycc.hessian(mp, route='2n+1').total
                         - pycc.hessian(mp).total)) < 1e-12
    assert np.max(np.abs(pycc.apt(mp, 'length', route='2n+1-nuclear').total
                         - pycc.apt(mp, 'length', route='2n+1-field').total)) < 1e-10
    assert np.all(pycc.hessian(hf, route='2n+1').correlation == 0.0)


def test_facade_orbital_gauge_option():
    """orbital_gauge (expert-only) is exposed on aat / apt(velocity); the tensor is invariant to
    it (the knob reaches the correlation method, which is gauge invariant), ignored for HF."""
    hf, mp = _wfns()
    assert np.max(np.abs(pycc.aat(mp, orbital_gauge='non-canonical').total
                         - pycc.aat(mp, orbital_gauge='canonical').total)) < 1e-9
    assert np.max(np.abs(pycc.apt(mp, 'velocity', orbital_gauge='non-canonical').total
                         - pycc.apt(mp, 'velocity', orbital_gauge='canonical').total)) < 1e-9
    assert np.all(pycc.apt(hf, 'velocity', orbital_gauge='canonical').correlation == 0.0)
