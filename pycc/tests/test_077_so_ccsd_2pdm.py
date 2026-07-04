"""
Spin-orbital CCSD two-particle density (Table 6.3 of the CC notes).

The nine antisymmetric blocks are validated by reconstructing the CCSD correlation energy from the
reduced densities, ``E_corr = contract(gamma, F) + 1/4 contract(Gamma, <pq||rs>)`` -- exercised via
``ccdensity.compute_energy`` (which now supports the spin-orbital 2-PDM).  Covered: RHF reference
through the spin-orbital path (which must match the spatial closed-shell energy), open-shell UHF
references, and frozen core.  The assembled full 2-PDM is checked for antisymmetry.
"""

import numpy as np
import pycc


E_CONV = R_CONV = 1e-11


def _so_density(wfn, frozen_core=False):
    """Converged spin-orbital CCSD wavefunction, Lambda, and ccdensity (with the 2-PDM)."""
    cc = pycc.ccwfn(wfn, orbital_basis='spinorbital', frozen_core=frozen_core)
    ecc = cc.solve_cc(E_CONV, R_CONV, 300)
    hbar = pycc.cchbar(cc)
    lam = pycc.cclambda(cc, hbar)
    lam.solve_lambda(E_CONV, R_CONV)
    dens = pycc.ccdensity(cc, lam)                 # onlyone=False -> builds the SO 2-PDM
    return cc, dens, ecc


def test_so_2pdm_reconstructs_energy_rhf(rhf_wfn):
    """Closed-shell (RHF) reference through the spin-orbital path: the SO 2-PDM reconstructs the
    CCSD correlation energy, and it equals the spatial closed-shell energy (SO == spatial).  All
    electron on both sides (the suite's default reference is frozen-core, so ask for all-electron
    explicitly to compare like with like)."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="false")
    cc, dens, ecc = _so_density(wfn)
    assert abs(dens.compute_energy() - ecc) < 1e-10

    # SO == spatial: the same molecule/basis through the spatial closed-shell CCSD
    spatial = pycc.ccwfn(wfn)
    e_spatial = spatial.solve_cc(E_CONV, R_CONV)
    assert abs(ecc - e_spatial) < 1e-10


def test_so_2pdm_reconstructs_energy_ccpvdz(rhf_wfn):
    """Larger basis (cc-pVDZ): the SO 2-PDM reconstructs E_corr for a real virtual space -- several
    virtuals per irrep and A2-symmetry MOs that STO-3G/H2O lacks -- and it matches the spatial
    closed-shell energy (SO == spatial)."""
    wfn = rhf_wfn("H2O", "cc-pVDZ", freeze_core="false")
    cc, dens, ecc = _so_density(wfn)
    assert abs(dens.compute_energy() - ecc) < 1e-9
    assert abs(ecc - pycc.ccwfn(wfn).solve_cc(E_CONV, R_CONV)) < 1e-9   # SO == spatial


def test_so_2pdm_reconstructs_energy_uhf(uhf_wfn):
    """Open-shell UHF references: the SO 2-PDM reconstructs the CCSD correlation energy."""
    for geom in ("0 2\nO\nH 1 0.97",                       # OH doublet
                 "0 2\nN\nH 1 1.02\nH 1 1.02 2 103.0"):    # NH2 doublet
        wfn = uhf_wfn(geom, "STO-3G", geom_extra="\nsymmetry c1")
        cc, dens, ecc = _so_density(wfn)
        assert abs(dens.compute_energy() - ecc) < 1e-9, geom


def test_so_2pdm_reconstructs_energy_frozen_core(rhf_wfn):
    """Frozen-core spin-orbital CCSD: the correlation density (active space) reconstructs the
    frozen-core correlation energy, which differs from the all-electron value."""
    wfn = rhf_wfn("H2O", "STO-3G", freeze_core="true")
    cc, dens, ecc = _so_density(wfn, frozen_core=True)
    assert cc.nfzc > 0
    assert abs(dens.compute_energy() - ecc) < 1e-10

    wfn_ae = rhf_wfn("H2O", "STO-3G")
    _, _, ecc_ae = _so_density(wfn_ae)
    assert abs(ecc - ecc_ae) > 1e-5                        # freezing the O 1s changes E_corr


def test_so_2pdm_full_assembly(uhf_wfn):
    """The assembled full spin-orbital 2-PDM is correct and antisymmetric.

    Two checks with different strengths:

    * **Energy from the full Gamma** -- ``contract(gamma, F) + 1/4 contract(Gamma, <pq||rs>)``
      reproduces the correlation energy.  This is the load-bearing check: it pins the *values* and
      *placements* of all sixteen o/v block types, independent of the block-wise energy in
      :meth:`ccdensity.compute_energy`.
    * **Antisymmetry** -- ``Gamma_pqrs = -Gamma_qprs`` (bra) and ``= -Gamma_pqsr`` (ket).  For the
      off-diagonal families :meth:`_so_full_twopdm` places the signed images explicitly, so those
      are antisymmetric by construction; the test still bites on the self-closed
      ``oooo``/``vvvv``/``oovv``/``vvoo`` blocks and on the intrinsic bra/ket antisymmetry of the
      stored representatives (e.g. ``Gamma_ijka = -Gamma_jika``)."""
    cc, dens, ecc = _so_density(uhf_wfn("0 2\nO\nH 1 0.97", "STO-3G", geom_extra="\nsymmetry c1"))
    o, v = cc.o, cc.v
    F = np.asarray(cc.H.F)
    ERI = np.asarray(cc.H.ERI)
    Gam = dens._so_full_twopdm()

    gamma = np.zeros_like(F)
    gamma[o, o] = dens.Doo; gamma[v, v] = dens.Dvv; gamma[o, v] = dens.Dov; gamma[v, o] = dens.Dvo
    e_recon = np.einsum('pq,pq->', gamma, F) + 0.25 * np.einsum('pqrs,pqrs->', Gam, ERI)
    assert abs(e_recon - ecc) < 1e-9

    assert np.max(np.abs(Gam + Gam.transpose(1, 0, 2, 3))) < 1e-12   # bra:  Gamma_pqrs = -Gamma_qprs
    assert np.max(np.abs(Gam + Gam.transpose(0, 1, 3, 2))) < 1e-12   # ket:  Gamma_pqrs = -Gamma_pqsr


def test_so_gradient_densities_convention(uhf_wfn):
    """``gradient_densities`` returns the spin-orbital ``D`` and ``Gamma`` in the gradient
    convention (the ``1/4`` absorbed into ``Gamma``), so the correlation energy is
    ``contract(D, F) + contract(Gamma, <pq||rs>) = E_corr`` -- no extra prefactor, matching the
    spatial path and the SO MP2 machinery.  The oovv block is ``1/4 * Gamma_ijab`` (cf.
    ``MPwfn._so_mp2_tpdm`` whose oovv block is ``1/4 t2``)."""
    cc, dens, ecc = _so_density(uhf_wfn("0 2\nN\nH 1 1.02\nH 1 1.02 2 103.0", "STO-3G",
                                        geom_extra="\nsymmetry c1"))
    D, Gam = dens.gradient_densities()
    F = np.asarray(cc.H.F)
    ERI = np.asarray(cc.H.ERI)
    e = np.einsum('pq,pq->', D, F) + np.einsum('pqrs,pqrs->', Gam, ERI)
    assert abs(e - ecc) < 1e-9
