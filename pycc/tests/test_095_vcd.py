"""
End-to-end VCD (vibrational circular dichroism) validation for the vibrational-analysis
utility (:mod:`pycc.vibanalysis`): frequencies, IR intensities, and rotatory strengths of
hydrogen peroxide against independent references.

The rotatory strength of each fundamental is Stephens' electric.magnetic transition-moment
product, ``R_k = sum_c (d mu_c / d Q_k)(d m_c / d Q_k)``, built from the analytic APT (electric
dipole gradient) and AAT (magnetic dipole gradient / atomic axial tensor) contracted with the
mass-weighted normal modes; the harmonic frequency factors cancel and the AAT already carries the
``Im`` (its ``i`` stripped).  Reported in 10^-44 esu^2 cm^2.

Two references, mirroring magpy's ``test_013_VCD_HF`` / ``test_021_VCD_MP2``:

* **SCF / STO-3G** (CFOUR HF/STO-3G optimized geometry): the whole analysis is pycc's own
  analytic SCF Hessian + APT + AAT via the ``pycc.vcd`` driver.  Frequencies and IR match CFOUR;
  the rotatory strengths match DALTON's *analytic* SCF values, so the tolerance is tight.

* **MP2 / STO-6G** (CID/6-31G* optimized geometry): pycc's analytic MP2 APT + AAT combined with an
  external CID/6-31G* Hessian (``fcm_h2o2_cid_631gd.txt``, the same file magpy reads), so the
  frequencies are CID/6-31G* (CFOUR) while the intensities are MP2/STO-6G.  The IR/VCD reference is
  magpy, whose APT/AAT are *finite-difference*, so the tolerance is looser -- the residual is
  magpy's FD truncation, not pycc's (pycc is analytic here).
"""

import os

import psi4
import pycc
import numpy as np


# CFOUR HF/STO-3G optimized geometry (bohr).
H2O2_SCF = """
O   0.0000000000   1.3192641900  -0.0952542913
O  -0.0000000000  -1.3192641900  -0.0952542913
H   1.6464858700   1.6841036400   0.7620343300
H  -1.6464858700  -1.6841036400   0.7620343300
no_com
no_reorient
symmetry c1
units bohr
"""

# CID/6-31G* optimized geometry (bohr) -- matches the external Hessian's frame (keep no_com /
# no_reorient so pycc's APT/AAT share that frame).
H2O2_MP2 = """
O   0.00000000   1.35909101  -0.103181170
O   0.00000000  -1.35909101  -0.103181170
H   1.53994623   1.68647251   0.825449361
H  -1.53994623  -1.68647251   0.825449361
symmetry c1
units bohr
no_com
no_reorient
"""

#: external CID/6-31G* Hessian (a.u.), read exactly as magpy reads it (header line, then 3N*3N
#: values row-major).
_HESSIAN_FILE = os.path.join(os.path.dirname(__file__), "fcm_h2o2_cid_631gd.txt")


def test_vcd_scf_h2o2(rhf_wfn):
    """SCF/STO-3G VCD, fully analytic through the ``pycc.vcd`` driver: frequencies and IR vs CFOUR,
    rotatory strengths vs DALTON (analytic).  The six vibrations are the six highest modes after
    projecting the rigid-body space."""
    wfn = rhf_wfn(H2O2_SCF, "STO-3G", freeze_core="false")
    hf = pycc.HFwfn(wfn, quiet=True)

    res = pycc.vcd(hf, project_trans=True, project_rot=True)
    freq = res['frequencies'][:6]
    ir = res['ir_intensities'][:6]
    vcd = res['rotatory_strengths'][:6]

    freq_ref = np.array([4148.2766, 4140.8893, 1781.0495, 1589.6394, 1486.9510, 184.6327])  # CFOUR
    ir_ref = np.array([30.3023, 12.7136, 2.0726, 46.0798, 0.0168, 134.1975])                # CFOUR
    vcd_ref = np.array([50.538, -53.528, 28.607, -14.650, 1.098, 101.378])                  # DALTON

    assert np.max(np.abs(freq - freq_ref)) < 1e-2, freq
    assert np.max(np.abs(ir - ir_ref)) < 1e-2, ir
    assert np.max(np.abs(vcd - vcd_ref)) < 1e-2, vcd
    psi4.core.clean()


def test_vcd_mp2_h2o2(rhf_wfn):
    """MP2/STO-6G VCD (all-electron): pycc's analytic MP2 APT + AAT with an external CID/6-31G*
    Hessian, versus CFOUR frequencies and magpy's finite-difference MP2 IR/VCD.  Looser tolerance
    because the reference intensities carry magpy's FD error (pycc is analytic)."""
    wfn = rhf_wfn(H2O2_MP2, "STO-6G", freeze_core="false")
    mp = pycc.MPwfn(wfn, quiet=True)
    mp.compute_energy()
    d = pycc.MPderiv(mp)

    hessian = np.genfromtxt(_HESSIAN_FILE, skip_header=1).reshape(12, 12)
    apt = np.asarray(pycc.apt(d).total)
    aat = np.asarray(pycc.aat(mp).total)

    res = pycc.harmonic_analysis(wfn.molecule(), hessian, apt=apt, aat=aat,
                                 project_trans=True, project_rot=True)
    freq = res['frequencies'][:6]
    ir = res['ir_intensities'][:6]
    vcd = res['rotatory_strengths'][:6]

    freq_ref = np.array([3853.69, 3852.33, 1528.10, 1392.62, 1021.70, 371.44])  # CFOUR
    ir_ref = np.array([24.745, 51.674, 1.372, 47.471, 0.020, 125.908])          # magpy (FD MP2)
    vcd_ref = np.array([-71.619, 65.754, 23.006, -14.998, 0.652, 108.598])      # magpy (FD MP2)

    assert np.max(np.abs(freq - freq_ref)) < 5e-2, freq
    assert np.max(np.abs(ir - ir_ref)) < 5e-2, ir
    assert np.max(np.abs(vcd - vcd_ref)) < 5e-2, vcd
    psi4.core.clean()
