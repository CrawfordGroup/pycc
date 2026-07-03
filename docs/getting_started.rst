Getting Started
===============

Installation
------------
PyCC runs on top of `Psi4 <https://psicode.org>`_, which supplies the reference
wavefunction and the integrals. The simplest way to get a working environment is with
conda (Psi4 is most easily installed that way). You will need:

* Python 3.8 or newer
* `Psi4 <https://psicode.org>`_
* `NumPy <https://numpy.org/>`_
* `SciPy <https://scipy.org/>`_
* `opt_einsum <https://optimized-einsum.readthedocs.io/>`_
* `PyTorch <https://pytorch.org/>`_ *(optional; enables the GPU and mixed-precision paths)*

With those in place, install PyCC in developer mode from the repository root::

    pip install -e .

Usage
-----
PyCC starts from a converged Psi4 RHF reference wavefunction; every method is built on
top of it. The examples below are self-contained.

Coupled-cluster energy
~~~~~~~~~~~~~~~~~~~~~~~~
::

    import psi4
    import pycc

    psi4.geometry("""
    O
    H 1 0.96
    H 1 0.96 2 104.5
    """)
    psi4.set_options({'basis': 'cc-pVDZ'})
    _, wfn = psi4.energy('SCF', return_wfn=True)

    # model can be 'CCD', 'CC2', 'CCSD', 'CCSD(T)', or 'CC3'
    cc = pycc.CCwfn(wfn, model='CCSD')
    ecc = cc.solve_cc(e_conv=1e-8, r_conv=1e-7, maxiter=75)

Lambda amplitudes and densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the ground-state amplitudes are converged, build the similarity-transformed
Hamiltonian, solve the lambda equations, and form the one- and two-particle reduced
density matrices::

    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lcc = cclambda.solve_lambda(e_conv=1e-8, r_conv=1e-7)
    density = pycc.ccdensity(cc, cclambda)

MP2 energy
~~~~~~~~~~
::

    emp2 = pycc.MPwfn(wfn).compute_energy()

Derivative properties (IR / VCD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyCC computes analytic MO-basis derivative properties for both the Hartree-Fock reference
(:class:`~pycc.hfwfn.HFwfn`) and MP2 (:class:`~pycc.mpwfn.MPwfn`) -- the ingredients for IR
and VCD spectra. The :mod:`pycc.properties` facade is the recommended interface: one call per
property, dispatching on the wavefunction type and returning a
:class:`~pycc.properties.PropertyComponents` with the physical decomposition
``total = nuclear + reference + correlation`` (the ``correlation`` block is an all-zeros array
for an :class:`~pycc.hfwfn.HFwfn`)::

    hf = pycc.HFwfn(wfn)
    mp = pycc.MPwfn(wfn); mp.compute_energy()

    r = pycc.hessian(mp)               # nuclear Hessian; works with pycc.hessian(hf) too
    r.total                            # nuclear + reference + correlation  (3*natom, 3*natom)
    r.reference                        # SCF contribution  (== r.scf == r.hf)
    r.correlation                      # MP2 correlation contribution
    r.nuclear                          # nuclear-repulsion second derivative

    pycc.gradient(mp)                  # nuclear gradient             (natom, 3)
    pycc.polarizability(mp)            # static dipole polarizability (3, 3)
    pycc.apt(mp, gauge='length')       # atomic polar tensors         (natom, 3, 3)
    pycc.apt(mp, gauge='velocity')     # velocity-gauge APTs          (natom, 3, 3)
    pycc.aat(mp)                       # atomic axial tensors (VCD)   (natom, 3, 3)

Every property is available for both spin paths (spin-adapted closed-shell RHF and spin-orbital,
selected by ``orbital_basis`` on the wavefunction), all-electron and frozen core. The MP2 second
derivatives (polarizability, APT, Hessian) offer two independent algorithms via ``route='explicit'``
(default) or ``route='2n+1'`` (O(N)-cheaper) -- an efficiency lever and a mutual cross-check.

The underlying per-wavefunction methods (``hf.gradient()``, ``mp.hessian()``,
``hf.atomic_axial_tensors()``, ...) remain available for the individual reference or correlation
contributions. The nuclear coupled-perturbed Hartree-Fock (CPHF) response is solved once and
cached, so computing several properties does not repeat that work.

GPU and mixed precision
~~~~~~~~~~~~~~~~~~~~~~~~~
The canonical ground-state and real-time CC methods can run on a GPU and/or in single
precision (PyTorch required)::

    cc = pycc.ccwfn(wfn, device='GPU', precision='SP')
