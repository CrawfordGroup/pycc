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
* `h5py <https://www.h5py.org/>`_ *(on-disk cache for derivative tensors; falls back to in-memory memoization if absent)*
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
PyCC computes analytic MO-basis derivative properties for the Hartree-Fock reference
(:class:`~pycc.hfwfn.HFwfn`) and the correlated methods MP2, CISD, and CCSD/CCSD(T) --
the ingredients for IR and VCD spectra. The :mod:`pycc.properties` facade is the interface:
one call per property, returning a :class:`~pycc.properties.PropertyComponents` with the
physical decomposition ``total = nuclear + reference + correlation`` (the ``correlation``
block is all zeros for an :class:`~pycc.hfwfn.HFwfn`).

For a correlated method, pass a **derivative driver** -- :class:`~pycc.mpderiv.MPderiv`,
:class:`~pycc.cideriv.CIderiv`, or :class:`~pycc.ccderiv.CCderiv`. The driver's constructor
runs the (one-time) amplitude/response solve and caches the perturbed responses, so build it
once and reuse it across properties::

    hf = pycc.HFwfn(wfn)
    mp = pycc.MPwfn(wfn); mp.compute_energy()
    d = pycc.MPderiv(mp)               # MP2 derivative driver (owns the solve + response cache)

    r = pycc.hessian(d)                # nuclear Hessian
    r.total                            # nuclear + reference + correlation  (3*natom, 3*natom)
    r.reference                        # SCF contribution
    r.correlation                      # MP2 correlation contribution
    r.nuclear                          # nuclear-repulsion second derivative

    pycc.gradient(d)                   # nuclear gradient             (natom, 3)
    pycc.polarizability(d)             # static dipole polarizability (3, 3)
    pycc.apt(d, gauge='length')        # atomic polar tensors         (natom, 3, 3)
    pycc.apt(d, gauge='velocity')      # velocity-gauge APTs          (natom, 3, 3)
    pycc.aat(d)                        # atomic axial tensors (VCD)   (natom, 3, 3)

The reference-only (SCF) property takes the :class:`~pycc.hfwfn.HFwfn` directly --
``pycc.hessian(hf)`` -- with a zero ``correlation`` block.

Every property is available for both spin paths (spin-adapted closed-shell RHF and
spin-orbital, selected by ``orbital_basis`` on the wavefunction), all-electron and frozen
core. The length-gauge APT has two equivalent algorithms, ``route='2n+1-field'`` (default,
the ``O(N)``-cheaper 3-field-solve route) and ``route='2n+1-nuclear'`` -- a mutual
cross-check. Because the response solves are cached on the driver, computing several
properties from one driver does not repeat that work.

GPU and mixed precision
~~~~~~~~~~~~~~~~~~~~~~~~~
The canonical ground-state and real-time CC methods can run on a GPU and/or in single
precision (PyTorch required)::

    cc = pycc.ccwfn(wfn, device='GPU', precision='SP')
