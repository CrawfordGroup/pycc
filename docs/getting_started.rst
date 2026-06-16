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
    cc = pycc.ccwfn(wfn, model='CCSD')
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

Hartree-Fock derivative properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~pycc.hfwfn.HFwfn` provides MO-basis analytic derivative properties of the RHF
reference -- the ingredients for IR and VCD spectra::

    hf = pycc.HFwfn(wfn)
    grad  = hf.gradient()              # nuclear gradient             (natom, 3)
    hess  = hf.hessian()               # nuclear Hessian              (3*natom, 3*natom)
    alpha = hf.polarizability()        # static dipole polarizability (3, 3)
    apt   = hf.dipole_derivatives()    # atomic polar tensors         (natom, 3, 3)
    aat   = hf.atomic_axial_tensors()  # atomic axial tensors         (natom, 3, 3)

The nuclear coupled-perturbed Hartree-Fock (CPHF) response is solved once and cached, so
computing the Hessian and then the dipole/axial tensors does not repeat that work.

GPU and mixed precision
~~~~~~~~~~~~~~~~~~~~~~~~~
The canonical ground-state and real-time CC methods can run on a GPU and/or in single
precision (PyTorch required)::

    cc = pycc.ccwfn(wfn, device='GPU', precision='SP')
