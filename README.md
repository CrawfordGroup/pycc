PyCC
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build
Status](https://github.com/CrawfordGroup/pycc/workflows/CI/badge.svg)](https://github.com/CrawfordGroup/pycc/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/CrawfordGroup/pycc/branch/main/graph/badge.svg)](https://codecov.io/gh/CrawfordGroup/pycc/branch/main)

A Python-based electronic-structure package centered on coupled cluster, built on a
shared `Wavefunction` base that also hosts Hartree-Fock, MP2, and CI.  Current capabilities
include:
  - RHF-/UHF-/ROHF-based CCD, CC2, CCSD, CCSD(T), and CC3 energies
  - Triples drivers for various approximate triples methods
  - One- and two-electron (reduced) density matrices for CC2 and CCSD (with CC3 and (T) contributions)
  - RHF-EOM-CCSD excitation energies
  - Linear response functions (dynamic polarizability, optical rotation)
  - Real-time (RT) CC2, CCSD, and CC3 with a selection of integrators
  - Local (PAO, PNO, PNO++) CCSD energies, and local RT-CC
  - Hartree-Fock (`HFwfn`) MO-basis analytic derivative properties: energy gradient,
    static dipole polarizability, nuclear Hessian (force-constant matrix), atomic polar
    tensors (APTs / dipole derivatives), and atomic axial tensors (AATs) -- the building
    blocks for IR and VCD spectra
  - MP2 (`MPwfn`) and CISD/CID (`CIwfn`) energies
  - GPU implementations for multiple methods
  - Single- and mixed-precision arithmetic

Future plans:
  - Quadratic response functions (in development)
  - CC2 and CC3 excited states
  - Analytic CC gradients

This repository is currently under development. To do a developmental install, download this repository and type `pip install -e .` in the repository directory.

This package requires the following:
  - [psi4](https://psicode.org)
  - [numpy](https://numpy.org/)
  - [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)
  - [scipy](https://www.scipy.org/)

Optional packages:
  - [pytorch](https://pytorch.org/)

### Documentation

The `docs/` directory holds the Sphinx documentation — a
[getting-started guide](docs/getting_started.rst) with runnable examples and an
[API reference](docs/api.rst). GitHub renders these `.rst` files in the browser; to
build the full HTML site locally, install Sphinx (and the Read the Docs theme) and run
`make html` from `docs/`.

### Authors

T. Daniel Crawford, Benjamin G. Peyton, Zhe Wang, Jose Madriaga, Aparna Krishnan

### Copyright

Copyright (c) 2026, T. Daniel Crawford


#### Acknowledgements
 
Project structure based on the 
[MolSSI's](https://molssi.org) [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) Version 1.5.
</content>
