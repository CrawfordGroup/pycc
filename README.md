PyCC
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build
Status](https://github.com/CrawfordGroup/pycc/workflows/CI/badge.svg)](https://github.com/CrawfordGroup/pycc/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/CrawfordGroup/pycc/branch/main/graph/badge.svg)](https://codecov.io/gh/CrawfordGroup/pycc/branch/main)
[![Documentation Status](https://readthedocs.org/projects/crawfordgrouppycc/badge/?version=latest)](https://crawfordgrouppycc.readthedocs.io/en/latest/?badge=latest)

A Python-based electronic-structure package centered on coupled cluster, built on a
shared `Wavefunction` base that also hosts Hartree-Fock, MP2, and CI.  Current capabilities
include:
  - RHF-/UHF-/ROHF-based CCD, CC2, CCSD, CCSD(T), and CC3 energies, via either a
    spin-adapted (closed-shell RHF) or a spin-orbital formalism; the latter enables the
    open-shell (UHF/ROHF) correlated methods
  - Triples drivers for various approximate triples methods
  - One- and two-electron (reduced) density matrices for CC2 and CCSD (with CC3 and (T)
    contributions), and CC one-electron properties (e.g. dipole moments)
  - RHF-EOM-CCSD excitation energies
  - Linear response functions (dynamic polarizability, optical rotation) for both the
    spin-adapted (closed-shell) and spin-orbital references
  - Real-time (RT) CC2, CCSD, and CC3 with a selection of integrators
  - Local (PAO, PNO, PNO++) CCSD energies, and local RT-CC
  - Analytic MO-basis derivative properties for Hartree-Fock (`HFwfn`) and MP2 (`MPwfn`):
    energy gradient, static dipole polarizability, nuclear Hessian (force-constant matrix),
    atomic polar tensors (APTs / dipole derivatives, in both the length and velocity gauge),
    and atomic axial tensors (AATs) -- the building blocks for IR and VCD spectra. Both spin
    paths (spin-adapted closed-shell RHF and spin-orbital), all-electron and frozen-core; the
    MP2 second derivatives come via two independent routes (explicit-derivative and 2n+1)
  - A uniform property interface -- `pycc.dipole`, `pycc.gradient`, `pycc.polarizability`,
    `pycc.hessian`, `pycc.apt`, `pycc.aat` -- that dispatches on wavefunction type and returns
    each property's nuclear/reference/correlation decomposition as a `PropertyComponents`
    dataclass (`.total`, `.nuclear`, `.reference`, `.correlation`, `.electronic`)
  - MP2 (`MPwfn`) and CISD/CID (`CIwfn`) energies (closed- and open-shell)
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
