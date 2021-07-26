PyCC
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build
Status](https://github.com/CrawfordGroup/pycc/workflows/CI/badge.svg)](https://github.com/lothian/pycc/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/CrawfordGroup/pycc/branch/main/graph/badge.svg)](https://codecov.io/gh/lothian/pycc/branch/main)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/lothian/pycc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lothian/pycc/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/lothian/pycc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/lothian/pycc/context:python)

A Python-based coupled cluster implementation.  Current capabilities include:
  - Spin-adapted RHF-CCSD and RHF-CCSD(T) energies
  - Triples-drivers for (T), CC3, and other approximate triples
  - RHF-CCSD densities
  - Real-time CCSD
  - LPNO-CCSD energies and RT-CC (preliminary)

Future plans:
  - CC2 and CC3 methods
  - Linear and quadratic response functions
  - EOM-CC
  - Single- and mixed-precision arithmetic
  - Analytic gradients

This repository is currently under development. To do a developmental install, download this repository and type `pip install -e .` in the repository directory.

This package requires the following:
  - [psi4](https://psicode.org)
  - [numpy](https://numpy.org/)
  - [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)
  - [scipy](https://www.scipy.org/)

### Copyright

Copyright (c) 2021, T. Daniel Crawford


#### Acknowledgements
 
Project structure based on the 
[MolSSI's](https://molssi.org) [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) Version 1.5.
