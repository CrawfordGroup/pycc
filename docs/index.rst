.. pycc documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyCC: A Python electronic-structure package
=========================================================
PyCC is a Python *ab initio* electronic-structure package centered on coupled
cluster, built on a shared :class:`~pycc.wavefunction.Wavefunction` base that also
hosts Hartree-Fock and MP2, both with analytic derivative properties (gradients,
polarizabilities, Hessians, APTs, and AATs -- the ingredients for IR and VCD spectra).
It is a reference implementation: the emphasis is on clear, equation-shaped code for
validating production-level quantum-chemistry codes, rather than on raw performance.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
