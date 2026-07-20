API Documentation
=================

Every module below is documented from its source docstrings (``automodule`` with
``:members:``, private members included), so each class, method, and function appears with
its full docstring.

Core infrastructure
-------------------

Wavefunction base class
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.wavefunction
   :members:

Hamiltonian and integrals
~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.hamiltonian
   :members:

Device / precision manager
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.device
   :members:

Wavefunction methods
--------------------

Hartree-Fock (HFwfn)
~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.hfwfn
   :members:

Moller-Plesset MP2 (MPwfn)
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.mpwfn
   :members:

Coupled cluster (CCwfn)
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.ccwfn
   :members:

Configuration interaction (CIwfn)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.ciwfn
   :members:

Coupled-cluster machinery
-------------------------

Similarity-transformed Hamiltonian (cchbar)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.cchbar
   :members:

Lambda amplitudes (cclambda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.cclambda
   :members:

CC reduced densities (ccdensity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.ccdensity
   :members:

Triples corrections (cctriples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.cctriples
   :members:

CC response functions (ccresponse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.ccresponse
   :members:

Equation-of-motion CC (cceom)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.cceom
   :members:

Analytic derivative properties
------------------------------

Property facade (properties)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.properties
   :members:

Correlated-derivative base (CorrelatedDerivs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.correlatedderivs
   :members:

CC derivative driver (CCderiv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.ccderiv
   :members:

MP2 derivative driver (MPderiv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.mpderiv
   :members:

CISD derivative driver (scaffold / stub)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.cideriv
   :members:

MO derivative integrals (Derivatives)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.derivatives
   :members:

Coupled-perturbed Hartree-Fock (cphf)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.cphf
   :members:

Real-time CC
------------

Real-time CC propagator (rtcc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.rt.rtcc
   :members:

ODE integrators (rt.integrators)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.rt.integrators
   :members:

Laser pulses (rt.lasers)
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.rt.lasers
   :members:

Real-time utilities (rt.utils)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.rt.utils
   :members:

Utilities and support
---------------------

Utilities (utils)
~~~~~~~~~~~~~~~~~
.. automodule:: pycc.utils
   :members:

Exceptions (exceptions)
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc.exceptions
   :members:

Type aliases (_typing)
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pycc._typing
   :members:

