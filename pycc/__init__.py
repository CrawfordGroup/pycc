"""
PyCC: A Python-based coupled cluster implementation.
====================================================


"""

# Add imports here
from .ccwfn import CCwfn, ccwfn  # ccwfn: backward-compat alias for CCwfn
from .mpwfn import MPwfn
from .hfwfn import HFwfn
from .ciwfn import CIwfn
from .cchbar import cchbar
from .cclambda import cclambda
from .ccdensity import ccdensity
from .ccresponse import ccresponse
from .ccresponse import pertbar
from pycc.rt.rtcc import rtcc
from .cceom import cceom
from .ccderiv import CCderiv
from .mpderiv import MPderiv
from .cideriv import CIderiv
from .properties import PropertyComponents, aat, apt, dipole, gradient, hessian, polarizability, register_deriv

# Route the property facade's correlation-derivative calls for each wavefunction to its downstream
# driver (the CorrelatedDerivs leaf carrying the correlation-property methods).
register_deriv(CCwfn, CCderiv)
register_deriv(MPwfn, MPderiv)
# CIderiv is a Phase-4 stub (density hooks not yet implemented); until they are, CISD properties
# stay on the transitional CIwfn code. Uncomment to route CISD through the driver once the hooks in
# cideriv.py work -- see the cideriv module docstring for the migration roadmap.
# register_deriv(CIwfn, CIderiv)

__all__ = ['CCwfn', 'ccwfn', 'MPwfn', 'HFwfn', 'CIwfn', 'cchbar', 'cclambda', 'ccdensity', 'ccresponse', 'pertbar', 'rtcc', 'cceom', 'CCderiv', 'MPderiv', 'CIderiv', 'PropertyComponents', 'aat', 'apt', 'dipole', 'gradient', 'hessian', 'polarizability']

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
