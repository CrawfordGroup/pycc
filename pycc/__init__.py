"""
PyCC: A Python-based coupled cluster implementation.
====================================================


"""

# Add imports here
from .ccenergy import ccenergy
from .cchbar import cchbar
from .cclambda import cclambda
from .ccdensity import ccdensity
from .cctriples import cctriples
from pycc.rt.rtcc import rtcc

__all__ = ['ccenergy', 'cchbar', 'cclambda', 'ccdensity', 'cctriples', 'rtcc']

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
