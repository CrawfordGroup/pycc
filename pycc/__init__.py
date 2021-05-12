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
from .rtcc import rtcc
from .lasers import gaussian_laser
from .lasers import sine_square_laser

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
