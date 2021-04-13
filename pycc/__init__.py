"""
PyCC: A Python-based coupled cluster implementation.
====================================================


"""

# Add imports here
from .ccenergy import ccenergy
from .hbar import cchbar
from .cclambda import cclambda

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
