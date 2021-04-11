"""
PyCC
A Python-based coupled cluster implementation.
"""

# Add imports here
from .pycc import ccwfn

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
