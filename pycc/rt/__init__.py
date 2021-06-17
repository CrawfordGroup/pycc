"""
RT: a subpackage handing real-time coupled cluster routines.
============================================================
"""

# Add imports here
from .rtcc import rtcc
from .lasers import gaussian_laser
from .lasers import sine_square_laser

## Handle versioneer
#from ._version import get_versions
#versions = get_versions()
#__version__ = versions['version']
#__git_revision__ = versions['full-revisionid']
#del get_versions, versions
