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
from .properties import PropertyComponents, aat, apt, dipole, gradient, hessian, polarizability, optical_rotation
from .vibanalysis import harmonic_analysis, ir, vcd
from .checkpoint import Checkpoint, save_checkpoint, load_checkpoint

__all__ = ['CCwfn', 'ccwfn', 'MPwfn', 'HFwfn', 'CIwfn', 'cchbar', 'cclambda', 'ccdensity', 'ccresponse', 'pertbar', 'rtcc', 'cceom', 'CCderiv', 'MPderiv', 'CIderiv', 'PropertyComponents', 'aat', 'apt', 'dipole', 'gradient', 'hessian', 'polarizability', 'optical_rotation', 'harmonic_analysis', 'ir', 'vcd', 'Checkpoint', 'save_checkpoint', 'load_checkpoint']

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
