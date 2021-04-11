"""
PyCC
A Python-based coupled cluster implementation.
"""

# Add imports here
from .pycc import ccwfn
from .ccsd_eqs import build_tau
from .ccsd_eqs import build_Fae, build_Fmi, build_Fme
from .ccsd_eqs import build_Wmnij, build_Wmbej, build_Wmbje, build_Zmbij
from .ccsd_eqs import r_T1, r_T2, ccsd_energy
from .hbar_eqs import build_Hov, build_Hvv, build_Hoo
from .hbar_eqs import build_Hoooo, build_Hvvvv, build_Hvovv, build_Hooov
from .hbar_eqs import build_Hovvo, build_Hovov, build_Hvvvo, build_Hovoo
from .lambda_eqs import r_L1, r_L2

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
