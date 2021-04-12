import numpy as np
import time
from .utils import helper_diis
from .lambda_eqs import build_l1, build_l2, build_Goo, build_Gvv, ccsd_pseudoenergy


class cclambda(object):


	def __init__(self, ccenergy, cchbar):

        self.l1 = 2.0 * ccenergy.t1
        self.l2 = 2.0 * (2.0 * ccenergy.t2 - ccenergy.t2.swapaxes(2,3))

