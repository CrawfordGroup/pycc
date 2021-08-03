"""
ccdensity.py: Builds the CC density.
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
from .density_eqs import build_Dov, build_Dvo, build_Dvv, build_Doo
from .density_eqs import build_Doooo, build_Dvvvv, build_Dooov, build_Dvvvo
from .density_eqs import build_Dovov, build_Doovv
import numpy as np
from opt_einsum import contract


class ccdensity(object):
    """
    An RHF-CCSD Density object.

    Attributes
    ----------
    Dov : NumPy array
        The occupied-virtual block of the one-body density.
    Dvo : NumPy array
        The virtual-occupied block of the one-body density.
    Dvv : NumPy array
        The virtual-virtual block of the one-body density.
    Doo : NumPy array
        The occupied-occupied block of the one-body density.
    Doooo : NumPy array
        The occ,occ,occ,occ block of the two-body density.
    Dvvvv : NumPy array
        The vir,vir,vir,vir block of the two-body density.
    Dooov : NumPy array
        The occ,occ,occ,vir block of the two-body density.
    Dvvvo : NumPy array
        The vir,vir,vir,occ block of the two-body density.
    Dovov : NumPy array
        The occ,vir,occ,vir block of the two-body density.
    Doovv : NumPy array
        The occ,occ,vir,vir block of the two-body density.
        The occ,vir,occ,occ block of the two-body density.

    Methods
    -------
    compute_energy() :
        Compute the CC energy from the density.  If only onepdm is available, just compute the one-electron energy.
    compute_onepdm() :
        Compute the one-electron density for a given set of amplitudes (useful for RTCC)
    """
    def __init__(self, ccwfn, cclambda, onlyone=False):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            contains the necessary T-amplitudes (either instantiated to defaults or converged)
        cclambda : PyCC cclambda object
            Contains the necessary Lambda-amplitudes (instantiated to defaults or converged)
        onlyone : Boolean
            only compute the onepdm if True

        Returns
        -------
        None
        """

        time_init = time.time()

        self.ccwfn = ccwfn
        self.cclambda = cclambda

        t1 = ccwfn.t1
        t2 = ccwfn.t2
        l1 = cclambda.l1
        l2 = cclambda.l2

        self.Dov = build_Dov(t1, t2, l1, l2)
        self.Dvo = build_Dvo(l1)
        self.Dvv = build_Dvv(t1, t2, l1, l2)
        self.Doo = build_Doo(t1, t2, l1, l2)

        self.onlyone = onlyone

        if onlyone is False:
            self.Doooo = build_Doooo(t1, t2, l2)
            self.Dvvvv = build_Dvvvv(t1, t2, l2)
            self.Dooov = build_Dooov(t1, t2, l1, l2)
            self.Dvvvo = build_Dvvvo(t1, t2, l1, l2)
            self.Dovov = build_Dovov(t1, t2, l1, l2)
            self.Doovv = build_Doovv(t1, t2, l1, l2)

        print("\nCCDENSITY constructed in %.3f seconds.\n" % (time.time() - time_init))

    def compute_energy(self):
        """
        Compute the CC energy from the density.  If only onepdm is available, just compute the one-electron energy.

        Parameters
        ----------
        None

        Returns
        -------
        ecc | float
            CCSD correlation energy computed using the one- and two-electron densities
        """

        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI

        oo_energy = contract('ij,ij->', F[o,o], self.Doo)
        vv_energy = contract('ab,ab->', F[v,v], self.Dvv)
        eone = oo_energy + vv_energy
        print("One-electron CCSD energy = %20.15f" % eone)

        if self.onlyone is True:
            print("Only one-electron density available.")
            ecc = eone
        else:
            oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], self.Doooo)
            vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], self.Dvvvv)
            ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], self.Dooov)
            vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], self.Dvvvo)
            ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], self.Dovov)
            oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], self.Doovv)
            etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy

            print("OOOV Energy = %20.15f" % oooo_energy)
            print("OOOV Energy = %20.15f" % vvvv_energy)
            print("OOOV Energy = %20.15f" % ooov_energy)
            print("VVVO Energy = %20.15f" % vvvo_energy)
            print("OVOV Energy = %20.15f" % ovov_energy)
            print("OOVV Energy = %20.15f" % oovv_energy)
            print("Two-electron CCSD energy = %20.15f" % etwo)
            ecc = eone + etwo

        print("CCSD Correlation Energy  = %20.15f" % ecc)

        self.ecc = ecc

        return ecc

    def compute_onepdm(self, t1, t2, l1, l2, withref=False):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes
        withref : Boolean (default: False)
            include the reference contribution if True

        Returns
        -------
        onepdm : NumPy array
            the CC one-electron density as a single, full matrix
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        nt = no + nv

        opdm = np.zeros((nt, nt), dtype='complex128')
        opdm[o,o] = build_Doo(t1, t2, l1, l2)
        if withref is True:
            opdm[o,o] += 2.0 * np.eye(no)  # Reference contribution
        opdm[v,v] = build_Dvv(t1, t2, l1, l2)
        opdm[o,v] = build_Dov(t1, t2, l1, l2)
        opdm[v,o] = build_Dvo(l1)

        return opdm
