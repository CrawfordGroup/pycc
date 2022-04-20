"""
ccdensity.py: Builds the CC density.
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np
from opt_einsum import contract


class ccdensity(object):
    """
    An RHF-CC Density object.

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

        self.Dov = self.build_Dov(t1, t2, l1, l2)
        self.Dvo = self.build_Dvo(l1)
        self.Dvv = self.build_Dvv(t1, t2, l1, l2)
        self.Doo = self.build_Doo(t1, t2, l1, l2)

        self.onlyone = onlyone

        if onlyone is False:
            self.Doooo = self.build_Doooo(t1, t2, l2)
            self.Dvvvv = self.build_Dvvvv(t1, t2, l2)
            self.Dooov = self.build_Dooov(t1, t2, l1, l2)
            self.Dvvvo = self.build_Dvvvo(t1, t2, l1, l2)
            self.Dovov = self.build_Dovov(t1, t2, l1, l2)
            self.Doovv = self.build_Doovv(t1, t2, l1, l2)

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
            CC correlation energy computed using the one- and two-electron densities
        """

        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI

        oo_energy = contract('ij,ij->', F[o,o], self.Doo)
        vv_energy = contract('ab,ab->', F[v,v], self.Dvv)
        eone = oo_energy + vv_energy
        print("One-electron CC energy = %20.15f" % eone)

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
            print("Two-electron CC energy = %20.15f" % etwo)
            ecc = eone + etwo

        print("CC Correlation Energy  = %20.15f" % ecc)

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
        opdm[o,o] = self.build_Doo(t1, t2, l1, l2)
        if withref is True:
            opdm[o,o] += 2.0 * np.eye(no)  # Reference contribution
        opdm[v,v] = self.build_Dvv(t1, t2, l1, l2)
        opdm[o,v] = self.build_Dov(t1, t2, l1, l2)
        opdm[v,o] = self.build_Dvo(l1)

        return opdm

    def build_Doo(self, t1, t2, l1, l2):  # complete
        if self.ccwfn.model == 'CCD':
            Doo = -contract('imef,jmef->ij', t2, l2)
        else:
            Doo = -1.0 * contract('ie,je->ij', t1, l1)
            Doo -= contract('imef,jmef->ij', t2, l2)
        return Doo


    def build_Dvv(self, t1, t2, l1, l2):  # complete
        if self.ccwfn.model == 'CCD':
            Dvv = contract('mnbe,mnae->ab', t2, l2)
        else:
            Dvv = contract('mb,ma->ab', t1, l1)
            Dvv += contract('mnbe,mnae->ab', t2, l2)
        return Dvv


    def build_Dvo(self, l1):  # complete
        return l1.T.copy()


    def build_Dov(self, t1, t2, l1, l2):  # complete
        if self.ccwfn.model == 'CCD':
            Dov = np.zeros_like(t1)
        else:
            Dov = 2.0 * t1.copy()
            Dov += 2.0 * contract('me,imae->ia', l1, t2)
            Dov -= contract('me,miae->ia', l1, self.ccwfn.build_tau(t1, t2))
            tmp = contract('mnef,inef->mi', l2, t2)
            Dov -= contract('mi,ma->ia', tmp, t1)
            tmp = contract('mnef,mnaf->ea', l2, t2)
            Dov -= contract('ea,ie->ia', tmp, t1)
        return Dov


    def build_Doooo(self, t1, t2, l2):  # complete
        if self.ccwfn.model == 'CCD':
            return contract('ijef,klef->ijkl', t2, l2)
        elif self.ccwfn.model == 'CC2':
            return contract('jf, klif->ijkl', t1, contract('ie, klef->klif', t1, l2))
        else:
            return contract('ijef,klef->ijkl', self.ccwfn.build_tau(t1, t2), l2)


    def build_Dvvvv(self, t1, t2, l2):  # complete
        if self.ccwfn.model == 'CCD':
            return contract('mnab,mncd->abcd', t2, l2)
        elif self.ccwfn.model == 'CC2':
            return contract('nb,ancd->abcd', t1, contract('ma,mncd->ancd', t1, l2))
        else:
            return contract('mnab,mncd->abcd', self.ccwfn.build_tau(t1, t2), l2)


    def build_Dooov(self, t1, t2, l1, l2):  # complete
        if self.ccwfn.model == 'CCD':
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            Dooov = np.zeros((no,no,no,nv))
        else:
            tmp = 2.0 * self.ccwfn.build_tau(t1, t2) - self.ccwfn.build_tau(t1, t2).swapaxes(2, 3)
            Dooov = -1.0 * contract('ke,ijea->ijka', l1, tmp)
            Dooov -= contract('ie,jkae->ijka', t1, l2)

            if self.ccwfn.model != 'CC2':

                Goo = self.cclambda.build_Goo(t2, l2)
                Dooov -= 2.0 * contract('ik,ja->ijka', Goo, t1)
                Dooov += contract('jk,ia->ijka', Goo, t1)
                tmp = contract('jmaf,kmef->jake', t2, l2)
                Dooov -= 2.0 * contract('jake,ie->ijka', tmp, t1)
                Dooov += contract('iake,je->ijka', tmp, t1)

                tmp = contract('ijef,kmef->ijkm', t2, l2)
                Dooov += contract('ijkm,ma->ijka', tmp, t1)
                tmp = contract('mjaf,kmef->jake', t2, l2)
                Dooov += contract('jake,ie->ijka', tmp, t1)
                tmp = contract('imea,kmef->iakf', t2, l2)
                Dooov += contract('iakf,jf->ijka', tmp, t1)

            tmp = contract('kmef,jf->kmej', l2, t1)
            tmp = contract('kmej,ie->kmij', tmp, t1)
            Dooov += contract('kmij,ma->ijka', tmp, t1)
        return Dooov


    def build_Dvvvo(self, t1, t2, l1, l2):  # complete
        if self.ccwfn.model == 'CCD':
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            Dvvvo = np.zeros((nv,nv,nv,no))
        else:
            tmp = 2.0 * self.ccwfn.build_tau(t1, t2) - self.ccwfn.build_tau(t1, t2).swapaxes(2, 3)
            Dvvvo = contract('mc,miab->abci', l1, tmp)
            Dvvvo += contract('ma,imbc->abci', t1, l2)

            if self.ccwfn.model != 'CC2':
                
                Gvv = self.cclambda.build_Gvv(t2, l2)
                Dvvvo -= 2.0 * contract('ca,ib->abci', Gvv, t1)
                Dvvvo += contract('cb,ia->abci', Gvv, t1)
                tmp = contract('imbe,nmce->ibnc', t2, l2)
                Dvvvo += 2.0 * contract('ibnc,na->abci', tmp, t1)
                Dvvvo -= contract('ianc,nb->abci', tmp, t1)

                tmp = contract('nmab,nmce->abce', t2, l2)
                Dvvvo -= contract('abce,ie->abci', tmp, t1)
                tmp = contract('niae,nmce->iamc', t2, l2)
                Dvvvo -= contract('iamc,mb->abci', tmp, t1)
                tmp = contract('mibe,nmce->ibnc', t2, l2)
                Dvvvo -= contract('ibnc,na->abci', tmp, t1)

            tmp = contract('nmce,ie->nmci', l2, t1)
            tmp = contract('nmci,na->amci', tmp, t1)
            Dvvvo -= contract('amci,mb->abci', tmp, t1)
        return Dvvvo


    def build_Dovov(self, t1, t2, l1, l2):  # complete
        if self.ccwfn.model == 'CCD':
            Dovov = -contract('mibe,jmea->iajb', t2, l2)
            Dovov -= contract('imbe,mjea->iajb', t2, l2)
        else:
            Dovov = -1.0 * contract('ia,jb->iajb', t1, l1)
            if self.ccwfn.model == 'CC2':
                Dovov -= contract('mb,jmia->iajb', t1, contract('ie,jmea->jmia', t1, l2))
            else:
                Dovov -= contract('mibe,jmea->iajb', self.ccwfn.build_tau(t1, t2), l2)
                Dovov -= contract('imbe,mjea->iajb', t2, l2)
        return Dovov


    def build_Doovv(self, t1, t2, l1, l2):
        tau = self.ccwfn.build_tau(t1, t2)
        tau_spinad = 2.0 * tau - tau.swapaxes(2,3)

        if self.ccwfn.model == 'CCD':
            Doovv = 2.0 * tau_spinad + l2

            Doovv += 4.0 * contract('imae,mjeb->ijab', t2, l2)
            Doovv -= 2.0 * contract('mjbe,imae->ijab', tau, l2)

            tmp_oooo = contract('ijef,mnef->ijmn', t2, l2)
            Doovv += contract('ijmn,mnab->ijab', tmp_oooo, t2)
            tmp1 = contract('njbf,mnef->jbme', t2, l2)
            Doovv += contract('jbme,miae->ijab', tmp1, t2)
            tmp1 = contract('imfb,mnef->ibne', t2, l2)
            Doovv += contract('ibne,njae->ijab', tmp1, t2)
            Gvv = self.cclambda.build_Gvv(t2, l2)
            Doovv += 4.0 * contract('eb,ijae->ijab', Gvv, tau)
            Doovv -= 2.0 * contract('ea,ijbe->ijab', Gvv, tau)
            Goo = self.cclambda.build_Goo(t2, l2)
            Doovv -= 4.0 * contract('jm,imab->ijab', Goo, tau)  # use tau_spinad?
            Doovv += 2.0 * contract('jm,imba->ijab', Goo, tau)
            tmp1 = contract('inaf,mnef->iame', t2, l2)
            Doovv -= 4.0 * contract('iame,mjbe->ijab', tmp1, tau)
            Doovv += 2.0 * contract('ibme,mjae->ijab', tmp1, tau)
            Doovv += 4.0 * contract('jbme,imae->ijab', tmp1, t2)
            Doovv -= 2.0 * contract('jame,imbe->ijab', tmp1, t2)

        else:
            Doovv = 4.0 * contract('ia,jb->ijab', t1, l1)
            Doovv += 2.0 * tau_spinad
            Doovv += l2

            tmp1 = 2.0 * t2 - t2.swapaxes(2,3)
            tmp2 = 2.0 * contract('me,jmbe->jb', l1, tmp1)
            Doovv += 2.0 * contract('jb,ia->ijab', tmp2, t1)
            Doovv -= contract('ja,ib->ijab', tmp2, t1)
            tmp2 = 2.0 * contract('ijeb,me->ijmb', tmp1, l1)
            Doovv -= contract('ijmb,ma->ijab', tmp2, t1)
            tmp2 = 2.0 * contract('jmba,me->jeba', tau_spinad, l1)
            Doovv -= contract('jeba,ie->ijab', tmp2, t1)

            if self.ccwfn.model == 'CC2':
                Doovv -= 2.0 * contract('mb,imaj->ijab', t1, contract('je,imae->imaj', t1, l2))
            else:
                Doovv += 4.0 * contract('imae,mjeb->ijab', t2, l2)
                Doovv -= 2.0 * contract('mjbe,imae->ijab', tau, l2)

                tmp_oooo = contract('ijef,mnef->ijmn', t2, l2)
                Doovv += contract('ijmn,mnab->ijab', tmp_oooo, t2)
                tmp1 = contract('njbf,mnef->jbme', t2, l2)
                Doovv += contract('jbme,miae->ijab', tmp1, t2)
                tmp1 = contract('imfb,mnef->ibne', t2, l2)
                Doovv += contract('ibne,njae->ijab', tmp1, t2)
                Gvv = self.cclambda.build_Gvv(t2, l2)
                Doovv += 4.0 * contract('eb,ijae->ijab', Gvv, tau)
                Doovv -= 2.0 * contract('ea,ijbe->ijab', Gvv, tau)
                Goo = self.cclambda.build_Goo(t2, l2)
                Doovv -= 4.0 * contract('jm,imab->ijab', Goo, tau)  # use tau_spinad?
                Doovv += 2.0 * contract('jm,imba->ijab', Goo, tau)
                tmp1 = contract('inaf,mnef->iame', t2, l2)
                Doovv -= 4.0 * contract('iame,mjbe->ijab', tmp1, tau)
                Doovv += 2.0 * contract('ibme,mjae->ijab', tmp1, tau)
                Doovv += 4.0 * contract('jbme,imae->ijab', tmp1, t2)
                Doovv -= 2.0 * contract('jame,imbe->ijab', tmp1, t2)

                # this can definitely be optimized better
                tmp = contract('nb,ijmn->ijmb', t1, tmp_oooo)
                Doovv += contract('ma,ijmb->ijab', t1, tmp)
                tmp = contract('ie,mnef->mnif', t1, l2)
                tmp = contract('jf,mnif->mnij', t1, tmp)
                Doovv += contract('mnij,mnab->ijab', tmp, t2)
                tmp = contract('ie,mnef->mnif', t1, l2)
                tmp = contract('mnif,njbf->mijb', tmp, t2)
                Doovv += contract('ma,mijb->ijab', t1, tmp)
                tmp = contract('jf,mnef->mnej', t1, l2)
                tmp = contract('mnej,miae->njia', tmp, t2)
                Doovv += contract('nb,njia->ijab', t1, tmp)
                tmp = contract('je,mnef->mnjf', t1, l2)
                tmp = contract('mnjf,imfb->njib', tmp, t2)
                Doovv += contract('na,njib->ijab', t1, tmp)
                tmp = contract('if,mnef->mnei', t1, l2)
                tmp = contract('mnei,njae->mija', tmp, t2)
                Doovv += contract('mb,mija->ijab', t1, tmp)

            tmp = contract('jf,mnef->mnej', t1, l2)
            tmp = contract('ie,mnej->mnij', t1, tmp)
            tmp = contract('nb,mnij->mbij', t1, tmp)
            Doovv += contract('ma,mbij->ijab', t1, tmp)

        return Doovv
