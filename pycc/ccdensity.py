"""
ccdensity.py: Builds the CC density.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import time
from typing import TYPE_CHECKING

import numpy as np
from pycc.ccwfn import HAS_TORCH
if HAS_TORCH:
    import torch
from .cctriples import t3c_ijk, t3c_abc, l3_ijk, l3_abc, t3c_bc, l3_bc, t3_pert_ijk, t3_pert_bc
from .utils import zeros, zeros_like, clone

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn
    from pycc.cclambda import cclambda

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
    def __init__(self, ccwfn: "CCwfn", cclambda: "cclambda", onlyone: bool = False) -> None:
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
        self.contract = self.ccwfn.contract

        o = ccwfn.o
        v = ccwfn.v
        no = ccwfn.no
        nv = ccwfn.nv
        F = ccwfn.H.F
        ERI = ccwfn.H.ERI
        L = ccwfn.H.L if ccwfn.orbital_basis == 'spatial' else None  # no L in spin orbitals
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
            if ccwfn.orbital_basis == 'spinorbital':
                # Spin-orbital 2-PDM: the nine unique (antisymmetric) blocks of Table 6.3.
                self.so_twopdm = self.build_so_twopdm(t1, t2, l1, l2)
            else:
                self.Doooo = self.build_Doooo(t1, t2, l2)
                self.Dvvvv = self.build_Dvvvv(t1, t2, l2)
                self.Dooov = self.build_Dooov(t1, t2, l1, l2)
                self.Dvvvo = self.build_Dvvvo(t1, t2, l1, l2)
                self.Dovov = self.build_Dovov(t1, t2, l1, l2)
                self.Doovv = self.build_Doovv(t1, t2, l1, l2)

        print("\nCCDENSITY constructed in %.3f seconds.\n" % (time.time() - time_init))

    def compute_energy(self) -> float:
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

        contract = self.contract 

        # We assume here that the Brillouin condition holds
        oo_energy = contract('ij,ij->', F[o,o], self.Doo)
        vv_energy = contract('ab,ab->', F[v,v], self.Dvv)
        eone = oo_energy + vv_energy
        print("One-electron CC energy = %20.15f" % eone)

        if self.onlyone is True:
            print("Only one-electron density available.")
            ecc = eone
        elif self.ccwfn.orbital_basis == 'spinorbital':
            # Spin-orbital two-electron energy, 1/4 sum_pqrs Gamma_pqrs <pq||rs>.  The nine stored
            # blocks are contracted with their antisymmetric-image multiplicities (a block whose
            # antisymmetry class has k members contributes k times, since Gamma and <pq||rs> pick up
            # the same sign under each index swap): oooo/vvvv/oovv/vvoo -> 1, the (3+1) blocks -> 2,
            # ovvo -> 4.
            G = self.so_twopdm
            etwo = 0.25 * (
                contract('ijkl,ijkl->', ERI[o,o,o,o], G['ijkl'])
                + contract('abcd,abcd->', ERI[v,v,v,v], G['abcd'])
                + contract('ijab,ijab->', ERI[o,o,v,v], G['ijab'])
                + contract('abij,abij->', ERI[v,v,o,o], G['abij'])
                + 2.0 * contract('ijka,ijka->', ERI[o,o,o,v], G['ijka'])
                + 2.0 * contract('kaij,kaij->', ERI[o,v,o,o], G['kaij'])
                + 2.0 * contract('abci,abci->', ERI[v,v,v,o], G['abci'])
                + 2.0 * contract('ciab,ciab->', ERI[v,o,v,v], G['ciab'])
                + 4.0 * contract('ibaj,ibaj->', ERI[o,v,v,o], G['ibaj']))
            print("Two-electron CC energy = %20.15f" % etwo)
            ecc = eone + etwo
        else:
            oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], self.Doooo)
            vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], self.Dvvvv)
            ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], self.Dooov)
            vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], self.Dvvvo)
            ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], self.Dovov)
            oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], self.Doovv)
            etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy

            print("OOOO Energy = %20.15f" % oooo_energy)
            print("VVVV Energy = %20.15f" % vvvv_energy)
            print("OOOV Energy = %20.15f" % ooov_energy)
            print("VVVO Energy = %20.15f" % vvvo_energy)
            print("OVOV Energy = %20.15f" % ovov_energy)
            print("OOVV Energy = %20.15f" % oovv_energy)
            print("Two-electron CC energy = %20.15f" % etwo)
            ecc = eone + etwo

        print("CC Correlation Energy  = %20.15f" % ecc)

        self.ecc = ecc
        self.eone = eone
        self.etwo = etwo

        return ecc

    def compute_onepdm(self, t1, t2, l1, l2, real_time=False):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes

        Returns
        -------
        onepdm : NumPy array
            the CC one-electron density as a single, full matrix (only the correlated contribution)
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        nt = no + nv
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L if self.ccwfn.orbital_basis == 'spatial' else None

        opdm = zeros((nt, nt), like=t1)
        opdm[o,o] = self.build_Doo(t1, t2, l1, l2)
        opdm[v,v] = self.build_Dvv(t1, t2, l1, l2)
        opdm[o,v] = self.build_Dov(t1, t2, l1, l2)
        opdm[v,o] = self.build_Dvo(l1)

        if self.ccwfn.model == 'CC3' and self.ccwfn.orbital_basis == 'spinorbital':
            # Spin-orbital CC3 Lambda exists, but the CC3 one-particle density
            # (T1-transformed blocks below) is still spatial-only -- a separate
            # follow-on. Fail clearly rather than crash on the spatial builders.
            raise NotImplementedError("Spin-orbital CC3 one-particle density is not "
                                      "implemented (Lambda is); use the spatial path.")

        if self.ccwfn.model == 'CC3':
            Fov = self.ccwfn.build_Fme(o, v, F, L, t1)
            Wvvvo = self.ccwfn.build_cc3_Wabei(o, v, ERI, t1)
            Woooo = self.ccwfn.build_cc3_Wmnij(o, v, ERI, t1)
            Wovoo = self.ccwfn.build_cc3_Wmbij(o, v, ERI, t1, Woooo)
            Wvovv = self.ccwfn.build_cc3_Wamef(o, v, ERI, t1)
            Wooov = self.ccwfn.build_cc3_Wmnie(o, v, ERI, t1)

            opdm[o,v] += self.build_cc3_Dov(o, v, no, nv, F, L, t1, t2, l1, l2, Wvvvo, Wovoo, Fov, Wvovv, Wooov, real_time=real_time)

            # Density matrix blocks in contractions with T1-transformed dipole integrals
            opdm_cc3 = zeros_like(opdm)
            opdm_cc3[o,o] += self.build_cc3_Doo(o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov)
            opdm_cc3[v,v] += self.build_cc3_Dvv(o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov)

            return (opdm, opdm_cc3)

        else:
            return opdm

    def gradient_densities(self):
        """Full-MO one- and two-particle correlation densities for the analytic gradient:
        ``D`` (``nmo x nmo``) and ``Gamma`` (``nmo^4``, physicist ``<pq|rs>`` ordering).

        The convention is the one the density-based gradient machinery assumes -- **no prefactor**
        on the two-particle term -- so the two-electron correlation energy is
        ``contract(Gamma, ERI)`` and the total correlation energy is
        ``contract(D, F) + contract(Gamma, ERI)`` (cf. :meth:`compute_energy`).

        ``D`` places the ``Doo``/``Dov``/``Dvo``/``Dvv`` blocks; ``Gamma`` places the six stored
        blocks and their bra-ket (``rspq``) images with per-block factors, then applies the four-fold
        permutational symmetrization ``Gamma <- 1/4 (Gamma + Gamma_qpsr + Gamma_rspq + Gamma_srqp)``
        so the result carries the proper spatial 2-PDM symmetry (``Gamma_pqrs = Gamma_rspq =
        Gamma_qpsr``).  This matters: the two-particle density stored here (:meth:`compute_energy`
        and the density builders) keeps only asymmetric *representatives* of each block --
        e.g. ``Doovv`` holds only the ``l2`` piece while the ``vvoo`` partner (``t2 + t1 t1 + ...``)
        is folded in under the implicit assumption of a subsequent contraction against a *symmetric*
        quantity (the ERIs, in the energy).  A full 4-index energy/gradient contraction against the
        (8-fold-symmetric) ``<pq|rs>`` only sees the symmetric projection, so the energy and the
        explicit-derivative gradient are insensitive to the missing symmetry.  But the
        generalized-Fock term ``4 sum_rst <pr|st> Gamma_qrst`` contracts only three indices and
        *does* require it -- so the Z-vector gradient needs the symmetrized ``Gamma`` here.  The
        symmetrization is energy-preserving (the antisymmetric complement contracts to zero against
        the symmetric ERI).  (A future re-derivation of the ccdensity 2-PDM equations could carry
        this symmetry natively.)  Densities are placed on the active ``o``/``v`` slices (frozen-core
        rows/columns stay zero).

        **Spin-orbital path:** :meth:`_so_full_twopdm` assembles the ``Gamma`` that is
        antisymmetric within the bra and within the ket (``Gamma_pqrs = -Gamma_qprs = -Gamma_pqsr``),
        but the CC density is **not Hermitian** (``Lambda != T-dagger``), so it is *not* bra-ket
        symmetric (``Gamma_ijab != Gamma_abij``).  The three-index generalized-Fock ``termC`` needs
        that bra-ket symmetry, so -- exactly as the spatial four-fold symmetrization does -- the
        bra-ket average ``Gamma <- 1/2 (Gamma + Gamma_rspq)`` is applied here (for the natively
        antisymmetric ``Gamma`` the spatial four-fold ``1/4(Gamma + Gamma_qpsr + Gamma_rspq +
        Gamma_srqp)`` collapses to this, since the bra/ket antisymmetry already supplies the
        ``qpsr``/``srqp`` images).  It is energy-preserving (the antisymmetric complement contracts
        to zero against the symmetric ``<pq||rs>``).  The remaining twist is the prefactor: the
        spin-orbital two-electron energy is ``1/4 sum Gamma <pq||rs>``, so to keep the same
        no-extra-prefactor convention (``contract(D, F) + contract(Gamma, ERI) = E_corr``) the ``1/4``
        is absorbed into the returned ``Gamma`` -- matching :meth:`MPwfn._so_mp2_tpdm` (whose oovv
        block is ``1/4 t2``) and the ``termC = 4 sum <pr||st> Gamma_qrst`` in
        :meth:`CorrelatedDerivs._so_lagrangian`.  (MP2's ``Gamma`` is already bra-ket symmetric --
        ``oovv = t2``, ``vvoo = t2.T`` -- so it needs no symmetrization.)"""
        if self.onlyone:
            raise RuntimeError("gradient_densities needs the two-particle density "
                               "(construct ccdensity with onlyone=False).")
        ccwfn = self.ccwfn
        o, v, nmo = ccwfn.o, ccwfn.v, ccwfn.nmo
        D = zeros((nmo, nmo), like=self.Doo)
        D[o, o] = self.Doo; D[v, v] = self.Dvv; D[o, v] = self.Dov; D[v, o] = self.Dvo
        if ccwfn.orbital_basis == 'spinorbital':
            Gam = self._so_full_twopdm()
            Gam = 0.5 * (Gam + Gam.transpose(2, 3, 0, 1))   # bra-ket symmetrize (non-Hermitian CC density)
            return D, 0.25 * Gam
        G = zeros((nmo, nmo, nmo, nmo), like=self.Doo)
        G[o, o, o, o] = 0.5 * self.Doooo
        G[v, v, v, v] = 0.5 * self.Dvvvv
        G[o, v, o, v] = self.Dovov
        G[o, o, v, v] = 0.25 * self.Doovv
        G[v, v, o, o] = 0.25 * self.Doovv.transpose(2, 3, 0, 1)
        G[o, o, o, v] = 0.5 * self.Dooov
        G[o, v, o, o] = 0.5 * self.Dooov.transpose(2, 3, 0, 1)
        G[v, v, v, o] = 0.5 * self.Dvvvo
        G[v, o, v, v] = 0.5 * self.Dvvvo.transpose(2, 3, 0, 1)
        G = 0.25 * (G + G.transpose(1, 0, 3, 2) + G.transpose(2, 3, 0, 1) + G.transpose(3, 2, 1, 0))
        return D, G

    def dipole(self, t1, t2, l1, l2):
        """Correlated (CC) contribution to the electric dipole moment.

        Returns the three Cartesian components of ``mu_axis = sum_pq mu_pq D_pq`` --
        the CC one-particle (correlation) density contracted with the MO-basis dipole
        integrals (``H.mu``). The reference (SCF) and nuclear dipole are NOT included;
        add them separately for the total molecular dipole. CCSD/CCD only -- the CC3
        dipole (T1-transformed integrals) is handled by ``rtcc.dipole``.
        """
        if self.ccwfn.model == 'CC3':
            raise NotImplementedError("CC3 dipole: use rtcc.dipole.")
        contract = self.contract
        opdm = self.compute_onepdm(t1, t2, l1, l2)
        mu = self.ccwfn.H.mu
        return [contract('pq,pq->', mu[axis], opdm) for axis in range(3)]

    def build_Doo(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            Doo = (-1.0 * contract('ie,je->ij', t1, l1)
                   - 0.5 * contract('imef,jmef->ij', t2, l2))
            if self.ccwfn.model == 'CCSD(T)':
                Doo = Doo + self.ccwfn.Doo
            return Doo
        if self.ccwfn.model == 'CCD':
            Doo = -contract('imef,jmef->ij', t2, l2)
        else:
            Doo = -1.0 * contract('ie,je->ij', t1, l1)
            Doo -= contract('imef,jmef->ij', t2, l2)
            # (T) contributions computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Doo += self.ccwfn.Doo

        return Doo


    def build_Dvv(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            Dvv = (contract('mb,ma->ab', t1, l1)
                   + 0.5 * contract('mnbe,mnae->ab', t2, l2))
            if self.ccwfn.model == 'CCSD(T)':
                Dvv = Dvv + self.ccwfn.Dvv
            return Dvv
        if self.ccwfn.model == 'CCD':
            Dvv = contract('mnbe,mnae->ab', t2, l2)
        else:
            Dvv = contract('mb,ma->ab', t1, l1)
            Dvv += contract('mnbe,mnae->ab', t2, l2)
            # (T) contributions computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Dvv += self.ccwfn.Dvv

        return Dvv


    def build_Dvo(self, l1):  # complete
        return clone(l1.T)

    def build_Dov(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.orbital_basis == 'spinorbital':
            Dov = clone(t1)
            Dov = Dov + contract('me,imae->ia', l1, t2)
            Dov = Dov - contract('me,ie,ma->ia', l1, t1, t1)
            tmp = contract('mnef,mnaf->ea', l2, t2)
            Dov = Dov - 0.5 * contract('ea,ie->ia', tmp, t1)
            tmp = contract('mnef,inef->mi', l2, t2)
            Dov = Dov - 0.5 * contract('mi,ma->ia', tmp, t1)
            if self.ccwfn.model == 'CCSD(T)':
                Dov = Dov + self.ccwfn.Dov
            return Dov
        if self.ccwfn.model == 'CCD':
            Dov = zeros_like(t1)
        else:
            Dov = 2.0 * clone(t1)

            Dov += 2.0 * contract('me,imae->ia', l1, t2)
            Dov -= contract('me,miae->ia', l1, self.ccwfn.build_tau(t1, t2))
            tmp = contract('mnef,inef->mi', l2, t2)
            Dov -= contract('mi,ma->ia', tmp, t1)
            tmp = contract('mnef,mnaf->ea', l2, t2)
            Dov -= contract('ea,ie->ia', tmp, t1)

            if self.ccwfn.model == 'CCSD(T)':
                Dov += self.ccwfn.Dov

            if HAS_TORCH and isinstance(tmp, torch.Tensor):
                del tmp

        return Dov

    # CC3 contributions to the one electron densities
    def build_cc3_Dov(self, o, v, no, nv, F, L, t1, t2, l1, l2, Wvvvo, Wovoo, Fov, Wvovv, Wooov, real_time=False):
        contract = self.contract
        Dov = zeros_like(t1)
        Zlmdi = zeros_like(t2[:,:,:,:no])
        for i in range(no):
            for j in range(no):
                for k in range(no):                    
                    l3 = l3_ijk(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract)  
                    # Intermediate for Dov_2
                    Zlmdi[i,j] += contract('def,ife->di', l3, t2[k])
                    # Dov_1
                    t3 = t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    if real_time is True:
                        V = F - clone(self.ccwfn.H.F)
                        t3 -= t3_pert_ijk(o, v, i, j, k, t2, V, F, contract)
                    Dov[i] +=  contract('abc,bc->a', t3 - t3.swapaxes(0,1), l2[j,k])
        # Dov_2
        Dov -= contract('lmdi, lmda->ia', Zlmdi, t2)

        return Dov
                                    
    def build_cc3_Doo(self, o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov, real_time=False):
        contract = self.contract
        Doo = zeros_like(l1[:,:no])
        for b in range(nv):
            for c in range(nv):
                t3 = t3c_bc(o, v, b, c, t2, Wvvvo, Wovoo, F, contract)
                if real_time is True:
                    V = F - clone(self.ccwfn.H.F)
                    t3 -= t3_pert_bc(o, v, b, c, t2, V, F, contract)
                l3 = l3_bc(b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract)
                Doo -= 0.5 * contract('lmia,lmja->ij', t3, l3)        

        return Doo        

    def build_cc3_Dvv(self, o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov, real_time=False):
        contract = self.contract
        # Dvv's leading axis is virtual (shape nv,nv), so allocate it directly
        # rather than zeros_like(l1) (no,nv) + pad.
        Dvv = zeros((nv, nv), like=l1)
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    if real_time is True:
                        V = F - clone(self.ccwfn.H.F)
                        t3 -= t3_pert_ijk(o, v, i, j, k, t2, V, F, contract)
                    l3 = l3_ijk(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract)
                    Dvv += 0.5 * contract('bdc,adc->ab', t3, l3)

        return Dvv

    def build_Doooo(self, t1, t2, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            return contract('ijef,klef->ijkl', t2, l2)
        elif self.ccwfn.model == 'CC2':
            return contract('jf, klif->ijkl', t1, contract('ie, klef->klif', t1, l2))
        else:
            return contract('ijef,klef->ijkl', self.ccwfn.build_tau(t1, t2), l2)

    def build_Dvvvv(self, t1, t2, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            return contract('mnab,mncd->abcd', t2, l2)
        elif self.ccwfn.model == 'CC2':
            return contract('nb,ancd->abcd', t1, contract('ma,mncd->ancd', t1, l2))
        else:
            return contract('mnab,mncd->abcd', self.ccwfn.build_tau(t1, t2), l2)

    def build_Dooov(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            Dooov = zeros((no,no,no,nv), like=t1)
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
	
                if HAS_TORCH and isinstance(tmp, torch.Tensor):
                    del tmp, Goo

            tmp = contract('kmef,jf->kmej', l2, t1)
            tmp = contract('kmej,ie->kmij', tmp, t1)
            Dooov += contract('kmij,ma->ijka', tmp, t1)

            # (T) contributions to twopdm computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Dooov += self.ccwfn.Gooov
            
            if HAS_TORCH and isinstance(tmp, torch.Tensor):
                del tmp

        return Dooov


    def build_Dvvvo(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            Dvvvo = zeros((nv,nv,nv,no), like=t1)
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
		
                if HAS_TORCH and isinstance(tmp, torch.Tensor):
                    del tmp, Gvv

            tmp = contract('nmce,ie->nmci', l2, t1)
            tmp = contract('nmci,na->amci', tmp, t1)
            Dvvvo -= contract('amci,mb->abci', tmp, t1)
 
            # (T) contributions to twopdm computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Dvvvo += self.ccwfn.Gvvvo

            if HAS_TORCH and isinstance(tmp, torch.Tensor):
                del tmp               

        return Dvvvo


    def build_Dovov(self, t1, t2, l1, l2):  # complete
        contract = self.contract
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
        contract = self.contract
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
            
            if HAS_TORCH and isinstance(tmp1, torch.Tensor):
                del tmp_oooo, tmp1, Gvv, Goo

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
		
                if HAS_TORCH and isinstance(tmp, torch.Tensor):
                    del tmp, tmp1, tmp2, Goo, Gvv

            tmp = contract('jf,mnef->mnej', t1, l2)
            tmp = contract('ie,mnej->mnij', t1, tmp)
            tmp = contract('nb,mnij->mbij', t1, tmp)
            Doovv += contract('ma,mbij->ijab', t1, tmp)
 
            # (T) contributions to twopdm computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Doovv += self.ccwfn.Goovv

            if HAS_TORCH and isinstance(tmp, torch.Tensor):
                del tmp

        return Doovv

    def build_so_twopdm(self, t1, t2, l1, l2):
        """Spin-orbital CCSD two-particle density -- the nine unique blocks of Table 6.3 of the
        coupled-cluster notes (Crawford), returned as a dict keyed by block label.

        Because the spin-orbital 2-PDM is fully antisymmetric (``Gamma_pqrs = -Gamma_qprs =
        -Gamma_pqsr``), these nine blocks are *representatives*: every one of the sixteen o/v
        block types is generated from them by index antisymmetry (see :meth:`_so_full_twopdm`).
        The CC density is not Hermitian (``Lambda != T-dagger``), so bra-ket partners are
        independent -- e.g. ``ijab`` and ``abij`` are distinct blocks, not transposes.

        The effective double-excitation operator is the spin-orbital tau,
        ``tau^{ab}_{ij} = t^{ab}_{ij} + t^a_i t^b_j - t^a_j t^b_i`` (= ``t2 + P(ij) t1 t1``), which
        also equals ``t2^{ab}_{ij} + 1/2 P(ij)P(ab) t^a_i t^b_j``.  Validated by reconstructing the
        CCSD correlation energy, ``contract(gamma, F) + 1/4 contract(Gamma, <pq||rs>) = E_corr``.
        """
        contract = self.contract
        tau = t2 + contract('ia,jb->ijab', t1, t1) - contract('ja,ib->ijab', t1, t1)
        Pij = lambda Y: Y - Y.swapaxes(0, 1)
        Pab = lambda Y: Y - Y.swapaxes(2, 3)
        Pijab = lambda Y: Pij(Pab(Y))
        # X_{miae} = t^{ae}_{mi} + 2 t^e_i t^a_m  (recurring in the ijab block)
        X = t2 + 2.0 * contract('ie,ma->miae', t1, t1)
        G = {}
        G['ijkl'] = 0.5 * contract('ijef,klef->ijkl', tau, l2)
        G['abcd'] = 0.5 * contract('mncd,mnab->abcd', tau, l2)
        G['ijka'] = (-contract('ke,ijea->ijka', l1, tau)
                     + 0.5 * contract('kmef,ijef,ma->ijka', l2, tau, t1)
                     + Pij(contract('mkef,imae,jf->ijka', l2, t2, t1))
                     - 0.5 * Pij(contract('kmef,imef,ja->ijka', l2, t2, t1)))
        # NB: the t^{ae}_{in} term carries a leading minus sign (erratum vs the printed table).
        G['ciab'] = (contract('mc,miab->ciab', l1, tau)
                     - 0.5 * contract('mnce,mnab,ie->ciab', l2, tau, t1)
                     - Pab(contract('mnce,inae,mb->ciab', l2, t2, t1))
                     + 0.5 * Pab(contract('mnce,mnae,ib->ciab', l2, t2, t1)))
        G['abci'] = contract('miab,mc->abci', l2, t1)
        G['kaij'] = -contract('ijea,ke->kaij', l2, t1)
        G['ibaj'] = (contract('ia,jb->ibaj', t1, l1)
                     + contract('jmbe,miea->ibaj', l2, t2)
                     - contract('jmbe,ma,ie->ibaj', l2, t1, t1))
        G['ijab'] = (tau
                     + 0.25 * contract('mnef,ijef,mnab->ijab', l2, tau, tau)
                     - 0.5 * Pij(contract('mnef,inef,mjab->ijab', l2, t2, tau))
                     - Pij(contract('me,mjab,ie->ijab', l1, tau, t1))
                     - 0.5 * Pab(contract('mnef,mnaf,ijeb->ijab', l2, t2, tau))
                     - Pab(contract('me,ijeb,ma->ijab', l1, tau, t1))
                     - 0.5 * Pijab(contract('miae,mnef,jnbf->ijab', X, l2, t2))
                     - Pijab(contract('miae,me,jb->ijab', X, l1, t1))
                     + 3.0 * Pijab(contract('me,ia,je,mb->ijab', l1, t1, t1, t1)))
        G['abij'] = contract('ijab->abij', l2)
        if self.ccwfn.model == 'CCSD(T)':
            G['ijab'] = G['ijab'] + self.ccwfn.Goovv
            G['ijka'] = G['ijka'] + self.ccwfn.Gooov
            G['ciab'] = G['ciab'] + self.ccwfn.Gvovv
            G['kaij'] = G['kaij'] + self.ccwfn.Govoo
            G['abci'] = G['abci'] + self.ccwfn.Gvvvo
        return G

    def _so_full_twopdm(self):
        """Assemble the full antisymmetric spin-orbital 2-PDM (``nmo^4``, physicist ``<pq|rs>``
        ordering) on the active ``o``/``v`` slices from the nine :meth:`build_so_twopdm` blocks,
        generating every block type by index antisymmetry.  Frozen-core rows/columns stay zero."""
        ccwfn = self.ccwfn
        o, v, nmo = ccwfn.o, ccwfn.v, ccwfn.nmo
        G = self.so_twopdm
        Gam = zeros((nmo, nmo, nmo, nmo), like=self.Doo)
        Gam[o, o, o, o] = G['ijkl']
        Gam[v, v, v, v] = G['abcd']
        Gam[o, o, v, v] = G['ijab']; Gam[v, v, o, o] = G['abij']
        Gam[o, o, o, v] = G['ijka']; Gam[o, o, v, o] = -G['ijka'].transpose(0, 1, 3, 2)
        Gam[o, v, o, o] = G['kaij']; Gam[v, o, o, o] = -G['kaij'].transpose(1, 0, 2, 3)
        Gam[v, v, v, o] = G['abci']; Gam[v, v, o, v] = -G['abci'].transpose(0, 1, 3, 2)
        Gam[v, o, v, v] = G['ciab']; Gam[o, v, v, v] = -G['ciab'].transpose(1, 0, 2, 3)
        ib = G['ibaj']
        Gam[o, v, v, o] = ib
        Gam[v, o, v, o] = -ib.transpose(1, 0, 2, 3)
        Gam[o, v, o, v] = -ib.transpose(0, 1, 3, 2)
        Gam[v, o, o, v] = ib.transpose(1, 0, 3, 2)
        return Gam

    # T1-transformed dipole integrals needed in CC3
    def build_Moo(self, no, nv, ints, t1):
        contract = self.contract
        Moo = clone(ints[:no,:no])
        Moo = Moo + contract('ma,ia->mi', ints[:no,-nv:], t1)

        return Moo

    def build_Mvv(self, no, nv, ints, t1):
        contract = self.contract
        Mvv = clone(ints[-nv:,-nv:])
        Mvv = Mvv - contract('ie,ia->ae', ints[:no,-nv:], t1)

        return Mvv

   
