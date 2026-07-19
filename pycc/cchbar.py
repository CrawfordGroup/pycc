"""
cchbar.py: Builds the similarity-transformed Hamiltonian (one- and two-body terms only).
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
from pycc.utils import clone

if TYPE_CHECKING:
    from pycc.ccwfn import CCwfn


class cchbar(object):
    """
    An RHF-CC Similarity-Transformed Hamiltonian object.

    Attributes
    ----------
    Hov : NumPy array
        The occupied-virtual block of the one-body component HBAR.
    Hvv : NumPy array
        The virtual-virtual block of the one-body component HBAR.
    Hoo : NumPy array
        The occupied-occupied block of the one-body component HBAR.
    Hoooo : NumPy array
        The occ,occ,occ,occ block of the two-body component HBAR.
    Hvvvv : NumPy array
        The vir,vir,vir,vir block of the two-body component HBAR.
    Hvovv : NumPy array
        The vir,occ,vir,vir block of the two-body component HBAR.
    Hooov : NumPy array
        The occ,occ,occ,vir block of the two-body component HBAR.
    Hovvo : NumPy array
        The occ,vir,vir,occ block of the two-body component HBAR.
    Hovov : NumPy array
        The occ,vir,occ,vir block of the two-body component HBAR.
    Hvvvo : NumPy array
        The vir,vir,vir,occ block of the two-body component HBAR.
    Hovoo : NumPy array
        The occ,vir,occ,occ block of the two-body component HBAR.

    """
    def __init__(self, ccwfn: "CCwfn") -> None:
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            amplitudes instantiated to defaults or converged

        Returns
        -------
        None
        """

        time_init = time.time()
  
        self.ccwfn = ccwfn
        self.contract = self.ccwfn.contract
 
        F = ccwfn.H.F
        ERI = ccwfn.H.ERI
        L = ccwfn.H.L if ccwfn.orbital_basis == 'spatial' else None  # no L in spin orbitals
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        o = self.o = ccwfn.o
        v = self.v = ccwfn.v
        self.no = ccwfn.no
        self.nv = ccwfn.nv

        if ccwfn.orbital_basis == 'spinorbital':
            self._so_build(o, v, F, ERI, t1, t2)
            print("\nHBAR constructed in %.3f seconds." % (time.time() - time_init))
            return

        self.Hov = self.build_Hov(o, v, F, L, t1)
        self.Hvv = self.build_Hvv(o, v, F, L, t1, t2)
        self.Hoo = self.build_Hoo(o, v, F, L, t1, t2)
        self.Hoooo = self.build_Hoooo(o, v, ERI, t1, t2)
        self.Hvvvv = self.build_Hvvvv(o, v, ERI, t1, t2)
        self.Hvovv = self.build_Hvovv(o, v, ERI, t1)
        self.Hooov = self.build_Hooov(o, v, ERI, t1)
        self.Hovvo = self.build_Hovvo(o, v, ERI, L, t1, t2)
        self.Hovov = self.build_Hovov(o, v, ERI, t1, t2)
        self.Hvvvo = self.build_Hvvvo(o, v, ERI, L, self.Hov, self.Hvvvv, t1, t2)
        self.Hovoo = self.build_Hovoo(o, v, ERI, L, self.Hov, self.Hoooo, t1, t2)
    
        print("\nHBAR constructed in %.3f seconds." % (time.time() - time_init))

    """
    For GPU implementation:
    2-index tensors are stored on GPU
    4-index tensors are stored on CPU
    """

    # Spin-orbital HBAR (open-shell UHF/ROHF references): each _so_build_* sibling sits
    # immediately after its spatial build_* counterpart and works directly from the
    # antisymmetrized ERI = <pq||rs> (no spin-adapted L). There is no separate Hovov block
    # (the spin-orbital Lambda residuals fold it into Hovvo, via the inline Zovov intermediate
    # used by _so_build_Hvvvo / _so_build_Hovoo).

    def _so_build(self, o, v, F, ERI, t1, t2):
        """Build and cache all spin-orbital HBAR blocks (the SO sibling of the spatial
        ``build`` path), on the antisymmetrized ERI = <pq||rs>.  Selected by
        ``orbital_basis == 'spinorbital'``."""
        self.Hov = self._so_build_Hov(o, v, F, ERI, t1)
        self.Hvv = self._so_build_Hvv(o, v, F, ERI, self.Hov, t1, t2)
        self.Hoo = self._so_build_Hoo(o, v, F, ERI, self.Hov, t1, t2)
        self.Hoooo = self._so_build_Hoooo(o, v, ERI, t1, t2)
        self.Hvvvv = self._so_build_Hvvvv(o, v, ERI, t1, t2)
        self.Hvovv = self._so_build_Hvovv(o, v, ERI, t1)
        self.Hooov = self._so_build_Hooov(o, v, ERI, t1)
        self.Hovvo = self._so_build_Hovvo(o, v, ERI, t1, t2)
        Zovov = self._so_build_Zovov(o, v, ERI, t2)
        self.Hvvvo = self._so_build_Hvvvo(o, v, ERI, self.Hov, self.Hvvvv, Zovov, t1, t2)
        self.Hovoo = self._so_build_Hovoo(o, v, ERI, self.Hov, self.Hoooo, Zovov, t1, t2)

    def build_Hov(self, o, v, F, L, t1):
        r"""Build the occupied-virtual block H_me of the one-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv)
            Indexed [m, e]. For CCD this is just f_me (no singles term).

        Notes
        -----
        CCSD form (repeated indices summed)::

            H_me = f_me + t_nf L_mnef

        .. math::

            \begin{aligned}
            H_{me} = f_{me} + t^f_n L_{mnef}
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hov = clone(F[o,v])
        else:
            Hov = clone(F[o,v])
            Hov = Hov + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Hov

    def _so_build_Hov(self, o, v, F, ERI, t1):
        r"""Spin-orbital H_me one-body HBAR block (the SO sibling of :meth:`build_Hov`,
        on antisymmetrized <pq||rs>).

        Notes
        -----
        Repeated indices summed::

            H_me = f_me + t_nf <mn||ef>

        .. math::

            \begin{aligned}
            H_{me} = f_{me} + t^f_n \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        Hov = clone(F[o,v])
        Hov = Hov + contract('nf,mnef->me', t1, ERI[o,o,v,v])
        return Hov

    def build_Hvv(self, o, v, F, L, t1, t2):
        r"""Build the virtual-virtual block H_ae of the one-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv)
            Indexed [a, e].

        Notes
        -----
        CCSD form (repeated indices summed; CCD keeps only f_ae and the final
        t2 term)::

            H_ae = f_ae - f_me t_ma + t_mf L_amef
                        - (t2_mnfa + t_mf t_na) L_mnfe

        .. math::

            \begin{aligned}
            H_{ae} = f_{ae} &- f_{me} t^a_m + t^f_m L_{amef} \\
            &- \left(t^{fa}_{mn} + t^f_m t^a_n\right) L_{mnfe}
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hvv = clone(F[v,v])
            Hvv = Hvv - contract('mnfa,mnfe->ae', t2, L[o,o,v,v])

        else:
            Hvv = clone(F[v,v])
            Hvv = Hvv - contract('me,ma->ae', F[o,v], t1)
            Hvv = Hvv + contract('mf,amef->ae', t1, L[v,o,v,v])
            Hvv = Hvv - contract('mnfa,mnfe->ae', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])
        return Hvv

    def _so_build_Hvv(self, o, v, F, ERI, Hov, t1, t2):
        r"""Spin-orbital H_ae one-body HBAR block (the SO sibling of :meth:`build_Hvv`,
        on antisymmetrized <pq||rs>; taut is :meth:`~pycc.ccwfn.CCwfn._so_build_tau` with
        fact2=1/2, and H_me is the already-built :meth:`_so_build_Hov`).

        Notes
        -----
        Repeated indices summed::

            H_ae = f_ae - 1/2 f_me t_ma - 1/2 H_me t_ma + t_mf <am||ef>
                        - 1/2 taut_mnaf <mn||ef>

        .. math::

            \begin{aligned}
            H_{ae} = f_{ae} &- \tfrac{1}{2} f_{me} t^a_m - \tfrac{1}{2} H_{me} t^a_m + t^f_m \langle am||ef \rangle \\
            &- \tfrac{1}{2} \tilde\tau^{af}_{mn} \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        taut = self.ccwfn._so_build_tau(t1, t2, 1.0, 0.5)
        Hvv = clone(F[v,v])
        Hvv = Hvv - 0.5 * contract('me,ma->ae', F[o,v], t1)
        Hvv = Hvv - 0.5 * contract('me,ma->ae', Hov, t1)
        Hvv = Hvv + contract('mf,amef->ae', t1, ERI[v,o,v,v])
        Hvv = Hvv - 0.5 * contract('mnaf,mnef->ae', taut, ERI[o,o,v,v])
        return Hvv

    def build_Hoo(self, o, v, F, L, t1, t2):
        r"""Build the occupied-occupied block H_mi of the one-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no)
            Indexed [m, i].

        Notes
        -----
        CCSD form (repeated indices summed; CCD keeps only f_mi and the final
        t2 term)::

            H_mi = f_mi + t_ie f_me + t_ne L_mnie
                        + (t2_inef + t_ie t_nf) L_mnef

        .. math::

            \begin{aligned}
            H_{mi} = f_{mi} &+ t^e_i f_{me} + t^e_n L_{mnie} \\
            &+ \left(t^{ef}_{in} + t^e_i t^f_n\right) L_{mnef}
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hoo = clone(F[o,o])
            Hoo = Hoo + contract('inef,mnef->mi', t2, L[o,o,v,v])

        else:
            Hoo = clone(F[o,o])
            Hoo = Hoo + contract('ie,me->mi', t1, F[o,v])
            Hoo = Hoo + contract('ne,mnie->mi', t1, L[o,o,o,v])
            Hoo = Hoo + contract('inef,mnef->mi', self.ccwfn.build_tau(t1, t2), L[o,o,v,v])
        return Hoo

    def _so_build_Hoo(self, o, v, F, ERI, Hov, t1, t2):
        r"""Spin-orbital H_mi one-body HBAR block (the SO sibling of :meth:`build_Hoo`,
        on antisymmetrized <pq||rs>; taut is :meth:`~pycc.ccwfn.CCwfn._so_build_tau` with
        fact2=1/2, and H_me is the already-built :meth:`_so_build_Hov`).

        Notes
        -----
        Repeated indices summed::

            H_mi = f_mi + 1/2 f_me t_ie + 1/2 H_me t_ie + t_ne <mn||ie>
                        + 1/2 taut_inef <mn||ef>

        .. math::

            \begin{aligned}
            H_{mi} = f_{mi} &+ \tfrac{1}{2} f_{me} t^e_i + \tfrac{1}{2} H_{me} t^e_i + t^e_n \langle mn||ie \rangle \\
            &+ \tfrac{1}{2} \tilde\tau^{ef}_{in} \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        taut = self.ccwfn._so_build_tau(t1, t2, 1.0, 0.5)
        Hoo = clone(F[o,o])
        Hoo = Hoo + 0.5 * contract('ie,me->mi', t1, F[o,v])
        Hoo = Hoo + 0.5 * contract('ie,me->mi', t1, Hov)
        Hoo = Hoo + contract('ne,mnie->mi', t1, ERI[o,o,o,v])
        Hoo = Hoo + 0.5 * contract('inef,mnef->mi', taut, ERI[o,o,v,v])
        return Hoo

    def build_Hoooo(self, o, v, ERI, t1, t2):
        r"""Build the occ-occ-occ-occ block H_mnij of the two-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, no, no)
            Indexed [m, n, i, j]. ERI is in Dirac order <pq|rs>.

        Notes
        -----
        CCSD form (repeated indices summed)::

            H_mnij = <mn|ij> + t_je <mn|ie> + t_ie <nm|je>
                            + (t2_ijef + t_ie t_jf) <mn|ef>

        .. math::

            \begin{aligned}
            H_{mnij} = \langle mn|ij \rangle &+ t^e_j \langle mn|ie \rangle + t^e_i \langle nm|je \rangle \\
            &+ \left(t^{ef}_{ij} + t^e_i t^f_j\right) \langle mn|ef \rangle
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hoooo = clone(ERI[o,o,o,o], device=self.ccwfn.device1)
            Hoooo = Hoooo + contract('ijef,mnef->mnij', t2, ERI[o,o,v,v])

        else:
            Hoooo = clone(ERI[o,o,o,o], device=self.ccwfn.device1)
            tmp = contract('je,mnie->mnij', t1, ERI[o,o,o,v])
            Hoooo = Hoooo + (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
            if self.ccwfn.model == 'CC2':
                Hoooo = Hoooo + contract('jf,mnif->mnij', t1, contract('ie,mnef->mnif', t1, ERI[o,o,v,v]))
            else:
                Hoooo = Hoooo + contract('ijef,mnef->mnij', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v]) 

        return Hoooo

    def _so_build_Hoooo(self, o, v, ERI, t1, t2):
        r"""Spin-orbital H_mnij two-body HBAR block (the SO sibling of :meth:`build_Hoooo`,
        on antisymmetrized <pq||rs>; tau is :meth:`~pycc.ccwfn.CCwfn._so_build_tau`).

        Notes
        -----
        Repeated indices summed::

            H_mnij = <mn||ij> + t_je <mn||ie> - t_ie <mn||je>
                             + 1/2 tau_ijef <mn||ef>

        .. math::

            \begin{aligned}
            H_{mnij} = \langle mn||ij \rangle &+ t^e_j \langle mn||ie \rangle - t^e_i \langle mn||je \rangle \\
            &+ \tfrac{1}{2} \tau^{ef}_{ij} \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        tau = self.ccwfn._so_build_tau(t1, t2)
        Hoooo = clone(ERI[o,o,o,o])
        Hoooo = Hoooo + (contract('je,mnie->mnij', t1, ERI[o,o,o,v])
                         - contract('ie,mnje->mnij', t1, ERI[o,o,o,v]))
        Hoooo = Hoooo + 0.5 * contract('ijef,mnef->mnij', tau, ERI[o,o,v,v])
        return Hoooo

    def build_Hvvvv(self, o, v, ERI, t1, t2):
        r"""Build the vir-vir-vir-vir block H_abef of the two-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv, nv, nv)
            Indexed [a, b, e, f].

        Notes
        -----
        CCSD form (repeated indices summed)::

            H_abef = <ab|ef> - t_mb <am|ef> - t_ma <bm|fe>
                            + (t2_mnab + t_ma t_nb) <mn|ef>

        .. math::

            \begin{aligned}
            H_{abef} = \langle ab|ef \rangle &- t^b_m \langle am|ef \rangle - t^a_m \langle bm|fe \rangle \\
            &+ \left(t^{ab}_{mn} + t^a_m t^b_n\right) \langle mn|ef \rangle
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hvvvv = clone(ERI[v,v,v,v], device=self.ccwfn.device1)
            Hvvvv = Hvvvv + contract('mnab,mnef->abef', t2, ERI[o,o,v,v])

        else:
            Hvvvv = clone(ERI[v,v,v,v], device=self.ccwfn.device1)
            tmp = contract('mb,amef->abef', t1, ERI[v,o,v,v])
            Hvvvv = Hvvvv - (tmp + tmp.swapaxes(0,1).swapaxes(2,3))
            if self.ccwfn.model == 'CC2':
                Hvvvv = Hvvvv + contract('nb,anef->abef', t1, contract('ma,mnef->anef', t1, ERI[o,o,v,v]))
            else:
                Hvvvv = Hvvvv + contract('mnab,mnef->abef', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])

        return Hvvvv

    def _so_build_Hvvvv(self, o, v, ERI, t1, t2):
        r"""Spin-orbital H_abef two-body HBAR block (the SO sibling of :meth:`build_Hvvvv`,
        on antisymmetrized <pq||rs>; tau is :meth:`~pycc.ccwfn.CCwfn._so_build_tau`).

        Notes
        -----
        Repeated indices summed::

            H_abef = <ab||ef> - t_mb <am||ef> + t_ma <bm||ef>
                             + 1/2 tau_mnab <mn||ef>

        .. math::

            \begin{aligned}
            H_{abef} = \langle ab||ef \rangle &- t^b_m \langle am||ef \rangle + t^a_m \langle bm||ef \rangle \\
            &+ \tfrac{1}{2} \tau^{ab}_{mn} \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        tau = self.ccwfn._so_build_tau(t1, t2)
        Hvvvv = clone(ERI[v,v,v,v])
        Hvvvv = Hvvvv - (contract('mb,amef->abef', t1, ERI[v,o,v,v])
                         - contract('ma,bmef->abef', t1, ERI[v,o,v,v]))
        Hvvvv = Hvvvv + 0.5 * contract('mnab,mnef->abef', tau, ERI[o,o,v,v])
        return Hvvvv

    def build_Hvovv(self, o, v, ERI, t1):
        r"""Build the vir-occ-vir-vir block H_amef of the two-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, no, nv, nv)
            Indexed [a, m, e, f]. For CCD this is just <am|ef>.

        Notes
        -----
        CCSD form (repeated indices summed)::

            H_amef = <am|ef> - t_na <nm|ef>

        .. math::

            \begin{aligned}
            H_{amef} = \langle am|ef \rangle - t^a_n \langle nm|ef \rangle
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hvovv = clone(ERI[v,o,v,v], device=self.ccwfn.device1)
        else:
            Hvovv = clone(ERI[v,o,v,v], device=self.ccwfn.device1)
            Hvovv = Hvovv - contract('na,nmef->amef', t1, ERI[o,o,v,v])

        return Hvovv

    def _so_build_Hvovv(self, o, v, ERI, t1):
        r"""Spin-orbital H_amef two-body HBAR block (the SO sibling of :meth:`build_Hvovv`,
        on antisymmetrized <pq||rs>).

        Notes
        -----
        Repeated indices summed::

            H_amef = <am||ef> - t_na <nm||ef>

        .. math::

            \begin{aligned}
            H_{amef} = \langle am||ef \rangle - t^a_n \langle nm||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        Hvovv = clone(ERI[v,o,v,v])
        Hvovv = Hvovv - contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return Hvovv

    def build_Hooov(self, o, v, ERI, t1):
        r"""Build the occ-occ-occ-vir block H_mnie of the two-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, no, no, nv)
            Indexed [m, n, i, e]. For CCD this is just <mn|ie>.

        Notes
        -----
        CCSD form (repeated indices summed)::

            H_mnie = <mn|ie> + t_if <nm|ef>

        .. math::

            \begin{aligned}
            H_{mnie} = \langle mn|ie \rangle + t^f_i \langle nm|ef \rangle
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hooov = clone(ERI[o,o,o,v], device=self.ccwfn.device1)
        else:
            Hooov = clone(ERI[o,o,o,v], device=self.ccwfn.device1)
            Hooov = Hooov + contract('if,nmef->mnie', t1, ERI[o,o,v,v])

        return Hooov

    def _so_build_Hooov(self, o, v, ERI, t1):
        r"""Spin-orbital H_mnie two-body HBAR block (the SO sibling of :meth:`build_Hooov`,
        on antisymmetrized <pq||rs>).

        Notes
        -----
        Repeated indices summed::

            H_mnie = <mn||ie> + t_if <mn||fe>

        .. math::

            \begin{aligned}
            H_{mnie} = \langle mn||ie \rangle + t^f_i \langle mn||fe \rangle
            \end{aligned}
        """
        contract = self.contract
        Hooov = clone(ERI[o,o,o,v])
        Hooov = Hooov + contract('if,mnfe->mnie', t1, ERI[o,o,v,v])
        return Hooov

    def build_Hovvo(self, o, v, ERI, L, t1, t2):
        r"""Build the occ-vir-vir-occ block H_mbej of the two-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, nv, no)
            Indexed [m, b, e, j].

        Notes
        -----
        CCSD form (repeated indices summed; CC2 drops the t2 terms)::

            H_mbej = <mb|ej> + t_jf <mb|ef> - t_nb <mn|ej>
                            - (t2_jnfb + t_jf t_nb) <mn|ef>
                            + t2_njfb L_mnef

        .. math::

            \begin{aligned}
            H_{mbej} = \langle mb|ej \rangle &+ t^f_j \langle mb|ef \rangle - t^b_n \langle mn|ej \rangle \\
            &- \left(t^{fb}_{jn} + t^f_j t^b_n\right) \langle mn|ef \rangle \\
            &+ t^{fb}_{nj} L_{mnef}
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hovvo = clone(ERI[o,v,v,o], device=self.ccwfn.device1)
            # clean th== up
            Hovvo = Hovvo - contract('jnfb,mnef->mbej', t2, ERI[o,o,v,v])
            Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])

        else:
            Hovvo = clone(ERI[o,v,v,o], device=self.ccwfn.device1)
            Hovvo = Hovvo + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
            Hovvo = Hovvo - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
            if self.ccwfn.model != 'CC2':
                Hovvo = Hovvo - contract('jnfb,mnef->mbej', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
                Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Hovvo

    def _so_build_Hovvo(self, o, v, ERI, t1, t2):
        r"""Spin-orbital H_mbej two-body HBAR block (the SO sibling of :meth:`build_Hovvo`,
        on antisymmetrized <pq||rs>).  Here tau = t2 + t1 t1 (the un-antisymmetrized
        product, as assembled inline).

        Notes
        -----
        Repeated indices summed::

            H_mbej = <mb||ej> + t_jf <mb||ef> - t_nb <mn||ej>
                             - (t2_jnfb + t_jf t_nb) <mn||ef>

        .. math::

            \begin{aligned}
            H_{mbej} = \langle mb||ej \rangle &+ t^f_j \langle mb||ef \rangle - t^b_n \langle mn||ej \rangle \\
            &- \left(t^{fb}_{jn} + t^f_j t^b_n\right) \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        Hovvo = clone(ERI[o,v,v,o])
        Hovvo = Hovvo + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
        Hovvo = Hovvo - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
        tau = t2 + contract('ia,jb->ijab', t1, t1)
        Hovvo = Hovvo - contract('jnfb,mnef->mbej', tau, ERI[o,o,v,v])
        return Hovvo

    def build_Hovov(self, o, v, ERI, t1, t2):
        r"""Build the occ-vir-occ-vir block H_mbje of the two-body HBAR.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, no, nv)
            Indexed [m, b, j, e].

        Notes
        -----
        CCSD form (repeated indices summed; CC2 drops the t2 term)::

            H_mbje = <mb|je> + t_jf <bm|ef> - t_nb <mn|je>
                            - (t2_jnfb + t_jf t_nb) <nm|ef>

        .. math::

            \begin{aligned}
            H_{mbje} = \langle mb|je \rangle &+ t^f_j \langle bm|ef \rangle - t^b_n \langle mn|je \rangle \\
            &- \left(t^{fb}_{jn} + t^f_j t^b_n\right) \langle nm|ef \rangle
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hovov = clone(ERI[o,v,o,v], device=self.ccwfn.device1)
            Hovov = Hovov - contract('jnfb,nmef->mbje', t2, ERI[o,o,v,v])
        else:
            Hovov = clone(ERI[o,v,o,v], device=self.ccwfn.device1)
            Hovov = Hovov + contract('jf,bmef->mbje', t1, ERI[v,o,v,v])
            Hovov = Hovov - contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
            if self.ccwfn.model != 'CC2':
                Hovov = Hovov - contract('jnfb,nmef->mbje', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hovov

    def build_Hvvvo(self, o, v, ERI, L, Hov, Hvvvv, t1, t2):
        r"""Build the vir-vir-vir-occ block H_abei of the two-body HBAR.

        Reuses the already-built Hov and Hvvvv blocks.

        Returns
        -------
        ndarray or torch.Tensor, shape (nv, nv, nv, no)
            Indexed [a, b, e, i].

        Notes
        -----
        CCSD form (repeated indices summed; the parenthesized groups are the
        nested intermediates assembled in the code)::

            H_abei = <ab|ei> - H_me t2_miab + t_if H_abef
                            + (t2_mnab + t_ma t_nb) <mn|ei>
                            - t2_imfa <bm|fe> - t2_imfb <am|ef>
                            + t2_mifb L_amef
                            - t_mb ( <am|ei> - t2_infa <mn|fe> )
                            - t_ma ( <bm|ie> - t2_infb <mn|ef>
                                              + t2_nifb L_mnef )

        .. math::

            \begin{aligned}
            H_{abei} &= \langle ab|ei \rangle - H_{me} t^{ab}_{mi} + t^f_i H_{abef} \\
            &\quad + \left(t^{ab}_{mn} + t^a_m t^b_n\right) \langle mn|ei \rangle \\
            &\quad - t^{fa}_{im} \langle bm|fe \rangle - t^{fb}_{im} \langle am|ef \rangle + t^{fb}_{mi} L_{amef} \\
            &\quad - t^b_m \left(\langle am|ei \rangle - t^{fa}_{in} \langle mn|fe \rangle\right) \\
            &\quad - t^a_m \left(\langle bm|ie \rangle - t^{fb}_{in} \langle mn|ef \rangle + t^{fb}_{ni} L_{mnef}\right)
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hvvvo = clone(ERI[v,v,v,o], device=self.ccwfn.device1)
            Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
            Hvvvo = Hvvvo + contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
            Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])

        elif self.ccwfn.model == 'CC2':
            Hvvvo = clone(ERI[v,v,v,o], device=self.ccwfn.device1)
            Hvvvo = Hvvvo - contract('me,miab->abei', self.ccwfn.H.F[o,v], t2)
            Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv)
            Hvvvo = Hvvvo + contract('nb,anei->abei', t1, contract('ma,mnei->anei', t1, ERI[o,o,v,o]))
            Hvvvo = Hvvvo - contract('mb,amei->abei', t1, ERI[v,o,v,o])
            Hvvvo = Hvvvo - contract('ma,bmie->abei', t1, ERI[v,o,o,v])
        else:
            Hvvvo = clone(ERI[v,v,v,o], device=self.ccwfn.device1)
            Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
            Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv)
            Hvvvo = Hvvvo + contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
            Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
            Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])   
            tmp = clone(ERI[v,o,v,o], device=self.ccwfn.device1)
            tmp = tmp - contract('infa,mnfe->amei', t2, ERI[o,o,v,v])
            Hvvvo = Hvvvo - contract('mb,amei->abei', t1, tmp)
            tmp = clone(ERI[v,o,o,v], device=self.ccwfn.device1)
            tmp = tmp - contract('infb,mnef->bmie', t2, ERI[o,o,v,v])
            tmp = tmp + contract('nifb,mnef->bmie', t2, L[o,o,v,v])
            Hvvvo = Hvvvo - contract('ma,bmie->abei', t1, tmp)
            if HAS_TORCH and isinstance(tmp, torch.Tensor):
                del tmp
        return Hvvvo

    def _so_build_Zovov(self, o, v, ERI, t2):
        r"""Spin-orbital Z_mbie auxiliary intermediate (a T2-dressed <mb||ie> reused by
        :meth:`_so_build_Hvvvo` and :meth:`_so_build_Hovoo`; the SO analogue of the
        inline ``tmp`` intermediates in the spatial :meth:`build_Hvvvo`/:meth:`build_Hovoo`).

        Notes
        -----
        Repeated indices summed::

            Z_mbie = <mb||ie> + t2_nibf <mn||ef>

        .. math::

            \begin{aligned}
            Z_{mbie} = \langle mb||ie \rangle + t^{bf}_{ni} \langle mn||ef \rangle
            \end{aligned}
        """
        contract = self.contract
        return clone(ERI[o,v,o,v]) + contract('nibf,mnef->mbie', t2, ERI[o,o,v,v])

    def _so_build_Hvvvo(self, o, v, ERI, Hov, Hvvvv, Zovov, t1, t2):
        r"""Spin-orbital H_abei two-body HBAR block (the SO sibling of :meth:`build_Hvvvo`,
        on antisymmetrized <pq||rs>).  Reuses the one-body H_me (:meth:`_so_build_Hov`),
        H_abef (:meth:`_so_build_Hvvvv`), the Z_mbie auxiliary (:meth:`_so_build_Zovov`),
        and tau = :meth:`~pycc.ccwfn.CCwfn._so_build_tau`.  The a<->b antisymmetry is
        carried inline by the explicit +/- term pairs.

        Notes
        -----
        Repeated indices summed::

            H_abei = <ab||ei> - H_me t2_miab + t_if H_abef + 1/2 tau_mnab <mn||ei>
                            - t2_miaf <mb||ef> + t2_mibf <ma||ef>
                            + t_ma Z_mbie - t_mb Z_maie

        .. math::

            \begin{aligned}
            H_{abei} &= \langle ab||ei \rangle - H_{me} t^{ab}_{mi} + t^f_i H_{abef} + \tfrac{1}{2} \tau^{ab}_{mn} \langle mn||ei \rangle \\
            &\quad - t^{af}_{mi} \langle mb||ef \rangle + t^{bf}_{mi} \langle ma||ef \rangle \\
            &\quad + t^a_m Z_{mbie} - t^b_m Z_{maie}
            \end{aligned}
        """
        contract = self.contract
        tau = self.ccwfn._so_build_tau(t1, t2)
        Hvvvo = clone(ERI[v,v,v,o])
        Hvvvo = Hvvvo - contract('me,miab->abei', Hov, t2)
        Hvvvo = Hvvvo + contract('if,abef->abei', t1, Hvvvv)
        Hvvvo = Hvvvo + 0.5 * contract('mnab,mnei->abei', tau, ERI[o,o,v,o])
        Hvvvo = Hvvvo - (contract('miaf,mbef->abei', t2, ERI[o,v,v,v])
                         - contract('mibf,maef->abei', t2, ERI[o,v,v,v]))
        Hvvvo = Hvvvo + (contract('ma,mbie->abei', t1, Zovov)
                         - contract('mb,maie->abei', t1, Zovov))
        return Hvvvo

    def build_Hovoo(self, o, v, ERI, L, Hov, Hoooo, t1, t2):
        r"""Build the occ-vir-occ-occ block H_mbij of the two-body HBAR.

        Reuses the already-built Hov and Hoooo blocks.

        Returns
        -------
        ndarray or torch.Tensor, shape (no, nv, no, no)
            Indexed [m, b, i, j].

        Notes
        -----
        CCSD form (repeated indices summed; the parenthesized groups are the
        nested intermediates assembled in the code)::

            H_mbij = <mb|ij> + H_me t2_ijeb - t_nb H_mnij
                            + (t2_ijef + t_ie t_jf) <mb|ef>
                            - t2_ineb <nm|je> - t2_jneb <mn|ie>
                            + t2_njeb L_mnie
                            + t_je ( <mb|ie> - t2_infb <mn|fe> )
                            + t_ie ( <bm|je> - t2_jnfb <mn|ef>
                                              + t2_njfb L_mnef )

        .. math::

            \begin{aligned}
            H_{mbij} &= \langle mb|ij \rangle + H_{me} t^{eb}_{ij} - t^b_n H_{mnij} \\
            &\quad + \left(t^{ef}_{ij} + t^e_i t^f_j\right) \langle mb|ef \rangle \\
            &\quad - t^{eb}_{in} \langle nm|je \rangle - t^{eb}_{jn} \langle mn|ie \rangle + t^{eb}_{nj} L_{mnie} \\
            &\quad + t^e_j \left(\langle mb|ie \rangle - t^{fb}_{in} \langle mn|fe \rangle\right) \\
            &\quad + t^e_i \left(\langle bm|je \rangle - t^{fb}_{jn} \langle mn|ef \rangle + t^{fb}_{nj} L_{mnef}\right)
            \end{aligned}
        """
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Hovoo = clone(ERI[o,v,o,o], device=self.ccwfn.device1)
            Hovoo = Hovoo + contract('me,ijeb->mbij', Hov, t2)
            Hovoo = Hovoo + contract('ijef,mbef->mbij', t2, ERI[o,v,v,v])
            Hovoo = Hovoo - contract('ineb,nmje->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo - contract('jneb,mnie->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo + contract('njeb,mnie->mbij', t2, L[o,o,o,v])

        elif self.ccwfn.model == 'CC2':
            Hovoo = clone(ERI[o,v,o,o], device=self.ccwfn.device1)
            Hovoo = Hovoo + contract('me,ijeb->mbij', self.ccwfn.H.F[o,v], t2)
            Hovoo = Hovoo - contract('nb,mnij->mbij', t1, Hoooo)
            Hovoo = Hovoo + contract('jf,mbif->mbij', t1, contract('ie,mbef->mbif', t1, ERI[o,v,v,v]))
            Hovoo = Hovoo + contract('je,mbie->mbij', t1, ERI[o,v,o,v])
            Hovoo = Hovoo + contract('ie,bmje->mbij', t1, ERI[v,o,o,v])     
  
        else:
            Hovoo = clone(ERI[o,v,o,o], device=self.ccwfn.device1)
            Hovoo = Hovoo + contract('me,ijeb->mbij', Hov, t2)
            Hovoo = Hovoo - contract('nb,mnij->mbij', t1, Hoooo)
            Hovoo = Hovoo + contract('ijef,mbef->mbij', self.ccwfn.build_tau(t1, t2), ERI[o,v,v,v])
            Hovoo = Hovoo - contract('ineb,nmje->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo - contract('jneb,mnie->mbij', t2, ERI[o,o,o,v])
            Hovoo = Hovoo + contract('njeb,mnie->mbij', t2, L[o,o,o,v])
            tmp = clone(ERI[o,v,o,v], device=self.ccwfn.device1)
            tmp = tmp - contract('infb,mnfe->mbie', t2, ERI[o,o,v,v])
            Hovoo = Hovoo + contract('je,mbie->mbij', t1, tmp)
            tmp = clone(ERI[v,o,o,v], device=self.ccwfn.device1)
            tmp = tmp - contract('jnfb,mnef->bmje', t2, ERI[o,o,v,v])
            tmp = tmp + contract('njfb,mnef->bmje', t2, L[o,o,v,v])
            Hovoo = Hovoo + contract('ie,bmje->mbij', t1, tmp)
            
            if HAS_TORCH and isinstance(tmp, torch.Tensor):
                del tmp
        return Hovoo

    def _so_build_Hovoo(self, o, v, ERI, Hov, Hoooo, Zovov, t1, t2):
        r"""Spin-orbital H_mbij two-body HBAR block (the SO sibling of :meth:`build_Hovoo`,
        on antisymmetrized <pq||rs>).  Reuses the one-body H_me (:meth:`_so_build_Hov`),
        H_mnij (:meth:`_so_build_Hoooo`), the Z_mbie auxiliary (:meth:`_so_build_Zovov`),
        and tau = :meth:`~pycc.ccwfn.CCwfn._so_build_tau`.  The i<->j antisymmetry is
        carried inline by the explicit +/- term pairs.

        Notes
        -----
        Repeated indices summed::

            H_mbij = <mb||ij> - H_me t2_ijbe - t_nb H_mnij + 1/2 tau_ijef <mb||ef>
                            + t2_jnbe <mn||ie> - t2_inbe <mn||je>
                            - t_ie Z_mbje + t_je Z_mbie

        .. math::

            \begin{aligned}
            H_{mbij} &= \langle mb||ij \rangle - H_{me} t^{be}_{ij} - t^b_n H_{mnij} + \tfrac{1}{2} \tau^{ef}_{ij} \langle mb||ef \rangle \\
            &\quad + t^{be}_{jn} \langle mn||ie \rangle - t^{be}_{in} \langle mn||je \rangle \\
            &\quad - t^e_i Z_{mbje} + t^e_j Z_{mbie}
            \end{aligned}
        """
        contract = self.contract
        tau = self.ccwfn._so_build_tau(t1, t2)
        Hovoo = clone(ERI[o,v,o,o])
        Hovoo = Hovoo - contract('me,ijbe->mbij', Hov, t2)
        Hovoo = Hovoo - contract('nb,mnij->mbij', t1, Hoooo)
        Hovoo = Hovoo + 0.5 * contract('ijef,mbef->mbij', tau, ERI[o,v,v,v])
        Hovoo = Hovoo + (contract('jnbe,mnie->mbij', t2, ERI[o,o,o,v])
                         - contract('inbe,mnje->mbij', t2, ERI[o,o,o,v]))
        Hovoo = Hovoo - (contract('ie,mbje->mbij', t1, Zovov)
                         - contract('je,mbie->mbij', t1, Zovov))
        return Hovoo
