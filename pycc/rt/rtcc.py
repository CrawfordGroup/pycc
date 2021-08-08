"""
rtcc.py: Real-time coupled object that provides data for an ODE propagator
"""

import psi4
import numpy as np
from opt_einsum import contract


class rtcc(object):
    """
    A Real-time CCSD object for ODE propagation.

    Attributes
    -----------
    ccwfn: PyCC ccwfn object
        the coupled cluster T amplitudes and supporting data structures
    cclambda : PyCC cclambda object
        the coupled cluster Lambda amplitudes and supporting data structures
    ccdensity : PyCC ccdensity object
        the coupled cluster one- and two-electron densities
    V: the time-dependent laser field
        must accept only the current time as an argument, e.g., as defined in lasers.py
    mu: list of NumPy arrays
        the dipole integrals for each Cartesian direction
    mu_tot: NumPy arrays
        1/sqrt(3) * sum of dipole integrals (for isotropic field)
    m: list of NumPy arrays
        the magnetic dipole integrals for each Cartesian direction (only if magnetic = True)

    Parameters
    ----------
    magnetic: bool
        optionally store magnetic dipole integrals (default = False)
    kick: bool or str
        optionally isolate 'x', 'y', or 'z' electric field kick (default = False)

    Methods
    -------
    f(): Returns a flattened NumPy array of cluster residuals
        The ODE defining function (the right-hand-side of a Runge-Kutta solver)
    collect_amps():
        Collect the cluster amplitudes into a single vector
    extract_amps():
        Separate a flattened array of cluster amplitudes into the t1, t2, l1, and l2 components
    dipole()
        Compute the electronic or magnetic dipole moment for a given time t
    energy()
        Compute the CC correlation energy for a given time t
    lagrangian()
        Compute the CC Lagrangian energy for a given time t
    """
    def __init__(self, ccwfn, cclambda, ccdensity, V, magnetic = False, kick = None):
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.ccdensity = ccdensity
        self.V = V

        # Prep the dipole integrals in MO basis
        mints = psi4.core.MintsHelper(ccwfn.ref.basisset())
        dipole_ints = mints.ao_dipole()
        C = np.asarray(self.ccwfn.C)  # May be localized MOs, so we take them from ccwfn
        self.mu = []
        for axis in range(3):
            self.mu.append(C.T @ np.asarray(dipole_ints[axis]) @ C)
        if kick:
            s_to_i = {"x":0, "y":1, "z":2}
            self.mu_tot = self.mu[s_to_i[kick.lower()]]
        else:
            self.mu_tot = sum(self.mu)/np.sqrt(3.0)  # isotropic field

        if magnetic:
            m_ints = mints.ao_angular_momentum()
            self.m = []
            for axis in range(3):
                m = (C.T @ (np.asarray(m_ints[axis])*-0.5) @ C)
                self.m.append(m*1.0j)

    def f(self, t, y):
        """
        Parameters
        ----------
        t : float
            Current time step in the external ODE solver
        y : NumPy array
            flattened array of cluster amplitudes

        Returns
        -------
        f(t, y): NumPy array
            flattened array of cluster residuals
        """
        # Extract amplitude tensors
        t1, t2, l1, l2 = self.extract_amps(y)

        # Add the field to the Hamiltonian
        F = self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)

        # Compute the current residuals
        rt1, rt2 = self.ccwfn.residuals(F, t1, t2)
        rt1 = rt1 * (-1.0j)
        rt2 = rt2 * (-1.0j)
        if self.ccwfn.local is not None:
            rt1, rt2 = self.ccwfn.Local.filter_res(rt1, rt2)

        rl1, rl2 = self.cclambda.residuals(F, t1, t2, l1, l2)
        rl1 = rl1 * (+1.0j)
        rl2 = rl2 * (+1.0j)
        if self.ccwfn.local is not None:
            rl1, rl2 = self.ccwfn.Local.filter_res(rl1, rl2)

        # Pack up the residuals
        y = self.collect_amps(rt1, rt2, rl1, rl2)

        return y

    def collect_amps(self, t1, t2, l1, l2):
        """
        Parameters
        ----------
        t1, t2, l2, l2 : NumPy arrays
            current cluster amplitudes or residuals

        Returns
        -------
        NumPy array
            amplitudes or residuals as a vector (flattened array)
        """
        return np.concatenate((t1, t2, l1, l2), axis=None)

    def extract_amps(self, y):
        """
        Parameters
        ----------
        y : NumPy array
            flattened array of cluster amplitudes or residuals

        Returns
        -------
        t1, t2, l2, l2 : NumPy arrays
            current cluster amplitudes or residuals
        """
        no = self.ccwfn.no
        nv = self.ccwfn.nv

        # Extract the amplitudes
        len1 = no*nv
        len2 = no*no*nv*nv
        t1 = np.reshape(y[:len1], (no, nv))
        t2 = np.reshape(y[len1:(len1+len2)], (no, no, nv, nv))
        l1 = np.reshape(y[(len1+len2):(len1+len2+len1)], (no, nv))
        l2 = np.reshape(y[(len1+len2+len1):], (no, no, nv, nv))

        return t1, t2, l1, l2

    def dipole(self, t1, t2, l1, l2, withref = True, magnetic = False):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes
        withref        : Bool (default = True)
            include reference contribution to the OPDM
        magnetic       : Bool (default = False)
            compute magnetic dipole rather than electric

        Returns
        -------
        x, y, z : complex128
            Cartesian components of the dipole moment
        """
        opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2, withref=withref)
        if magnetic:
            ints = self.m
        else:
            ints = self.mu
        x = ints[0].flatten().dot(opdm.flatten())
        y = ints[1].flatten().dot(opdm.flatten())
        z = ints[2].flatten().dot(opdm.flatten())
        return x, y, z

    def energy(self, t, t1, t2, l1, l2):
        """
        Parameters
        ----------
        t : float
            current time step in external ODE solver
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes

        Returns
        -------
        ecc : complex128
            CC correlation energy
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)
        ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
        L = self.ccwfn.H.L
        ecc = ecc + contract('ijab,ijab->', build_tau(t1, t2), L[o,o,v,v])
        return ecc

    def lagrangian(self, t, t1, t2, l1, l2):
        """
        Parameters
        ----------
        t : float
            current time step in external ODE solver
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes

        Returns
        -------
        ecc : complex128
            CC Lagrangian energy (including reference contribution, but excluding nuclear repulsion)
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        ERI = self.ccwfn.H.ERI
        opdm = self.ccdensity.compute_onepdm(t1, t2, l1, l2)
        Doooo = self.ccdensity.build_Doooo(t1, t2, l2)
        Dvvvv = self.ccdensity.build_Dvvvv(t1, t2, l2)
        Dooov = self.ccdensity.build_Dooov(t1, t2, l1, l2)
        Dvvvo = self.ccdensity.build_Dvvvo(t1, t2, l1, l2)
        Dovov = self.ccdensity.build_Dovov(t1, t2, l1, l2)
        Doovv = self.ccdensity.build_Doovv(t1, t2, l1, l2)

        F = self.ccwfn.H.F.copy() + self.mu_tot * self.V(t)

        eref = 2.0 * np.trace(F[o,o])
        eref -= np.trace(np.trace(self.ccwfn.H.L[o,o,o,o], axis1=1, axis2=3))

        eone = F.flatten().dot(opdm.flatten())

        oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], Doooo)
        vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], Dvvvv)
        ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], Dooov)
        vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], Dvvvo)
        ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], Dovov)
        oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], Doovv)
        etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy
        return eref + eone + etwo
