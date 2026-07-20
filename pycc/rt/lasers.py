"""
lasers.py: time-dependent electric-field pulse shapes ``V(t)`` for the RT-CC propagator.

Each class is a callable ``laser(t) -> V(t)`` (scaled by the field strength ``F_str``) that
supplies the external field to :meth:`pycc.rt.rtcc.rtcc.f`, where it enters the Hamiltonian as
``H'(t) = -mu . V(t)``.  The Gaussian and sine-square pulses are the work of Håkon E. Kristiansen
(U. Oslo); the delta-kick and ramped-continuous-wave (lrcw/qrcw) pulses were added by Zhe Wang.
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import numpy as np


class gaussian_laser:
    r"""Gaussian-envelope carrier pulse::

        V(t) = F_str exp(-(t-t0)^2 / (2 sigma^2)) cos(omega (t-t0))

    .. math::

        V(t) = F_\mathrm{str}\, e^{-(t-t_0)^2 / (2\sigma^2)}\, \cos(\omega (t - t_0))

    Parameters
    ----------
    F_str : float
        field strength (peak amplitude)
    omega : float
        carrier frequency
    sigma : float
        Gaussian envelope width
    center : float
        pulse center time ``t0`` (default 0)
    """
    def __init__(self, F_str, omega, sigma, center=0.):
        self.F_str = F_str
        self.omega = omega
        self.sigma2 = sigma**2
        self.t0 = center

    def _envelope(self, t):
        dt = t - self.t0
        return np.exp(-dt**2/(2*self.sigma2))

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-dt**2/(2*self.sigma2))
            * np.cos(self.omega * dt)
        )
        return self.F_str*pulse


class sine_square_laser:
    r"""Sine-squared envelope pulse, active only on ``[0, tprime]``::

        V(t) = F_str sin^2(pi t / tprime) Theta(t) Theta(tprime - t) cos(omega t + phase)

    .. math::

        V(t) = F_\mathrm{str}\, \sin^2\!\Big(\frac{\pi t}{T'}\Big)\, \Theta(t)\,\Theta(T' - t)\, \cos(\omega t + \phi)

    Parameters
    ----------
    F_str, omega : float
        field strength and carrier frequency
    tprime : float
        pulse duration ``T'`` (the envelope vanishes outside ``[0, T']``)
    phase : float
        carrier phase ``phi`` (default 0)
    """
    def __init__(self, F_str, omega, tprime, phase=0):
        self.F_str = F_str
        self.omega = omega
        self.tprime = tprime
        self.phase = phase

    def __call__(self, t):
        pulse = (
            (np.sin(np.pi * t / self.tprime) ** 2)
            * np.heaviside(t, 1.0)
            * np.heaviside(self.tprime - t, 1.0)
            * np.cos(self.omega * t + self.phase)
            * self.F_str
        )
        return pulse

class delta_pulse_laser:
    r"""Delta-kick pulse: a constant amplitude within ``tol`` of ``center``, zero elsewhere -- a
    numerical approximation to a delta-function kick::

        V(t) = F_str  if |t - center| <= tol  else  0

    .. math::

        V(t) = \begin{cases} F_\mathrm{str} & |t - t_c| \le \mathrm{tol} \\ 0 & \text{otherwise} \end{cases}

    Parameters
    ----------
    F_str : float
        kick strength
    center : float
        kick time ``t_c`` (default 0)
    tol : float
        half-width of the kick window (default 1e-7)
    """
    def __init__(self, F_str, center=0.0, tol=1e-7):
        self.F_str = F_str
        self.center = center
        self.tol = tol
    def __call__(self, t):
        if abs(t - self.center) <= self.tol:
            pulse = self.F_str * 1.0
        else:
            pulse = 0
        return pulse

# ramped continuous wave (RCW)
# set nr=0 for a regular cosine wave
class lrcw_laser:
    r"""Linearly-ramped continuous wave: the carrier amplitude ramps linearly to full strength over
    ``nr`` cycles (``tc = 2 pi nr / omega``), then stays constant (set ``nr=0`` for a plain cosine)::

        V(t) = (t/tc) F_str cos(omega t)   for t <= tc
             =         F_str cos(omega t)   for t >  tc

    .. math::

        V(t) = \begin{cases} (t/t_c)\, F_\mathrm{str}\cos(\omega t) & t \le t_c \\
            F_\mathrm{str}\cos(\omega t) & t > t_c \end{cases}, \qquad t_c = \frac{2\pi\, n_r}{\omega}

    Parameters
    ----------
    F_str, omega : float
        field strength and carrier frequency
    nr : int
        number of ramp cycles (0 = no ramp, plain cosine)
    """
    def __init__(self, F_str, omega, nr):
        self.F_str = F_str
        self.omega = omega
        self.nr = nr
    def __call__(self, t):
        tc = 2 * np.pi / self.omega * self.nr
        if t <= tc:
            pulse = t / tc * self.F_str * np.cos(self.omega * t)
        else:
            pulse = self.F_str * np.cos(self.omega * t)
        return pulse

class qrcw_laser:
    r"""Quadratically-ramped continuous wave: a smooth (quadratic) ramp to full strength over ``nr``
    cycles (``tc = 2 pi nr / omega``), then constant::

        V(t) = 2 t^2/tc^2            F_str cos(omega t)   for t <= tc/2
             = (1 - 2 (t-tc)^2/tc^2) F_str cos(omega t)   for tc/2 < t <= tc
             =                       F_str cos(omega t)   for t >  tc

    .. math::

        V(t) = F_\mathrm{str}\cos(\omega t)\,\times
            \begin{cases} 2 t^2/t_c^2 & t \le t_c/2 \\
            1 - 2 (t-t_c)^2/t_c^2 & t_c/2 < t \le t_c \\
            1 & t > t_c \end{cases}

    Parameters
    ----------
    F_str, omega : float
        field strength and carrier frequency
    nr : int
        number of ramp cycles
    """
    def __init__(self, F_str, omega, nr):
        self.F_str = F_str
        self.omega = omega
        self.nr = nr
    def __call__(self, t):
        tc = 2 * np.pi / self.omega * self.nr
        if t <= 0.5 * tc:
            pulse = 2 * t ** 2 / tc ** 2 * self.F_str * np.cos(self.omega * t)
        elif t <= tc:
            pulse = (1 - 2 * (t - tc) ** 2 / tc ** 2)* self.F_str * np.cos(self.omega * t)
        else:
            pulse = self.F_str * np.cos(self.omega * t)
        return pulse


