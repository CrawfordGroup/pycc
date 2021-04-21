import numpy as np

class gaussian_laser:
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
