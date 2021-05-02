import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
import sys

def get_spectral_lines(
    time_points, dipole_moment, stop_laser=0, xlim=(None, None), detrend=None
):
    mask = time_points >= stop_laser

    dt = time_points[1] - time_points[0]

    freq = (
        scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(time_points[mask])))
        * 2
        * np.pi
        / dt
    )

    dip = 0

    if detrend:
        dip = scipy.signal.detrend(dipole_moment[mask], type="constant")
    else:
        dip = dipole_moment[mask]

    a = np.abs(scipy.fftpack.fftshift(scipy.fftpack.fft(dip)))

    a /= a.max()

    lower_bound = xlim[0] if xlim[0] is not None else np.min(freq)
    upper_bound = xlim[1] if xlim[1] is not None else np.max(freq)

    mask = (freq >= lower_bound) & (freq <= upper_bound)

    return freq[mask], a[mask]

#name = str(sys.argv[1])
#F_str = float(sys.argv[3])
#omega=float(sys.argv[4])
#nfrozen = 0
#basis = str(sys.argv[2])

samples = np.load(
    f"helium_cc-pvdz_F_str=10.0_omega=2.87.npz",
    allow_pickle=True,
)

time_points = samples["time_points"]
energy = samples["energy"]
dip_x = samples["dip_x"]
dip_y = samples["dip_y"]
dip_z = samples["dip_z"]

#auto_corr = samples["A_t_t0"]
#tau0 = np.abs(np.exp(samples["tau0"]))**2

plt.figure()
plt.subplot(211)
plt.plot(time_points,energy.real,label=r'$\Re E(t)$')
plt.grid()
plt.legend()
plt.subplot(212)
plt.plot(time_points,np.abs(energy.imag), label=r'$|\Im E(t)|$')
plt.grid()
plt.legend()
plt.xlabel('time (a.u.)')

#plt.figure()
#plt.plot(time_points, np.abs(auto_corr)**2, label=r'$|A(t,0)|^2$')
#plt.plot(time_points, np.abs(tau0)**2, label=r'$|e^{\tau_0(t)}|^2$')
#plt.grid()
#plt.legend()
#plt.xlabel('time (a.u.)')

plt.figure()
plt.subplot(311)
plt.plot(time_points, dip_x.real, label=r'$\Re d_x(t)$')
plt.legend()
plt.grid()
plt.subplot(312)
plt.plot(time_points, dip_y.real, label=r'$\Re d_y(t)$')
plt.legend()
plt.grid()
plt.subplot(313)
plt.plot(time_points, dip_z.real, label=r'$\Re d_z(t)$')
plt.legend()
plt.grid()
plt.xlabel('time (a.u.)')

dip_tot = dip_x+dip_y+dip_z
freq, a = get_spectral_lines(time_points, dip_tot, 5, detrend=True)
one_ev = 27.211386245988468

plt.figure()
plt.plot(freq*one_ev, a)
plt.xlim(0,90)
plt.ylabel('Relative intensity')
plt.xlabel('frequency (eV)')
plt.show()


