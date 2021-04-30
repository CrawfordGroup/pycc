import numpy as np

samples = np.load("helium_cc-pvdz_F_str=1.0_omega=2.87.npz")

#print(sorted(samples.files))
time_points = samples["time_points"]
energy = samples["energy"]
dip_z = samples["dip_z"]

print("Time(s)                  Energy (a.u.)                               Z-Dipole (a.u.)     ")
for i in range(701):
    print("%7.2f  %20.15f + %20.15fi  %20.15f + %20.15fi" % (time_points[i], energy[i].real, energy[i].imag, dip_z[i].real, dip_z[i].imag))

