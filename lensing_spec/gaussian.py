import matplotlib.pyplot as plt
from lensing_sim.lensing import LensingSim

# Lensing Parameters
exposure = 1610.0
n_xy = 64
pixel_size = 0.1
mag_zero = 25.5
mag_iso = 22.5
mag_s = 23.0
fwhm_psf = 0.18

coordinate_limit = pixel_size * n_xy / 2.0
f_iso = LensingSim._mag_to_flux(mag_iso, mag_zero)
S_tot = LensingSim._mag_to_flux(mag_s, mag_zero)

observation_dict = {
    "n_x": n_xy,
    "n_y": n_xy,
    "theta_x_lims": (-coordinate_limit, coordinate_limit),
    "theta_y_lims": (-coordinate_limit, coordinate_limit),
    "exposure": exposure,
    "f_iso": f_iso,
}
