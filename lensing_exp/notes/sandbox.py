import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

from lensing_sim.lensing import LensingSim

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

### Can play around with changing these ###
# Size of source
theta_s_e = 0.4

# Redshift (distance) of source and lens
global_dict = {"z_s": 1., "z_l": 0.2}

# theta_x_0 and theta_y_0 are the position of lens, theta_E is its size, q is its ellipticity
hst_param_dict = {"profile": "SIE", "theta_x_0": 0.12, "theta_y_0": -0.2, "theta_E": 2.15, "q": 0.8}

src_param_dict = {"profile": "Sersic", "theta_x_0": 0., "theta_y_0": 0., "S_tot": S_tot, "theta_e": theta_s_e,
                  "n_srsc": 1}

# Lensed image
lens_list = [hst_param_dict]

lsi = LensingSim(lens_list, [src_param_dict], global_dict, observation_dict)
image = lsi.lensed_image()

plt.imshow(image)
plt.show()

# In[7]:
# Change the lens position
hst_param_dict = {"profile": "SIE", "theta_x_0": 0.3, "theta_y_0": 0.4, "theta_E": 2.15, "q": 0.8}

lens_list = [hst_param_dict]

lsi = LensingSim(lens_list, [src_param_dict], global_dict, observation_dict)
image = lsi.lensed_image()

plt.imshow(image)

# Unlense image---just don't add a lens
lens_list = []  # No lens

lsi = LensingSim(lens_list, [src_param_dict], global_dict, observation_dict)
image = lsi.lensed_image()

plt.imshow(image)
plt.show()
