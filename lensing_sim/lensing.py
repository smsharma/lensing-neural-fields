import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
from astropy.convolution import convolve, Gaussian2DKernel

from lensing_sim.units import *
from lensing_sim.profiles import MassProfileExternalShear, MassProfileSIE, MassProfileNFW, LightProfileSersic

import jax.numpy as jnp

from jaxinterp2d import CartesianGrid


class LensingSim:
    def __init__(self, lenses_list=[{}], sources_list=[{}], global_dict={}, observation_dict={}):
        """
        Class for simulation of strong lensing images
        """

        self.lenses_list = lenses_list
        self.sources_list = sources_list

        self.global_dict = global_dict
        self.observation_dict = observation_dict

        self.set_up_global()
        self.set_up_observation()

    def set_up_global(self):
        """ Set some global variables so don't need to recompute each time
        """
        self.z_s = self.global_dict["z_s"]
        self.z_l = self.global_dict["z_l"]

        self.D_s = Planck15.angular_diameter_distance(z=self.z_s).value  # * Mpc
        self.D_l = Planck15.angular_diameter_distance(z=self.z_l).value  # * Mpc

        self.Sigma_crit = 1.0 / (4 * jnp.pi * GN) * self.D_s / ((self.D_s - self.D_l) * self.D_l) / M_s

        # print(self.Sigma_crit)

    def set_up_observation(self):
        """ Set up observational grid and parameters
        """
        # Coordinate limits (in arcsecs)
        self.theta_x_lims = self.observation_dict["theta_x_lims"]
        self.theta_y_lims = self.observation_dict["theta_y_lims"]

        # Size of grid
        self.n_x = self.observation_dict["n_x"]
        self.n_y = self.observation_dict["n_y"]

        # Exposure and background noise level
        self.exposure = self.observation_dict["exposure"]
        self.f_iso = self.observation_dict["f_iso"]

        # x/y-coordinates of grid and pixel area in arcsec**2

        self.theta_x, self.theta_y = jnp.meshgrid(
            jnp.linspace(self.theta_x_lims[0], self.theta_x_lims[1], self.n_x),
            jnp.linspace(self.theta_y_lims[0], self.theta_y_lims[1], self.n_y)
        )

        self.x, self.y = self.D_l * self.theta_x * asctorad, self.D_l * self.theta_y * asctorad

        self.x_lims, self.y_lims = self.D_l * asctorad * jnp.array(self.theta_x_lims), self.D_l * asctorad * jnp.array(
            self.theta_y_lims)

        self.pix_area = ((self.theta_x_lims[1] - self.theta_x_lims[0]) / self.n_x) * (
                    (self.theta_y_lims[1] - self.theta_y_lims[0]) / self.n_y)

    def lensed_image(self, return_deflection_maps=False):
        """ Get strongly lensed image
        """

        # Get lensing potential gradients

        x_d, y_d = jnp.zeros((self.n_x, self.n_y)), jnp.zeros((self.n_x, self.n_y))

        if return_deflection_maps:
            x_d_host, y_d_host = jnp.zeros((self.n_x, self.n_y)), jnp.zeros((self.n_x, self.n_y))
            x_d_sub, y_d_sub = jnp.zeros((self.n_x, self.n_y)), jnp.zeros((self.n_x, self.n_y))

        for lens_dict in self.lenses_list:
            if lens_dict["profile"] == "SIE":
                _x_d, _y_d = MassProfileSIE(
                    x_0=lens_dict["theta_x_0"] * self.D_l * asctorad,
                    y_0=lens_dict["theta_y_0"] * self.D_l * asctorad,
                    r_E=lens_dict["theta_E"] * self.D_l * asctorad,
                    q=lens_dict["q"],
                ).deflection(self.x, self.y)
            elif lens_dict["profile"] == "NFW":
                # print(lens_dict["rho_s"]* (lens_dict["r_s"])  / self.Sigma_crit)
                # print((lens_dict["rho_s"]) * (lens_dict["r_s"])  / self.Sigma_crit / Mpc)
                _x_d, _y_d = MassProfileNFW(
                    x_0=lens_dict["theta_x_0"] * self.D_l * asctorad,
                    y_0=lens_dict["theta_y_0"] * self.D_l * asctorad,
                    M_200=lens_dict["M_200"],
                    kappa_s=(lens_dict["rho_s"]) * (lens_dict["r_s"]) / self.Sigma_crit / Mpc,
                    r_s=lens_dict["r_s"],
                ).deflection(self.x, self.y)
            elif lens_dict["profile"] == "ExtShear":
                _x_d, _y_d = MassProfileExternalShear(
                    x_0=lens_dict["theta_x_0"] * self.D_l * asctorad,
                    y_0=lens_dict["theta_y_0"] * self.D_l * asctorad,
                    gamma_1=lens_dict["gamma_1"],
                    gamma_2=lens_dict["gamma_2"],
                ).deflection(self.x, self.y)
            else:
                raise Exception("Unknown lens profile specification!")

            x_d += _x_d
            y_d += _y_d
            if return_deflection_maps and lens_dict["profile"] == "SIE":
                x_d_host += _x_d
                y_d_host += _y_d
            elif return_deflection_maps and lens_dict["profile"] == "NFW":
                x_d_sub += _x_d
                y_d_sub += _y_d

        if return_deflection_maps:
            return (x_d, y_d), (x_d_host, y_d_host), (x_d_sub, y_d_sub), (
                        self.x.flatten() ** 2 + self.y.flatten() ** 2) ** 2

        # Evaluate source image on deflected lens plane to get lensed image

        f_lens = jnp.zeros((self.n_x, self.n_y))

        for source_dict in self.sources_list:
            if source_dict["profile"] == "Sersic":

                f_lens += (
                        LightProfileSersic(
                            x_0=source_dict["theta_x_0"] * self.D_l * asctorad,
                            y_0=source_dict["theta_y_0"] * self.D_l * asctorad,
                            S_tot=source_dict["S_tot"],
                            r_e=source_dict["theta_e"] * self.D_l * asctorad,
                            n_srsc=source_dict["n_srsc"],
                        ).flux(self.x - x_d, self.y - y_d)
                        * self.D_l ** 2
                        / radtoasc ** 2
                )
            elif source_dict["profile"] == "CartesianGrid":
                src_ary = source_dict["src_ary"]
                f_lens += CartesianGrid(limits=[self.x_lims, self.y_lims], values=src_ary)(self.x - x_d, self.y - y_d)
            else:
                raise Exception("Unknown source profile specification!")

        f_iso = self.f_iso * jnp.ones((self.n_x, self.n_y))  # Isotropic background
        i_tot = (f_lens + f_iso) * self.exposure * self.pix_area  # Total lensed image

        return i_tot

    @staticmethod
    def _mag_to_flux(mag, mag_zp):
        """
        Returns total flux of the integrated profile corresponding to magnitude `mag`, in ADU relative to `mag_zp`
        """
        return 10 ** (-0.4 * (mag - mag_zp))
