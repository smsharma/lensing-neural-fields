import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile

from jaxinterp2d import CartesianGrid

class Interp(gigalens.profile.LightProfile):
    _name = "INTERP"
    _params = ["img", "x_lims", "y_lims"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, img, x_lims, y_lims):
        ret = CartesianGrid(limits=[x_lims, y_lims], values=img)(x, y)
        return ret[jnp.newaxis, ...] if self.use_lstsq else ret
