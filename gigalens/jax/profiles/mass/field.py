import functools

import jax.numpy as jnp
from jax import jit, jacfwd

import gigalens.profile

from jaxinterp2d import CartesianGrid


class FIELD(gigalens.profile.MassProfile):
    _name = "FIELD"
    _params = ["img", "x_lims", "y_lims"]

    @functools.partial(jit, static_argnums=(0))
    def deriv(self, x, y, img, x_lims, y_lims):
        def rho(coords):
            x, y = coords
            return CartesianGrid(limits=[x_lims, y_lims], values=img)(x, y)[:, :, 0]

        print(x.shape, y.shape)
        grad_x, grad_y = jacfwd(rho)([x, y])
        print(grad_x.shape, grad_y.shape)

        return grad_x, grad_y
