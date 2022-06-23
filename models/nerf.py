from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class NeRFModel(nn.Module):
    """ NeRF MLP
    """

    dtype: Any = jnp.float32
    num_dense_layers: int = 4
    dense_layer_width: int = 256
    add_positional_encoding: bool = True
    add_random_fourier: bool = False
    add_skip_connection: bool = False
    positional_encoding_dims: int = 5
    bandwidth_rff: int = 5

    @nn.compact
    def __call__(self, input_points):

        loc = self.param("loc_params", self.initial_loc_params)
        scale_tri = self.param("scale_tri_params", self.initial_scale_tri_params)

        # Apply positional encodings or random Fourier features (but not both) to the input points

        if self.add_positional_encoding and self.add_random_fourier:
            raise NotImplementedError("Can't have both positional encodings and RFFs")

        if self.add_positional_encoding:
            x = self.positional_encoding(input_points, self.positional_encoding_dims)
        elif self.add_random_fourier:
            x = nn.Dense(self.dense_layer_width, dtype=self.dtype, kernel_init=nn.initializers.normal(stddev=self.bandwidth_rff / 2.0), bias_init=nn.initializers.uniform(scale=1))(input_points)
            x = jnp.sin(2 * np.pi * x)
        else:
            x = input_points

        for i in range(self.num_dense_layers):

            x = nn.Dense(self.dense_layer_width, dtype=self.dtype,)(x)

            x = nn.relu(x)

            if self.add_skip_connection:
                x = jnp.concatenate([x, input_points], axis=-1) if i in [4] else x

        x = nn.Dense(2, dtype=self.dtype)(x)
        return x, loc, scale_tri

    def positional_encoding(self, inputs, positional_encoding_dims):
        """ Sinusoidal positional encodings
        """
        batch_size, _ = inputs.shape
        inputs_freq = jax.vmap(lambda x: inputs * 2.0 ** x)(jnp.arange(positional_encoding_dims))
        periodic_fns = jnp.stack([jnp.sin(inputs_freq), jnp.cos(inputs_freq)])
        periodic_fns = periodic_fns.swapaxes(0, 2).reshape([batch_size, -1])
        periodic_fns = jnp.concatenate([inputs, periodic_fns], axis=-1)
        return periodic_fns

    def initial_scale_tri_params(self, key, n_lens_params=7):
        """ Initial Cholesky matrix lens params
        """
        ary = -7.0 * jnp.ones(int(n_lens_params * (n_lens_params + 1) / 2))
        ary = ary.at[:-n_lens_params].set(0.0)
        return ary

    def initial_loc_params(self, key):
        """ Initial mean lens params
        """
        return jnp.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

