import jax.numpy as jnp
from jax import grad, jit, vmap, device_put, jacfwd, jacrev
from jax import random


class Args:
    seed = 100


def hessian(fn):
    return jit(jacfwd(jacrev(fn)))


if __name__ == '__main__':
    key = random.PRNGKey(Args.seed)
    x = random.normal(key, (10,))
    x = device_put(x)
    print(x)

    size = 3000
    x = random.normal(key, (size, size), dtype=jnp.float32)
    from ml_logger import logger

    logger.start('start')
    (x @ x.T).block_until_ready()
    logger.print("Took:", logger.since('start'))
