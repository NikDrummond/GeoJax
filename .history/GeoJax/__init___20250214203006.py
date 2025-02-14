from .core import *
from .circstats import *

@jit
def components(arr: jnp.ndarray,
               p: float = 1.0,
               phi: jnp.ndarray = jnp.array([0.0]),
               axis: int = -1,
               weights: jnp.ndarray = jnp.array([0])
               ) -> jnp.array:
    
    # if weights are 0
    if jnp.sum(weights) == 0:
        use_weights = jnp.ones_like(arr)
    try:
        use_weights = jnp.broadcast_to(weights, arr.shape)
    except ValueError:
        raise ValueError("Weights and data have inconsistent shape.")
    
    if axis == -1:
        use_axis = None
    else:
        use_axis = axis

    C = jnp.sum(use_weights * jnp.cos(p * (arr - phi)), axis) / jnp.sum(use)
