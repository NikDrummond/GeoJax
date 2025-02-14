from jax import jit, lax
import jax.numpy as jnp

@jit
def _components(
    arr: jnp.ndarray,
    p: float = 1.0,
    phi: jnp.ndarray = jnp.array([0.0]),
    axis: int = -1,
    weights: jnp.ndarray = jnp.array([0], dtype=jnp.float32)
) -> jnp.array:
    """Jitted Jax implementation of astopy's _components utility function

    Computes the generalized rectangular components of circular data

    """

    # if weights are 0
    weights = lax.cond(
        jnp.sum(weights) == 0,
        lambda _: jnp.ones_like(arr, dtype = jnp.float32),
        lambda w: jnp.broadcast_to(w, arr.shape),
        operand=weights
    )

    # sort if axis is -1
    # lax.cond(axis == -1, lambda axis: None, lambda axis: axis)
    if axis == -1:
        axis = None

    C = jnp.sum(weights * jnp.cos(p * (arr - phi)), axis) / jnp.sum(weights, axis)
    S = jnp.sum(weights * jnp.sin(p * (arr - phi)), axis) / jnp.sum(weights, axis)

    return C, S