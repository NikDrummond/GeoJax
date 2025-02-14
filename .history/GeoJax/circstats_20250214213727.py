from jax import jit, lax
import jax.numpy as jnp

@jit
def components(
    arr: jnp.ndarray,
    p: float = 1.0,
    phi: jnp.ndarray = jnp.array([0.0]),
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


    C = jnp.sum(weights * jnp.cos(p * (arr - phi))) / jnp.sum(weights)
    S = jnp.sum(weights * jnp.sin(p * (arr - phi))) / jnp.sum(weights)

    return C, S

@jit
def _angle(arr = jnp.ndarray,
              p: float = 1.0,
              phi: jnp.ndarray = jnp.array([0.0]),
              weights: jnp.ndarray = jnp.array([0], dtype = jnp.float32)
              ) -> jnp.ndarray:
    
    C,S = components(arr, p, phi, weights)

    # theta will be an angle in the interval [-np.pi, np.pi)
    # [-180, 180)*u.deg in case data is a Quantity
    theta = jnp.arctan2(S, C)

    return theta

@jit
def circmean(arr:jnp.ndarray,
             weights: jnp.ndarray = jnp.array([0], dtype = jnp.float32)
             ) -> jnp.ndarray:
    assert arr.ndim == 1, 'Input array must be 1-dimensional'
    return jit_angle(arr,1.0,0.0,weights)

