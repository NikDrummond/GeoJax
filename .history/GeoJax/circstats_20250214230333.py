from jax import jit, lax
import jax.numpy as jnp


@jit
def _components(
    arr: jnp.ndarray,
    p: float = 1.0,
    phi: jnp.ndarray = jnp.array([0.0]),
    weights: jnp.ndarray = jnp.array([0], dtype=jnp.float32),
) -> jnp.array:
    """Jitted Jax implementation of astopy's _components utility function

    Computes the generalized rectangular components of circular data

    """

    # if weights are 0
    weights = lax.cond(
        jnp.sum(weights) == 0,
        lambda _: jnp.ones_like(arr, dtype=jnp.float32),
        lambda w: jnp.broadcast_to(w, arr.shape),
        operand=weights,
    )

    C = jnp.sum(weights * jnp.cos(p * (arr - phi))) / jnp.sum(weights)
    S = jnp.sum(weights * jnp.sin(p * (arr - phi))) / jnp.sum(weights)

    return C, S


@jit
def _angle(
    arr=jnp.ndarray,
    p: float = 1.0,
    phi: jnp.ndarray = jnp.array([0.0]),
    weights: jnp.ndarray = jnp.array([0], dtype=jnp.float32),
) -> jnp.ndarray:

    C, S = _components(arr, p, phi, weights)

    # theta will be an angle in the interval [-np.pi, np.pi)
    theta = jnp.arctan2(S, C)

    return theta

@jit
def _length(arr:jnp.ndarray,
           p:float = 1.0,
           phi:float = 0.0,
           weights: jnp.ndarray = jnp.array([0], dtype = jnp.float32)
           ) -> jnp.ndarray:
    C, S = _components(arr, p, phi, weights)
    return jnp.hypot(S,C)


@jit
def circmean(
    arr: jnp.ndarray, weights: jnp.ndarray = jnp.array([0], dtype=jnp.float32), to_degree:bool = False
) -> jnp.ndarray:
    
    assert arr.ndim == 1, "Input array must be 1-dimensional"
    out_angle_rad = _angle(arr, 1.0, 0.0, weights)
    # Optionally convert to degrees.
    out_angle = lax.cond(
        to_degree, lambda a: jnp.degrees(a), lambda a: a, out_angle_rad
    )
    return out_angle

@jit
def _circstd(
    arr: jnp.ndarray,
    weights: jnp.ndarray = jnp.array([0], dtype=jnp.float32),
    method: int = 0,
) -> jnp.ndarray:
    # if angular or circular
    std = lax.cond(
        method == 0,
        lambda: jnp.sqrt(2.0 * (1.0 - _length(arr, 1.0, 0.0, weights))),
        lambda: jnp.sqrt(-2.0 * jnp.log(_length(arr, 1.0, 0.0, weights))),
    )

    return std

def circ_std(
    arr: jnp.ndarray,
    weights: jnp.ndarray = jnp.array([0], dtype=jnp.float32),
    method: str = 'angular',
) -> jnp.ndarray:

    assert method in ('angular','circular'), "method must be angular or circular"

    if method == 'angular':
        m = 0
    else:
        m = 1

    std = _circ_std(arr,weights,m)

    return std