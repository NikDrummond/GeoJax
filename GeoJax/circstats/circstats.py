"""Circular statistics with JAX."""

from functools import partial

from jax import jit, lax
import jax.numpy as jnp


@jit
def _components(
    angles: jnp.ndarray,
    p: float = 1.0,
    phi: float = 0.0,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized rectangular components of circular data."""
    weights = lax.cond(
        jnp.sum(weights) == 0,
        lambda _: jnp.ones_like(angles, dtype=jnp.float32),
        lambda w: jnp.broadcast_to(w, angles.shape),
        operand=weights,
    )
    C = jnp.sum(weights * jnp.cos(p * (angles - phi))) / jnp.sum(weights)
    S = jnp.sum(weights * jnp.sin(p * (angles - phi))) / jnp.sum(weights)
    return C, S


@jit
def _angle(
    angles: jnp.ndarray,
    p: float = 1.0,
    phi: float = 0.0,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
) -> jnp.ndarray:
    """Return mean angle in radians."""
    C, S = _components(angles, p, phi, weights)
    return jnp.arctan2(S, C)


@jit
def _length(
    angles: jnp.ndarray,
    p: float = 1.0,
    phi: float = 0.0,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
) -> jnp.ndarray:
    """Return mean resultant length."""
    C, S = _components(angles, p, phi, weights)
    return jnp.hypot(S, C)


@partial(jit, static_argnames=["to_degree"])
def circmean(
    angles: jnp.ndarray,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
    to_degree: bool = False,
) -> jnp.ndarray:
    """Compute the circular mean of angles."""
    mean_angle = _angle(angles, 1.0, 0.0, weights)
    return lax.cond(to_degree, lambda x: jnp.degrees(x), lambda x: x, mean_angle)


@jit
def _circstd(
    angles: jnp.ndarray,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
    method: int = 0,
) -> jnp.ndarray:
    """Internal standard deviation helper."""
    length = _length(angles, 1.0, 0.0, weights)
    return lax.cond(
        method == 0,
        lambda: jnp.sqrt(2.0 * (1.0 - length)),
        lambda: jnp.sqrt(-2.0 * jnp.log(length)),
    )


def circstd(
    angles: jnp.ndarray,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
    method: str = "angular",
) -> jnp.ndarray:
    """Compute circular or angular standard deviation."""
    method_id = 0 if method == "angular" else 1
    return _circstd(angles, weights, method_id)


@jit
def circvar(
    angles: jnp.ndarray,
    weights: jnp.ndarray = jnp.array([0.0], dtype=jnp.float32),
) -> jnp.ndarray:
    """Compute circular variance: 1 - mean resultant length."""
    return 1.0 - _length(angles, weights=weights)
