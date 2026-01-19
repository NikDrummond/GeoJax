"""Vector property checks and geometric predicates."""

from functools import partial

from jax import jit, lax
import jax.numpy as jnp

from ..core.core import magnitude, dot, cross, normalize


@jit
def is_unit_vector(v: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """Check if a vector is unit length within tolerance.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector(s) of shape (..., D).
    atol : float, optional
        Absolute tolerance for length check. Default is 1e-6.

    Returns
    -------
    jnp.ndarray
        Boolean array indicating if vector(s) are unit length, shape (...,).
    """
    return jnp.isclose(magnitude(v), 1.0, atol=atol)


@jit
def is_collinear(v1: jnp.ndarray, v2: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """Check if two vectors are collinear (zero cross product).

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector(s) of shape (..., 3).
    v2 : jnp.ndarray
        Second vector(s) of shape (..., 3), broadcastable with v1.
    atol : float, optional
        Absolute tolerance for cross product magnitude. Default is 1e-6.

    Returns
    -------
    jnp.ndarray
        Boolean array indicating if vectors are collinear, shape (...,).
    """
    return jnp.allclose(cross(v1, v2), 0.0, atol=atol)


@jit
def is_orthogonal(v1: jnp.ndarray, v2: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """Check if two vectors are orthogonal (dot product near 0).

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector(s) of shape (..., D).
    v2 : jnp.ndarray
        Second vector(s) of shape (..., D), broadcastable with v1.
    atol : float, optional
        Absolute tolerance for dot product. Default is 1e-6.

    Returns
    -------
    jnp.ndarray
        Boolean array indicating if vectors are orthogonal, shape (...,).
    """
    return jnp.isclose(dot(v1, v2), 0.0, atol=atol)


@jit
def orthonormal_basis_from_vector(v: jnp.ndarray) -> jnp.ndarray:
    """Construct a right-handed orthonormal basis given one vector.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector of shape (3,).

    Returns
    -------
    jnp.ndarray
        Orthonormal basis vectors, shape (3, 3). First row is normalized v.
    """
    v = normalize(v)
    arbitrary = jnp.array([1.0, 0.0, 0.0])
    alt = jnp.array([0.0, 1.0, 0.0])
    helper = jnp.where(is_collinear(v, arbitrary), alt, arbitrary)
    x = v
    y = normalize(cross(helper, x))
    z = cross(x, y)
    return jnp.stack([x, y, z], axis=0)
