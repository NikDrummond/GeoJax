"""Vector property checks and geometric predicates."""

from functools import partial

from jax import jit, lax
import jax.numpy as jnp

from ..core.core import magnitude, dot, cross, normalize


@jit
def is_unit_vector(v: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """Check if a vector is unit length within tolerance."""
    return jnp.isclose(magnitude(v), 1.0, atol=atol)


@jit
def is_collinear(v1: jnp.ndarray, v2: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """Check if two vectors are collinear (zero cross product)."""
    return jnp.allclose(cross(v1, v2), 0.0, atol=atol)


@jit
def is_orthogonal(v1: jnp.ndarray, v2: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """Check if two vectors are orthogonal (dot product near 0)."""
    return jnp.isclose(dot(v1, v2), 0.0, atol=atol)


@jit
def orthonormal_basis_from_vector(v: jnp.ndarray) -> jnp.ndarray:
    """Construct a right-handed orthonormal basis given one vector."""
    v = normalize(v)
    arbitrary = jnp.array([1.0, 0.0, 0.0])
    alt = jnp.array([0.0, 1.0, 0.0])
    helper = jnp.where(is_collinear(v, arbitrary), alt, arbitrary)
    x = v
    y = normalize(cross(helper, x))
    z = cross(x, y)
    return jnp.stack([x, y, z], axis=0)
