### checks.py
# checks.py
# Vector property checks and geometric predicates

import jax.numpy as jnp
from jax import jit
from functools import partial
from .core import magnitude, dot, cross, normalize

@jit
def is_unit_vector(v: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """
    Check if a vector is unit length within tolerance.
    """
    return jnp.isclose(magnitude(v), 1.0, atol=atol)

@jit
def is_collinear(v1: jnp.ndarray, v2: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """
    Check if two vectors are collinear (zero cross product).
    """
    return jnp.allclose(cross(v1, v2), 0.0, atol=atol)

@jit
def is_orthogonal(v1: jnp.ndarray, v2: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    """
    Check if two vectors are orthogonal (dot product near 0).
    """
    return jnp.isclose(dot(v1, v2), 0.0, atol=atol)

@partial(jit, static_argnames=['to_degree'])
def angle_between_planes(n1: jnp.ndarray, n2: jnp.ndarray, to_degree: bool = False) -> jnp.ndarray:
    """
    Compute the angle between two planes, given their normals.

    Parameters
    ----------
    n1, n2 : jnp.ndarray
        Normal vectors to the planes.
    to_degree : bool
        Return angle in degrees if True.

    Returns
    -------
    jnp.ndarray
        The angle between the planes.
    """
    cos_theta = dot(n1, n2)
    angle = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
    return jnp.degrees(angle) if to_degree else angle

@jit
def orthonormal_basis_from_vector(v: jnp.ndarray) -> jnp.ndarray:
    """
    Construct a right-handed orthonormal basis given one vector.

    Parameters
    ----------
    v : jnp.ndarray
        A normalized (unit) vector.

    Returns
    -------
    jnp.ndarray
        A (3, 3) matrix whose rows are an orthonormal basis.
    """
    v = normalize(v)
    arbitrary = jnp.array([1.0, 0.0, 0.0])
    alt = jnp.array([0.0, 1.0, 0.0])
    helper = jnp.where(is_collinear(v, arbitrary), alt, arbitrary)
    x = v
    y = normalize(cross(helper, x))
    z = cross(x, y)
    return jnp.stack([x, y, z], axis=0)