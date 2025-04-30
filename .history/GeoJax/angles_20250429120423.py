### angles.py
# angles.py
# Angle-related operations using JAX
# Includes: angle, signed_angle, minimum_signed_angle

from jax import jit, lax
import jax.numpy as jnp
from functools import partial
from .core import dot, cross, reject, magnitude

@partial(jit, static_argnames=['to_degree'])
def angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray | None = None,
    to_degree: bool = False,
    assume_normalized: bool = False,
) -> jnp.ndarray:
    """
    Compute the angle between two vectors, optionally projected to a plane.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector(s).
    v2 : jnp.ndarray
        Second vector(s).
    plane_normal : jnp.ndarray | None, optional
        Plane normal for projection, by default None.
    to_degree : bool, optional
        Return angle in degrees if True, radians otherwise.
    assume_normalized : bool, optional
        Set True to skip normalization (vectors must be unit length).

    Returns
    -------
    jnp.ndarray
        Angle(s) in radians or degrees.
    """
    if plane_normal is not None:
        v1 = reject(v1, plane_normal)
        v2 = reject(v2, plane_normal)

    v1, v2 = jnp.atleast_2d(v1), jnp.atleast_2d(v2)
    v1, v2 = jnp.broadcast_arrays(v1, v2)

    dot_products = jnp.sum(v1 * v2, axis=-1)
    magnitudes = magnitude(v1) * magnitude(v2)
    cosines = jnp.where(assume_normalized, dot_products, dot_products / (magnitudes + 1e-10))

    angles = jnp.arccos(jnp.clip(cosines, -1.0, 1.0))
    return lax.cond(to_degree, lambda x: jnp.degrees(x), lambda x: x, angles)


@partial(jit, static_argnames=['to_degree'])
def signed_angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray,
    to_degree: bool = False,
) -> jnp.ndarray:
    """
    Compute the signed angle between v1 and v2 in the plane defined by plane_normal.

    Sign follows right-hand rule w.r.t. the normal.

    Parameters
    ----------
    v1, v2 : jnp.ndarray
        Input vectors or vector batches.
    plane_normal : jnp.ndarray
        Normal vector of the plane.
    to_degree : bool
        If True, convert to degrees.

    Returns
    -------
    jnp.ndarray
        Signed angle(s).
    """
    v1_proj = reject(v1, plane_normal)
    v2_proj = reject(v2, plane_normal)

    v1_proj, v2_proj = jnp.broadcast_arrays(v1_proj, v2_proj)

    x = jnp.sum(v1_proj * v2_proj, axis=-1)
    y = jnp.sum(cross(v1_proj, v2_proj) * plane_normal, axis=-1)

    theta = jnp.arctan2(y, x)
    # return jnp.degrees(theta) if to_degree else theta
    return lax.cond(to_degree, lambda x: jnp.degrees(theta), lambda x: theta, result)


@jit
def minimum_signed_angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray,
    to_degree: bool = False,
) -> jnp.ndarray:
    """
    Compute minimal signed angle between v1 and v2, within [-pi/2, pi/2].

    This assumes an undirected interpretation of the second vector.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector.
    v2 : jnp.ndarray
        Second vector (interpreted as a direction).
    plane_normal : jnp.ndarray
        Normal of the projection plane.
    to_degree : bool
        Whether to return result in degrees.

    Returns
    -------
    jnp.ndarray
        Minimal signed angle(s).
    """
    theta = signed_angle(v1, v2, plane_normal=plane_normal, to_degree=False)
    minimal = jnp.where(jnp.abs(theta) > (jnp.pi / 2), theta - jnp.sign(theta) * jnp.pi, theta)
    return jnp.degrees(minimal) if to_degree else minimal
