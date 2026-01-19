"""Angle-related operations."""

from functools import partial

from jax import jit, lax
import jax.numpy as jnp

from ..core import dot, cross, reject, magnitude


@partial(jit, static_argnames=["to_degree"])
def angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray | None = None,
    to_degree: bool = False,
    assume_normalized: bool = False,
) -> jnp.ndarray:
    """Compute the angle between two vectors, optionally projected to a plane."""
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


@partial(jit, static_argnames=["to_degree"])
def signed_angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray,
    to_degree: bool = False,
) -> jnp.ndarray:
    """Compute the signed angle between v1 and v2 in the plane defined by plane_normal."""
    v1_proj = reject(v1, plane_normal)
    v2_proj = reject(v2, plane_normal)

    v1_proj, v2_proj = jnp.broadcast_arrays(v1_proj, v2_proj)

    x = jnp.sum(v1_proj * v2_proj, axis=-1)
    y = jnp.sum(cross(v1_proj, v2_proj) * plane_normal, axis=-1)

    theta = jnp.arctan2(y, x)
    return lax.cond(to_degree, lambda x: jnp.degrees(theta), lambda x: x, theta)


@partial(jit, static_argnames=["to_degree"])
def angle_between_planes(n1: jnp.ndarray, n2: jnp.ndarray, to_degree: bool = False) -> jnp.ndarray:
    """Compute the angle between two planes, given their normals."""
    cos_theta = dot(n1, n2)
    angle_val = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
    return lax.cond(to_degree, lambda x: jnp.degrees(x), lambda x: x, angle_val)


@jit
def minimum_signed_angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray,
    to_degree: bool = False,
) -> jnp.ndarray:
    """Compute minimal signed angle between v1 and v2, within [-pi/2, pi/2]."""
    theta = signed_angle(v1, v2, plane_normal=plane_normal, to_degree=False)
    minimal = jnp.where(jnp.abs(theta) > (jnp.pi / 2), theta - jnp.sign(theta) * jnp.pi, theta)
    return jnp.degrees(minimal) if to_degree else minimal
