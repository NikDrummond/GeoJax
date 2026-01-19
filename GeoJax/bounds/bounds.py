"""Bounding shape utilities for point clouds."""

from jax import jit
import jax.numpy as jnp

from ..core.core import magnitude
from ..alignment.alignment import coord_eig_decomp


@jit
def aabb_bounds(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the axis-aligned bounding box (AABB) for a point cloud.

    Parameters
    ----------
    points : jnp.ndarray
        Input point cloud of shape (N, D).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Minimum and maximum corners of the AABB, each of shape (D,).
    """
    return jnp.min(points, axis=0), jnp.max(points, axis=0)


@jit
def bounding_sphere(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute a bounding sphere (center is mean, radius is max distance to center).

    Parameters
    ----------
    points : jnp.ndarray
        Input point cloud of shape (N, D).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Center and radius of the bounding sphere. Center shape (D,), radius is scalar.
    """
    center = jnp.mean(points, axis=0)
    distances = magnitude(points - center)
    return center, jnp.max(distances)


def oriented_bounding_box(points: jnp.ndarray) -> dict:
    """Compute an oriented bounding box (OBB) using PCA axes.

    Parameters
    ----------
    points : jnp.ndarray
        Input point cloud of shape (N, D).

    Returns
    -------
    dict
        Dictionary with keys:
        - "center": OBB center, shape (D,)
        - "axes": OBB axes (principal directions), shape (D, D)
        - "extents": Half-extents along each axis, shape (D,)
    """
    center = jnp.mean(points, axis=0)
    _, eigvecs = coord_eig_decomp(
        points, robust=False, center=True, PCA=False, sort=True, transpose=True
    )

    local_coords = jnp.dot(points - center, eigvecs.T)
    min_corner, max_corner = jnp.min(local_coords, axis=0), jnp.max(local_coords, axis=0)
    extents = (max_corner - min_corner) / 2.0
    obb_center_local = (min_corner + max_corner) / 2.0
    obb_center = jnp.dot(obb_center_local, eigvecs) + center

    return {"center": obb_center, "axes": eigvecs, "extents": extents}


def bounding_cylinder(points: jnp.ndarray) -> dict:
    """Approximate a bounding cylinder aligned to the first principal axis.

    Parameters
    ----------
    points : jnp.ndarray
        Input point cloud of shape (N, D).

    Returns
    -------
    dict
        Dictionary with keys:
        - "axis": Cylinder axis (first principal direction), shape (D,)
        - "center": Cylinder center, shape (D,)
        - "radius": Cylinder radius (scalar)
        - "height": Cylinder height (scalar)
    """
    center = jnp.mean(points, axis=0)
    _, eigvecs = coord_eig_decomp(
        points, robust=False, center=True, PCA=False, sort=True, transpose=True
    )
    major_axis = eigvecs[0]

    projections = points - jnp.outer(jnp.dot(points - center, major_axis), major_axis)
    radius = jnp.max(magnitude(projections - center))

    heights = jnp.dot(points - center, major_axis)
    height = jnp.max(heights) - jnp.min(heights)

    return {"axis": major_axis, "center": center, "radius": radius, "height": height}


def tight_aabb_in_frame(points: jnp.ndarray, frame_axes: jnp.ndarray) -> dict:
    """Compute an axis-aligned bounding box in a custom frame.

    Parameters
    ----------
    points : jnp.ndarray
        Input point cloud of shape (N, D).
    frame_axes : jnp.ndarray
        Frame basis vectors, shape (D, D).

    Returns
    -------
    dict
        Dictionary with keys:
        - "center": AABB center in original frame, shape (D,)
        - "extents": Half-extents along each frame axis, shape (D,)
        - "axes": Frame axes (same as input), shape (D, D)
    """
    local = jnp.dot(points, frame_axes.T)
    min_corner, max_corner = jnp.min(local, axis=0), jnp.max(local, axis=0)
    extents = (max_corner - min_corner) / 2.0
    center_local = (min_corner + max_corner) / 2.0
    center = jnp.dot(center_local, frame_axes)

    return {"center": center, "extents": extents, "axes": frame_axes}

