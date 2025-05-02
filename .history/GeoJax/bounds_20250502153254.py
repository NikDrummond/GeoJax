# bounds.py
# Bounding shape utilities for point clouds

import jax.numpy as jnp
from jax import jit
from .core import magnitude, apply_affine
from .alignment import coord_eig_decomp

@jit
def aabb_bounds(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the axis-aligned bounding box (AABB) for a point cloud.

    Parameters
    ----------
    points : jnp.ndarray
        A (N, 3) array of 3D points.

    Returns
    -------
    (min_corner, max_corner) : tuple of jnp.ndarray
        The corners of the bounding box.
    """
    return jnp.min(points, axis=0), jnp.max(points, axis=0)


@jit
def bounding_sphere(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute a bounding sphere for a point cloud.
    (Center is the mean, radius is max distance to center.)

    Parameters
    ----------
    points : jnp.ndarray
        A (N, 3) array of 3D points.

    Returns
    -------
    center : jnp.ndarray
        Center of the sphere.
    radius : jnp.ndarray
        Radius of the sphere.
    """
    center = jnp.mean(points, axis=0)
    distances = magnitude(points - center)
    return center, jnp.max(distances)


def oriented_bounding_box(points: jnp.ndarray) -> dict:
    """
    Compute an oriented bounding box (OBB) using PCA axes.

    Parameters
    ----------
    points : jnp.ndarray
        A (N, 3) array of 3D points.

    Returns
    -------
    dict with:
        - 'center': jnp.ndarray (3,)
        - 'axes': jnp.ndarray (3, 3) (rows are basis vectors)
        - 'extents': jnp.ndarray (3,) (half-widths along each axis)
    """
    center = jnp.mean(points, axis=0)
    _, eigvecs = coord_eig_decomp(points, robust=False, center=True, PCA=False, sort=True, transpose=True)

    local_coords = jnp.dot(points - center, eigvecs.T)
    min_corner, max_corner = jnp.min(local_coords, axis=0), jnp.max(local_coords, axis=0)
    extents = (max_corner - min_corner) / 2.0
    obb_center_local = (min_corner + max_corner) / 2.0
    obb_center = jnp.dot(obb_center_local, eigvecs) + center

    return {
        "center": obb_center,
        "axes": eigvecs,
        "extents": extents
    }


def bounding_cylinder(points: jnp.ndarray) -> dict:
    """
    Approximate a bounding cylinder aligned to the first principal axis.
    Useful for elongated shapes.
    """
    center = jnp.mean(points, axis=0)
    _, eigvecs = coord_eig_decomp(points, robust=False, center=True, PCA=False, sort=True, transpose=True)
    major_axis = eigvecs[0]

    projections = points - jnp.outer(jnp.dot(points - center, major_axis), major_axis)
    radius = jnp.max(magnitude(projections - center))

    heights = jnp.dot(points - center, major_axis)
    height = jnp.max(heights) - jnp.min(heights)

    return {
        "axis": major_axis,
        "center": center,
        "radius": radius,
        "height": height
    }


def tight_aabb_in_frame(points: jnp.ndarray, frame_axes: jnp.ndarray) -> dict:
    """
    Compute an axis-aligned bounding box in a custom frame.

    Parameters
    ----------
    points : jnp.ndarray
        (N, 3) array of points.
    frame_axes : jnp.ndarray
        (3, 3) orthonormal basis, e.g., from PCA.

    Returns
    -------
    dict with 'min', 'max', 'center', 'extents', and 'axes'.
    """
    local = jnp.dot(points, frame_axes.T)
    min_corner, max_corner = jnp.min(local, axis=0), jnp.max(local, axis=0)
    extents = (max_corner - min_corner) / 2.0
    center_local = (min_corner + max_corner) / 2.0
    center = jnp.dot(center_local, frame_axes)

    return {
        "center": center,
        "extents": extents,
        "axes": frame_axes
    }
