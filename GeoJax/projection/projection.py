"""Geometric projection utilities."""

from functools import partial

from jax import jit, lax, vmap
import jax.numpy as jnp

from ..core import magnitude, normalize, reject


@partial(jit, static_argnames=["axis", "squash"])
def reject_axis(vectors: jnp.ndarray, axis: int, squash: bool = False) -> jnp.ndarray:
    """Reject a component of the vector(s) along a specified axis.

    Parameters
    ----------
    vectors : jnp.ndarray
        Input vectors of shape (..., 3).
    axis : int
        Axis index (0, 1, or 2) to reject.
    squash : bool, optional
        If True, remove the axis dimension; otherwise set it to zero. Default is False.

    Returns
    -------
    jnp.ndarray
        Vectors with rejected component, shape (..., 2) if squash=True, else (..., 3).

    Raises
    ------
    ValueError
        If vectors last dimension is not 3.
    """
    if vectors.shape[-1] != 3:
        raise ValueError("reject_axis expects input vectors with last dimension 3")
    if squash:
        return vectors[..., jnp.array([i for i in range(3) if i != axis])]
    return vectors.at[..., axis].set(0.0)


@jit
def project_to_sphere(points: jnp.ndarray, radius: float, center: jnp.ndarray) -> jnp.ndarray:
    """Project a set of points onto a sphere of given radius and center."""
    shifted = points - center
    norm = magnitude(shifted)[..., None]
    scaled = (radius / (norm + 1e-10)) * shifted
    return scaled + center


@jit
def project_to_plane(vector: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """Project a vector onto a plane defined by a normal vector.

    Parameters
    ----------
    vector : jnp.ndarray
        Vector(s) to project, shape (..., D).
    normal : jnp.ndarray
        Normal vector of the plane, shape (D,) or broadcastable.

    Returns
    -------
    jnp.ndarray
        Projected vector(s) on the plane, same shape as input.
    """
    return reject(vector, normal)


@jit
def project_to_xy_plane(points: jnp.ndarray) -> jnp.ndarray:
    """Project points onto the XY plane by dropping the Z-component.

    Parameters
    ----------
    points : jnp.ndarray
        Input points of shape (..., 3).

    Returns
    -------
    jnp.ndarray
        Projected points, shape (..., 2).
    """
    return points[..., :2]


@jit
def project_to_xz_plane(points: jnp.ndarray) -> jnp.ndarray:
    """Drop Y-component to project onto the XZ plane."""
    return points[..., [0, 2]]


@jit
def project_to_yz_plane(points: jnp.ndarray) -> jnp.ndarray:
    """Project points onto the YZ plane by dropping the X-component.

    Parameters
    ----------
    points : jnp.ndarray
        Input points of shape (..., 3).

    Returns
    -------
    jnp.ndarray
        Projected points, shape (..., 2).
    """
    return points[..., 1:]


@jit
def orthographic_projection(points: jnp.ndarray) -> jnp.ndarray:
    """Orthographic projection onto the XY plane."""
    return points[..., :2]


def _rotate_to_align_with_z(points: jnp.ndarray, north_pole: jnp.ndarray) -> jnp.ndarray:
    """Rotate `points` so that `north_pole` aligns with the z-axis (assumes normalized).

    Parameters
    ----------
    points : jnp.ndarray
        Input points of shape (..., 3), assumed to be normalized.
    north_pole : jnp.ndarray
        Vector to align with z-axis, shape (3,).

    Returns
    -------
    jnp.ndarray
        Rotated points, same shape as input.
    """
    north_pole = normalize(north_pole)
    z_axis = jnp.array([0.0, 0.0, 1.0])

    axis = jnp.cross(north_pole, z_axis)
    angle = jnp.arccos(jnp.clip(jnp.dot(north_pole, z_axis), -1.0, 1.0))
    axis_norm = normalize(axis)

    def rotate(v):
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        return v * c + jnp.cross(axis_norm, v) * s + axis_norm * jnp.dot(axis_norm, v) * (1 - c)

    is_aligned = jnp.allclose(north_pole, z_axis, atol=1e-6)
    return lax.cond(is_aligned, lambda: points, lambda: vmap(rotate)(points))


@jit
def stereographic_projection(points: jnp.ndarray, north_pole: jnp.ndarray = None) -> jnp.ndarray:
    """Apply stereographic projection from the north pole of a unit sphere onto the equatorial plane.

    Parameters
    ----------
    points : jnp.ndarray
        Input 3D points of shape (..., 3).
    north_pole : jnp.ndarray, optional
        North pole direction for projection. If None, uses standard z-axis. Default is None.

    Returns
    -------
    jnp.ndarray
        Projected 2D points, shape (..., 2).
    """
    norm_points = normalize(points)
    if north_pole is not None:
        norm_points = _rotate_to_align_with_z(norm_points, north_pole)

    x, y, z = norm_points[..., 0], norm_points[..., 1], norm_points[..., 2]
    denom = jnp.clip(1.0 - z, 1e-6, jnp.inf)
    return jnp.stack([x / denom, y / denom], axis=-1)

@jit
def project_to_vector(vector: jnp.ndarray, onto: jnp.ndarray) -> jnp.ndarray:
    """
    Project `vector` onto another vector `onto`.

    Parameters
    ----------
    vector : jnp.ndarray
        Input vector or array of vectors (..., 3).
    onto : jnp.ndarray
        Target direction vector (3,) or broadcastable to `vector`.

    Returns
    -------
    jnp.ndarray
        Vector projection(s) of `vector` onto `onto`.
    """
    if vector.shape[-1] != 3 or onto.shape[-1] != 3:
        raise ValueError("project_to_vector expects vectors with shape (..., 3)")
    onto_unit = normalize(onto)
    projection_length = jnp.sum(vector * onto_unit, axis=-1, keepdims=True)
    return projection_length * onto_unit

@jit
def equirectangular_projection(points: jnp.ndarray, north_pole: jnp.ndarray = None) -> jnp.ndarray:
    """Apply equirectangular projection converting spherical coordinates to (longitude, latitude).

    Parameters
    ----------
    points : jnp.ndarray
        Input 3D points of shape (..., 3).
    north_pole : jnp.ndarray, optional
        North pole direction for projection. If None, uses standard z-axis. Default is None.

    Returns
    -------
    jnp.ndarray
        Projected coordinates (lon, lat) in radians, shape (..., 2).
    """
    norm_points = normalize(points)
    if north_pole is not None:
        norm_points = _rotate_to_align_with_z(norm_points, north_pole)

    x, y, z = norm_points[..., 0], norm_points[..., 1], norm_points[..., 2]
    lon = jnp.arctan2(y, x)
    lat = jnp.arcsin(jnp.clip(z, -1.0, 1.0))
    return jnp.stack([lon, lat], axis=-1)


@jit
def mercator_projection(points: jnp.ndarray, north_pole: jnp.ndarray = None) -> jnp.ndarray:
    """Apply Mercator projection mapping spherical coordinates to a cylindrical surface.

    Parameters
    ----------
    points : jnp.ndarray
        Input 3D points of shape (..., 3).
    north_pole : jnp.ndarray, optional
        North pole direction for projection. If None, uses standard z-axis. Default is None.

    Returns
    -------
    jnp.ndarray
        Projected coordinates (lon, merc_y), shape (..., 2).
    """
    norm_points = normalize(points)
    if north_pole is not None:
        norm_points = _rotate_to_align_with_z(norm_points, north_pole)

    x, y, z = norm_points[..., 0], norm_points[..., 1], norm_points[..., 2]
    lon = jnp.arctan2(y, x)
    lat = jnp.arcsin(jnp.clip(z, -0.9999, 0.9999))
    merc_y = jnp.log(jnp.tan((jnp.pi / 4) + (lat / 2)))
    return jnp.stack([lon, merc_y], axis=-1)


@jit
def lambert_azimuthal_projection(points: jnp.ndarray, north_pole: jnp.ndarray = None) -> jnp.ndarray:
    """Lambert azimuthal equal-area projection centered at the north pole."""
    norm_points = normalize(points)
    if north_pole is not None:
        norm_points = _rotate_to_align_with_z(norm_points, north_pole)

    x, y, z = norm_points[..., 0], norm_points[..., 1], norm_points[..., 2]
    k = jnp.sqrt(2.0 / (1.0 + jnp.clip(z, -1.0, 1.0)))
    return jnp.stack([k * x, k * y], axis=-1)


def project_to_2d(points: jnp.ndarray, method: str = "orthographic", **kwargs) -> jnp.ndarray:
    """Project 3D points to 2D using a specified projection method.

    Parameters
    ----------
    points : jnp.ndarray
        Input 3D points of shape (..., 3).
    method : str, optional
        Projection method: "orthographic", "stereographic", "equirectangular",
        "mercator", "lambert", "axis_plane", or "plane". Default is "orthographic".
    **kwargs
        Additional arguments for specific methods:
        - "north_pole": jnp.ndarray for spherical projections
        - "drop_axis": str ("x", "y", "z") for "axis_plane" method
        - "normal": jnp.ndarray for "plane" method

    Returns
    -------
    jnp.ndarray
        Projected 2D points, shape (..., 2).

    Raises
    ------
    ValueError
        If method is unknown or required kwargs are missing.
    """
    if method == "axis_plane":
        drop_axis = kwargs.get("drop_axis", None)
        if drop_axis not in ["x", "y", "z"]:
            raise ValueError("Provide 'drop_axis' as 'x', 'y', or 'z' for 'axis_plane'.")
        if drop_axis == "x":
            return project_to_yz_plane(points)
        if drop_axis == "y":
            return project_to_xz_plane(points)
        if drop_axis == "z":
            return project_to_xy_plane(points)

    elif method == "plane":
        normal = kwargs.get("normal", None)
        if normal is None:
            raise ValueError("For 'plane' projection, you must provide 'normal' vector.")
        return project_to_plane(points, normal=normal)[..., :2]

    elif method == "orthographic":
        return orthographic_projection(points)

    elif method == "stereographic":
        north_pole = kwargs.get("north_pole", None)
        return stereographic_projection(points, north_pole=north_pole)

    elif method == "equirectangular":
        north_pole = kwargs.get("north_pole", None)
        return equirectangular_projection(points, north_pole=north_pole)

    elif method == "mercator":
        north_pole = kwargs.get("north_pole", None)
        return mercator_projection(points, north_pole=north_pole)

    elif method == "lambert":
        north_pole = kwargs.get("north_pole", None)
        return lambert_azimuthal_projection(points, north_pole=north_pole)

    raise ValueError(f"Unknown projection method: {method}")
