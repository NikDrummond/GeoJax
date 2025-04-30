# projection.py
# Geometric projections using JAX
# Includes: reject_axis, project_to_sphere, project_to_vector, project_to_plane

from jax import jit, lax
import jax.numpy as jnp
from functools import partial
from .core import magnitude, normalize, reject


@partial(jit, static_argnames=["axis", "squash"])
def reject_axis(vectors: jnp.ndarray, axis: int, squash: bool = False) -> jnp.ndarray:
    """
    Reject a component of the vector(s) along a specified axis.

    Parameters
    ----------
    vectors : jnp.ndarray
        Input vector or array of vectors (..., 3).
    axis : int
        Axis to reject (0=x, 1=y, 2=z).
    squash : bool
        If True, removes the specified axis. If False, zeroes out that component.

    Returns
    -------
    jnp.ndarray
        Modified vectors with the specified axis removed or zeroed.
    """
    if squash:
        return vectors[..., jnp.array([i for i in range(3) if i != axis])]
    else:
        return vectors.at[..., axis].set(0.0)


@jit
def project_to_sphere(
    points: jnp.ndarray, radius: float, center: jnp.ndarray
) -> jnp.ndarray:
    """
    Project a set of points onto a sphere of given radius and center.

    Parameters
    ----------
    points : jnp.ndarray
        Input array of points (..., 3).
    radius : float
        Radius of the target sphere.
    center : jnp.ndarray
        Sphere center (3,).

    Returns
    -------
    jnp.ndarray
        Points projected onto the sphere surface.
    """
    shifted = points - center
    norm = magnitude(shifted)[..., None]
    scaled = (radius / (norm + 1e-10)) * shifted
    return scaled + center


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
    onto_unit = normalize(onto)
    projection_length = jnp.sum(vector * onto_unit, axis=-1, keepdims=True)
    return projection_length * onto_unit


@jit
def project_to_plane(vector: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """
    Project a vector onto a plane defined by a normal vector.

    Parameters
    ----------
    vector : jnp.ndarray
        Input vector or array of vectors (..., 3).
    normal : jnp.ndarray
        Plane normal vector (3,).

    Returns
    -------
    jnp.ndarray
        Projection of the input onto the plane (same shape as input).
    """
    return reject(vector, normal)


@jit
def project_to_xy_plane(points: jnp.ndarray) -> jnp.ndarray:
    """
    Drop Z-component to project onto the XY plane.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3).

    Returns
    -------
    jnp.ndarray
        Projected points (..., 2).
    """
    return points[..., :2]


@jit
def project_to_xz_plane(points: jnp.ndarray) -> jnp.ndarray:
    """
    Drop Y-component to project onto the XZ plane.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3).

    Returns
    -------
    jnp.ndarray
        Projected points (..., 2).
    """
    return points[..., [0, 2]]


@jit
def project_to_yz_plane(points: jnp.ndarray) -> jnp.ndarray:
    """
    Drop X-component to project onto the YZ plane.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3).

    Returns
    -------
    jnp.ndarray
        Projected points (..., 2).
    """
    return points[..., 1:]


@jit
def orthographic_projection(points: jnp.ndarray) -> jnp.ndarray:
    """
    Orthographic projection onto the XY plane.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3).

    Returns
    -------
    jnp.ndarray
        Projected 2D points (..., 2).
    """
    return points[..., :2]


@jit
def stereographic_projection(points: jnp.ndarray) -> jnp.ndarray:
    """
    Stereographic projection from the north pole of the unit sphere onto the plane z=0.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3), assumed to lie on a unit sphere.

    Returns
    -------
    jnp.ndarray
        Projected points (..., 2).
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    denom = jnp.clip(1.0 - z, 1e-6, jnp.inf)
    return jnp.stack([x / denom, y / denom], axis=-1)


@jit
def equirectangular_projection(points: jnp.ndarray) -> jnp.ndarray:
    """
    Equirectangular projection (longitude, latitude).

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3), assumed to lie on a unit sphere.

    Returns
    -------
    jnp.ndarray
        (longitude, latitude) in radians (..., 2).
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    lon = jnp.arctan2(y, x)
    lat = jnp.arcsin(jnp.clip(z, -1.0, 1.0))
    return jnp.stack([lon, lat], axis=-1)


@jit
def mercator_projection(points: jnp.ndarray) -> jnp.ndarray:
    """
    Mercator projection of spherical coordinates.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3), assumed to lie on a unit sphere.

    Returns
    -------
    jnp.ndarray
        (longitude, Mercator Y) coordinates (..., 2).
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    lon = jnp.arctan2(y, x)
    lat = jnp.arcsin(jnp.clip(z, -0.9999, 0.9999))
    merc_y = jnp.log(jnp.tan((jnp.pi / 4) + (lat / 2)))
    return jnp.stack([lon, merc_y], axis=-1)


@jit
def lambert_azimuthal_projection(points: jnp.ndarray) -> jnp.ndarray:
    """
    Lambert azimuthal equal-area projection.

    Parameters
    ----------
    points : jnp.ndarray
        Input points (..., 3), assumed to lie on a unit sphere.

    Returns
    -------
    jnp.ndarray
        Projected 2D coordinates (..., 2).
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    k = jnp.sqrt(2.0 / (1.0 + jnp.clip(z, -1.0, 1.0)))
    return jnp.stack([k * x, k * y], axis=-1)


def project_to_2d(
    points: jnp.ndarray, method: str = "orthographic", **kwargs
) -> jnp.ndarray:
    """
    Project 3D points to 2D using a specified method.

    Parameters
    ----------
    points : jnp.ndarray
        Array of shape (..., 3).
    method : str
        Projection method:
        - "xy_plane", "xz_plane", "yz_plane"
        - "axis_plane" (drop specified axis: 'x', 'y', or 'z')
        - "plane" (custom plane normal required)
        - "orthographic"
        - "stereographic"
        - "equirectangular"
        - "mercator"
        - "lambert"
    **kwargs
        Extra arguments depending on method:
        - drop_axis : str, for 'axis_plane'
        - normal : jnp.ndarray, for 'plane'

    Returns
    -------
    jnp.ndarray
        Projected 2D points (..., 2).
    """
    if method == "axis_plane":
        drop_axis = kwargs.get("drop_axis", None)
        if drop_axis not in ["x", "y", "z"]:
            raise ValueError(
                "For 'axis_plane' projection you must provide 'drop_axis' as 'x', 'y', or 'z'."
            )
        if drop_axis == "x":
            return project_to_yz_plane(points)
        elif drop_axis == "y":
            return project_to_xz_plane(points)
        elif drop_axis == "z":
            return project_to_xy_plane(points)

    elif method == "plane":
        normal = kwargs.get("normal", None)
        if normal is None:
            raise ValueError(
                "For 'plane' projection, you must provide 'normal' vector."
            )
        return project_to_plane(points, normal=normal)[..., :2]

    elif method == "orthographic":
        return orthographic_projection(points)

    elif method == "stereographic":
        return stereographic_projection(points)

    elif method == "equirectangular":
        return equirectangular_projection(points)

    elif method == "mercator":
        return mercator_projection(points)

    elif method == "lambert":
        return lambert_azimuthal_projection(points)

    else:
        raise ValueError(f"Unknown projection method: {method}")
