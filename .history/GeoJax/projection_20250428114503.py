### projection.py
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
        Input vector or array of vectors (N, 3).
    axis : int
        Axis to reject (0=x, 1=y, 2=z).
    squash : bool
        If True, drops the axis instead of zeroing it.

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
def project_to_sphere(points: jnp.ndarray, radius: float, center: jnp.ndarray) -> jnp.ndarray:
    """
    Project a set of points onto a sphere of given radius and center.

    Parameters
    ----------
    points : jnp.ndarray
        Input array of points (N, 3).
    radius : float
        Radius of the target sphere.
    center : jnp.ndarray
        Sphere center (3,).

    Returns
    -------
    jnp.ndarray
        Points projected onto the sphere.
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
        Vector or stack of vectors (N, 3).
    onto : jnp.ndarray
        Reference vector or stack (N, 3) or (3,).

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
        Input vector or array of vectors.
    normal : jnp.ndarray
        Plane normal vector.

    Returns
    -------
    jnp.ndarray
        The component of the vector within the plane (i.e., projection onto the plane).
    """
    return reject(vector, normal)

@jit
def project_to_xy_plane(points: jnp.ndarray) -> jnp.ndarray:
    return points[..., :2]  # simply drop z

@jit
def project_to_xz_plane(points: jnp.ndarray) -> jnp.ndarray:
    return points[..., [0, 2]]  # drop y

@jit
def project_to_yz_plane(points: jnp.ndarray) -> jnp.ndarray:
    return points[..., 1:]  # drop x

@jit
def orthographic_projection(points: jnp.ndarray) -> jnp.ndarray:
    # Assume points are on a sphere, project onto xy-plane orthographically
    return points[..., :2]

@jit
def stereographic_projection(points: jnp.ndarray) -> jnp.ndarray:
    # Assume unit sphere centered at origin
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    denom = 1.0 - z
    denom = jnp.where(denom < 1e-6, 1e-6, denom)  # Avoid division by zero
    return jnp.stack([x / denom, y / denom], axis=-1)

# Future: Lambert, etc.

# -------------------------------
# Dispatcher
# -------------------------------

def project_to_2d(points: jnp.ndarray, method: str = "orthographic", **kwargs) -> jnp.ndarray:
    """
    Project 3D points to 2D using a specified method.

    Parameters
    ----------
    points : jnp.ndarray
        Array of shape (..., 3)
    method : str
        Projection method:
        - "xy_plane", "xz_plane", "yz_plane"
        - "plane" (custom plane normal required)
        - "orthographic"
        - "stereographic"
    **kwargs
        Extra arguments depending on method.

    Returns
    -------
    jnp.ndarray
        Projected 2D points (..., 2)
    """

    # drop a specified axis
    if method == "axis_plane":
        drop_axis = kwargs.get('drop_axis', None)
        if drop_axis not in ['x','y','z']:
            raise ValueError("For 'Axis_plane' projection you must provide x,y, or z axis to drop.")
        elif drop_axis == 'x':
            return project_to_yz_plane(points)
        elif drop_axis == 'y':
            return project_to_xz_plane(points)
        elif drop_axis == 'z':
            return project_to_xy_plane(points)
    #
    elif method == "plane":
        normal = kwargs.get("normal", None)
        if normal is None:
            raise ValueError("For 'plane' projection, you must provide 'normal' vector.")
        return project_to_plane(points, normal=normal)[..., :2]  # project to plane and take 2D
    elif method == "orthographic":
        return orthographic_projection(points)
    elif method == "stereographic":
        return stereographic_projection(points)
    else:
        raise ValueError(f"Unknown projection method: {method}")