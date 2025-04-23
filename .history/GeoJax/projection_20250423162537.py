### projection.py
# projection.py
# Geometric projections using JAX
# Includes: reject_axis, project_to_sphere, project_to_vector, project_to_plane

from jax import jit, lax
import jax.numpy as jnp
from functools import partial
from .core import magnitude, normalize, reject

from functools import partial

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