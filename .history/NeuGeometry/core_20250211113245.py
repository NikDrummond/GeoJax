from typing import Union, Optional
from numpy import ndarray
from jax import jit
import jax.numpy as jnp

from .helpers import raise_dim_error

Array = Union[jnp.ndarray, jnp.DeviceArray, ndarray]

@jit
def normalise(arr: Array) -> jnp.ndarray:
    """Normalizes a 1D or 2D array using the L2 norm.

    Parameters
    ----------
    arr : Array
        Input array to be normalized.

    Returns
    -------
    jnp.ndarray
        Normalized array with the same shape as the input.
    """

    # return nans if not one or two dimensions!
    if arr.ndim not in {1, 2}:
        return jnp.full_like(arr, jnp.nan)  

    norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / norm

@jit
def magnitude(arr: Array) -> jnp.ndarray:
    """Calculate the Euclidean norm (magnitude) of a given array.

    Parameters
    ----------
    arr : Array
        Input array, expected to be either 1D or 2D.

    Returns
    -------
    jnp.ndarray
        The magnitude of the input array.
    """
    # return nans if not one or two dimensions!
    if arr.ndim not in {1, 2}:
        return jnp.full_like(arr, jnp.nan)  
    

def pairwise_euclidean(v1: Array, v2: Array) -> jnp.ndarray:
    """Calculate the pairwise Euclidean distance between two sets of points.

    Parameters
    ----------
    v1 : Array
        An array representing a single point or a collection of points.
    v2 : Array
        An array representing a single point or a collection of points.

    Returns
    -------
    jnp.ndarray
        An array of Euclidean distances between corresponding points in v1 and v2.

    Raises
    ------
    ValueError
        If both inputs are 2D but have different numbers of rows.
    """
    v1, v2 = jnp.atleast_2d(v1), jnp.atleast_2d(v2)

    # Case 1: Both 2D with matching rows
    if v1.shape[0] == v2.shape[0]:  
        return jnp.linalg.norm(v1 - v2, axis=1)
    # Case 2: v1 is a single point, v2 is multiple points
    elif v1.shape[0] == 1:  
        return jnp.linalg.norm(v2 - v1, axis=1)
    # Case 3: v2 is a single point, v1 is multiple points
    elif v2.shape[0] == 1:  
        return jnp.linalg.norm(v1 - v2, axis=1)
    else:
        raise ValueError("If both inputs are 2D, they must have the same number of rows.")

def magnitude(v: Array) -> Array:
    """Compute the Euclidean norm of a vector or set of vectors."""
    return jnp.linalg.norm(v, axis=-1)

def reject(v: Array, from_v: Array) -> Array:
    """Rejects component of v along from_v (projection subtraction)."""
    projection = jnp.dot(v, from_v) / jnp.dot(from_v, from_v) * from_v
    return v - projection

def angle(
    v1: Array,
    v2: Array,
    plane_normal: Optional[Array] = None,
    assume_normalized: bool = False,
    units: str = "deg"
) -> Union[Array, float]:
    """Calculate the angle between two vectors (or sets of vectors) in 3D space.

    Parameters
    ----------
    v1 : Array
        First vector or set of vectors.
    v2 : Array
        Second vector or set of vectors.
    plane_normal : Array, optional
        If provided, rejects components of v1 and v2 along this normal before computing the angle.
    assume_normalized : bool, default=False
        If True, assumes input vectors are already unit vectors (skips normalization step).
    units : {"deg", "rad"}, default="deg"
        The unit of the output angle.

    Returns
    -------
    Array or float
        Angle(s) in the specified unit. Returns a scalar if both v1 and v2 are 1D.
    """
    # Ensure units are valid
    if units not in {"deg", "rad"}:
        raise ValueError(f"Unrecognized units '{units}'; expected 'deg' or 'rad'.")

    # Reject components along the normal plane if a normal is given
    if plane_normal is not None:
        v1, v2 = reject(v1, plane_normal), reject(v2, plane_normal)

    # Compute dot product for angle calculation
    dot_products = jnp.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))

    # Normalize if not assumed to be normalized
    if assume_normalized:
        cosines = dot_products
    else:
        cosines = dot_products / (magnitude(v1) * magnitude(v2))

    # Clip to avoid precision errors outside [-1, 1]
    angles = jnp.arccos(jnp.clip(cosines, -1.0, 1.0))

    # Convert to degrees if requested
    if units == "deg":
        angles = jnp.degrees(angles)

    # Return a scalar if both inputs are 1D
    return angles[0] if v1.ndim == 1 and v2.ndim == 1 else angles
