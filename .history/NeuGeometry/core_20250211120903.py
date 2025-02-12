from jax import jit
import jax.numpy as jnp
from typing import Union


@jit
def normalise(arr: jnp.ndarray) -> jnp.ndarray:
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
def magnitude(arr: jnp.ndarray) -> jnp.ndarray:
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
    norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
    return norm

@jit
def pairwise_euclidean(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Calculate the pairwise Euclidean distance between two sets of points.

    Parameters
    ----------
    v1 : jnp.ndarray
        An array representing a single point or a collection of points.
    v2 : jnp.ndarray
        An array representing a single point or a collection of points.

    Returns
    -------
    jnp.ndarray
        An array of Euclidean distances between corresponding points in v1 and v2.
        If both inputs are 2D and have different numbers of rows, returns NaNs.
    """
    v1, v2 = jnp.atleast_2d(v1), jnp.atleast_2d(v2)

    # Case 1: Both 2D with matching rows
    cond1 = v1.shape[0] == v2.shape[0]
    cond2 = v1.shape[0] == 1
    cond3 = v2.shape[0] == 1

    # Compute pairwise distances where valid
    distances = jnp.where(
        cond1 | cond2 | cond3, 
        jnp.linalg.norm(v1 - v2, axis=1), 
        jnp.full((max(v1.shape[0], v2.shape[0]),), jnp.nan)  # Return NaNs for invalid cases
    )
    return distances

@jit
def reject(v: jnp.ndarray, from_v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the rejection of the input vector v from the reference vector from_v.

    Parameters
    ----------
    v : jnp.ndarray
        The vector to be decomposed.
    from_v : jnp.ndarray
        The reference vector used to calculate the rejection.

    Returns
    -------
    jnp.ndarray
        The component of v that is orthogonal to from_v.
    """
    # Ensure both v and from_v are at least 2D for consistent dot product behavior
    v, from_v = jnp.atleast_2d(v), jnp.atleast_2d(from_v)

    # Compute dot products
    dot_v_from_v = jnp.sum(v * from_v, axis=-1, keepdims=True)  # Equivalent to jnp.dot(v, from_v)
    dot_from_v_from_v = jnp.sum(from_v * from_v, axis=-1, keepdims=True)

    # Avoid division by zero by adding a small epsilon
    projection = (dot_v_from_v / (dot_from_v_from_v + 1e-10)) * from_v

    return v - projection

@jit
def angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: jnp.ndarray | None = None,
    assume_normalized: bool = False,
    to_degree: bool = False,
) -> Union[jnp.ndarray, float]:
    """
    Compute the angle between two vectors, optionally projecting them onto a plane defined by a normal vector.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector.
    v2 : jnp.ndarray
        Second vector.
    plane_normal : jnp.ndarray | None, optional
        Normal vector of the plane to project v1 and v2 onto, by default None.
    assume_normalized : bool, optional
        Indicates whether the input vectors are already normalized, by default False.
    to_degree : bool, optional
        If True, returns the angle in degrees; otherwise in radians, by default False.

    Returns
    -------
    Union[jnp.ndarray, float]
        The angle between v1 and v2. Returns a scalar if the inputs are 1D, otherwise an array.
    """
    
    # Reject components along the plane normal if given
    if plane_normal is not None:
        v1, v2 = reject(v1, plane_normal), reject(v2, plane_normal)

    # Ensure vectors are at least 2D and have shape (N, 3)
    v1, v2 = jnp.atleast_2d(v1), jnp.atleast_2d(v2)
    
    # Compute dot product
    dot_products = jnp.sum(v1 * v2, axis=-1)

    # Normalize if required
    magnitudes = magnitude(v1) * magnitude(v2)
    cosines = jnp.where(assume_normalized, dot_products, dot_products / (magnitudes + 1e-10))

    # Compute angle and clip to prevent numerical errors
    angles = jnp.arccos(jnp.clip(cosines, -1.0, 1.0))

    # Convert to degrees if needed
    if to_degree:
        angles = jnp.degrees(angles)

    # Return scalar if inputs are 1D
    return angles if v1.shape[0] > 1 else angles[0]

def signed_angle(v1: jnp.ndarray, v2: jnp.ndarray, plane_normal: jnp.ndarray, to_degree: bool = False) -> Union[jnp.ndarray, float]:


    # Compute the cross product between v1 and v2.
    cross_prod = jnp.cross(v1, v2)
    # For each vector (or row), take the dot with `look`. If v1 and v2 are (3,)
    # vectors, jnp.sum(cross_prod * look, axis=-1) returns a scalar.
    dot_val = jnp.sum(cross_prod * plane_normal, axis=-1)
    sign = jnp.sign(dot_val)
    # Replace zeros with +1 (i.e. consider collinear cases as "clockwise")
    sign = jnp.where(sign == 0, 1, sign)

    # Compute the unsigned angle 
    unsigned_angle = angle(v1, v2, plane_normal=plane_normal, to_degree=to_degree)

    return sign * un