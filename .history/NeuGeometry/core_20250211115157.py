from jax import jit
import jax.numpy as jnp


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

def angle(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    plane_normal: Optional[jnp.ndarray] = None,
    assume_normalized: bool = False,
    return_degree: str = "deg"
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
