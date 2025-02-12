from jax import jit, lax
import jax.numpy as jnp
from typing import Union

__all__ = ['_norm_jit', 'normalise','_mag_jit','magnitude']

@jit
def _norm_jit(arr: jnp.ndarray) -> jnp.ndarray:


    # return nans if not one or two dimensions!
    if arr.ndim not in {1, 2}:
        return jnp.full_like(arr, jnp.nan)

    norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / norm

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
    if arr.ndim not in {1, 2}:
        raise ValueError('Input array must be either a single vector, or a stack of vectors')
    
    return _norm_jit(arr)


@jit
def _mag_jit(arr: jnp.ndarray) -> jnp.ndarray:
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

def magnitude(arr:)

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
    # Case 2: v1 in a single vector
    cond2 = v1.shape[0] == 1
    # case 3: v2 is a single vector
    cond3 = v2.shape[0] == 1

    # Compute pairwise distances where valid
    distances = jnp.where(
        cond1 | cond2 | cond3,
        jnp.linalg.norm(v1 - v2, axis=1),
        jnp.full(
            (max(v1.shape[0], v2.shape[0]),), jnp.nan
        ),  # Return NaNs for invalid cases
    )
    return distances

@jit
def reject(v: jnp.ndarray, from_v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the rejection of the input vector v from the reference vector from_v.

    The function supports several broadcasting modes:
      - If both v and from_v are single vectors (1D arrays), a single vector is returned.
      - If one of v or from_v is a single vector and the other is a 2D array (a stack of vectors),
        the single vector is broadcast to each row.
      - If both are 2D arrays (i.e. stacks of vectors), then they are interpreted as pairwise inputs
        (so that v[i] is rejected from from_v[i]). If the batch dimensions differ and neither is 1,
        a ValueError is raised.

    Parameters
    ----------
    v : jnp.ndarray
        The vector (or stack of vectors) to be decomposed.
    from_v : jnp.ndarray
        The reference vector (or stack of vectors) used for the rejection.

    Returns
    -------
    jnp.ndarray
        The component of v that is orthogonal to from_v.
        * If both inputs are single vectors, a 1D vector is returned.
        * Otherwise, a 2D array is returned.
    """
    # Save whether the inputs were originally 1D.
    orig_v_is_1d = v.ndim == 1
    orig_from_v_is_1d = from_v.ndim == 1

    # Convert both inputs to at least 2D.
    v = jnp.atleast_2d(v)  # Now shape is (a, N)
    from_v = jnp.atleast_2d(from_v)  # Now shape is (b, N)

    # If both inputs are multi-vector (i.e. more than one row) but their batch sizes differ,
    # we cannot do pairwise rejection. (We allow one of them to be a singleton.)
    if (v.shape[0] != from_v.shape[0]) and (v.shape[0] != 1) and (from_v.shape[0] != 1):
        raise ValueError(
            "If both v and from_v are multi-vector, they must have the same number of rows."
        )

    # Compute dot products along the last axis.
    dot_v_from_v = jnp.sum(v * from_v, axis=-1, keepdims=True)
    dot_from_v_from_v = jnp.sum(from_v * from_v, axis=-1, keepdims=True)

    # Compute the projection of v onto from_v.
    projection = (dot_v_from_v / (dot_from_v_from_v + 1e-10)) * from_v

    # The rejection is v minus its projection onto from_v.
    result = v - projection

    # If both inputs were originally 1D, squeeze out the added batch dimension.
    if orig_v_is_1d and orig_from_v_is_1d:
        result = result[0]

    return result

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

    Broadcasting behavior:
      - If both v1 and v2 are 1D, the result is a scalar.
      - If one is 1D and the other is a batch (2D), the 1D vector is broadcast to all rows.
      - If both are 2D, pairwise angles are computed. If their batch sizes differ (and neither is 1), a ValueError is raised.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector or batch of vectors.
    v2 : jnp.ndarray
        Second vector or batch of vectors.
    plane_normal : jnp.ndarray | None, optional
        Normal vector of the plane to project v1 and v2 onto, by default None.
    assume_normalized : bool, optional
        Indicates whether the input vectors are already normalized, by default False.
    to_degree : bool, optional
        If True, returns the angle in degrees; otherwise in radians, by default False.

    Returns
    -------
    Union[jnp.ndarray, float]
        The angle between v1 and v2. Returns a scalar if the inputs are single vectors; otherwise an array.
    """

    # Record original 1D status.
    orig_v1_is_1d = v1.ndim == 1
    orig_v2_is_1d = v2.ndim == 1

    # If a plane_normal is provided, apply rejection.
    if plane_normal is not None:
        v1 = reject(v1, plane_normal)
        v2 = reject(v2, plane_normal)

    # Convert to at least 2D.
    v1 = jnp.atleast_2d(v1)
    v2 = jnp.atleast_2d(v2)

    # Ensure the two inputs are broadcastable along the first axis.
    if (v1.shape[0] != v2.shape[0]) and (v1.shape[0] != 1) and (v2.shape[0] != 1):
        raise ValueError(
            "v1 and v2 must be broadcastable along the first axis (or have the same number of rows)."
        )

    # Broadcast v1 and v2 to a common shape.
    # v1, v2 = jnp.broadcast_arrays(v1, v2)

    # Compute the dot products along the feature axis.
    dot_products = jnp.sum(v1 * v2, axis=-1, keepdims = True)

    # Compute magnitudes and the cosine of the angle.
    magnitudes = magnitude(v1) * magnitude(v2)
    cosines = jnp.where(
        assume_normalized, dot_products, dot_products / (magnitudes + 1e-10)
    )

    # Compute the angle (in radians) and clip to safe numerical range.
    angles = jnp.arccos(jnp.clip(cosines, -1.0, 1.0))

    # Convert to degrees if requested, using lax.cond for JIT-compatibility.
    angles = lax.cond(to_degree, lambda a: jnp.degrees(a), lambda a: a, angles)

    # If both inputs were originally single vectors, return a scalar.
    if orig_v1_is_1d and orig_v2_is_1d:
        return angles[0]
    return angles


@jit
def signed_angle(
    v1: jnp.ndarray, v2: jnp.ndarray, plane_normal: jnp.ndarray, to_degree: bool = False
) -> Union[jnp.ndarray, float]:
    """
    Compute the signed angle between two vectors relative to a specified plane.

    Broadcasting behavior:
      - If v1, v2, and plane_normal are each single vectors (1D), a scalar is returned.
      - If one or more are 1D and the others are batches (2D), broadcasting occurs.
      - If all are 2D, pairwise signed angles are computed. If their batch sizes differ (and none is a singleton),
        a ValueError is raised.

    Parameters
    ----------
    v1 : jnp.ndarray
        The first input vector or batch of vectors.
    v2 : jnp.ndarray
        The second input vector or batch of vectors.
    plane_normal : jnp.ndarray
        The normal vector (or batch of normals) defining the reference plane.
    to_degree : bool, optional
        If True, the result is returned in degrees; otherwise in radians, by default False.

    Returns
    -------
    Union[jnp.ndarray, float]
        The signed angle between v1 and v2. Returns a scalar for single vectors or an array for multiple vectors.
    """
    # Record original dimensionality.
    orig_v1_is_1d = v1.ndim == 1
    orig_v2_is_1d = v2.ndim == 1
    orig_normal_is_1d = plane_normal.ndim == 1

    # Convert all inputs to at least 2D.
    v1 = jnp.atleast_2d(v1)
    v2 = jnp.atleast_2d(v2)
    plane_normal = jnp.atleast_2d(plane_normal)

    # Ensure broadcastability along the first axis.
    if ((v1.shape[0] != v2.shape[0]) or (v1.shape[0] != plane_normal.shape[0])) and (
        v1.shape[0] != 1 and v2.shape[0] != 1 and plane_normal.shape[0] != 1
    ):
        raise ValueError(
            "v1, v2, and plane_normal must be broadcastable along the first axis."
        )

    # Broadcast to a common shape.
    v1, v2, plane_normal = jnp.broadcast_arrays(v1, v2, plane_normal)

    # Compute the cross product between v1 and v2.
    cross_prod = jnp.cross(v1, v2)
    # The dot of the cross product with the plane normal gives the sign.
    dot_val = jnp.sum(cross_prod * plane_normal, axis=-1)
    sign = jnp.sign(dot_val)
    # Replace zeros (collinear case) with +1.
    sign = jnp.where(sign == 0, 1, sign)

    # Compute the unsigned angle.
    unsigned_angle = angle(v1, v2, plane_normal=plane_normal, to_degree=to_degree)
    result = sign * unsigned_angle

    # If all inputs were originally 1D, return a scalar.
    if orig_v1_is_1d and orig_v2_is_1d and orig_normal_is_1d:
        return result[0]
    return result
