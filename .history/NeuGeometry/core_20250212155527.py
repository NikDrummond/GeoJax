from jax import jit, lax
import jax.numpy as jnp
import jax.random as jr


@jit
def normalise(arr: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalizes a 1D or 2D array using the L2 norm, avoiding division by zero.

    Parameters
    ----------
    arr : jnp.ndarray
        Input array to be normalized.
    eps : float, optional
        Small constant to prevent division by zero (default is 1e-12).

    Returns
    -------
    jnp.ndarray
        Normalized array with the same shape as the input.
    """
    assert arr.ndim in {1, 2}, "Input arr must be 1D or 2D"
    norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, eps)  # Avoid division by zero
    return arr / norm


@jit
def magnitude(arr: jnp.ndarray) -> jnp.ndarray:
    """Calculate the Euclidean norm (magnitude) of a given array.

    Parameters
    ----------
    arr : jnp.ndarray
        Input array, expected to be either 1D or 2D.

    Returns
    -------
    jnp.ndarray
        The magnitude of the input array.
    """
    if arr.ndim not in {1, 2}:
        return jnp.full((), jnp.nan)  # Return a scalar NaN instead of full_like(arr)

    return jnp.sqrt(
        jnp.sum(arr**2, axis=-1)
    )  # Equivalent to jnp.linalg.norm but faster


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
) -> jnp.ndarray | float:
    """
    Compute the angle between two vectors, optionally projecting them onto a plane defined by a normal vector.

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
    jnp.ndarray | float
        The angle between v1 and v2. Returns a scalar if both inputs are 1D, otherwise an array.
    """

    # Record original shape to determine return type
    orig_v1_is_1d = v1.ndim == 1
    orig_v2_is_1d = v2.ndim == 1

    # If a plane_normal is provided, apply rejection.
    if plane_normal is not None:
        v1 = reject(v1, plane_normal)
        v2 = reject(v2, plane_normal)

    # Convert both to at least 2D (ensures broadcasting works correctly)
    v1 = jnp.atleast_2d(v1)
    v2 = jnp.atleast_2d(v2)

    # Ensure broadcastability along the first axis
    if (v1.shape[0] != v2.shape[0]) and (v1.shape[0] != 1) and (v2.shape[0] != 1):
        raise ValueError(
            "v1 and v2 must have the same number of rows or be broadcastable."
        )

    # Handle broadcasting by expanding the singleton dimension if necessary
    if v1.shape[0] == 1:
        v1 = jnp.broadcast_to(v1, v2.shape)
    if v2.shape[0] == 1:
        v2 = jnp.broadcast_to(v2, v1.shape)

    # Compute dot product along the last axis
    dot_products = jnp.sum(v1 * v2, axis=-1)

    # Compute magnitudes without keeping dimensions
    magnitudes = magnitude(v1) * magnitude(v2)

    # Compute cosines, handle normalization errors
    cosines = jnp.where(
        assume_normalized, dot_products, dot_products / (magnitudes + 1e-10)
    )

    # Compute the angle (in radians) and clip to safe numerical range
    angles = jnp.arccos(jnp.clip(cosines, -1.0, 1.0))

    # Convert to degrees if requested
    angles = lax.cond(to_degree, lambda a: jnp.degrees(a), lambda a: a, angles)

    # Ensure correct return type
    if orig_v1_is_1d and orig_v2_is_1d:
        return angles.item()  # Return a float for single vectors
    return angles  # Return a 1D array otherwise


@jit
def signed_angle(
    v1: jnp.ndarray, v2: jnp.ndarray, plane_normal: jnp.ndarray, to_degree: bool = False
) -> jnp.ndarray | float:
    """
    Compute the signed angle between two vectors relative to a specified plane.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector or batch of vectors.
    v2 : jnp.ndarray
        Second vector or batch of vectors.
    plane_normal : jnp.ndarray
        Normal vector (or batch of normals) defining the reference plane.
    to_degree : bool, optional
        If True, returns the angle in degrees; otherwise in radians, by default False.

    Returns
    -------
    jnp.ndarray | float
        The signed angle between v1 and v2. Returns a scalar for single vectors or an array for multiple vectors.
    """

    # Record original dimensionality
    orig_v1_is_1d = v1.ndim == 1
    orig_v2_is_1d = v2.ndim == 1
    orig_normal_is_1d = plane_normal.ndim == 1

    # Convert all inputs to at least 2D
    v1 = jnp.atleast_2d(v1)
    v2 = jnp.atleast_2d(v2)
    plane_normal = jnp.atleast_2d(plane_normal)

    # Ensure broadcastability along the first axis
    if (v1.shape[0] != v2.shape[0] or v1.shape[0] != plane_normal.shape[0]) and (
        v1.shape[0] != 1 and v2.shape[0] != 1 and plane_normal.shape[0] != 1
    ):
        raise ValueError(
            "v1, v2, and plane_normal must be broadcastable along the first axis."
        )

    # Broadcast each input to match the maximum batch size
    if v1.shape[0] == 1:
        v1 = jnp.broadcast_to(
            v1, (max(v2.shape[0], plane_normal.shape[0]), v1.shape[1])
        )
    if v2.shape[0] == 1:
        v2 = jnp.broadcast_to(
            v2, (max(v1.shape[0], plane_normal.shape[0]), v2.shape[1])
        )
    if plane_normal.shape[0] == 1:
        plane_normal = jnp.broadcast_to(
            plane_normal, (max(v1.shape[0], v2.shape[0]), plane_normal.shape[1])
        )

    # Compute the cross product between v1 and v2
    cross_prod = jnp.cross(v1, v2)

    # Compute the signed component using dot product with the plane normal
    dot_val = jnp.sum(cross_prod * plane_normal, axis=-1)
    sign = jnp.sign(dot_val)

    # Replace zeros (collinear case) with +1
    sign = jnp.where(sign == 0, 1, sign)

    # Compute the unsigned angle
    unsigned_angle = angle(v1, v2, plane_normal=plane_normal, to_degree=to_degree)
    result = sign * unsigned_angle

    # Ensure correct return type
    if orig_v1_is_1d and orig_v2_is_1d and orig_normal_is_1d:
        return result.item()  # Return scalar float
    return result  # Return 1D array otherwise
