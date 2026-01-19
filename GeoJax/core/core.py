"""Core vector utilities (JAX-compatible)."""

from jax import jit
import jax.numpy as jnp


@jit
def magnitude(v: jnp.ndarray) -> jnp.ndarray:
    """Compute the L2 norm (Euclidean length) of vectors.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector(s) of shape (..., D) where D is the dimension.

    Returns
    -------
    jnp.ndarray
        Euclidean norm(s) of the input vector(s), shape (...,).
    """
    return jnp.linalg.norm(v, axis=-1)


@jit
def normalize(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize vectors using their L2 norm.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector(s) of shape (..., D).
    eps : float, optional
        Small epsilon value to prevent division by zero. Default is 1e-12.

    Returns
    -------
    jnp.ndarray
        Normalized vector(s) with the same shape as input, unit length along the last axis.
    """
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)


@jit
def dot(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Compute normalized dot product between vectors.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector(s) of shape (..., D).
    v2 : jnp.ndarray
        Second vector(s) of shape (..., D), broadcastable with v1.

    Returns
    -------
    jnp.ndarray
        Dot product(s) between normalized v1 and v2, shape (...,).
    """
    return jnp.sum(normalize(v1) * normalize(v2), axis=-1)


@jit
def cross(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Compute 3D cross product between vectors.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector(s) of shape (..., 3).
    v2 : jnp.ndarray
        Second vector(s) of shape (..., 3), broadcastable with v1.

    Returns
    -------
    jnp.ndarray
        Cross product(s) v1 × v2, shape (..., 3).
    """
    return jnp.cross(v1, v2)


@jit
def reject(v: jnp.ndarray, from_v: jnp.ndarray) -> jnp.ndarray:
    """Reject v from from_v (subtract projection of v onto from_v).

    The rejection of v from from_v is the component of v perpendicular to from_v.

    Parameters
    ----------
    v : jnp.ndarray
        Vector(s) to reject, shape (..., D).
    from_v : jnp.ndarray
        Vector(s) to reject from, shape (..., D), broadcastable with v.

    Returns
    -------
    jnp.ndarray
        Rejected vector(s), component of v perpendicular to from_v, shape (..., D).
    """
    from ..projection.projection import project_to_vector  # avoid circular import at module load
    proj = project_to_vector(v, from_v)
    return v - proj


@jit
def scalar_triple(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray) -> jnp.ndarray:
    """Compute the scalar triple product (v1 . (v2 x v3)).

    The scalar triple product gives the signed volume of the parallelepiped
    formed by the three vectors.

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector(s) of shape (..., 3).
    v2 : jnp.ndarray
        Second vector(s) of shape (..., 3), broadcastable with v1.
    v3 : jnp.ndarray
        Third vector(s) of shape (..., 3), broadcastable with v1 and v2.

    Returns
    -------
    jnp.ndarray
        Scalar triple product(s), shape (...,).
    """
    return jnp.sum(v1 * cross(v2, v3), axis=-1)


@jit
def reflect(v: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """Reflect vector v across a surface with the given normal.

    Parameters
    ----------
    v : jnp.ndarray
        Vector(s) to reflect, shape (..., D).
    normal : jnp.ndarray
        Surface normal vector(s), shape (..., D), broadcastable with v.

    Returns
    -------
    jnp.ndarray
        Reflected vector(s), shape (..., D).
    """
    from ..projection.projection import project_to_vector  # avoid circular import at module load
    return v - 2 * project_to_vector(v, normal)


@jit
def _scale_along_basis_jit(vectors: jnp.ndarray, scale: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """JIT core for scaling vectors along provided basis directions.

    Parameters
    ----------
    vectors : jnp.ndarray
        Input vectors of shape (N, D).
    scale : jnp.ndarray
        Scale factors for each basis direction, shape (D,).
    basis : jnp.ndarray
        Basis vectors, shape (D, D).

    Returns
    -------
    jnp.ndarray
        Scaled vectors, shape (N, D).
    """
    projections = jnp.einsum("nd,bd->nb", vectors, basis)
    scaled_projections = projections * scale
    return scaled_projections @ basis


def scale_along_basis(
    vectors: jnp.ndarray,
    scale: jnp.ndarray = 1.0,
    basis: jnp.ndarray = None,
) -> jnp.ndarray:
    """Scale vectors along a given set of basis directions."""
    D = vectors.shape[1]

    if basis is None:
        basis = jnp.eye(D)

    scale = jnp.broadcast_to(jnp.asarray(scale), (D,))
    return _scale_along_basis_jit(vectors, scale, basis)


@jit
def gram_schmidt(vectors: jnp.ndarray) -> jnp.ndarray:
    """Apply Gram-Schmidt process to obtain an orthonormal basis.

    Parameters
    ----------
    vectors : jnp.ndarray
        Input vectors of shape (2, 3) or (3, 3).

    Returns
    -------
    jnp.ndarray
        Orthonormal basis vectors, shape (2, 3) or (3, 3).

    Raises
    ------
    AssertionError
        If vectors shape is not (2, 3) or (3, 3).
    """
    assert vectors.shape[0] in [2, 3] and vectors.shape[1] == 3

    def step(i, val):
        q, vs = val
        v = vs[i]
        for j in range(i):
            v -= jnp.dot(v, q[j]) * q[j]
        q = q.at[i].set(normalize(v))
        return q, vs

    Q = jnp.zeros_like(vectors)
    Q = Q.at[0].set(normalize(vectors[0]))
    if vectors.shape[0] > 1:
        Q, _ = step(1, (Q, vectors))
    if vectors.shape[0] == 3:
        Q, _ = step(2, (Q, vectors))
    return Q


@jit
def scale_point_cloud_by_robust_axis_extent(
    points: jnp.ndarray,
    axis: int = 1,
    lower: float = 1.0,
    upper: float = 99.0,
    scale_factor: float = 1.0,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Scale a point cloud based on robust percentile extent along an axis.

    Uses percentiles to compute extent, making it robust to outliers.

    Parameters
    ----------
    points : jnp.ndarray
        Point cloud of shape (N, D).
    axis : int, optional
        Axis along which to compute extent. Default is 1.
    lower : float, optional
        Lower percentile for extent calculation. Default is 1.0.
    upper : float, optional
        Upper percentile for extent calculation. Default is 99.0.
    scale_factor : float, optional
        Overall scaling factor. Default is 1.0.
    eps : float, optional
        Small epsilon to prevent division by zero. Default is 1e-6.

    Returns
    -------
    jnp.ndarray
        Scaled point cloud, shape (N, D).
    """
    values = points[:, axis]
    q_low = jnp.percentile(values, lower)
    q_high = jnp.percentile(values, upper)
    robust_extent = q_high - q_low
    scale = scale_factor / (robust_extent + eps)
    return points * scale


@jit
def apply_affine(points: jnp.ndarray, matrix: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
    """Apply an affine transformation to a batch of points.

    Transforms points as: points @ matrix.T + offset.

    Parameters
    ----------
    points : jnp.ndarray
        Input points of shape (N, D).
    matrix : jnp.ndarray
        Transformation matrix of shape (D, D).
    offset : jnp.ndarray
        Translation offset of shape (D,).

    Returns
    -------
    jnp.ndarray
        Transformed points, shape (N, D).
    """
    return jnp.dot(points, matrix.T) + offset


@jit
def point_to_plane_distance(point: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray) -> jnp.ndarray:
    """Compute signed distance from a point to a plane.

    The sign indicates which side of the plane the point is on.

    Parameters
    ----------
    point : jnp.ndarray
        Point(s) of shape (..., D).
    plane_point : jnp.ndarray
        A point on the plane, shape (D,) or broadcastable.
    plane_normal : jnp.ndarray
        Normal vector of the plane, shape (D,) or broadcastable.

    Returns
    -------
    jnp.ndarray
        Signed distance(s) from point to plane, shape (...,).
    """
    return jnp.dot(point - plane_point, normalize(plane_normal))


@jit
def ray_plane_intersect(
    ray_origin: jnp.ndarray,
    ray_dir: jnp.ndarray,
    plane_point: jnp.ndarray,
    plane_normal: jnp.ndarray,
) -> jnp.ndarray:
    """Compute intersection point of a ray and a plane.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Origin point of the ray, shape (D,).
    ray_dir : jnp.ndarray
        Direction vector of the ray, shape (D,).
    plane_point : jnp.ndarray
        A point on the plane, shape (D,).
    plane_normal : jnp.ndarray
        Normal vector of the plane, shape (D,).

    Returns
    -------
    jnp.ndarray
        Intersection point, shape (D,). If ray is parallel to plane, result may be invalid.
    """
    denom = jnp.dot(ray_dir, plane_normal)
    d = jnp.dot(plane_point - ray_origin, plane_normal) / (denom + 1e-10)
    return ray_origin + d * ray_dir


@jit
def tetrahedron_volume(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Compute volume of a tetrahedron from its vertices.

    Parameters
    ----------
    a : jnp.ndarray
        First vertex, shape (3,).
    b : jnp.ndarray
        Second vertex, shape (3,).
    c : jnp.ndarray
        Third vertex, shape (3,).
    d : jnp.ndarray
        Fourth vertex, shape (3,).

    Returns
    -------
    jnp.ndarray
        Volume of the tetrahedron (scalar).
    """
    return jnp.abs(scalar_triple(b - a, c - a, d - a)) / 6.0
