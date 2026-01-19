"""Core vector utilities (JAX-compatible)."""

from jax import jit
import jax.numpy as jnp


@jit
def magnitude(v: jnp.ndarray) -> jnp.ndarray:
    """Compute the L2 norm (Euclidean length) of vectors."""
    return jnp.linalg.norm(v, axis=-1)


@jit
def normalize(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize vectors using their L2 norm."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)


@jit
def dot(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Normalized dot product between vectors."""
    return jnp.sum(normalize(v1) * normalize(v2), axis=-1)


@jit
def cross(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """3D cross product between vectors."""
    return jnp.cross(v1, v2)


@jit
def reject(v: jnp.ndarray, from_v: jnp.ndarray) -> jnp.ndarray:
    """Reject v from from_v (subtract projection of v onto from_v)."""
    from ..projection.projection import project_to_vector  # avoid circular import at module load
    proj = project_to_vector(v, from_v)
    return v - proj


@jit
def scalar_triple(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray) -> jnp.ndarray:
    """Compute the scalar triple product (v1 . (v2 x v3))."""
    return jnp.sum(v1 * cross(v2, v3), axis=-1)


@jit
def reflect(v: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """Reflect vector v across a surface with the given normal."""
    from ..projection.vector import project_to_vector  # avoid circular import at module load
    return v - 2 * project_to_vector(v, normal)


@jit
def _scale_along_basis_jit(vectors: jnp.ndarray, scale: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """JIT core for scaling vectors along provided basis directions."""
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
    """Apply Gram-Schmidt to obtain an orthonormal basis."""
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
    """Scale a point cloud based on robust percentile extent along an axis."""
    values = points[:, axis]
    q_low = jnp.percentile(values, lower)
    q_high = jnp.percentile(values, upper)
    robust_extent = q_high - q_low
    scale = scale_factor / (robust_extent + eps)
    return points * scale


@jit
def apply_affine(points: jnp.ndarray, matrix: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
    """Apply an affine transformation to a batch of points."""
    return jnp.dot(points, matrix.T) + offset


@jit
def point_to_plane_distance(point: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray) -> jnp.ndarray:
    """Compute signed distance from a point to a plane."""
    return jnp.dot(point - plane_point, normalize(plane_normal))


@jit
def ray_plane_intersect(
    ray_origin: jnp.ndarray,
    ray_dir: jnp.ndarray,
    plane_point: jnp.ndarray,
    plane_normal: jnp.ndarray,
) -> jnp.ndarray:
    """Compute intersection point of a ray and a plane."""
    denom = jnp.dot(ray_dir, plane_normal)
    d = jnp.dot(plane_point - ray_origin, plane_normal) / (denom + 1e-10)
    return ray_origin + d * ray_dir


@jit
def tetrahedron_volume(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Compute volume of a tetrahedron from its vertices."""
    return jnp.abs(scalar_triple(b - a, c - a, d - a)) / 6.0
