# core.py
# Low-level vector operations (JAX-compatible)

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
    """Reject v from from_v (i.e., subtract projection of v onto from_v)."""
    proj = project_to_vector(v, from_v)
    return v - proj

@jit
def project_to_vector(v: jnp.ndarray, onto: jnp.ndarray) -> jnp.ndarray:
    """Project v onto vector 'onto'."""
    onto_unit = normalize(onto)
    proj_len = jnp.sum(v * onto_unit, axis=-1, keepdims=True)
    return proj_len * onto_unit

@jit
def scalar_triple(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray) -> jnp.ndarray:
    """Compute the scalar triple product (v1 . (v2 x v3))."""
    return jnp.sum(v1 * cross(v2, v3), axis=-1)

@jit
def reflect(v: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """
    Reflect vector v across a surface with the given normal.
    """
    return v - 2 * project_to_vector(v, normal)


@jit
def _scale_along_basis_jit(vectors: jnp.ndarray, scale: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """
    Core JIT-compatible function to scale vectors along given basis directions.
    """
    # Project onto basis
    projections = jnp.einsum("nd,bd->nb", vectors, basis)

    # Scale each projection
    scaled_projections = projections * scale

    # Transform back to original space
    return scaled_projections @ basis

def scale_along_basis(
    vectors: jnp.ndarray,
    scale: jnp.ndarray = 1.0,
    basis: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Scale vectors along a given set of basis directions.

    Parameters
    ----------
    vectors : jnp.ndarray
        Array of shape (N, D), each row a vector.
    scale : float or jnp.ndarray
        Scalar or array of shape (D,) for scaling along each axis.
    basis : jnp.ndarray, optional
        Orthonormal basis of shape (D, D). Defaults to Euclidean basis.

    Returns
    -------
    jnp.ndarray
        Scaled vectors of shape (N, D).
    """
    D = vectors.shape[1]

    # Handle default basis
    if basis is None:
        basis = jnp.eye(D)

    # Broadcast scale to (D,) shape
    scale = jnp.broadcast_to(jnp.asarray(scale), (D,))

    # Call JIT-compiled function
    return _scale_along_basis_jit(vectors, scale, basis)

@jit
def gram_schmidt(vectors: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Gram-Schmidt process to a 2x3 or 3x3 stack of vectors to obtain an orthonormal basis.

    Parameters
    ----------
    vectors : jnp.ndarray
        A (2, 3) or (3, 3) array of linearly independent vectors.

    Returns
    -------
    jnp.ndarray
        Orthonormalized vectors with the same shape as input.
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
def apply_affine(points: jnp.ndarray, matrix: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
    """
    Apply an affine transformation to a batch of points.

    Parameters
    ----------
    points : jnp.ndarray
        Points of shape (N, 3).
    matrix : jnp.ndarray
        Rotation or scale matrix of shape (3, 3).
    offset : jnp.ndarray
        Translation vector of shape (3,).

    Returns
    -------
    jnp.ndarray
        Transformed points of shape (N, 3).
    """
    return jnp.dot(points, matrix.T) + offset

@jit
def point_to_plane_distance(point: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray) -> jnp.ndarray:
    """
    Compute signed distance from point to plane.

    Parameters
    ----------
    point : jnp.ndarray
        Point of interest (3,).
    plane_point : jnp.ndarray
        A point on the plane (3,).
    plane_normal : jnp.ndarray
        Normal vector of the plane (3,).

    Returns
    -------
    jnp.ndarray
        Signed distance.
    """
    return jnp.dot(point - plane_point, normalize(plane_normal))

@jit
def ray_plane_intersect(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray) -> jnp.ndarray:
    """
    Compute intersection point of a ray and a plane.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Ray origin (3,).
    ray_dir : jnp.ndarray
        Direction of the ray (3,), assumed normalized.
    plane_point : jnp.ndarray
        A point on the plane (3,).
    plane_normal : jnp.ndarray
        Plane's normal (3,).

    Returns
    -------
    jnp.ndarray
        Point of intersection.
    """
    denom = jnp.dot(ray_dir, plane_normal)
    d = jnp.dot(plane_point - ray_origin, plane_normal) / (denom + 1e-10)
    return ray_origin + d * ray_dir

@jit
def tetrahedron_volume(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Compute volume of a tetrahedron given four vertices.

    Parameters
    ----------
    a, b, c, d : jnp.ndarray
        Vertices of the tetrahedron (3,).

    Returns
    -------
    jnp.ndarray
        Volume (scalar).
    """
    return jnp.abs(scalar_triple(b - a, c - a, d - a)) / 6.0