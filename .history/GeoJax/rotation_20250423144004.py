## rotation.py
# rotation.py
# Rotation utilities using Rodrigues' rotation formula with JAX
# Includes: rotation_matrix_from_vector, rotate_points

from jax import jit, lax
import jax.numpy as jnp

@jit
def rotation_matrix_from_vector(rot_vec: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a 3x3 rotation matrix from a rotation vector (axis * angle).

    Parameters
    ----------
    rot_vec : jnp.ndarray
        A vector in R^3 representing axis-angle (axis * theta).

    Returns
    -------
    jnp.ndarray
        A 3x3 rotation matrix.
    """
    angle = jnp.linalg.norm(rot_vec)

    def compute():
        u = rot_vec / angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        one_minus_cos = 1.0 - cos_a

        u_cross = jnp.array([
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0]
        ])
        outer = jnp.outer(u, u)
        return cos_a * jnp.eye(3) + one_minus_cos * outer + sin_a * u_cross

    return lax.cond(angle > 1e-8, compute, lambda: jnp.eye(3))


@jit
def rotate_points(points: jnp.ndarray, theta: float, axis: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate points around an axis using an angle theta.

    Parameters
    ----------
    points : jnp.ndarray
        Points to rotate (N, 3) or a single point (3,).
    theta : float
        Rotation angle in radians.
    axis : jnp.ndarray
        Rotation axis (3,).

    Returns
    -------
    jnp.ndarray
        Rotated points with same shape as input.
    """
    rot_vec = axis * theta
    R = rotation_matrix_from_vector(rot_vec)
    pts = jnp.atleast_2d(points)
    rotated = jnp.dot(pts, R.T)
    return rotated[0] if points.ndim == 1 else rotated
