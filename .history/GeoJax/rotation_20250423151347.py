# rotation.py
# Rotation operations in 3D space using JAX

from jax import jit, lax
import jax.numpy as jnp
from .core import magnitude

@jit
def rotation_matrix_from_rotvec(rot_vec: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a 3x3 rotation matrix from a rotation vector (axis * angle)
    using Rodrigues' rotation formula.

    Parameters
    ----------
    rot_vec : jnp.ndarray
        A 1D array of shape (3,) where the direction is the axis of rotation
        and the magnitude is the angle in radians.

    Returns
    -------
    jnp.ndarray
        A 3x3 rotation matrix.
    """
    angle = jnp.linalg.norm(rot_vec)

    def compute_rotation(_):
        axis = rot_vec / angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        one_minus_cos = 1.0 - cos_a
        
        x, y, z = axis
        cross_matrix = jnp.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])
        outer = jnp.outer(axis, axis)
        return cos_a * jnp.eye(3) + sin_a * cross_matrix + one_minus_cos * outer

    return lax.cond(angle > 1e-8, compute_rotation, lambda _: jnp.eye(3), operand=None)


@jit
def rotate_around_axis(coords: jnp.ndarray, theta: float, axis: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a set of coordinates around a given axis by a specified angle.

    Parameters
    ----------
    coords : jnp.ndarray
        An array of shape (N, 2) or (N, 3). 2D coords will be promoted to 3D.
    theta : float
        The rotation angle in radians.
    axis : jnp.ndarray
        A 3D unit vector indicating the axis of rotation.

    Returns
    -------
    jnp.ndarray
        The rotated coordinates, same shape as input.
    """
    rot_vec = axis * theta
    R = rotation_matrix_from_rotvec(rot_vec)

    is_2d = coords.shape[1] == 2
    coords_3d = jnp.concatenate([coords, jnp.zeros((coords.shape[0], 1))], axis=1) if is_2d else coords
    rotated = coords_3d @ R.T
    return rotated[:, :2] if is_2d else rotated
