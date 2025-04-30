# rotation.py
# Rotation operations in 3D space using JAX

from jax import jit, lax
import jax.numpy as jnp
from .core import magnitude, normalize

from jax import jit, lax
import jax.numpy as jnp
from .core import normalize, cross, dot, magnitude

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

@jit
def rotation_matrix_between_vectors(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the rotation matrix that aligns vector v1 to v2.

    Parameters
    ----------
    v1, v2 : jnp.ndarray
        3D vectors.

    Returns
    -------
    jnp.ndarray
        3x3 rotation matrix.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    cross_prod = cross(v1, v2)
    dot_prod = dot(v1, v2)
    I = jnp.eye(3)

    def no_rotation(_: None) -> jnp.ndarray:
        return I

    def opposite_rotation(_: None) -> jnp.ndarray:
        # Find an orthogonal vector
        orthogonal = jnp.array([1.0, 0.0, 0.0])
        alt = jnp.array([0.0, 1.0, 0.0])
        axis = lax.cond(
            jnp.allclose(v1, orthogonal, atol=1e-3),
            lambda _: normalize(cross(v1, alt)),
            lambda _: normalize(cross(v1, orthogonal)),
            operand=None
        )
        return rotation_matrix_from_rotvec(axis * jnp.pi)

    def general_case(_: None) -> jnp.ndarray:
        skew = jnp.array([
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0]
        ])
        sin = magnitude(cross_prod)
        cos = dot_prod
        return I + skew + (skew @ skew) * ((1 - cos) / (sin**2 + 1e-8))

    return lax.cond(
        jnp.isclose(dot_prod, 1.0, atol=1e-6),
        no_rotation,
        lambda _: lax.cond(
            jnp.isclose(dot_prod, -1.0, atol=1e-6),
            opposite_rotation,
            general_case,
            operand=None
        ),
        operand=None
    )


@jit
def angle_between_rotations(R1: jnp.ndarray, R2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the angular distance between two rotation matrices.

    Parameters
    ----------
    R1 : jnp.ndarray
        First rotation matrix (3x3).
    R2 : jnp.ndarray
        Second rotation matrix (3x3).

    Returns
    -------
    jnp.ndarray
        Angle in radians.
    """
    R = jnp.matmul(R1, R2.T)
    trace = jnp.clip(jnp.trace(R), -1.0, 3.0)
    return jnp.arccos(jnp.clip((trace - 1) / 2, -1.0, 1.0))

@jit
def rotation_between_vectors(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the rotation matrix that rotates vector v1 to align with vector v2.

    Parameters
    ----------
    v1 : jnp.ndarray
        Starting vector (3,).
    v2 : jnp.ndarray
        Target vector (3,).

    Returns
    -------
    jnp.ndarray
        Rotation matrix (3, 3).
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    cross_prod = jnp.cross(v1, v2)
    dot_prod = jnp.dot(v1, v2)
    skew = jnp.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    I = jnp.eye(3)
    factor = 1 / (1 + dot_prod + 1e-10)
    return I + skew + skew @ skew * factor
