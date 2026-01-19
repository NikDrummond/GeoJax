"""Rotation operations in 3D space."""

from jax import jit, lax
import jax.numpy as jnp

from ..core import normalize, cross, dot, magnitude


@jit
def rotation_matrix_from_rotvec(rot_vec: jnp.ndarray) -> jnp.ndarray:
    """Compute a 3x3 rotation matrix from a rotation vector (Rodrigues' formula).

    Parameters
    ----------
    rot_vec : jnp.ndarray
        Rotation vector (axis * angle), shape (3,).

    Returns
    -------
    jnp.ndarray
        3x3 rotation matrix.
    """
    angle = jnp.linalg.norm(rot_vec)

    def compute_rotation(_):
        axis = rot_vec / angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        one_minus_cos = 1.0 - cos_a

        x, y, z = axis
        cross_matrix = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        outer = jnp.outer(axis, axis)
        return cos_a * jnp.eye(3) + sin_a * cross_matrix + one_minus_cos * outer

    return lax.cond(angle > 1e-8, compute_rotation, lambda _: jnp.eye(3), operand=None)


@jit
def rotate_around_axis(coords: jnp.ndarray, theta: float, axis: jnp.ndarray) -> jnp.ndarray:
    """Rotate coordinates around a given axis by an angle."""
    rot_vec = axis * theta
    R = rotation_matrix_from_rotvec(rot_vec)

    is_2d = coords.shape[1] == 2
    coords_3d = (
        jnp.concatenate([coords, jnp.zeros((coords.shape[0], 1))], axis=1)
        if is_2d
        else coords
    )
    rotated = coords_3d @ R.T
    return rotated[:, :2] if is_2d else rotated


@jit
def rotation_matrix_between_vectors(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Compute the rotation matrix that aligns vector v1 to v2.

    Parameters
    ----------
    v1 : jnp.ndarray
        Source vector, shape (3,).
    v2 : jnp.ndarray
        Target vector, shape (3,).

    Returns
    -------
    jnp.ndarray
        3x3 rotation matrix such that R @ v1 ≈ v2.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    cross_prod = cross(v1, v2)
    dot_prod = dot(v1, v2)
    I = jnp.eye(3)

    def no_rotation(_: None) -> jnp.ndarray:
        return I

    def opposite_rotation(_: None) -> jnp.ndarray:
        orthogonal = jnp.array([1.0, 0.0, 0.0])
        alt = jnp.array([0.0, 1.0, 0.0])
        axis = lax.cond(
            jnp.allclose(v1, orthogonal, atol=1e-3),
            lambda _: normalize(jnp.cross(v1, alt)),
            lambda _: normalize(jnp.cross(v1, orthogonal)),
            operand=None,
        )
        return rotation_matrix_from_rotvec(axis * jnp.pi)

    def general_case(_: None) -> jnp.ndarray:
        skew = jnp.array(
            [
                [0, -cross_prod[2], cross_prod[1]],
                [cross_prod[2], 0, -cross_prod[0]],
                [-cross_prod[1], cross_prod[0], 0],
            ]
        )
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
            operand=None,
        ),
        operand=None,
    )


@jit
def angle_between_rotations(R1: jnp.ndarray, R2: jnp.ndarray) -> jnp.ndarray:
    """Compute the angular distance between two rotation matrices.

    Parameters
    ----------
    R1 : jnp.ndarray
        First rotation matrix, shape (3, 3).
    R2 : jnp.ndarray
        Second rotation matrix, shape (3, 3).

    Returns
    -------
    jnp.ndarray
        Angular distance in radians (scalar).
    """
    R = jnp.matmul(R1, R2.T)
    trace = jnp.clip(jnp.trace(R), -1.0, 3.0)
    return jnp.arccos(jnp.clip((trace - 1) / 2, -1.0, 1.0))


@jit
def rotation_between_vectors(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """Compute the rotation matrix that rotates vector v1 to align with vector v2.

    Uses a simplified formula for the rotation matrix.

    Parameters
    ----------
    v1 : jnp.ndarray
        Source vector, shape (3,).
    v2 : jnp.ndarray
        Target vector, shape (3,).

    Returns
    -------
    jnp.ndarray
        3x3 rotation matrix such that R @ v1 ≈ v2.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    cross_prod = jnp.cross(v1, v2)
    dot_prod = jnp.dot(v1, v2)
    skew = jnp.array(
        [
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0],
        ]
    )
    I = jnp.eye(3)
    factor = 1 / (1 + dot_prod + 1e-10)
    return I + skew + skew @ skew * factor
