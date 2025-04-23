import pytest
import jax.numpy as jnp
from GeoJax.core import magnitude, normalize
from GeoJax.rotation import (
    rotation_matrix_from_rotvec,
    rotate_around_axis,
    rotation_matrix_between_vectors,
    angle_between_rotations,
)

# Sample vectors
v1 = jnp.array([1.0, 0.0, 0.0])
v2 = jnp.array([0.0, 1.0, 0.0])
v3 = jnp.array([0.0, 0.0, 1.0])

def test_rotation_matrix_from_rotvec_identity():
    rot_vec = jnp.array([0.0, 0.0, 0.0])
    R = rotation_matrix_from_rotvec(rot_vec)
    assert jnp.allclose(R, jnp.eye(3))

def test_rotation_matrix_from_rotvec_90deg_z():
    rot_vec = jnp.array([0.0, 0.0, jnp.pi / 2])
    R = rotation_matrix_from_rotvec(rot_vec)
    rotated = R @ v1
    expected = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(rotated, expected, atol=1e-6)

def test_rotate_around_axis_2d():
    coords = jnp.array([[1.0, 0.0]])
    theta = jnp.pi / 2
    axis = jnp.array([0.0, 0.0, 1.0])
    rotated = rotate_around_axis(coords, theta, axis)
    expected = jnp.array([[0.0, 1.0]])
    assert jnp.allclose(rotated, expected, atol=1e-6)

def test_rotate_around_axis_3d():
    coords = jnp.array([[1.0, 0.0, 0.0]])
    theta = jnp.pi
    axis = jnp.array([0.0, 1.0, 0.0])
    rotated = rotate_around_axis(coords, theta, axis)
    expected = jnp.array([[-1.0, 0.0, 0.0]])
    assert jnp.allclose(rotated, expected, atol=1e-6)

def test_rotation_matrix_between_vectors():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    R = rotation_matrix_between_vectors(v1, v2)
    rotated = jnp.dot(R, v1)
    assert jnp.allclose(rotated, v2, atol=1e-5)

def test_angle_between_rotations_identity():
    R1 = jnp.eye(3)
    R2 = jnp.eye(3)
    angle = angle_between_rotations(R1, R2)
    assert jnp.isclose(angle, 0.0)

def test_angle_between_rotations_90deg():
    R1 = rotation_matrix_from_rotvec(jnp.array([0.0, 0.0, 0.0]))
    R2 = rotation_matrix_from_rotvec(jnp.array([0.0, 0.0, jnp.pi / 2]))
    angle = angle_between_rotations(R1, R2)
    assert jnp.isclose(angle, jnp.pi / 2, atol=1e-6)
