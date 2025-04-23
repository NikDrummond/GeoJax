# test_geojax_rotation.py
# Tests for GeoJax rotation functions

import pytest
import jax.numpy as jnp
from GeoJax import rotation_matrix_from_rotvec, rotate_around_axis

def test_rotation_matrix_identity():
    rotvec = jnp.array([0.0, 0.0, 0.0])
    R = rotation_matrix_from_rotvec(rotvec)
    assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

def test_rotation_matrix_axis():
    # 90 deg rotation around z axis
    theta = jnp.pi / 2
    rotvec = jnp.array([0.0, 0.0, theta])
    R = rotation_matrix_from_rotvec(rotvec)
    expected = jnp.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    assert jnp.allclose(R, expected, atol=1e-6)

def test_rotate_around_axis_2d():
    coords = jnp.array([[1.0, 0.0]])
    theta = jnp.pi / 2
    axis = jnp.array([0.0, 0.0, 1.0])
    rotated = rotate_around_axis(coords, theta, axis)
    expected = jnp.array([[0.0, 1.0]])
    assert jnp.allclose(rotated, expected, atol=1e-6)

def test_rotate_around_axis_3d():
    coords = jnp.array([[1.0, 0.0, 0.0]])
    theta = jnp.pi / 2
    axis = jnp.array([0.0, 0.0, 1.0])
    rotated = rotate_around_axis(coords, theta, axis)
    expected = jnp.array([[0.0, 1.0, 0.0]])
    assert jnp.allclose(rotated, expected, atol=1e-6)