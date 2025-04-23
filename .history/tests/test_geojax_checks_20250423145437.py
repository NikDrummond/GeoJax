# test_geojax_checks.py
# Tests for GeoJax vector and plane property checks

import pytest
import jax.numpy as jnp
from GeoJax import (
    is_unit_vector,
    is_collinear,
    is_orthogonal,
    angle_between_planes,
    orthonormal_basis_from_vector,
)

def test_is_unit_vector():
    v = jnp.array([1.0, 0.0, 0.0])
    assert is_unit_vector(v)
    v2 = jnp.array([1.0, 1.0, 0.0])
    assert not is_unit_vector(v2)

def test_is_collinear():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([2.0, 4.0, 6.0])
    assert is_collinear(a, b)

def test_is_orthogonal():
    x = jnp.array([1.0, 0.0, 0.0])
    y = jnp.array([0.0, 1.0, 0.0])
    assert is_orthogonal(x, y)

def test_angle_between_planes():
    n1 = jnp.array([0.0, 0.0, 1.0])
    n2 = jnp.array([0.0, 1.0, 0.0])
    theta = angle_between_planes(n1, n2, to_degree=True)
    assert pytest.approx(theta.item(), abs=1e-6) == 90.0

def test_orthonormal_basis_from_vector():
    v = jnp.array([0.0, 0.0, 1.0])
    B = orthonormal_basis_from_vector(v)
    I = jnp.dot(B, B.T)
    assert jnp.allclose(I, jnp.eye(3), atol=1e-6)
    assert jnp.allclose(B[0], normalize(v), atol=1e-6)
