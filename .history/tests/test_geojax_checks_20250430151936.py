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
    not_unit = jnp.array([2.0, 0.0, 0.0])
    assert not is_unit_vector(not_unit)


def test_is_collinear():
    v1 = jnp.array([1.0, 2.0, 3.0])
    v2 = 2 * v1
    assert is_collinear(v1, v2)
    v3 = jnp.array([0.0, 1.0, 0.0])
    assert not is_collinear(v1, v3)


def test_is_orthogonal():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    assert is_orthogonal(v1, v2)
    assert not is_orthogonal(v1, jnp.array([1.0, 1.0, 0.0]))



def test_orthonormal_basis_from_vector():
    v = jnp.array([1.0, 0.0, 0.0])
    basis = orthonormal_basis_from_vector(v)
    assert basis.shape == (3, 3)
    assert is_unit_vector(basis[0])
    assert is_orthogonal(basis[0], basis[1])
    assert is_orthogonal(basis[0], basis[2])
    assert is_orthogonal(basis[1], basis[2])
