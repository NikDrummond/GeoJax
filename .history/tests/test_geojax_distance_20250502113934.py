import pytest
import jax.numpy as jnp
from distance_module import compute_distance


# ---------------------------
# Utility: Close Comparison
# ---------------------------
def approx(a, b, tol=1e-5):
    return jnp.allclose(a, b, atol=tol)


# ---------------------------
# 1. Basic Functionality Tests
# ---------------------------
def test_euclidean():
    a = jnp.array([0.0, 0.0])
    b = jnp.array([3.0, 4.0])
    assert approx(compute_distance(a, b, method="euclidean"), 5.0)


def test_manhattan():
    a = jnp.array([1.0, 2.0])
    b = jnp.array([4.0, 6.0])
    assert approx(compute_distance(a, b, method="manhattan"), 7.0)


def test_chebyshev():
    a = jnp.array([1.0, 5.0])
    b = jnp.array([4.0, 9.0])
    assert approx(compute_distance(a, b, method="chebyshev"), 4.0)


def test_minkowski():
    a = jnp.array([0.0, 0.0])
    b = jnp.array([1.0, 1.0])
    assert approx(compute_distance(a, b, method="minkowski", p=3), (2.0) ** (1 / 3))


def test_cosine():
    a = jnp.array([1.0, 0.0])
    b = jnp.array([0.0, 1.0])
    assert approx(compute_distance(a, b, method="cosine"), 1.0)


def test_haversine():
    a = jnp.array([0.0, 0.0])
    b = jnp.array([0.0, 1.0])
    d = compute_distance(a, b, method="haversine")
    assert approx(d, 111.1949, tol=0.01)


# ---------------------------
# 2. Broadcasting Tests
# ---------------------------
def test_one_to_many():
    a = jnp.array([0.0, 0.0])
    b = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    d = compute_distance(a, b, method="euclidean")
    assert approx(d, jnp.array([1.0, 2.0]))


def test_many_to_one():
    a = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    b = jnp.array([0.0, 0.0])
    d = compute_distance(a, b, method="euclidean")
    assert approx(d, jnp.array([1.0, 2.0]))


def test_many_to_many():
    a = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    b = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    d = compute_distance(a, b, method="euclidean")
    assert approx(d, jnp.array([1.0, 2.0]))


def test_full_matrix():
    a = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    b = jnp.array([[0.0, 1.0], [1.0, 1.0]])
    d = compute_distance(a, b, method="euclidean", full_matrix=True)
    expected = jnp.array([[1.0, jnp.sqrt(2)], [jnp.sqrt(2), 1.0]])
    assert approx(d, expected)


# ---------------------------
# 3. Error Tests
# ---------------------------


def test_mismatched_dimensions():
    a = jnp.array([0.0, 0.0])
    b = jnp.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        compute_distance(a, b, method="euclidean")


def test_invalid_method():
    a = jnp.array([0.0, 0.0])
    b = jnp.array([0.0, 1.0])
    with pytest.raises(ValueError):
        compute_distance(a, b, method="not_a_real_method")


def test_invalid_shape():
    a = jnp.ones((3, 3, 3))
    b = jnp.ones((3,))
    with pytest.raises(ValueError):
        compute_distance(a, b, method="euclidean")


def test_haversine_wrong_dim():
    a = jnp.array([0.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        compute_distance(a, b, method="haversine")
