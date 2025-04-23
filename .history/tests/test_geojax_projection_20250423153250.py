# tests/test_geojax_projection.py

import jax.numpy as jnp
import pytest
from GeoJax.projection import (
    reject_axis,
    project_to_sphere,
    project_to_vector,
    project_to_plane,
)

def test_reject_axis_zero():
    vec = jnp.array([[1.0, 2.0, 3.0]])
    result = reject_axis(vec, axis=1)
    expected = jnp.array([[1.0, 0.0, 3.0]])
    assert jnp.allclose(result, expected)

def test_reject_axis_squash():
    vec = jnp.array([[1.0, 2.0, 3.0]])
    result = reject_axis(vec, axis=1, squash=True)
    expected = jnp.array([[1.0, 3.0]])
    assert jnp.allclose(result, expected)

def test_project_to_sphere():
    points = jnp.array([[1.0, 0.0, 0.0]])
    center = jnp.array([0.0, 0.0, 0.0])
    radius = 2.0
    result = project_to_sphere(points, radius, center)
    expected = jnp.array([[2.0, 0.0, 0.0]])
    assert jnp.allclose(result, expected)

def test_project_to_vector():
    vector = jnp.array([[1.0, 1.0, 0.0]])
    onto = jnp.array([[1.0, 0.0, 0.0]])
    result = project_to_vector(vector, onto)
    expected = jnp.array([[1.0, 0.0, 0.0]])
    assert jnp.allclose(result, expected)

def test_project_to_plane():
    vector = jnp.array([[1.0, 1.0, 1.0]])
    normal = jnp.array([0.0, 0.0, 1.0])
    result = project_to_plane(vector, normal)
    expected = jnp.array([[1.0, 1.0, 0.0]])
    assert jnp.allclose(result, expected)
