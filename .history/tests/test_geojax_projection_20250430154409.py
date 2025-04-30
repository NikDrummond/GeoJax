# tests/test_projection.py

import pytest
import jax.numpy as jnp
from GeoJax.projection import (
    reject_axis,
    project_to_sphere,
    project_to_vector,
    project_to_plane,
    project_to_xy_plane,
    project_to_xz_plane,
    project_to_yz_plane,
    orthographic_projection,
    stereographic_projection,
    equirectangular_projection,
    mercator_projection,
    lambert_azimuthal_projection,
)

# ==== Fixtures ====

unit_points = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

non_unit_points = jnp.array([
    [2.0, 0.0, 0.0]
])

north_pole = jnp.array([0.0, 0.0, 1.0])
alt_north_pole = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3)


# ==== Reject Axis ====

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

def test_reject_axis_bad_shape():
    vec = jnp.array([[1.0, 2.0]])  # Not 3D
    with pytest.raises(ValueError):
        reject_axis(vec, axis=1)

# ==== Sphere ====

def test_project_to_sphere_on_axis():
    points = jnp.array([[1.0, 0.0, 0.0]])
    result = project_to_sphere(points, radius=2.0, center=jnp.array([0.0, 0.0, 0.0]))
    expected = jnp.array([[2.0, 0.0, 0.0]])
    assert jnp.allclose(result, expected)

# ==== Vector ====

def test_project_to_vector_single():
    vec = jnp.array([[1.0, 1.0, 0.0]])
    onto = jnp.array([[1.0, 0.0, 0.0]])
    result = project_to_vector(vec, onto)
    expected = jnp.array([[1.0, 0.0, 0.0]])
    assert jnp.allclose(result, expected)

def test_project_to_vector_bad_shape():
    vec = jnp.array([[1.0, 1.0]])
    onto = jnp.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        project_to_vector(vec, onto)

# ==== Plane ====

def test_project_to_plane_orthogonal():
    vec = jnp.array([[1.0, 2.0, 3.0]])
    normal = jnp.array([0.0, 0.0, 1.0])
    projected = project_to_plane(vec, normal)
    expected = jnp.array([[1.0, 2.0, 0.0]])
    assert jnp.allclose(projected, expected)

# ==== Axis Planes ====

def test_project_to_xy_plane():
    vec = jnp.array([[1.0, 2.0, 3.0]])
    result = project_to_xy_plane(vec)
    expected = jnp.array([[1.0, 2.0]])
    assert jnp.allclose(result, expected)

def test_project_to_xz_plane():
    vec = jnp.array([[1.0, 2.0, 3.0]])
    result = project_to_xz_plane(vec)
    expected = jnp.array([[1.0, 3.0]])
    assert jnp.allclose(result, expected)

def test_project_to_yz_plane():
    vec = jnp.array([[1.0, 2.0, 3.0]])
    result = project_to_yz_plane(vec)
    expected = jnp.array([[2.0, 3.0]])
    assert jnp.allclose(result, expected)

# ==== Orthographic ====

def test_orthographic_projection():
    vec = jnp.array([[0.6, 0.8, 1.0]])
    result = orthographic_projection(vec)
    expected = jnp.array([[0.6, 0.8]])
    assert jnp.allclose(result, expected)

# ==== Stereographic ====

def test_stereographic_projection_equator():
    point = jnp.array([[1.0, 0.0, 0.0]])
    result = stereographic_projection(point)
    assert jnp.allclose(result, jnp.array([[1.0, 0.0]]), atol=1e-3)

def test_stereographic_projection_alt_pole():
    result = stereographic_projection(unit_points, north_pole=alt_north_pole)
    assert result.shape == (3, 2)
    assert jnp.all(jnp.isfinite(result))

# ==== Equirectangular ====

def test_equirectangular_projection_basic():
    point = jnp.array([[1.0, 0.0, 0.0]])
    result = equirectangular_projection(point)
    lon, lat = result[0]
    assert jnp.isclose(lat, 0.0, atol=1e-5)
    assert jnp.isclose(lon, 0.0, atol=1e-5)

def test_equirectangular_projection_rotated():
    result = equirectangular_projection(unit_points, north_pole=alt_north_pole)
    assert result.shape == (3, 2)
    assert jnp.all(jnp.isfinite(result))

# ==== Mercator ====

def test_mercator_projection_equator():
    result = mercator_projection(jnp.array([[1.0, 0.0, 0.0]]))
    assert jnp.isclose(result[0, 1], 0.0, atol=1e-4)

def test_mercator_projection_pole_handling():
    near_pole = jnp.array([[0.0, 0.0, 0.9999]])
    result = mercator_projection(near_pole)
    assert jnp.isfinite(result[0, 1])

# ==== Lambert Azimuthal ====

def test_lambert_projection_centered():
    point = jnp.array([[0.0, 0.0, 1.0]])
    result = lambert_azimuthal_projection(point)
    assert jnp.allclose(result, jnp.array([[0.0, 0.0]]), atol=1e-5)

def test_lambert_projection_rotated():
    result = lambert_azimuthal_projection(unit_points, north_pole=alt_north_pole)
    assert result.shape == (3, 2)
    assert jnp.all(jnp.isfinite(result))
