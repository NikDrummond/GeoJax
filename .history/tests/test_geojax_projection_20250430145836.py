# tests/test_geojax_projection.py

import jax.numpy as jnp
import pytest
from GeoJax.projection import (
    reject_axis,
    project_to_sphere,
    project_to_vector,
    project_to_plane,
    project_to_2d,
    equirectangular_projection,
    mercator_projection,
    lambert_azimuthal_projection,
    stereographic_projection,
)

# === Reject Axis Tests ===

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

def test_reject_axis_batch():
    vecs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = reject_axis(vecs, axis=0)
    expected = jnp.array([[0.0, 2.0, 3.0], [0.0, 5.0, 6.0]])
    assert jnp.allclose(result, expected)


# === Sphere Projection ===

def test_project_to_sphere_on_axis():
    points = jnp.array([[1.0, 0.0, 0.0]])
    center = jnp.array([0.0, 0.0, 0.0])
    radius = 2.0
    result = project_to_sphere(points, radius, center)
    expected = jnp.array([[2.0, 0.0, 0.0]])
    assert jnp.allclose(result, expected)

def test_project_to_sphere_off_center():
    points = jnp.array([[2.0, 0.0, 0.0]])
    center = jnp.array([1.0, 0.0, 0.0])
    radius = 1.0
    result = project_to_sphere(points, radius, center)
    expected = jnp.array([[2.0, 0.0, 0.0]])  # already on sphere
    assert jnp.allclose(result, expected)


# === Vector Projection ===

def test_project_to_vector_single():
    vector = jnp.array([[1.0, 1.0, 0.0]])
    onto = jnp.array([[1.0, 0.0, 0.0]])
    result = project_to_vector(vector, onto)
    expected = jnp.array([[1.0, 0.0, 0.0]])
    assert jnp.allclose(result, expected)

def test_project_to_vector_broadcast():
    vector = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    onto = jnp.array([1.0, 1.0, 0.0])
    result = project_to_vector(vector, onto)
    assert result.shape == (2, 3)
    # projections should be aligned with (1,1,0)
    dot = jnp.sum(result * onto, axis=-1)
    assert jnp.all(dot > 0)


# === Plane Projection ===

def test_project_to_plane_basic():
    vector = jnp.array([[1.0, 1.0, 1.0]])
    normal = jnp.array([0.0, 0.0, 1.0])
    result = project_to_plane(vector, normal)
    expected = jnp.array([[1.0, 1.0, 0.0]])
    assert jnp.allclose(result, expected)

def test_project_to_plane_oblique():
    vector = jnp.array([[1.0, 1.0, 1.0]])
    normal = jnp.array([1.0, 1.0, 1.0])
    result = project_to_plane(vector, normal)
    expected = jnp.array([[0.0, 0.0, 0.0]])  # fully aligned with normal
    assert jnp.allclose(result, expected, atol=1e-5)


# === Axis Plane Projections ===

def test_project_to_xy_plane():
    points = jnp.array([[1.0, 2.0, 3.0]])
    projected = project_to_2d(points, method="axis_plane", drop_axis="z")
    assert jnp.allclose(projected, jnp.array([[1.0, 2.0]]))

def test_project_to_xz_plane():
    points = jnp.array([[1.0, 2.0, 3.0]])
    projected = project_to_2d(points, method="axis_plane", drop_axis="y")
    assert jnp.allclose(projected, jnp.array([[1.0, 3.0]]))

def test_project_to_yz_plane():
    points = jnp.array([[1.0, 2.0, 3.0]])
    projected = project_to_2d(points, method="axis_plane", drop_axis="x")
    assert jnp.allclose(projected, jnp.array([[2.0, 3.0]]))


# === 2D Projection Variants ===

def test_project_to_plane_custom_normal():
    points = jnp.array([[1.0, 2.0, 3.0]])
    normal = jnp.array([0.0, 0.0, 1.0])
    projected = project_to_2d(points, method="plane", normal=normal)
    assert jnp.allclose(projected, jnp.array([[1.0, 2.0]]), atol=1e-5)

def test_orthographic_projection():
    points = jnp.array([[0.6, 0.8, 0.0]])
    projected = project_to_2d(points, method="orthographic")
    assert jnp.allclose(projected, jnp.array([[0.6, 0.8]]))

def test_stereographic_projection_equator():
    points = jnp.array([[0.7071, 0.7071, 0.0]])
    projected = project_to_2d(points, method="stereographic")
    assert jnp.allclose(projected, jnp.array([[0.7071, 0.7071]]), atol=1e-3)

def test_stereographic_projection_pole_safe():
    points = jnp.array([[0.0, 0.0, 1.0]])
    result = stereographic_projection(points)
    assert jnp.all(jnp.isfinite(result)), f"Non-finite stereographic result: {result}"


# === Equirectangular & Mercator ===

def test_equirectangular_projection_basic():
    points = jnp.array([[1.0, 0.0, 0.0]])
    result = equirectangular_projection(points)
    lon, lat = result[0]
    assert jnp.isclose(lat, 0.0, atol=1e-5)
    assert jnp.isclose(lon, 0.0, atol=1e-5)

def test_mercator_projection_basic():
    points = jnp.array([[1.0, 0.0, 0.0]])
    result = mercator_projection(points)
    lon, merc_y = result[0]
    assert jnp.isclose(merc_y, 0.0, atol=1e-5)
    assert jnp.isclose(lon, 0.0, atol=1e-5)

def test_mercator_projection_handles_poles():
    points = jnp.array([[0.0, 0.0, 0.9999]])
    result = mercator_projection(points)
    _, merc_y = result[0]
    assert jnp.isfinite(merc_y)

def test_lambert_azimuthal_projection_basic():
    points = jnp.array([[0.0, 0.0, 1.0]])
    result = lambert_azimuthal_projection(points)
    assert jnp.allclose(result, jnp.array([[0.0, 0.0]]), atol=1e-5)

def test_lambert_projection_scale():
    points = jnp.array([[1.0, 1.0, 1.0]]) / jnp.sqrt(3)
    result = lambert_azimuthal_projection(points)
    assert jnp.all(jnp.isfinite(result))


# === Error Handling ===

def test_invalid_method_raises():
    points = jnp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        project_to_2d(points, method="unknown")

def test_missing_plane_normal_raises():
    points = jnp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        project_to_2d(points, method="plane")

def test_invalid_axis_plane_choice_raises():
    points = jnp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        project_to_2d(points, method="axis_plane", drop_axis="bad")

def test_invalid_shapes_reject_axis():
    with pytest.raises(Exception):
        reject_axis(jnp.array([1.0, 2.0]), axis=1)

def test_invalid_shapes_project_to_vector():
    with pytest.raises(Exception):
        project_to_vector(jnp.array([[1.0, 2.0]]), jnp.array([[1.0]]))
