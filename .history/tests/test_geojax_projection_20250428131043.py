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

def test_project_to_xy_plane():
    points = jnp.array([[1.0, 2.0, 3.0]])
    projected = project_to_2d(points, method="axis_plane", drop_axis='z')
    assert projected.shape == (1, 2)
    assert jnp.allclose(projected, jnp.array([[1.0, 2.0]])), f"Got {projected}"

def test_project_to_xz_plane():
    points = jnp.array([[1.0, 2.0, 3.0]])
    projected = project_to_2d(points, method="axis_plane", drop_axis='y')
    assert projected.shape == (1, 2)
    assert jnp.allclose(projected, jnp.array([[1.0, 3.0]])), f"Got {projected}"

def test_project_to_yz_plane():
    points = jnp.array([[1.0, 2.0, 3.0]])
    projected = project_to_2d(points, method="axis_plane", drop_axis='x')
    assert projected.shape == (1, 2)
    assert jnp.allclose(projected, jnp.array([[2.0, 3.0]])), f"Got {projected}"

def test_project_to_plane_custom_normal():
    points = jnp.array([[1.0, 2.0, 3.0]])
    normal = jnp.array([0.0, 0.0, 1.0])  # project onto xy plane
    projected = project_to_2d(points, method="plane", normal=normal)
    assert projected.shape == (1, 2)
    # In this case, should drop z component effectively
    assert jnp.allclose(projected, jnp.array([[1.0, 2.0]]), atol=1e-5), f"Got {projected}"

def test_orthographic_projection():
    points = jnp.array([[0.6, 0.8, 0.0]])
    projected = project_to_2d(points, method="orthographic")
    assert projected.shape == (1, 2)
    assert jnp.allclose(projected, jnp.array([[0.6, 0.8]])), f"Got {projected}"

def test_stereographic_projection_on_equator():
    points = jnp.array([[0.7071, 0.7071, 0.0]])  # (x,y,z) on unit sphere equator
    projected = project_to_2d(points, method="stereographic")
    assert projected.shape == (1, 2)
    expected = jnp.array([[0.7071, 0.7071]])  # (since z=0, projection is same)
    assert jnp.allclose(projected, expected, atol=1e-3), f"Got {projected}"

def test_invalid_method_raises():
    points = jnp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        project_to_2d(points, method="unknown")

def test_missing_plane_normal_raises():
    points = jnp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        project_to_2d(points, method="plane")  # missing normal

def test_invalid_axis_plane_choice_raises():
    points = jnp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        project_to_2d(points, method="axis_plane", drop_axis="invalid")


def test_equirectangular_projection_basic():
    # A point on the equator (x=1, y=0, z=0) should have lat=0
    points = jnp.array([[1.0, 0.0, 0.0]])
    result = equirectangular_projection(points)
    lon, lat = result[0]
    assert jnp.isclose(lat, 0.0, atol=1e-5), f"Latitude should be zero, got {lat}"
    assert jnp.isclose(lon, 0.0, atol=1e-5), f"Longitude should be zero, got {lon}"

def test_mercator_projection_basic():
    # A point on the equator (x=1, y=0, z=0) should have merc_y=0
    points = jnp.array([[1.0, 0.0, 0.0]])
    result = mercator_projection(points)
    lon, merc_y = result[0]
    assert jnp.isclose(merc_y, 0.0, atol=1e-5), f"Mercator Y should be zero, got {merc_y}"
    assert jnp.isclose(lon, 0.0, atol=1e-5), f"Longitude should be zero, got {lon}"

def test_lambert_azimuthal_projection_basic():
    # North pole (z=1) should map to (0,0)
    points = jnp.array([[0.0, 0.0, 1.0]])
    result = lambert_azimuthal_projection(points)
    assert jnp.allclose(result, jnp.array([[0.0, 0.0]]), atol=1e-5), f"Expected origin, got {result}"

def test_mercator_projection_handles_poles():
    # Near North Pole (very close to z=1)
    points = jnp.array([[0.0, 0.0, 0.9999]])
    result = mercator_projection(points)
    _, merc_y = result[0]
    assert jnp.isfinite(merc_y), f"Expected finite Mercator Y at near-pole, got {merc_y}"

def test_lambert_projection_scale():
    # Test that Lambert projection does not produce NaN for a generic point
    points = jnp.array([[1.0, 1.0, 1.0]]) / jnp.sqrt(3)
    result = lambert_azimuthal_projection(points)
    assert jnp.all(jnp.isfinite(result)), f"Expected finite Lambert projection, got {result}"
