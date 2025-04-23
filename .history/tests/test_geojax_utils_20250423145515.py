# test_geojax_utils.py
# Tests for GeoJax utility functions

import pytest
import jax.numpy as jnp
from GeoJax import normalize_angle_array, center_points, scale_coords, origin_flip

def test_normalize_angle_array_radians():
    angles = jnp.array([-jnp.pi, 0.0, jnp.pi])
    normed = normalize_angle_array(angles)
    assert jnp.all((0.0 <= normed) & (normed < 2 * jnp.pi))

def test_normalize_angle_array_degrees():
    angles = jnp.array([-90.0, 0.0, 450.0])
    normed = normalize_angle_array(angles)
    assert jnp.all((0.0 <= normed) & (normed < 360.0))

def test_center_points_mean():
    pts = jnp.array([[1, 2, 3], [3, 2, 1]])
    centered = center_points(pts)
    assert jnp.allclose(jnp.mean(centered, axis=0), jnp.zeros(3), atol=1e-6)

def test_scale_coords_scalar():
    coords = jnp.array([[1.0, 2.0, 3.0]])
    scaled = scale_coords(coords, 2.0)
    assert jnp.allclose(scaled, [[2.0, 4.0, 6.0]])

def test_origin_flip_away():
    starts = jnp.array([[0.0, 0.0, 0.0]])
    stops = jnp.array([[1.0, 1.0, 1.0]])
    flipped = origin_flip(starts, stops, method="away")
    assert jnp.linalg.norm(flipped) > jnp.linalg.norm(starts)

def test_origin_flip_towards():
    starts = jnp.array([[1.0, 1.0, 1.0]])
    stops = jnp.array([[2.0, 2.0, 2.0]])
    flipped = origin_flip(starts, stops, method="towards")
    assert jnp.linalg.norm(flipped) < jnp.linalg.norm(starts)

def test_origin_flip_invalid():
    with pytest.raises(ValueError):
        origin_flip(jnp.zeros((1, 3)), jnp.ones((1, 3)), method="invalid")
