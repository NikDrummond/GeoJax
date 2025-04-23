# test_geojax_bounds.py
# Tests for GeoJax bounding shapes utilities

import pytest
import jax.numpy as jnp
from GeoJax import aabb_bounds, bounding_sphere, oriented_bounding_box

def test_aabb_bounds():
    points = jnp.array([[0, 0, 0], [1, 2, 3], [-1, -2, -3]])
    bmin, bmax = aabb_bounds(points)
    assert jnp.allclose(bmin, [-1, -2, -3])
    assert jnp.allclose(bmax, [1, 2, 3])

def test_bounding_sphere():
    points = jnp.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    center, radius = bounding_sphere(points)
    assert jnp.allclose(center, [0.0, 0.0, 0.0], atol=1e-6)
    assert pytest.approx(radius.item(), abs=1e-6) == 1.0

def test_oriented_bounding_box():
    cube = jnp.array([
        [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
        [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
    ])
    obb = oriented_bounding_box(cube)
    assert jnp.allclose(obb['center'], jnp.zeros(3), atol=1e-6)
    assert obb['axes'].shape == (3, 3)
    assert obb['extents'].shape == (3,)
