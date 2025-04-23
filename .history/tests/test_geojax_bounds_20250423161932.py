import jax.numpy as jnp
from GeoJax import aabb_bounds, bounding_sphere, oriented_bounding_box

def test_aabb_bounds():
    points = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0]
    ])
    min_corner, max_corner = aabb_bounds(points)
    assert jnp.allclose(min_corner, jnp.array([-1.0, -2.0, -3.0]))
    assert jnp.allclose(max_corner, jnp.array([1.0, 2.0, 3.0]))

def test_bounding_sphere():
    points = jnp.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    center, radius = bounding_sphere(points)
    assert jnp.allclose(center, jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isclose(radius, 1.0)

def test_oriented_bounding_box_structure():
    points = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    obb = oriented_bounding_box(points)
    assert "center" in obb
    assert "axes" in obb
    assert "extents" in obb
    assert obb["center"].shape == (3,)
    assert obb["axes"].shape == (3, 3)
    assert obb["extents"].shape == (3,)
