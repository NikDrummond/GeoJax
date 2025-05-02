import jax.numpy as jnp
from GeoJax import (
    aabb_bounds,
    bounding_sphere,
    oriented_bounding_box,
    bounding_cylinder,
    tight_aabb_in_frame
)

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

def test_bounding_cylinder():
    points = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.5, 0.0, 1.0],
        [-0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, -0.5, 1.0],
    ])
    cyl = bounding_cylinder(points)
    assert "axis" in cyl
    assert "center" in cyl
    assert "radius" in cyl
    assert "height" in cyl
    assert cyl["axis"].shape == (3,)
    assert jnp.isclose(cyl["radius"], 0.5, atol=1e-2)
    assert jnp.isclose(cyl["height"], 2.0, atol=1e-2)

def test_tight_aabb_in_frame():
    points = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    frame_axes = jnp.eye(3)
    tight = tight_aabb_in_frame(points, frame_axes)
    assert "center" in tight
    assert "extents" in tight
    assert "axes" in tight
    assert tight["center"].shape == (3,)
    assert tight["extents"].shape == (3,)
    assert jnp.allclose(tight["extents"], jnp.array([1.0, 1.0, 1.0]))
