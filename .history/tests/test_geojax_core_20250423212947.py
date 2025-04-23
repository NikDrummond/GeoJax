# tests/test_geojax_core.py

import jax.numpy as jnp
import numpy as np
import pytest
from GeoJax import *
# from GeoJax.core import scale_along_basis

def test_magnitude():
    assert jnp.isclose(magnitude(jnp.array([3.0, 4.0, 0.0])), 5.0)

def test_normalize():
    v = jnp.array([3.0, 0.0, 4.0])
    result = normalize(v)
    assert jnp.isclose(magnitude(result), 1.0)

def test_dot():
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    assert jnp.isclose(dot(a, b), 0.0)
    assert jnp.isclose(dot(a, a), 1.0)

def test_cross():
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    result = cross(a, b)
    np.testing.assert_array_almost_equal(result, jnp.array([0.0, 0.0, 1.0]))

def test_project_to_vector():
    v = jnp.array([2.0, 2.0, 0.0])
    onto = jnp.array([1.0, 0.0, 0.0])
    proj = project_to_vector(v, onto)
    np.testing.assert_array_almost_equal(proj, jnp.array([2.0, 0.0, 0.0]))

def test_reject():
    v = jnp.array([2.0, 2.0, 0.0])
    from_v = jnp.array([1.0, 0.0, 0.0])
    rej = reject(v, from_v)
    np.testing.assert_array_almost_equal(rej, jnp.array([0.0, 2.0, 0.0]))

def test_scalar_triple():
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    c = jnp.array([0.0, 0.0, 1.0])
    result = scalar_triple(a, b, c)
    assert jnp.isclose(result, 1.0)

def test_reflect():
    v = jnp.array([1.0, -1.0, 0.0])
    normal = jnp.array([0.0, 1.0, 0.0])
    result = reflect(v, normal)
    np.testing.assert_array_almost_equal(result, jnp.array([1.0, 1.0, 0.0]))

def test_gram_schmidt():
    input_vectors = jnp.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    Q = gram_schmidt(input_vectors)
    assert Q.shape == input_vectors.shape
    assert jnp.isclose(dot(Q[0], Q[1]), 0.0)

def test_apply_affine():
    points = jnp.array([[1.0, 0.0, 0.0]])
    matrix = jnp.eye(3)
    offset = jnp.array([0.0, 1.0, 0.0])
    result = apply_affine(points, matrix, offset)
    np.testing.assert_array_almost_equal(result, jnp.array([[1.0, 1.0, 0.0]]))

def test_point_to_plane_distance():
    p = jnp.array([0.0, 0.0, 2.0])
    plane_pt = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    dist = point_to_plane_distance(p, plane_pt, normal)
    assert jnp.isclose(dist, 2.0)

def test_ray_plane_intersect():
    ray_origin = jnp.array([0.0, 0.0, -1.0])
    ray_dir = jnp.array([0.0, 0.0, 1.0])
    plane_point = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    result = ray_plane_intersect(ray_origin, ray_dir, plane_point, normal)
    np.testing.assert_array_almost_equal(result, jnp.array([0.0, 0.0, 0.0]))

def test_tetrahedron_volume():
    a = jnp.array([0.0, 0.0, 0.0])
    b = jnp.array([1.0, 0.0, 0.0])
    c = jnp.array([0.0, 1.0, 0.0])
    d = jnp.array([0.0, 0.0, 1.0])
    volume = tetrahedron_volume(a, b, c, d)
    assert jnp.isclose(volume, 1/6)

def test_uniform_scale_euclidean_basis():
    vectors = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    scaled = scale_along_basis(vectors, scale=2.0)
    expected = vectors * 2.0
    assert jnp.allclose(scaled, expected), f"Expected {expected}, got {scaled}"

def test_per_axis_scale_euclidean_basis():
    vectors = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    scale = jnp.array([2.0, 0.5])
    scaled = scale_along_basis(vectors, scale=scale)
    expected = vectors * scale
    assert jnp.allclose(scaled, expected), f"Expected {expected}, got {scaled}"

def test_scale_with_custom_basis():
    vectors = jnp.array([[1.0, 1.0]])
    # 45-degree rotated basis
    basis = jnp.array([
        [1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
        [-1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
    ])
    scale = jnp.array([2.0, 0.5])
    scaled = scale_along_basis(vectors, scale=scale, basis=basis)

    # Manual construction for expected result:
    # Rotate -> scale -> rotate back
    rotated = vectors @ basis.T
    scaled_manual = rotated * scale
    expected = scaled_manual @ basis
    assert jnp.allclose(scaled, expected, atol=1e-5), f"Expected {expected}, got {scaled}"

def test_output_shape_and_dtype():
    vectors = jnp.ones((10, 3), dtype=jnp.float32)
    result = scale_along_basis(vectors, scale=2.0)
    assert result.shape == vectors.shape
    assert result.dtype == jnp.float32

import jax.numpy as jnp
from GeoJax.core import scale_point_cloud_by_robust_axis_extent

def test_scale_point_cloud_by_robust_axis_extent_basic():
    # Create a vertical line with some outliers
    points = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 100.0, 0.0],  # outlier
        [0.0, -50.0, 0.0]   # outlier
    ])

    # Scale based on central percentiles, ignoring the extreme values
    scaled = scale_point_cloud_by_robust_axis_extent(points, axis=1, lower=10.0, upper=90.0)

    # Check that the central span was used for scaling (not influenced by outliers)
    magnitudes = jnp.linalg.norm(scaled, axis=1)
    central_mags = magnitudes[0:3]

    # Should not be huge values like the outliers would induce
    assert jnp.all(central_mags < 2.0), f"Unexpectedly large magnitudes: {central_mags}"
