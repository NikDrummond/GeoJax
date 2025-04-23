# test_geojax_projection.py
# Tests for GeoJax projection utilities

import pytest
import jax.numpy as jnp
from GeoJax import project_to_plane, project_to_sphere

def test_project_to_plane():
    point = jnp.array([1.0, 2.0, 3.0])
    plane_point = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    projected = project_to_plane(point, plane_point, normal)
    expected = jnp.array([1.0, 2.0, 0.0])
    assert jnp.allclose(projected, expected, atol=1e-6)

def test_project_to_sphere():
    points = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, -3.0]
    ])
    radius = 1.0
    center = jnp.array([0.0, 0.0, 0.0])
    projected = project_to_sphere(points, r=radius, c=center)
    # All points should now lie on unit sphere (distance == 1.0)
    norms = jnp.linalg.norm(projected, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)
