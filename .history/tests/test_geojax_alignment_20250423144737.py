# test_geojax_alignment.py
# Tests for alignment and decomposition functions in GeoJax

import pytest
import jax.numpy as jnp
from GeoJax import (
    coord_eig_decomp,
    align_point_cloud,
    minimum_theta,
)

def test_coord_eig_decomp_identity():
    # Points aligned with X axis
    pts = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    evals, evecs = coord_eig_decomp(pts, robust=False, center=True, PCA=False)
    assert jnp.allclose(evecs[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-4) or \
           jnp.allclose(evecs[0], jnp.array([-1.0, 0.0, 0.0]), atol=1e-4)

def test_align_point_cloud_basic():
    # Cube aligned with world axes
    cube = jnp.array([
        [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
        [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
    ])
    aligned = align_point_cloud(cube, order=jnp.array([0, 1, 2]),
                                 target_basis=jnp.eye(3),
                                 robust=False, center=True)
    # Should still be symmetric about origin
    assert jnp.allclose(jnp.mean(aligned, axis=0), jnp.zeros(3), atol=1e-6)

def test_minimum_theta_flips():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([-1.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    angle = minimum_theta(v1, v2, normal, to_degree=True)
    assert pytest.approx(angle, abs=1e-5) == 0.0