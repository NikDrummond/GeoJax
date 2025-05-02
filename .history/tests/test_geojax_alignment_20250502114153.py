python3 import jax.numpy as jnp
import numpy as np
from GeoJax.alignment import (
    robust_covariance_mest,
    coord_eig_decomp,
    align_point_cloud,
    minimum_theta,
    alignment_matrix,
)

def test_robust_covariance_mest_returns_symmetric_matrix():
    X = jnp.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])
    cov = robust_covariance_mest(X)
    assert cov.shape == (3, 3)
    assert jnp.allclose(cov, cov.T)

def test_coord_eig_decomp_shapes_and_order():
    coords = jnp.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])
    evals, evecs = coord_eig_decomp(coords)
    assert evals.shape == (3,)
    assert evecs.shape == (3, 3)
    assert jnp.allclose(jnp.linalg.norm(evecs, axis=1), 1.0, atol=1e-3)

def test_align_point_cloud_shape_consistency():
    coords = jnp.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
    order = jnp.array([0, 1, 2])
    basis = jnp.eye(3)
    result = align_point_cloud(coords, order, basis)
    assert result.shape == coords.shape

def test_minimum_theta_identity_rotation():
    R = jnp.eye(3)
    theta = minimum_theta(R)
    assert jnp.isclose(theta, 0.0)

def test_alignment_matrix_identity_case():
    basis = jnp.eye(3)
    R = alignment_matrix(basis, basis)
    assert jnp.allclose(R, jnp.eye(3), atol=1e-6)
