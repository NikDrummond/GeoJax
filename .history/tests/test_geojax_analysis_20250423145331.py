# test_geojax_analysis.py
# Tests for GeoJax statistical analysis module

import pytest
import jax.numpy as jnp
from GeoJax import (
    mahalanobis_distance,
    detect_outliers_mahalanobis,
    ellipsoid_axes_from_covariance,
)

def test_mahalanobis_distance():
    x = jnp.array([[2.0, 2.0], [0.0, 0.0]])
    mean = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    d = mahalanobis_distance(x, mean, cov)
    assert jnp.allclose(d[0], jnp.sqrt(8), atol=1e-5)
    assert pytest.approx(d[1].item(), abs=1e-5) == 0.0

def test_detect_outliers():
    X = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0]  # clear outlier
    ])
    mask = detect_outliers_mahalanobis(X, threshold=3.0)
    assert jnp.sum(mask) == 1  # one outlier expected

def test_ellipsoid_axes():
    cov = jnp.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    axes = ellipsoid_axes_from_covariance(cov)
    assert axes.shape == (3, 3)
    assert jnp.allclose(jnp.linalg.norm(axes, axis=1), 1.0, atol=1e-6)
