import jax.numpy as jnp
from GeoJax import (
    mahalanobis_distance,
    detect_outliers_mahalanobis,
    ellipsoid_axes_from_covariance,
)


def test_mahalanobis_distance():
    X = jnp.array([[1.0, 2.0], [2.0, 1.0], [4.0, 4.0]])
    mean = jnp.array([2.0, 2.0])
    cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    distances = mahalanobis_distance(X, mean, cov)
    assert distances.shape == (3,)
    assert jnp.all(distances >= 0)


def test_detect_outliers_mahalanobis():
    X = jnp.array([[1.0, 2.0], [2.0, 2.0], [100.0, 100.0]])
    outliers = detect_outliers_mahalanobis(X, threshold=5.0)
    assert outliers.shape == (3,)
    assert outliers[-1]  # The last point should be an outlier


def test_ellipsoid_axes_from_covariance():
    cov = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
    ])
    axes = ellipsoid_axes_from_covariance(cov, scale=1.0)
    assert axes.shape == (3, 3)
    lengths = jnp.linalg.norm(axes, axis=1)
    expected = jnp.sqrt(jnp.array([2.0, 1.0, 0.5]))
    assert jnp.allclose(lengths, expected, atol=1e-5)
