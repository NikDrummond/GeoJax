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


# def test_detect_outliers_mahalanobis():
#     X = jnp.array([
#         [1.0, 2.0],
#         [1.1, 2.1],
#         [0.9, 1.9],
#         [50.0, 80.0]
#     ], dtype=jnp.float64)

#     outliers = detect_outliers_mahalanobis(X, alpha=0.99)
#     assert outliers[-1] == True, "Last point should be detected as an outlier"
#     assert jnp.sum(outliers) == 1, "Only one outlier should be detected"



def test_ellipsoid_axes_from_covariance():
    cov = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ])
    axes = ellipsoid_axes_from_covariance(cov)
    lengths = jnp.linalg.norm(axes, axis=1)
    expected = jnp.array([jnp.sqrt(2.0), 1.0, jnp.sqrt(0.5)])
    assert jnp.allclose(jnp.sort(lengths), jnp.sort(expected), atol=1e-5)

