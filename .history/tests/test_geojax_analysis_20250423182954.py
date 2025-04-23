import jax.numpy as jnp
import jax.random as jr
from GeoJax import (
    mahalanobis_distance,
    detect_outliers_mahalanobis,
    ellipsoid_axes_from_covariance,
    robust_proportional_dispersion,
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

def test_robust_proportional_dispersion_output():

    # Deterministic randomness
    key = jax.random.PRNGKey(42)

    # x-axis: wide spread, y-axis: tight spread
    x = jnp.linspace(-100.0, 100.0, 500)
    y = jax.random.normal(key, shape=(500,)) * 0.5
    points = jnp.stack([x, y], axis=1)

    ratios = robust_proportional_dispersion(points)

    assert ratios.shape == (2,), "Unexpected shape"
    assert jnp.isclose(jnp.sum(ratios), 1.0, atol=1e-5), "Ratios must sum to 1"
    assert ratios[0] > ratios[1], f"Expected more dispersion in axis 0, got: {ratios}"
