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

def test_robust_proportional_dispersion_output():
    from GeoJax.analysis import robust_proportional_dispersion
    import jax.numpy as jnp

    # Simple 2D example: spread more along x-axis
    X = jnp.array([
        [1.0, 0.0],
        [2.0, 0.1],
        [3.0, -0.1],
        [4.0, 0.05],
        [5.0, -0.05]
    ])

    # Should be more dispersed along the first principal component
    ratios = robust_proportional_dispersion(X)

    # Output should be 1D array of length equal to dimension
    assert ratios.shape == (2,), "Unexpected output shape"
    assert jnp.all(ratios >= 0), "All proportions must be non-negative"
    assert jnp.isclose(jnp.sum(ratios), 1.0, atol=1e-5), "Ratios must sum to 1"

    # Expect more dispersion in the first axis
    assert ratios[0] > ratios[1], f"Expected more dispersion in axis 0, got: {ratios}"
