"""Statistical and geometric analysis utilities."""

from jax import jit, lax
import jax.numpy as jnp
from scipy.stats import chi2

from ..core import magnitude
from ..alignment.alignment import coord_eig_decomp


@jit
def mahalanobis_distance(X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Compute Mahalanobis distance for each row in X."""
    diff = X - mean
    inv_cov = jnp.linalg.inv(cov + jnp.eye(cov.shape[0]) * 1e-6)
    return jnp.sqrt(jnp.sum(diff @ inv_cov * diff, axis=1))


@jit
def mahalanobis_squared(X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    diff = X - mean
    inv_cov = jnp.linalg.inv(cov + jnp.eye(cov.shape[0]) * 1e-6)
    return jnp.sum(diff @ inv_cov * diff, axis=1)


def detect_outliers_mahalanobis(X: jnp.ndarray, alpha: float = 0.99) -> jnp.ndarray:
    """Detect outliers based on Mahalanobis distance using a chi-squared threshold."""
    N, D = X.shape
    mean = jnp.mean(X, axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / X.shape[0]
    cov += jnp.eye(D) * 1e-6

    sq_dists = mahalanobis_squared(X, mean, cov)
    threshold = chi2.ppf(alpha, df=D)
    return sq_dists > threshold


@jit
def ellipsoid_axes_from_covariance(cov: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
    """Compute ellipsoid semi-axis directions and lengths from covariance matrix."""
    evals, evecs = jnp.linalg.eigh(cov)
    lengths = jnp.sqrt(evals) * scale
    return evecs.T * lengths[:, None]


@jit
def robust_proportional_dispersion(X: jnp.ndarray) -> jnp.ndarray:
    """Compute dispersion proportions along each eigenvector via robust covariance."""
    evals, _ = coord_eig_decomp(X, PCA=False)
    evals = jnp.clip(evals, min=0.0)
    return evals / jnp.sum(evals)