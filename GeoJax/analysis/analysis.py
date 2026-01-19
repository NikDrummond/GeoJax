"""Statistical and geometric analysis utilities."""

from jax import jit, lax
import jax.numpy as jnp
from scipy.stats import chi2

from ..core.core import magnitude
from ..alignment.alignment import coord_eig_decomp


@jit
def mahalanobis_distance(X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Compute Mahalanobis distance for each row in X.

    Parameters
    ----------
    X : jnp.ndarray
        Input data points of shape (N, D).
    mean : jnp.ndarray
        Mean vector of shape (D,).
    cov : jnp.ndarray
        Covariance matrix of shape (D, D).

    Returns
    -------
    jnp.ndarray
        Mahalanobis distances, shape (N,).
    """
    diff = X - mean
    inv_cov = jnp.linalg.inv(cov + jnp.eye(cov.shape[0]) * 1e-6)
    return jnp.sqrt(jnp.sum(diff @ inv_cov * diff, axis=1))


@jit
def mahalanobis_squared(X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Compute squared Mahalanobis distance for each row in X.

    Parameters
    ----------
    X : jnp.ndarray
        Input data points of shape (N, D).
    mean : jnp.ndarray
        Mean vector of shape (D,).
    cov : jnp.ndarray
        Covariance matrix of shape (D, D).

    Returns
    -------
    jnp.ndarray
        Squared Mahalanobis distances, shape (N,).
    """
    diff = X - mean
    inv_cov = jnp.linalg.inv(cov + jnp.eye(cov.shape[0]) * 1e-6)
    return jnp.sum(diff @ inv_cov * diff, axis=1)


def detect_outliers_mahalanobis(X: jnp.ndarray, alpha: float = 0.99) -> jnp.ndarray:
    """Detect outliers based on Mahalanobis distance using a chi-squared threshold.

    Parameters
    ----------
    X : jnp.ndarray
        Input data points of shape (N, D).
    alpha : float, optional
        Confidence level for threshold (e.g., 0.99 for 99% confidence). Default is 0.99.

    Returns
    -------
    jnp.ndarray
        Boolean array indicating outliers, shape (N,).
    """
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
    """Compute ellipsoid semi-axis directions and lengths from covariance matrix.

    Parameters
    ----------
    cov : jnp.ndarray
        Covariance matrix of shape (D, D).
    scale : float, optional
        Scaling factor for axis lengths. Default is 1.0.

    Returns
    -------
    jnp.ndarray
        Ellipsoid axes (directions scaled by lengths), shape (D, D).
    """
    evals, evecs = jnp.linalg.eigh(cov)
    lengths = jnp.sqrt(evals) * scale
    return evecs.T * lengths[:, None]


@jit
def robust_proportional_dispersion(X: jnp.ndarray) -> jnp.ndarray:
    """Compute dispersion proportions along each eigenvector via robust covariance.

    Parameters
    ----------
    X : jnp.ndarray
        Input data points of shape (N, D).

    Returns
    -------
    jnp.ndarray
        Proportional dispersion along each eigenvector, shape (D,), sums to 1.
    """
    evals, _ = coord_eig_decomp(X, PCA=False)
    evals = jnp.clip(evals, min=0.0)
    return evals / jnp.sum(evals)