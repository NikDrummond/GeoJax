### analysis.py

# analysis.py
# Statistical and geometric analysis utilities

import jax.numpy as jnp
from jax import jit, lax
from scipy.stats import chi2
from .core import magnitude
from .alignment import robust_covariance_mest

@jit
def mahalanobis_distance(X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Mahalanobis distance for each row in X.

    Parameters
    ----------
    X : jnp.ndarray
        Data points of shape (N, D).
    mean : jnp.ndarray
        Mean vector of shape (D,).
    cov : jnp.ndarray
        Covariance matrix of shape (D, D).

    Returns
    -------
    jnp.ndarray
        Mahalanobis distances (N,).
    """
    diff = X - mean
    inv_cov = jnp.linalg.inv(cov + jnp.eye(cov.shape[0]) * 1e-6)
    return jnp.sqrt(jnp.sum(diff @ inv_cov * diff, axis=1))



@jit
def mahalanobis_squared(X: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    diff = X - mean
    inv_cov = jnp.linalg.inv(cov + jnp.eye(cov.shape[0]) * 1e-6)
    return jnp.sum(diff @ inv_cov * diff, axis=1)

def detect_outliers_mahalanobis(
    X: jnp.ndarray, alpha: float = 0.99
) -> jnp.ndarray:
    """
    Detect outliers based on Mahalanobis distance using a chi-squared threshold.

    Parameters
    ----------
    X : jnp.ndarray
        Input data of shape (N, D).
    alpha : float
        Significance level (e.g., 0.99 for 99% confidence).

    Returns
    -------
    jnp.ndarray
        Boolean mask of shape (N,), where True indicates an outlier.
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
    """
    Compute ellipsoid semi-axis directions and lengths from covariance matrix.

    Parameters
    ----------
    cov : jnp.ndarray
        Covariance matrix (3x3).
    scale : float
        Scaling factor applied to axes (e.g. confidence level).

    Returns
    -------
    jnp.ndarray
        Array of shape (3, 3): scaled semi-axes as rows.
    """
    evals, evecs = jnp.linalg.eigh(cov)
    lengths = jnp.sqrt(evals) * scale
    return evecs.T * lengths[:, None]

@jit
def robust_proportional_dispersion(X: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the proportion of total dispersion along each eigenvector
    using a robust M-estimated covariance matrix.

    Parameters
    ----------
    X : jnp.ndarray
        Input point cloud of shape (N, D).

    Returns
    -------
    jnp.ndarray
        Normalized variance proportions along each principal axis (D,).
    """
    evals, _ = coord_eig_decomp
    evals = jnp.clip(evals, a_min=0.0)
    return evals / jnp.sum(evals)

