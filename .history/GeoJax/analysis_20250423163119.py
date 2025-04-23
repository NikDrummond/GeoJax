### analysis.py

# analysis.py
# Statistical and geometric analysis utilities

import jax.numpy as jnp
from jax import jit, lax
from .core import magnitude

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
def detect_outliers_mahalanobis(
    X: jnp.ndarray, threshold: float = 3.0
) -> jnp.ndarray:
    """
    Detect outliers based on Mahalanobis distance.

    Parameters
    ----------
    X : jnp.ndarray
        Data array (N, D).
    threshold : float
        Threshold multiplier for outlier detection.

    Returns
    -------
    jnp.ndarray
        Boolean array where True indicates an outlier.
    """
    mean = jnp.mean(X, axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / X.shape[0]
    cov += jnp.eye(X.shape[1]) * 1e-6

    dist = mahalanobis_distance(X, mean, cov)
    return dist > threshold


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
