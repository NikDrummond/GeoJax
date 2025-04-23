### alignment.py
# alignment.py
# PCA and point cloud alignment utilities using JAX
# Includes: coord_eig_decomp, align_point_cloud, robust_covariance_mest

from jax import jit, lax
import jax.numpy as jnp
from functools import partial

@jit
def robust_covariance_mest(
    X: jnp.ndarray, c: float = 1.5, tol: float = 1e-6, max_iter: int = 100
) -> jnp.ndarray:
    """
    Compute a robust covariance matrix using a Huber-like M-estimator.

    Parameters
    ----------
    X : jnp.ndarray
        Input data of shape (n_samples, n_features).
    c : float
        Tuning constant (default: 1.5).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    jnp.ndarray
        Robust covariance matrix (n_features x n_features).
    """
    n, d = X.shape
    mu0 = jnp.mean(X, axis=0)
    sigma0 = jnp.cov(X - mu0, rowvar=False, bias=True)
    state = (mu0, sigma0, 0, False)

    def cond_fn(state):
        _, _, i, done = state
        return jnp.logical_and(i < max_iter, jnp.logical_not(done))

    def body_fn(state):
        mu, sigma, i, _ = state
        diff = X - mu
        inv_sigma = jnp.linalg.inv(sigma + jnp.eye(d) * 1e-6)
        mahal = jnp.sum((diff @ inv_sigma) * diff, axis=1)
        weights = jnp.where(mahal < c**2, 1.0, c**2 / mahal)
        new_mu = jnp.sum(weights[:, None] * X, axis=0) / jnp.sum(weights)
        wdiff = X - new_mu
        new_sigma = (wdiff.T @ (weights[:, None] * wdiff)) / jnp.sum(weights)
        converged = jnp.linalg.norm(new_mu - mu) < tol
        return (new_mu, new_sigma, i + 1, converged)

    _, sigma_final, _, _ = lax.while_loop(cond_fn, body_fn, state)
    return sigma_final


@partial(jit, static_argnames=['robust', 'center', 'PCA', 'sort', 'transpose'])
def coord_eig_decomp(
    coords: jnp.ndarray,
    robust: bool = True,
    center: bool = False,
    PCA: bool = True,
    sort: bool = True,
    transpose: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Eigendecomposition of the covariance matrix of coordinates.

    Parameters
    ----------
    coords : jnp.ndarray
        Input coordinate array (n_samples, n_features).
    robust : bool
        Whether to use robust covariance estimation.
    center : bool
        Whether to center the data.
    PCA : bool
        Normalize eigenvalues to sum to 1.
    sort : bool
        Sort eigenvalues/eigenvectors in descending order.
    transpose : bool
        Return eigenvectors as rows if True.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Eigenvalues, eigenvectors.
    """
    coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, coords)

    def degenerate_case(_: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.zeros(coords.shape[1]), jnp.eye(coords.shape[1])

    def eig_case(coords):
        cov = lax.cond(
            robust,
            lambda c: robust_covariance_mest(c),
            lambda c: jnp.cov(c, rowvar=False, bias=True),
            coords,
        )
        evals, evecs = jnp.linalg.eigh(cov)
        evals = lax.cond(PCA, lambda e: e / jnp.sum(e), lambda e: e, evals)
        evals, evecs = lax.cond(
            sort,
            lambda _: (evals[jnp.argsort(evals)[::-1]], evecs[:, jnp.argsort(evals)[::-1]]),
            lambda _: (evals, evecs),
            operand=None
        )
        evecs = lax.cond(transpose, lambda e: e.T, lambda e: e, evecs)
        return evals, evecs

    is_degenerate = jnp.all(jnp.isclose(coords, coords[0]))
    return lax.cond(is_degenerate, degenerate_case, eig_case, coords)


@jit
def align_point_cloud(
    coords: jnp.ndarray,
    order: jnp.ndarray,
    target_basis: jnp.ndarray,
    robust: bool = True,
    center: bool = True,
    center_coord: jnp.ndarray = jnp.zeros(3),
) -> jnp.ndarray:
    """
    Align a point cloud to a target basis using PCA decomposition and axis reordering.

    Parameters
    ----------
    coords : jnp.ndarray
        Input point cloud (N, 3).
    order : jnp.ndarray
        Reordering of eigenvector indices.
    target_basis : jnp.ndarray
        Target basis (3, 3).
    robust : bool
        Whether to use robust PCA.
    center : bool
        Center cloud before/after alignment.
    center_coord : jnp.ndarray
        Optional explicit center coordinate.

    Returns
    -------
    jnp.ndarray
        Aligned point cloud.
    """
    def center_on_mean(c):
        return c - jnp.mean(c, axis=0)

    def center_on_point(c):
        return c - jnp.mean(c, axis=0) + center_coord

    centered = lax.cond(
        center,
        lambda c: lax.cond(jnp.all(center_coord == 0), center_on_mean, center_on_point, c),
        lambda c: c,
        coords,
    )

    evals, eigvecs = coord_eig_decomp(
        centered, robust=robust, center=True, PCA=True, sort=True, transpose=True
    )
    sorted_vecs = eigvecs[jnp.argsort(evals)[::-1]]
    E = sorted_vecs[order]
    signs = jnp.sign(jnp.sum(E * target_basis, axis=1))
    E_adjusted = E * signs[:, None]
    R = jnp.matmul(jnp.linalg.pinv(target_basis), E_adjusted)
    rotated = jnp.matmul(centered, R.T)

    recentered = lax.cond(
        center,
        lambda r: lax.cond(jnp.all(center_coord == 0), center_on_mean, center_on_point, r),
        lambda r: r,
        rotated,
    )
    return recentered

@jit
def minimum_theta(R: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the minimum angle (in radians) of rotation from a rotation matrix.

    Parameters
    ----------
    R : jnp.ndarray
        A 3x3 rotation matrix.

    Returns
    -------
    jnp.ndarray
        Angle of rotation.
    """
    trace = jnp.trace(R)
    return jnp.arccos((trace - 1.0) / 2.0)

@jit
def alignment_matrix(from_basis: jnp.ndarray, to_basis: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the rotation matrix aligning one basis to another.

    Parameters
    ----------
    from_basis : jnp.ndarray
        Source basis (3, 3).
    to_basis : jnp.ndarray
        Target basis (3, 3).

    Returns
    -------
    jnp.ndarray
        Rotation matrix (3, 3).
    """
    return jnp.matmul(to_basis, jnp.linalg.pinv(from_basis))
