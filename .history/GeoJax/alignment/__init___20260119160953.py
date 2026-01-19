"""PCA and point cloud alignment utilities using JAX."""

from functools import partial

from jax import jit, lax
import jax.numpy as jnp


@jit
def robust_covariance_mest(
    X: jnp.ndarray, c: float = 1.5, tol: float = 1e-6, max_iter: int = 100
) -> jnp.ndarray:
    """Compute a robust covariance matrix using a Huber-like M-estimator."""
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


@partial(jit, static_argnames=["robust", "center", "PCA", "sort", "transpose"])
def coord_eig_decomp(
    coords: jnp.ndarray,
    robust: bool = True,
    center: bool = False,
    PCA: bool = True,
    sort: bool = True,
    transpose: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition of the covariance matrix of coordinates."""
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
            operand=None,
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
    """Align a point cloud to a target basis using PCA decomposition and axis reordering."""

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

    evals, eigvecs = lax.cond(
        robust,
        lambda _: coord_eig_decomp(centered, robust=True, center=True, PCA=True, sort=True, transpose=True),
        lambda _: coord_eig_decomp(centered, robust=False, center=True, PCA=True, sort=True, transpose=True),
        operand=None,
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
    """Compute the minimum angle (in radians) of rotation from a rotation matrix."""
    trace = jnp.trace(R)
    return jnp.arccos((trace - 1.0) / 2.0)


@jit
def alignment_matrix(from_basis: jnp.ndarray, to_basis: jnp.ndarray) -> jnp.ndarray:
    """Compute the rotation matrix aligning one basis to another."""
    return jnp.matmul(to_basis, jnp.linalg.pinv(from_basis))


__all__ = [
    "robust_covariance_mest",
    "coord_eig_decomp",
    "align_point_cloud",
    "minimum_theta",
    "alignment_matrix",
]
