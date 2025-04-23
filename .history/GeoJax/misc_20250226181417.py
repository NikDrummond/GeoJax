from GeoJax import vector_geometry
import jax.numpy as jnp
from jax import jit

from vector_geometry import 
def normalize_angle_array(angles):
    """
    Normalize an array of angles so that they lie in the "positive" range.
    
    If the maximum absolute value in the array is <= 2π (~6.283), the function
    assumes the angles are in radians and normalizes them to the range [0, 2π).
    Otherwise, it assumes the angles are in degrees and normalizes them to [0, 360).
    
    Parameters
    ----------
    angles : array-like of float
        Array of angles (in radians or degrees).
    
    Returns
    -------
    norm_angles : numpy.ndarray
        Array of normalized angles, in the same unit as the input.
    unit : str
        The unit of the input angles ('radians' or 'degrees').
    """
    angles = np.asarray(angles)
    
    # Determine the unit based on the range of values.
    # Heuristic: if max(abs(angle)) <= 2π then assume radians.
    if np.max(np.abs(angles)) <= 2 * np.pi:

        # For radians, add 2π and take modulo 2π.
        norm_angles = (angles + 2 * np.pi) % (2 * np.pi)
    else:

        # For degrees, add 360 and take modulo 360.
        norm_angles = (angles + 360) % 360
    
    return norm_angles

@jit
def _center_points(pnts: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
    """ Jitterd function to center points"""
    return pnts - center

def center_points(pnts:jnp.ndarray, center:jnp.ndarray | None = None) -> jnp.ndarray:
    """Center and array of points on a specified coordinate

    Parameters
    ----------
    pnts : jnp.ndarray
        a point, or array of points
    center : jnp.ndarray | None
        a coordinate to center points on. If None, we will center the points on their mean.

    Returns
    -------
    jnp.ndarray
        points, centred on given coordinate

    Raises
    ------
    ValueError
        If the Given coordinate doesn't have the same dimensions as the given array
    """

    if center is None:
        center = pnts.mean(axis = 0)

    if pnts.shape[-1] == center.shape[0]:
        return _center_points(pnts,center)
    else:
        raise ValueError('Input points and coordinate to centre to must have same dimensions')

@jit
def _mat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a @ b
@jit
def robust_covariance_mest(
    X: jnp.ndarray, c: float = 1.5, tol: float = 1e-6, max_iter: int = 100
) -> jnp.ndarray:
    """
    Compute a robust covariance matrix using an M‐estimator with a Huber‐like weighting scheme.

    Parameters
    ----------
    X : jnp.ndarray
        Input data of shape (n_samples, n_features).
    c : float, optional
        Tuning constant for the Huber‐like weight function (default: 1.5).
    tol : float, optional
        Convergence tolerance (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 100).

    Returns
    -------
    jnp.ndarray
        Robust covariance matrix of shape (n_features, n_features).
    """
    n, d = X.shape
    # Initialize with the classical mean and covariance.
    mu0 = jnp.mean(X, axis=0)
    diff0 = X - mu0
    sigma0 = jnp.cov(diff0, rowvar=False, bias=True)

    # State: (current mean, current covariance, iteration counter, converged flag)
    state0 = (mu0, sigma0, 0, False)

    def cond_fn(state):
        mu, sigma, i, converged = state
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_fn(state):
        mu, sigma, i, _ = state
        diff = X - mu
        # Add a small regularization term for numerical stability.
        inv_sigma = jnp.linalg.inv(sigma + jnp.eye(d) * 1e-6)
        # Compute squared Mahalanobis distances.
        mahal = jnp.sum((diff @ inv_sigma) * diff, axis=1)
        # Compute weights: downweight points with large Mahalanobis distances.
        weights = jnp.where(mahal < c**2, 1.0, c**2 / mahal)
        # Update the weighted mean.
        new_mu = jnp.sum(weights[:, None] * X, axis=0) / jnp.sum(weights)
        weighted_diff = X - new_mu
        # Update the weighted covariance.
        new_sigma = (weighted_diff.T @ (weights[:, None] * weighted_diff)) / jnp.sum(
            weights
        )
        # Check convergence (using the change in the mean).
        converged = jnp.linalg.norm(new_mu - mu) < tol
        return (new_mu, new_sigma, i + 1, converged)

    mu_final, sigma_final, _, _ = lax.while_loop(cond_fn, body_fn, state0)
    return sigma_final


# @jit
# def coord_eig_decomp(
#     coords: jnp.ndarray,
#     robust: bool = True,
#     center: bool = False,
#     PCA: bool = True,
#     sort: bool = True,
#     transpose: bool = True,
# ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Compute the eigendecomposition of the covariance matrix for a set of coordinates,
#     with options for robust covariance estimation and PCA normalization.

#     Parameters
#     ----------
#     coords : jnp.ndarray
#         Input coordinates of shape (n_samples, n_features).
#     robust : bool, optional
#         If True, use robust covariance estimation (default: True).
#     center : bool, optional
#         If True, center the data by subtracting the mean (default: False).
#     PCA : bool, optional
#         If True, normalize eigenvalues so that they represent the fraction of variance explained (default: True).
#     sort : bool, optional
#         If True, sort eigenvalues and eigenvectors in descending order (default: True).
#     transpose : bool, optional
#         If True, return eigenvectors as rows rather than columns (default: True).

#     Returns
#     -------
#     tuple[jnp.ndarray, jnp.ndarray]
#         A tuple containing:
#           - eigenvalues (as a 1D array)
#           - eigenvectors (as a 2D array, transposed if requested)
#     """
#     # (1) Conditionally center the coordinates.
#     coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, coords)

#     # (2) Compute the covariance matrix using robust estimation or the standard method.
#     cov = lax.cond(
#         robust,
#         lambda c: robust_covariance_mest(c),
#         lambda c: jnp.cov(c, rowvar=False, bias=True),
#         coords,
#     )

#     # (3) Compute the eigendecomposition (using eigh for symmetric matrices).
#     evals, evecs = jnp.linalg.eigh(cov)

#     # (4) Conditionally normalize eigenvalues so that they sum to 1 (PCA mode).
#     evals = lax.cond(PCA, lambda e: e / jnp.sum(e), lambda e: e, evals)

#     # (5) Conditionally sort eigenvalues (and corresponding eigenvectors) in descending order.
#     def sort_fn(args):
#         ev, evec = args
#         sort_inds = jnp.argsort(ev)[::-1]
#         return (ev[sort_inds], evec[:, sort_inds])

#     evals, evecs = lax.cond(sort, sort_fn, lambda args: args, (evals, evecs))

#     # (6) Conditionally transpose the eigenvector matrix.
#     evecs = lax.cond(transpose, lambda ev: ev.T, lambda ev: ev, evecs)

#     return evals, evecs

# @jit
# def coord_eig_decomp(
#     coords: jnp.ndarray,
#     robust: bool = True,
#     center: bool = False,
#     PCA: bool = True,
#     sort: bool = True,
#     transpose: bool = True,
# ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Compute the eigendecomposition of the covariance matrix for a set of coordinates,
#     with options for robust covariance estimation and PCA normalization.
#     """
#     # Center the coordinates if needed.
#     coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, coords)

#     # Handle degenerate cases: If all points are identical or degenerate, return zeros.
#     if jnp.allclose(coords, coords[0]):
#         return jnp.zeros(3), jnp.eye(3)

#     # Compute covariance matrix using robust method or standard method.
#     cov = lax.cond(
#         robust,
#         lambda c: robust_covariance_mest(c),
#         lambda c: jnp.cov(c, rowvar=False, bias=True),
#         coords,
#     )

#     # Compute eigen decomposition.
#     evals, evecs = jnp.linalg.eigh(cov)

#     # Normalize eigenvalues if PCA is enabled.
#     evals = lax.cond(PCA, lambda e: e / jnp.sum(e), lambda e: e, evals)

#     # Sort eigenvalues and eigenvectors in descending order.
#     def sort_fn(args):
#         ev, evec = args
#         sort_inds = jnp.argsort(ev)[::-1]
#         return ev[sort_inds], evec[:, sort_inds]

#     evals, evecs = lax.cond(sort, sort_fn, lambda args: args, (evals, evecs))

#     # Ensure handedness (right-hand rule) by checking determinant.
#     if jnp.linalg.det(evecs) < 0:
#         evecs = evecs.at[:, 0].set(-evecs[:, 0])  # Flip one axis to maintain orientation.

#     # Transpose if requested.
#     evecs = lax.cond(transpose, lambda ev: ev.T, lambda ev: ev, evecs)

#     return evals, evecs


@jit
def minimum_theta(
    v1: jnp.ndarray, v2: jnp.ndarray, plane_normal: jnp.ndarray, to_degree: bool = False
) -> jnp.ndarray:
    """
    Compute the minimal signed angle between v1 and the undirected line spanned by v2,
    measured on the projection onto the plane defined by plane_normal.

    Because the line defined by v2 is undirected, we consider both v2 and -v2.
    If the absolute signed angle between v1 and v2 exceeds π/2, the alternative (flipped) angle
    is used, ensuring the result lies in [-π/2, π/2].

    Parameters:
      v1: 1D array representing the first vector.
      v2: 1D array representing the second vector (defines a line, direction is undirected).
      plane_normal: 1D array representing the normal of the plane.
      to_degree: if True, return the angle in degrees; otherwise, in radians.

    Returns:
      A JAX array scalar (or batch) with the minimal signed angle.
    """
    # Compute the signed angle (in radians) between v1 and v2 (after projection).
    angle_rad = signed_angle(v1, v2, plane_normal, to_degree=False)

    # Adjust: if the absolute angle exceeds π/2, flip by subtracting sign(angle)*π.
    minimal_angle_rad = jnp.where(
        jnp.abs(angle_rad) > (jnp.pi / 2),
        angle_rad - jnp.sign(angle_rad) * jnp.pi,
        angle_rad,
    )

    # Optionally convert to degrees using lax.cond for JIT compatibility.
    return lax.cond(to_degree, lambda a: jnp.degrees(a), lambda a: a, minimal_angle_rad)


@jit
def rotation_matrix_from_rotvec(rot_vec: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a 3x3 rotation matrix from a rotation vector (axis * theta)
    using Rodrigues rotation formula.

    Parameters
    ----------
    rot_vec : jnp.ndarray
        A 1D array of shape (3,). The rotation vector; its norm is the angle.

    Returns
    -------
    jnp.ndarray
        A 3x3 rotation matrix.
    """
    # Compute the rotation angle.
    angle = jnp.linalg.norm(rot_vec)

    # When the angle is nearly zero, return the identity matrix.
    def nonzero(_):
        u = rot_vec / angle
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        one_minus_cos = 1 - cos_a
        # Build the skew-symmetric matrix of u.
        u_cross = jnp.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        u_outer = jnp.outer(u, u)
        R = cos_a * jnp.eye(3) + one_minus_cos * u_outer + sin_a * u_cross
        return R

    def zero(_):
        return jnp.eye(3)

    # Use lax.cond to select the branch in a JIT compatible way.
    return lax.cond(angle > 1e-8, nonzero, zero, operand=None)


@jit
def rotate_around_axis(coords: jnp.ndarray, theta, axis: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a set of coordinates by a signed angle around a given axis.

    This function replicates the behavior of:

        def rotate_around_axis(coords, theta, axis):
            rot_vec = axis * theta
            rot = R.from_rotvec(rot_vec)
            coords = rot.apply(coords)
            return coords

    Parameters
    ----------
    coords : jnp.ndarray
        An array of coordinates of shape (N, 2) or (N, 3).
        For 2D input (x, y), the rotation is performed about the provided 3D axis
        by temporarily lifting the points to 3D.
    theta : float or scalar-like
        The rotation angle (in radians). The sign indicates rotation direction.
    axis : jnp.ndarray
        A 1D array of shape (3,) specifying the rotation axis.

    Returns
    -------
    jnp.ndarray
        The rotated coordinates in the same shape as the input.
        (For 2D input, returns an array of shape (N, 2).)
    """
    # Compute the rotation vector.
    rot_vec = axis * theta
    # Compute the rotation matrix from the rotation vector.
    R = rotation_matrix_from_rotvec(rot_vec)

    # Determine if the input is 2D or 3D.
    orig_dim = coords.shape[1]
    # We use a Python conditional here because the coordinate dimension is static.
    if orig_dim == 2:
        # Lift 2D points to 3D by appending a zero z-coordinate.
        coords_3d = jnp.concatenate([coords, jnp.zeros((coords.shape[0], 1))], axis=1)
    elif orig_dim == 3:
        coords_3d = coords
    else:
        raise ValueError("Coordinates must have 2 or 3 columns.")

    # Rotate the (3D) coordinates. (We multiply by R^T since points are row vectors.)
    rotated_3d = jnp.dot(coords_3d, R.T)

    # If the input was 2D, discard the third coordinate.
    if orig_dim == 2:
        return rotated_3d[:, :2]
    else:
        return rotated_3d


# @jit
# def align_point_cloud(
#     coords: jnp.ndarray,
#     order: jnp.ndarray,
#     target_basis: jnp.ndarray,
#     robust: bool,
#     center: bool,
#     center_coord: jnp.ndarray = jnp.zeros(3),
# ) -> jnp.ndarray:
#     """Aligns a point cloud by centering it (using a mean or provided offset), extracting its eigenvectors via a robust decomposition, reordering and sign-correcting them against a target basis, and finally applying the corresponding rotation.

#     Parameters
#     ----------
#     coords : jnp.ndarray
#         The input point cloud coordinates.
#     order : jnp.ndarray
#         Indices to reorder the eigenvectors.
#     target_basis : jnp.ndarray
#         The target basis for aligning the eigenvectors.
#     robust : bool
#         If True, use robust eigen decomposition.
#     center : bool or jnp.ndarray
#         If a boolean, indicates whether to center by subtracting the mean (True) or leave as is (False).
#         Otherwise, it is used directly as the centering offset.

#     Returns
#     -------
#     jnp.ndarray
#         The rotated and aligned point cloud coordinates.
#     """
#     # Center the point cloud.
#     # if isinstance(center, bool):
#     #     centered = coords - (jnp.mean(coords, axis=0) if center else 0)
#     # else:
#     #     centered = coords - center

#     # not 100% this will work and it is a bit hacky
#     coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, coords)
#     centered = coords + (center_coord - coords.mean(axis=0))

#     # Get the eigen decomposition.
#     _, eigvecs = coord_eig_decomp(centered, robust, False, False, True, True)
#     # Reorder and sign-correct the eigenvectors.
#     E = jnp.take(eigvecs, order, axis=0)
#     dots = jnp.sum(E * target_basis, axis=1, keepdims=True)
#     E_adjusted = E * jnp.where(dots < 0, -1.0, 1.0)
#     # Compute overall rotation directly:
#     R_total = jnp.matmul(target_basis.T, E_adjusted)
#     rotated = jnp.matmul(centered, R_total.T)
#     # we may need to re-center to the mean if wanted
#     coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, rotated)
#     centered = coords + (center_coord - coords.mean(axis=0))
#     return centered

# @jit
# def align_point_cloud(
#     coords: jnp.ndarray,
#     order: jnp.ndarray,
#     target_basis: jnp.ndarray,
#     robust: bool,
#     center: bool,
#     center_coord: jnp.ndarray = jnp.zeros(3),
# ) -> jnp.ndarray:
#     """
#     Aligns a point cloud by centering, performing eigen decomposition, reordering and sign-correcting
#     eigenvectors, and applying the corresponding rotation.
#     """
#     # Center the point cloud.
#     coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, coords)
#     centered = coords + (center_coord - coords.mean(axis=0))

#     # Compute eigen decomposition.
#     _, eigvecs = coord_eig_decomp(centered, robust, center=True, PCA=True, sort=True, transpose=True)

#     # Reorder eigenvectors.
#     E = jnp.take(eigvecs, order, axis=0)

#     # Adjust signs to match target basis.
#     signs = jnp.sign(jnp.sum(E * target_basis, axis=1))
#     E_adjusted = E * signs[:, None]

#     # Handle non-orthogonal target bases by using pseudo-inverse.
#     R_total = jnp.matmul(jnp.linalg.pinv(target_basis), E_adjusted)

#     # Apply the rotation.
#     rotated = jnp.matmul(centered, R_total.T)

#     # Re-center if needed.
#     coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, rotated)
#     centered = coords + (center_coord - coords.mean(axis=0))

#     # Ensure scale preservation by normalizing back to original scale.
#     scale_factor = jnp.linalg.norm(coords) / jnp.linalg.norm(rotated)
#     aligned = rotated * scale_factor

#     return aligned


@jit
def coord_eig_decomp(
    coords: jnp.ndarray,
    robust: bool = True,
    center: bool = False,
    PCA: bool = True,
    sort: bool = True,
    transpose: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the eigendecomposition of the covariance matrix for a set of coordinates.
    """
    coords = lax.cond(center, lambda c: c - jnp.mean(c, axis=0), lambda c: c, coords)

    # Handle degenerate cases using lax.cond
    def degenerate_case(_):
        return jnp.zeros(coords.shape[1]), jnp.eye(coords.shape[1])

    def non_degenerate_case(coords):
        # Compute covariance matrix
        cov = lax.cond(
            robust,
            lambda c: robust_covariance_mest(c),
            lambda c: jnp.cov(c, rowvar=False, bias=True),
            coords,
        )
        evals, evecs = jnp.linalg.eigh(cov)
        evals = lax.cond(PCA, lambda e: e / jnp.sum(e), lambda e: e, evals)
        evals, evecs = lax.cond(
            sort, lambda x: (x[0][::-1], x[1][:, ::-1]), lambda x: x, (evals, evecs)
        )
        evecs = lax.cond(transpose, lambda e: e.T, lambda e: e, evecs)
        return evals, evecs

    # Check if all points are identical (degenerate case)
    is_degenerate = jnp.all(jnp.isclose(coords, coords[0]))

    return lax.cond(is_degenerate, degenerate_case, non_degenerate_case, coords)


@jit
def align_point_cloud(
    coords: jnp.ndarray,
    order: jnp.ndarray,
    target_basis: jnp.ndarray,
    robust: bool,
    center: bool,
    center_coord: jnp.ndarray = jnp.zeros(3),
) -> jnp.ndarray:
    """
    Aligns a point cloud by centering, performing eigen decomposition, reordering and sign-correcting
    eigenvectors, and applying the corresponding rotation.
    """

    def center_on_mean(c):
        return c - jnp.mean(c, axis=0)

    def center_on_point(c):
        return c - jnp.mean(c, axis=0) + center_coord

    # Conditionally center the point cloud
    centered = lax.cond(
        center,
        lambda c: lax.cond(
            jnp.all(center_coord == 0), center_on_mean, center_on_point, c
        ),
        lambda c: c,
        coords,
    )

    # Compute eigen decomposition
    evals, eigvecs = coord_eig_decomp(
        centered, robust, center=True, PCA=True, sort=True, transpose=True
    )

    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = jnp.argsort(evals)[::-1]
    eigvecs_sorted = eigvecs[sort_indices]

    # Apply order to the sorted eigenvectors
    E = eigvecs_sorted[order]

    # Adjust signs for each eigenvector based on target basis
    dots = jnp.sum(E * target_basis, axis=1)
    E_adjusted = E * jnp.where(dots < 0, -1.0, 1.0)[:, None]

    # Compute rotation matrix
    R_total = jnp.matmul(jnp.linalg.pinv(target_basis), E_adjusted)

    # Apply rotation
    rotated = jnp.matmul(centered, R_total.T)

    # Conditionally re-center the point cloud after rotation
    aligned = lax.cond(
        center,
        lambda r: lax.cond(
            jnp.all(center_coord == 0), center_on_mean, center_on_point, r
        ),
        lambda r: r,
        rotated,
    )

    return aligned


@jit
def scale_coords(coords: jnp.ndarray, s) -> jnp.ndarray:
    """
    Scale the given coordinates by s.

    Parameters
    ----------
    coords : jnp.ndarray
        An array of shape (N, n) where n is either 2 or 3.
    s : scalar or jnp.ndarray
        If a scalar, each coordinate is multiplied by s.
        If a 1D array of length n, the jth column in coords is scaled by s[j].

    Returns
    -------
    jnp.ndarray
        The scaled coordinates.
    """
    s = jnp.asarray(s)
    # Optional: if s is a 1D array, assert its length equals number of columns.
    if s.ndim == 1:
        # Note: This assert is executed at trace time.
        assert (
            s.shape[0] == coords.shape[1]
        ), "Length of s must equal number of coordinate columns."
    return coords * s

@jit
def project_to_sphere(arr: jnp.ndarray, r: float, c: jnp.ndarray) -> jnp.ndarray:
    """Project points onto a sphere with a given radius and center.

    Parameters
    ----------
    arr : jnp.ndarray
        Input array of points.
    r : float
        Radius of the target sphere.
    c : jnp.ndarray
        Center offset to apply before projection.

    Returns
    -------
    jnp.ndarray
        Points scaled and shifted to lie on the sphere.
    """
    l = magnitude(arr)
    s = (r / jnp.expand_dims(l, axis=-1)) * (arr - c)
    return s

"""
### To Do

- rotate to ensure given points are in given quadrants.

"""


@jit
def _flip_towards(starts: jnp.ndarray, stops: jnp.ndarray) -> jnp.ndarray:

    # figure out which need to be flipped - if end is further than start
    flip = magnitude(stops) > magnitude(starts)

    # stops.at[flip].set(2 * starts[flip] - stops[flip])
    new_stops = jnp.where(flip[..., None], 2 * starts - stops, stops)
    return new_stops

@jit
def _flip_away(starts: jnp.ndarray, stops: jnp.ndarray) -> jnp.ndarray:

    # figure out which need to be flipped - if end is closer than start
    flip = magnitude(stops) < magnitude(starts)

    # stops.at[flip].set(2 * starts[flip] - stops[flip])
    new_stops = jnp.where(flip[..., None], 2 * starts - stops, stops)
    return new_stops
    
def origin_flip(starts: jnp.ndarray,stops:jnp.ndarray,method: str = 'away') -> jnp.ndarray:
    """Given a set of vectors defined by their start and end points, flips the end points so they
    are either oriented away from (if method is away) or towards (if method is towards) the origin.

    Parameters
    ----------
    starts : jnp.ndarray
        set of vector start points 
    stops : jnp.ndarray
        set of vector end points
    method : str, optional
        If away, will flip end points so they are further form the origin than start points.
        Alternatively, if towards will flip end points so they are closer to the origin than the start points.
        By default 'away'

    Returns
    -------
    jnp.ndarray
        End points array, with points flipped where necessary

    Raises
    ------
    ValueError
        If neither away nor towards is given as a method
    """
    allowed_methods = ['away','towards']
    if method == 'away':
        return _flip_away(starts,stops)
    elif method == 'towards':
        return _flip_towards(starts,stops)
    else:
        raise ValueError(f'Given method {method} is not valid, expecting one of {allowed_methods}')
