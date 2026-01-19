"""Distance metric implementations (JAX)."""

from jax import jit, vmap
import jax.numpy as jnp


@jit
def euclidean(v1, v2):
    """Compute Euclidean (L2) distance between two points or arrays of points."""
    return jnp.linalg.norm(v1 - v2, axis=-1)


@jit
def manhattan(v1, v2):
    """Compute Manhattan (L1) distance between two points or arrays of points."""
    return jnp.sum(jnp.abs(v1 - v2), axis=-1)


@jit
def chebyshev(v1, v2):
    """Compute Chebyshev (L∞) distance between two points or arrays of points."""
    return jnp.max(jnp.abs(v1 - v2), axis=-1)


@jit
def minkowski(v1, v2, p=3):
    """Compute Minkowski distance between two points or arrays of points."""
    return jnp.sum(jnp.abs(v1 - v2) ** p, axis=-1) ** (1.0 / p)


@jit
def cosine(v1, v2):
    """Compute cosine distance between two vectors."""
    v1_norm = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = v2 / jnp.linalg.norm(v2, axis=-1, keepdims=True)
    return 1.0 - jnp.sum(v1_norm * v2_norm, axis=-1)


@jit
def haversine(v1, v2, R=6371.0):
    """Compute Haversine distance (great-circle) between points on a sphere."""
    lon1, lat1 = jnp.radians(v1[..., 0]), jnp.radians(v1[..., 1])
    lon2, lat2 = jnp.radians(v2[..., 0]), jnp.radians(v2[..., 1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = jnp.sin(dlat / 2) ** 2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2) ** 2
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    return R * c


def _broadcast(v1, v2):
    """Ensure v1 and v2 are at least 2D for broadcasting purposes."""
    if v1.ndim == 1:
        v1 = v1[None, :]
    if v2.ndim == 1:
        v2 = v2[None, :]
    return v1, v2


def validate_inputs(v1, v2, method: str):
    """Validate inputs for distance computation."""
    if v1.ndim > 2 or v2.ndim > 2:
        raise ValueError("v1 and v2 must be 1D or 2D arrays.")

    d1 = v1.shape[-1]
    d2 = v2.shape[-1]
    if d1 != d2:
        raise ValueError("Point dimensionality mismatch: v1.shape[-1] != v2.shape[-1]")

    if d1 not in (2, 3):
        raise ValueError(f"Points must be 2D or 3D (found dimension {d1}).")

    if jnp.any(jnp.isnan(v1)) or jnp.any(jnp.isnan(v2)):
        raise ValueError("Input contains NaNs.")
    if jnp.any(jnp.isinf(v1)) or jnp.any(jnp.isinf(v2)):
        raise ValueError("Input contains Infs.")

    if method == "haversine" and d1 != 2:
        raise ValueError("Haversine distance requires 2D (lon, lat) points.")


def compute_distance(v1, v2, method="euclidean", full_matrix=False, **kwargs):
    """Compute distances between v1 and v2 using the specified method."""
    validate_inputs(v1, v2, method=method)
    method_fns = {
        "euclidean": euclidean,
        "manhattan": manhattan,
        "chebyshev": chebyshev,
        "minkowski": lambda x, y: minkowski(x, y, p=kwargs.get("p", 3)),
        "cosine": cosine,
        "haversine": lambda x, y: haversine(x, y, R=kwargs.get("R", 6371.0)),
    }

    if method not in method_fns:
        raise ValueError(f"Unsupported method: {method}")

    fn = method_fns[method]
    v1, v2 = _broadcast(v1, v2)

    if full_matrix:
        return vmap(lambda x: vmap(lambda y: fn(x, y))(v2))(v1)

    if v1.shape[0] == v2.shape[0]:
        return vmap(fn)(v1, v2)
    if v1.shape[0] == 1:
        return vmap(lambda x: fn(v1[0], x))(v2)
    if v2.shape[0] == 1:
        return vmap(lambda x: fn(x, v2[0]))(v1)
    raise ValueError("Mismatched input shapes: must be one-to-one or one-to-many.")

