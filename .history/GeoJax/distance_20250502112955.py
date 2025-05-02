import jax.numpy as jnp
from jax import jit, vmap

# -----------------------------------------
# Distance Metric Implementations (JIT)
# -----------------------------------------

@jit
def euclidean(v1, v2):
    return jnp.linalg.norm(v1 - v2, axis=-1)

@jit
def manhattan(v1, v2):
    return jnp.sum(jnp.abs(v1 - v2), axis=-1)

@jit
def chebyshev(v1, v2):
    return jnp.max(jnp.abs(v1 - v2), axis=-1)

@jit
def minkowski(v1, v2, p=3):
    return jnp.sum(jnp.abs(v1 - v2) ** p, axis=-1) ** (1.0 / p)

@jit
def cosine(v1, v2):
    v1_norm = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = v2 / jnp.linalg.norm(v2, axis=-1, keepdims=True)
    return 1.0 - jnp.sum(v1_norm * v2_norm, axis=-1)

@jit
def haversine(v1, v2):
    # Expect input as (lon, lat) in degrees
    R = 6371.0  # Earth radius in kilometers
    lon1, lat1 = jnp.radians(v1[..., 0]), jnp.radians(v1[..., 1])
    lon2, lat2 = jnp.radians(v2[..., 0]), jnp.radians(v2[..., 1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = jnp.sin(dlat / 2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2)**2
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    return R * c

# -----------------------------------------
# Dispatcher and Shape Handling
# -----------------------------------------

def _broadcast(v1, v2):
    # Expand dimensions if necessary
    if v1.ndim == 1:
        v1 = v1[None, :]
    if v2.ndim == 1:
        v2 = v2[None, :]
    return v1, v2

def compute_distance(v1, v2, method='euclidean', full_matrix=False, **kwargs):
    # Available methods
    method_fns = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
        'minkowski': lambda x, y: minkowski(x, y, p=kwargs.get('p', 3)),
        'cosine': cosine,
        'haversine': haversine,
    }

    if method not in method_fns:
        raise ValueError(f"Unsupported method: {method}")

    fn = method_fns[method]
    v1, v2 = _broadcast(v1, v2)

    if full_matrix:
        return vmap(lambda x: vmap(lambda y: fn(x, y))(v2))(v1)

    if v1.shape[0] == v2.shape[0]:
        return vmap(fn)(v1, v2)
    elif v1.shape[0] == 1:
        return vmap(lambda x: fn(v1[0], x))(v2)
    elif v2.shape[0] == 1:
        return vmap(lambda x: fn(x, v2[0]))(v1)
    else:
        raise ValueError("Mismatched input shapes: must be one-to-one or one-to-many.")
