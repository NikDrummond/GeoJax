import jax.numpy as jnp
from typing import Union
from numpy import ndarray
from jax import jit
import jax.numpy as jnp


from .helpers import raise_dim_error

Array = Union[jnp.ndarray, jnp.DeviceArray, ndarray]

def normalise(arr: Array) -> jnp.ndarray:
    """Normalizes a 1D or 2D array using the L2 norm.

    Parameters
    ----------
    arr : Array
        Input array to be normalized.

    Returns
    -------
    jnp.ndarray
        Normalized array with the same shape as the input.
    """
    if arr.ndim in {1, 2}:
        norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / norm
    raise_dim_error(arr)

def magnitude(arr: Array) -> jnp.ndarray:
    """Calculate the Euclidean norm (magnitude) of a given array.

    Parameters
    ----------
    arr : Array
        Input array, expected to be either 1D or 2D.

    Returns
    -------
    jnp.ndarray
        The magnitude of the input array.
    """
    if arr.ndim in {1, 2}:
        return jnp.linalg.norm(arr, axis=-1, keepdims=True)
    raise_dim_error(arr)
    
def pairwise_euclidean(v1: Array, v2: jnp.ndarray) -> jnp.ndarray:
    # Ensure inputs are at least 2D (so we can generalize broadcasting)
    v1 = jnp.atleast_2d(v1)
    v2 = jnp.atleast_2d(v2)

    if v1.shape[0] == 1 and v2.shape[0] > 1:
        # v1 is a single point, v2 is a collection of points
        return jnp.linalg.norm(v2 - v1, axis=1)
    elif v2.shape[0] == 1 and v1.shape[0] > 1:
        # v2 is a single point, v1 is a collection of points
        return jnp.linalg.norm(v1 - v2, axis=1)
    elif v1.shape[0] == v2.shape[0]:
        # v1 and v2 have the same number of rows, compute distances row-wise
        return jnp.linalg.norm(v1 - v2, axis=1)
    else:
        raise ValueError("If both inputs are 2D, they must have the same number of rows.")

