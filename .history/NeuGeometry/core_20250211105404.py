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
    

