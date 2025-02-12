import numpy as np
from jax import jit
import jax.numpy as jnp
from typing import TypeAlias

from .helpers import raise_dim_error

Array: TypeAlias = np.ndarray | jnp.ndarray


def normalise(arr: Array) -> jnp.ndarray:

    if arr.ndim == 1:
        norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / norm
    elif arr.ndim == 2:
        norm = 
        return arr / 
    norms =   # Works for both 1D and 2D arrays
    return arr / norms

