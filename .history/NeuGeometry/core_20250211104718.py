import numpy as np
from jax import jit
import jax.numpy as jnp
from typing import TypeAlias

from .helpers import raise_dim_error

Array: TypeAlias = np.ndarray | jnp.ndarray


def normalise(arr: Array) -> jnp.ndarray:

    if arr.ndim == 1:
        return arr / jnp.linalg.norm(arr)
    elif arr.ndim == 2:
        return
    norms = jnp.linalg.norm(arr, axis=-1, keepdims=True)  # Works for both 1D and 2D arrays
    return arr / norms

