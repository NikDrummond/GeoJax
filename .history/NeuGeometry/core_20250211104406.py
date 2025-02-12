import numpy as np
from jax import jit
import jax.numpy as jnp
from typing import Union

from .helpers import raise_dim_error


def normalise(arr: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(arr, axis=-1, keepdims=True)  # Works for both 1D and 2D arrays
    return arr / norms

