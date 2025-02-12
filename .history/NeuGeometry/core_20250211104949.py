import numpy as np
from jax import jit
import jax.numpy as jnp
from typing import TypeAlias

from .helpers import raise_dim_error

Array: TypeAlias = np.ndarray | jnp.ndarray


    """_summary_
    """def normalise(arr: Array) -> jnp.ndarray:
    """_summary_

    Parameters
    ----------
    arr : Array
        _description_

    Returns
    -------
    jnp.ndarray
        _description_
    """

    if arr.ndim in [1,2]:
        norm = jnp.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / norm
    else:
        raise_dim_error(arr)
    

