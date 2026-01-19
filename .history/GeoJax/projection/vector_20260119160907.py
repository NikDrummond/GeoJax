"""Vector projection utilities."""

from jax import jit
import jax.numpy as jnp

from ..core import normalize


@jit
def project_to_vector(vector: jnp.ndarray, onto: jnp.ndarray) -> jnp.ndarray:
    """
    Project `vector` onto another vector `onto`.

    Parameters
    ----------
    vector : jnp.ndarray
        Input vector or array of vectors (..., 3).
    onto : jnp.ndarray
        Target direction vector (3,) or broadcastable to `vector`.

    Returns
    -------
    jnp.ndarray
        Vector projection(s) of `vector` onto `onto`.
    """
    if vector.shape[-1] != 3 or onto.shape[-1] != 3:
        raise ValueError("project_to_vector expects vectors with shape (..., 3)")
    onto_unit = normalize(onto)
    projection_length = jnp.sum(vector * onto_unit, axis=-1, keepdims=True)
    return projection_length * onto_unit


__all__ = ["project_to_vector"]
