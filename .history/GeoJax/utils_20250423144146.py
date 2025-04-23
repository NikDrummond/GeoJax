### utils.py
# utils.py
# Miscellaneous utility functions for GeoJax
# Includes: normalize_angle_array, origin_flip

import jax.numpy as jnp
import numpy as np
from jax import jit
from .core import magnitude


def normalize_angle_array(angles: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Normalize angles to lie within [0, 2pi) or [0, 360), based on magnitude.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians or degrees.

    Returns
    -------
    tuple[np.ndarray, str]
        Normalized angles and the inferred unit ('radians' or 'degrees').
    """
    angles = np.asarray(angles)
    if np.max(np.abs(angles)) <= 2 * np.pi:
        return (angles + 2 * np.pi) % (2 * np.pi), 'radians'
    else:
        return (angles + 360) % 360, 'degrees'


@jit
def _flip_towards(starts: jnp.ndarray, stops: jnp.ndarray) -> jnp.ndarray:
    """Flip stop points to be closer to the origin than starts."""
    flip = magnitude(stops) > magnitude(starts)
    return jnp.where(flip[..., None], 2 * starts - stops, stops)


@jit
def _flip_away(starts: jnp.ndarray, stops: jnp.ndarray) -> jnp.ndarray:
    """Flip stop points to be farther from the origin than starts."""
    flip = magnitude(stops) < magnitude(starts)
    return jnp.where(flip[..., None], 2 * starts - stops, stops)


def origin_flip(
    starts: jnp.ndarray, stops: jnp.ndarray, method: str = 'away'
) -> jnp.ndarray:
    """
    Flip stop vectors to ensure direction is away from or toward origin.

    Parameters
    ----------
    starts : jnp.ndarray
        Starting points.
    stops : jnp.ndarray
        Ending points.
    method : str
        'away' to flip towards further, 'towards' to flip towards origin.

    Returns
    -------
    jnp.ndarray
        Adjusted stop vectors.

    Raises
    ------
    ValueError
        If method is not 'away' or 'towards'.
    """
    if method == 'away':
        return _flip_away(starts, stops)
    elif method == 'towards':
        return _flip_towards(starts, stops)
    else:
        raise ValueError(f"Invalid method '{method}', expected 'away' or 'towards'")