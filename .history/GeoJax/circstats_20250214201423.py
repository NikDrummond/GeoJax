from jax import jit
import jax.numpy as 

@jit
def circ_mean(samples, high=2 * jnp.pi, low=0, axis=None, weights=None):
    """
    Compute the circular (angular) mean of an array of angles.

    If the weighted sum of sines and cosines is zero (i.e. the resultant vector length is zero),
    return the midpoint of the interval (low + period/2).

    Parameters
    ----------
    samples : array_like
        Input angles.
    high : float, optional
        The high end of the interval (default is 2*pi).
    low : float, optional
        The low end of the interval (default is 0).
    axis : int or None, optional
        Axis along which to compute the mean. If None, the input is flattened.
    weights : array_like or None, optional
        Weights for the angles.

    Returns
    -------
    mean : jnp.ndarray
        The circular mean, expressed in the same units as the input.
    """
    samples = jnp.asarray(samples)
    period = high - low

    # If axis is None, flatten the array.
    if axis is None:
        samples = samples.ravel()
        axis = 0

    # Wrap the samples into the interval [low, high)
    samples = (samples - low) % period

    # Map the angles to [0, 2*pi)
    ang = 2 * jnp.pi * samples / period

    if weights is None:
        sum_sin = jnp.sum(jnp.sin(ang), axis=axis)
        sum_cos = jnp.sum(jnp.cos(ang), axis=axis)
        count = samples.shape[axis]
    else:
        weights = jnp.asarray(weights)
        sum_sin = jnp.sum(jnp.sin(ang) * weights, axis=axis)
        sum_cos = jnp.sum(jnp.cos(ang) * weights, axis=axis)
        count = jnp.sum(weights, axis=axis)

    # Compute the mean resultant length.
    R = jnp.sqrt(sum_sin**2 + sum_cos**2) / count
    # Compute the mean angle in [0, 2*pi)
    mean_angle = jnp.arctan2(sum_sin, sum_cos)
    mean_angle = (mean_angle + 2 * jnp.pi) % (2 * jnp.pi)

    # If the resultant length is nearly zero, return the midpoint.
    result_angle = jnp.where(
        R < 1e-6, low + period / 2, low + (mean_angle / (2 * jnp.pi)) * period
    )
    return result_angle


@jit
def circ_var(samples, high=2 * jnp.pi, low=0, axis=None, weights=None):
    """
    Compute the circular variance of an array of angles.

    Circular variance is defined as 1 - R, where R is the mean resultant length.
    If R is nearly zero, return 1.
    """
    samples = jnp.asarray(samples)
    period = high - low

    if axis is None:
        samples = samples.ravel()
        axis = 0

    # Wrap samples into [low, high)
    samples = (samples - low) % period

    # Map angles to [0, 2*pi)
    ang = 2 * jnp.pi * samples / period

    if weights is None:
        sum_sin = jnp.sum(jnp.sin(ang), axis=axis)
        sum_cos = jnp.sum(jnp.cos(ang), axis=axis)
        count = samples.shape[axis]
    else:
        weights = jnp.asarray(weights)
        sum_sin = jnp.sum(jnp.sin(ang) * weights, axis=axis)
        sum_cos = jnp.sum(jnp.cos(ang) * weights, axis=axis)
        count = jnp.sum(weights, axis=axis)

    R = jnp.sqrt(sum_sin**2 + sum_cos**2) / count
    # If R is very small, return 1 (the maximum variance)
    return jnp.where(R < 1e-6, 1.0, 1 - R)


@jit
def circ_std(samples, high=2 * jnp.pi, low=0, axis=None, weights=None):
    """
    Compute the circular standard deviation of an array of angles.

    It is defined as sqrt(-2 * log(R)). If R is nearly zero, return infinity.
    The result is mapped back to the original units.
    """
    samples = jnp.asarray(samples)
    period = high - low

    if axis is None:
        samples = samples.ravel()
        axis = 0

    # Corrected: Wrap samples into [low, high) without shifting after modulo
    wrapped = (samples - low) % period

    # Map angles to [0, 2*pi)
    ang = 2 * jnp.pi * wrapped / period

    if weights is None:
        sum_sin = jnp.sum(jnp.sin(ang), axis=axis)
        sum_cos = jnp.sum(jnp.cos(ang), axis=axis)
        count = samples.shape[axis]
    else:
        weights = jnp.asarray(weights)
        sum_sin = jnp.sum(jnp.sin(ang) * weights, axis=axis)
        sum_cos = jnp.sum(jnp.cos(ang) * weights, axis=axis)
        count = jnp.sum(weights, axis=axis)

    # Mean resultant length, R
    R = jnp.sqrt(sum_sin**2 + sum_cos**2) / count
    std_rad = jnp.where(R < 1e-6, jnp.inf, jnp.sqrt(-2 * jnp.log(R)))
    # Correct mapping back to original units
    std = jnp.where(jnp.isinf(std_rad), jnp.inf, std_rad * (period / (2 * jnp.pi)))
    return std
