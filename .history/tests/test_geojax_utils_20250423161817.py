import numpy as np
import jax.numpy as jnp
from GeoJax.core import magnitude
from GeoJax import normalize_angle_array, origin_flip


def test_normalize_angle_array_radians():
    angles = np.array([-np.pi, 0, np.pi])
    norm, unit = normalize_angle_array(angles)
    assert unit == "radians"
    assert np.allclose(norm, (angles + 2 * np.pi) % (2 * np.pi))


def test_normalize_angle_array_degrees():
    angles = np.array([-90, 0, 450])
    norm, unit = normalize_angle_array(angles)
    assert unit == "degrees"
    assert np.allclose(norm, (angles + 360) % 360)


def test_origin_flip_away():
    starts = jnp.array([[1.0, 0.0, 0.0]])
    stops = jnp.array([[0.5, 0.0, 0.0]])  # closer than start
    flipped = origin_flip(starts, stops, method='away')
    assert jnp.all(magnitude(flipped) > magnitude(starts))


def test_origin_flip_towards():
    starts = jnp.array([[0.5, 0.0, 0.0]])
    stops = jnp.array([[1.5, 0.0, 0.0]])  # farther than start
    flipped = origin_flip(starts, stops, method='towards')
    assert jnp.all(magnitude(flipped) < magnitude(starts) + 1e)


def test_origin_flip_invalid():
    starts = jnp.array([[1.0, 0.0, 0.0]])
    stops = jnp.array([[2.0, 0.0, 0.0]])
    try:
        origin_flip(starts, stops, method='invalid')
        assert False, "Expected ValueError"
    except ValueError:
        pass
