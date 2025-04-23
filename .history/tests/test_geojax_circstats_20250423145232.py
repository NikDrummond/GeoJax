# test_geojax_circstats.py
# Tests for GeoJax circular statistics

import pytest
import jax.numpy as jnp
from GeoJax import circmean, circstd

def test_circmean_uniform():
    angles = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
    mean = circmean(angles)
    # Uniform distribution around circle -> mean should be ambiguous (near 0)
    assert jnp.abs(mean) < 1e-6 or jnp.abs(mean - 2 * jnp.pi) < 1e-6

def test_circmean_weighted():
    angles = jnp.array([0.0, jnp.pi])
    weights = jnp.array([2.0, 1.0])
    mean = circmean(angles, weights)
    assert pytest.approx(mean.item(), abs=1e-6) == 0.0

def test_circstd_angular():
    angles = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    std = circstd(angles, method="angular")
    assert std > 0 and std < jnp.pi

def test_circstd_circular():
    angles = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    std = circstd(angles, method="circular")
    assert std > 0 and std < jnp.pi

def test_invalid_method():
    with pytest.raises(AssertionError):
        circstd(jnp.array([0.0, 1.0]), method="nonsense")
