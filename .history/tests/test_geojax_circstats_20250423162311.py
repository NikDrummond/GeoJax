# tests/test_geojax_circstats.py

import jax.numpy as jnp
import numpy as np
import pytest
from GeoJax import circmean, circstd, circvar

def test_circmean_basic():
    angles = jnp.array([0.0, jnp.pi * 2])
    mean = circmean(angles)
    expected = 0.0
    assert jnp.isclose((mean % (2 * jnp.pi)), expected, atol=1e-5)
    
def test_circmean_degrees():
    angles = jnp.array([0, jnp.pi/2, jnp.pi])
    mean = circmean(angles, to_degree=True)
    assert 0 <= mean <= 360

def test_circmean_weighted():
    angles = jnp.array([0, jnp.pi])
    weights = jnp.array([2.0, 1.0])
    result = circmean(angles, weights=weights)
    expected = 0.0  # Heavier weight on angle 0
    assert jnp.isclose(result, expected, atol=1e-5)

def test_circstd_angular():
    angles = jnp.array([0.0, jnp.pi/2, jnp.pi])
    std = circstd(angles, method="angular")
    assert std > 0

def test_circstd_circular():
    angles = jnp.array([0.0, jnp.pi/2, jnp.pi])
    std = circstd(angles, method="circular")
    assert std > 0

def test_circstd_weighted():
    angles = jnp.array([0.0, jnp.pi])
    weights = jnp.array([1.0, 1.0])
    std = circstd(angles, weights=weights, method="angular")
    assert std > 0

def test_circvar():
    angles = jnp.array([0, jnp.pi/4, jnp.pi/2])
    var = circvar(angles)
    assert 0 <= var <= 1

def test_circmean_edge_case_identical():
    angles = jnp.array([jnp.pi, jnp.pi, jnp.pi])
    result = circmean(angles)
    assert jnp.isclose(result, jnp.pi)

def test_circstd_zero_variance():
    angles = jnp.array([jnp.pi, jnp.pi, jnp.pi])
    result = circstd(angles)
    assert jnp.isclose(result, 0.0)

def test_circvar_zero_variance():
    angles = jnp.array([jnp.pi, jnp.pi, jnp.pi])
    var = circvar(angles)
    assert jnp.isclose(var, 0.0)
