# test_geojax_angles.py
# Tests for GeoJax angle functions

import pytest
import jax.numpy as jnp
from GeoJax import angle, signed_angle

def test_angle_between_vectors():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    ang = angle(v1, v2, to_degree=True)
    assert pytest.approx(ang.item(), abs=1e-4) == 90.0

def test_angle_in_plane():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([1.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    ang = angle(v1, v2, plane_normal=normal, to_degree=True)
    assert 44 < ang < 46  # Approximately 45 degrees

def test_signed_angle_ccw():
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    n = jnp.array([0.0, 0.0, 1.0])
    ang = signed_angle(a, b, n, to_degree=True)
    assert pytest.approx(ang.item(), abs=1e-4) == 90.0

def test_signed_angle_cw():
    a = jnp.array([0.0, 1.0, 0.0])
    b = jnp.array([1.0, 0.0, 0.0])
    n = jnp.array([0.0, 0.0, 1.0])
    ang = signed_angle(a, b, n, to_degree=True)
    assert pytest.approx(ang.item(), abs=1e-4) == -90.0

def test_signed_angle_collinear():
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([-1.0, 0.0, 0.0])
    n = jnp.array([0.0, 0.0, 1.0])
    ang = signed_angle(a, b, n, to_degree=True)
    assert pytest.approx(ang.item(), abs=1e-4) == 180.0

def test_angle_between_planes():
    n1 = jnp.array([0.0, 0.0, 1.0])
    n2 = jnp.array([0.0, 1.0, 0.0])
    rad = angle_between_planes(n1, n2)
    deg = angle_between_planes(n1, n2, to_degree=True)
    assert jnp.isclose(rad, jnp.pi / 2, atol=1e-6)
    assert jnp.isclose(deg, 90.0, atol=1e-3)