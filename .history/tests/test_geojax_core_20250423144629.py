# test_geojax_core.py
# Unit tests for GeoJax core functions

import pytest
import jax.numpy as jnp
from GeoJax import (
    magnitude, normalize, dot, cross, reject, reflect,
    scalar_triple, project_to_vector, gram_schmidt,
    apply_affine, point_to_plane_distance, ray_plane_intersect,
    tetrahedron_volume
)

def test_magnitude_and_normalize():
    v = jnp.array([3.0, 4.0, 0.0])
    assert pytest.approx(magnitude(v).item(), abs=1e-6) == 5.0
    nv = normalize(v)
    assert pytest.approx(magnitude(nv).item(), abs=1e-6) == 1.0

def test_dot_and_cross():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    assert pytest.approx(dot(v1, v2).item(), abs=1e-6) == 0.0
    cp = cross(v1, v2)
    assert jnp.allclose(cp, jnp.array([0.0, 0.0, 1.0]))

def test_reject_and_project():
    v = jnp.array([2.0, 2.0, 0.0])
    onto = jnp.array([1.0, 0.0, 0.0])
    r = reject(v, onto)
    proj = project_to_vector(v, onto)
    assert jnp.allclose(r + proj, v, atol=1e-6)

def test_reflect():
    v = jnp.array([1.0, -1.0, 0.0])
    normal = jnp.array([0.0, 1.0, 0.0])
    reflected = reflect(v, normal)
    assert jnp.allclose(reflected, jnp.array([1.0, 1.0, 0.0]), atol=1e-6)

def test_scalar_triple():
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    c = jnp.array([0.0, 0.0, 1.0])
    assert pytest.approx(scalar_triple(a, b, c).item(), abs=1e-6) == 1.0

def test_gram_schmidt():
    vectors = jnp.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    Q = gram_schmidt(vectors)
    I = jnp.dot(Q, Q.T)
    assert jnp.allclose(I, jnp.eye(3), atol=1e-6)

def test_apply_affine():
    points = jnp.array([[1.0, 0.0, 0.0]])
    matrix = jnp.eye(3)
    offset = jnp.array([0.0, 1.0, 0.0])
    transformed = apply_affine(points, matrix, offset)
    assert jnp.allclose(transformed, jnp.array([[1.0, 1.0, 0.0]]))

def test_point_to_plane_distance():
    p = jnp.array([0.0, 0.0, 1.0])
    plane_p = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    d = point_to_plane_distance(p, plane_p, normal)
    assert pytest.approx(d.item(), abs=1e-6) == 1.0

def test_ray_plane_intersect():
    origin = jnp.array([0.0, 0.0, -1.0])
    direction = jnp.array([0.0, 0.0, 1.0])
    plane_p = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    hit = ray_plane_intersect(origin, direction, plane_p, normal)
    assert jnp.allclose(hit, jnp.array([0.0, 0.0, 0.0]), atol=1e-6)

def test_tetrahedron_volume():
    a = jnp.array([0.0, 0.0, 0.0])
    b = jnp.array([1.0, 0.0, 0.0])
    c = jnp.array([0.0, 1.0, 0.0])
    d = jnp.array([0.0, 0.0, 1.0])
    vol = tetrahedron_volume(a, b, c, d)
    assert pytest.approx(vol.item(), abs=1e-6) == 1 / 6
