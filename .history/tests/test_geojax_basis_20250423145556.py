# test_geojax_basis.py
# Tests for GeoJax basis vector access

import jax.numpy as jnp
from GeoJax import basis_vectors

def test_basis_vectors():
    b = basis_vectors()
    assert jnp.allclose(b.x, jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(b.y, jnp.array([0.0, 1.0, 0.0]))
    assert jnp.allclose(b.z, jnp.array([0.0, 0.0, 1.0]))
    assert jnp.allclose(b.neg_x, jnp.array([-1.0, 0.0, 0.0]))
    assert jnp.allclose(b.neg_y, jnp.array([0.0, -1.0, 0.0]))
    assert jnp.allclose(b.neg_z, jnp.array([0.0, 0.0, -1.0]))
