import jax.numpy as jnp
from GeoJax import basis


def test_basis_vectors():
    assert jnp.allclose(basis.x, jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(basis.y, jnp.array([0.0, 1.0, 0.0]))
    assert jnp.allclose(basis.z, jnp.array([0.0, 0.0, 1.0]))

    assert jnp.allclose(basis.neg_x, jnp.array([-1.0, 0.0, 0.0]))
    assert jnp.allclose(basis.neg_y, jnp.array([0.0, -1.0, 0.0]))
    assert jnp.allclose(basis.neg_z, jnp.array([0.0, 0.0, -1.0]))


def test_orthogonality_and_norms():
    vectors = [basis.x, basis.y, basis.z]
    for i in range(3):
        for j in range(3):
            dot = jnp.dot(vectors[i], vectors[j])
            if i == j:
                assert jnp.isclose(dot, 1.0)
            else:
                assert jnp.isclose(dot, 0.0)
