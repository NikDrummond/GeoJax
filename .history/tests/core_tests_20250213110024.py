import jax.numpy as jnp
import pytest
from NeuGeometry import core

    def test_normalise_1d_nonzero_array(self):
        import jax.numpy as jnp
        input_arr = jnp.array([1.0, 2.0, 3.0])
        result = normalise(input_arr)
        assert jnp.allclose(jnp.linalg.norm(result), 1.0)