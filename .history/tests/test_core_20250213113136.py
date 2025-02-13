import jax.numpy as jnp
import pytest
from NeuGeometry import core

### normalise

# function normalises 1d arrays properly
def test_normalise_1d_nonzero_array():
    input_arr = jnp.array([1.0, 2.0, 3.0])
    result = core.normalise(input_arr)
    assert jnp.allclose(jnp.linalg.norm(result), 1.0)

# function normalises 2d arrays properly
def test_normalise_2d_nonzero_array():
    input_arr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = core.normalise(input_arr)
    norms = jnp.linalg.norm(result, axis=1)
    assert jnp.allclose(norms, jnp.ones_like(norms))

# Function rejects 3D or higher dimensional arrays with assertion error
def test_rejects_3d_array():
    input_arr = jnp.ones((2, 2, 2))
    try:
        core.normalise(input_arr)
    except AssertionError as e:
        assert str(e) == "Input arr must be 1D or 2D"

### magnitude

# function gets magnitude of 1d array
def test_magnitude_1d_positive_array():
    input_arr = jnp.array([3.0, 4.0])
    expected = 5.0
    result = core.magnitude(input_arr)
    assert jnp.allclose(result, expected)

# function gets magnitudes of 2d array
def test_magnitude_2d_positive_array():
    input_arr = jnp.array([[3.0, 4.0], [6.0, 8.0]])
    expected = jnp.array([5.0, 10.0])
    result = core.magnitude(input_arr)
    assert jnp.allclose(result, expected)

# handles 3 or more dimensions
def test_magnitude_3d_array():
    input_arr = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    expected = jnp.nan
    result = core.magnitude(input_arr)
    assert jnp.isnan(result)

### pairwise euclidean

# single pint distance

def test_single_points_distance(self):
    point1 = jnp.array([[1.0, 0.0]])
    point2 = jnp.array([[0.0, 1.0]]) 
    expected = jnp.array([jnp.sqrt(2.0)])
    result = pairwise_euclidean(point1, point2)
    assert jnp.allclose(result, expected)