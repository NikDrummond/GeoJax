import jax.numpy as jnp
import jax.random as jr
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

# single point distance

def test_single_points_distance():
    point1 = jnp.array([[1.0, 0.0]])
    point2 = jnp.array([[0.0, 1.0]]) 
    expected = jnp.array([jnp.sqrt(2.0)])
    result = core.euclidean_dist(point1, point2)
    assert jnp.allclose(result, expected)

# pairwise
def test_matching_row_counts():
    v1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    v2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    expected = jnp.array([jnp.sqrt(32.0), jnp.sqrt(32.0)])
    result = core.euclidean_dist(v1, v2)
    assert jnp.allclose(result, expected)

# one to many
def test_single_vector_with_multiple_points():
    single_vector = jnp.array([[1.0, 2.0]])
    multiple_points = jnp.array([[4.0, 6.0], [7.0, 8.0], [1.0, 2.0]])
    expected = jnp.array([5.0, jnp.sqrt(29.0), 0.0])
    result = core.euclidean_dist(single_vector, multiple_points)
    assert jnp.allclose(result, expected)

# make sure many to 1 works regardless of which is the one
def test_single_vector_with_multiple_points():
    # Single vector and multiple points
    single_vector = jnp.array([[1.0, 2.0]])
    multiple_points = jnp.array([[3.0, 4.0], [5.0, 6.0]])

    # Expected results when single_vector is v1
    expected_v1 = jnp.array([jnp.sqrt(8.0), jnp.sqrt(32.0)])
    result_v1 = core.euclidean_dist(single_vector, multiple_points)
    assert jnp.allclose(result_v1, expected_v1)

    # Expected results when single_vector is v2
    expected_v2 = jnp.array([jnp.sqrt(8.0), jnp.sqrt(32.0)])
    result_v2 = core.euclidean_dist(multiple_points, single_vector)
    assert jnp.allclose(result_v2, expected_v2)

### reject

# 1d returns correct rejection vector

def test_single_1d_vectors_rejection():
    v = jnp.array([3.0, 4.0, 0.0])
    from_v = jnp.array([0.0, 1.0, 0.0])
    result = core.reject(v, from_v)
    expected = jnp.array([3.0, 0.0, 0.0])
    assert jnp.allclose(result, expected, rtol=1e-7)

# 2d arrays return correct pairwise rejection vectors
def test_2d_array_pairwise_rejection():
    v = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    from_v = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = core.reject(v, from_v)
    expected = jnp.array([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0]])
    assert jnp.allclose(result, expected, rtol=1e-7)

# one to many correctly broadcasts
def test_single_vector_and_2d_array_broadcast():
    v = jnp.array([1.0, 2.0, 3.0])
    from_v = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = core.reject(v, from_v)
    expected = jnp.array([[0.0, 2.0, 3.0], [1.0, 0.0, 3.0], [1.0, 2.0, 0.0]])
    assert jnp.allclose(result, expected, rtol=1e-7)

# many to one
def test_broadcast_single_from_v_against_multiple_v():
    v = jnp.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])
    from_v = jnp.array([0.0, 1.0, 0.0])
    result = core.reject(v, from_v)
    expected = jnp.array([[3.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    assert jnp.allclose(result, expected, rtol=1e-7)

### (unsigned) angle

# test radians between two 1d
def test_angle_between_1d_vectors_in_radians():
    from math import pi

    v1 = jnp.array([1.0, 0.0])
    v2 = jnp.array([0.0, 1.0]) 

    result = core.angle(v1, v2)

    assert jnp.isclose(result, pi/2, rtol=1e-7)

# test pairwise comparison
def test_angle_between_2d_arrays_of_vectors_in_radians():
    from math import pi

    v1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    v2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    result = core.angle(v1, v2)

    expected = jnp.array([pi/2, pi/2])
    assert jnp.allclose(result, expected, rtol=1e-7)

# test plane projection
def test_angle_with_plane_projection():
    from math import pi

    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    plane_normal = jnp.array([0.0, 0.0, 1.0])

    result = core.angle(v1, v2, plane_normal=plane_normal)

    assert jnp.isclose(result, pi/2, rtol=1e-7)

# make sure we convert to degrees properly
def test_angle_between_1d_vectors_in_degrees():

    v1 = jnp.array([1.0, 0.0])
    v2 = jnp.array([0.0, 1.0]) 

    result = core.angle(v1, v2, to_degree=True)

    assert jnp.isclose(result, 90.0, rtol=1e-7)

# Skip normalization when assume_normalized is True
def test_skip_normalization_when_assume_normalized_true():
    from math import pi

    v1 = jnp.array([1.0, 0.0])
    v2 = jnp.array([0.0, 1.0])

    # Assume vectors are already normalized
    result = core.angle(v1, v2, assume_normalized=True)

    assert jnp.isclose(result, pi/2, rtol=1e-7)

# Handle broadcasting when one input is single vector and other is array
def test_angle_broadcasting_single_vector_with_array():
    from math import pi

    v1 = jnp.array([1.0, 0.0])  # Single vector
    v2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # Array of vectors

    result = core.angle(v1, v2)

    expected = jnp.array([pi/2, 0.0])  # Expected angles in radians

    assert jnp.allclose(result, expected, rtol=1e-7)

### angle (signed)

# Compute signed angle between two 1D vectors with plane normal in radians
def test_signed_angle_1d_vectors_radians():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0]) 
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=False)

    assert jnp.allclose(angle, jnp.pi/2)

# Compute signed angle between two 1D vectors with plane normal in degrees
def test_signed_angle_1d_vectors_degrees():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=True)

    assert jnp.allclose(angle, 90.0)

# Handle batched inputs (2D arrays) of vectors correctly
def test_signed_angle_batched_inputs():
    v1 = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v2 = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    normal = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    angles = core.signed_angle(v1, v2, normal, to_degree=False)

    expected_angles = jnp.array([jnp.pi/2, -jnp.pi/2])
    assert jnp.allclose(angles, expected_angles)

# Project vectors onto plane defined by normal vector before angle calculation
def test_signed_angle_projection_on_plane():
    v1 = jnp.array([1.0, 1.0, 0.0])
    v2 = jnp.array([1.0, -1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=False)

    # Expected angle is pi/2 since v1 and v2 are orthogonal on the plane defined by the normal
    assert jnp.allclose(angle, -jnp.pi/2)


# Maintain correct sign based on cross product with plane normal
def test_signed_angle_sign_with_plane_normal():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=False)

    assert jnp.allclose(angle, jnp.pi/2), "Angle should be positive when normal is [0, 0, 1]"

    # Test with opposite normal direction
    normal_opposite = jnp.array([0.0, 0.0, -1.0])
    angle_opposite = core.signed_angle(v1, v2, normal_opposite, to_degree=False)

    assert jnp.allclose(angle_opposite, -jnp.pi/2), "Angle should be negative when normal is [0, 0, -1]"

# Return scalar for 1D inputs and array for batched inputs
def test_signed_angle_scalar_and_array_outputs():
    v1_1d = jnp.array([1.0, 0.0, 0.0])
    v2_1d = jnp.array([0.0, 1.0, 0.0])
    normal_1d = jnp.array([0.0, 0.0, 1.0])

    # Test for 1D inputs, expecting a scalar output
    angle_scalar = core.signed_angle(v1_1d, v2_1d, normal_1d, to_degree=False)
    assert angle_scalar.shape == (), "Expected scalar output for 1D inputs"

    v1_2d = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v2_2d = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    normal_2d = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    # Test for batched inputs, expecting an array output
    angle_array = core.signed_angle(v1_2d, v2_2d, normal_2d, to_degree=False)
    assert angle_array.shape == (2,), "Expected array output for batched inputs"

### robust covariance matrix calculation

# Compute robust covariance matrix for well-behaved input data with default parameters
def test_robust_covariance_well_behaved_data():

    # Generate well-behaved test data
    rng = jr.PRNGKey(0)
    X = jr.normal(rng, shape=(100, 3))

    # Compute robust covariance
    cov_matrix = core.robust_covariance_mest(X)

    # Check shape and properties
    assert cov_matrix.shape == (3, 3)
    assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
    assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite

# Handle input data with zero variance in some dimensions
def test_robust_covariance_zero_variance():


    # Create data with zero variance in one dimension
    X = jnp.array([
        [1.0, 2.0, 0.0],
        [2.0, 3.0, 0.0],
        [3.0, 4.0, 0.0],
        [4.0, 5.0, 0.0]
    ])

    # Compute robust covariance
    cov_matrix = core.robust_covariance_mest(X)

    # Check shape and properties
    assert cov_matrix.shape == (3, 3)
    assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
    assert jnp.isfinite(cov_matrix).all()  # No NaN or inf values
    assert jnp.abs(cov_matrix[2, 2]) < 1e-6  # Zero variance dimension

# Converge within max_iter iterations for typical datasets
def test_convergence_within_max_iter():


    # Generate typical dataset
    rng = jr.PRNGKey(42)
    X = jr.normal(rng, shape=(100, 5))

    # Compute robust covariance
    cov_matrix = core.robust_covariance_mest(X)

    # Check shape and properties
    assert cov_matrix.shape == (5, 5)
    assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
    assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite

# Return symmetric positive definite covariance matrix
def test_robust_covariance_symmetric_positive_definite():


    # Generate test data
    rng = jr.PRNGKey(42)
    X = jr.normal(rng, shape=(50, 5))

    # Compute robust covariance
    cov_matrix = core.robust_covariance_mest(X)

    # Check shape and properties
    assert cov_matrix.shape == (5, 5)
    assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
    assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite


# Downweight outliers appropriately using Huber-like weights
def test_downweight_outliers_with_huber_weights():
    # Generate test data with outliers
    rng = jr.PRNGKey(0)
    rng1, rng2 = jr.split(rng)
    X = jnp.concatenate([
        jr.normal(rng1, shape=(95, 3)),              # Well-behaved data
        10.0 + 5.0 * jr.normal(rng2, shape=(5, 3))     # Outliers
    ])

    # Compute robust covariance
    cov_matrix = core.robust_covariance_mest(X)

    # Check shape and properties
    assert cov_matrix.shape == (3, 3)
    assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
    assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite

    # Additional check: ensure covariance is not overly influenced by outliers
    cov_no_outliers = core.robust_covariance_mest(X[:95])
    assert jnp.allclose(cov_matrix, cov_no_outliers, atol=1.0)  # Allow some tolerance

# Handle different input data shapes and dimensions correctly
def test_robust_covariance_varied_shapes():

    # Test with different shapes
    shapes = [(50, 2), (100, 5), (200, 10)]
    for shape in shapes:
        rng = jr.PRNGKey(0)
        X = jr.normal(rng, shape=shape)

        # Compute robust covariance
        cov_matrix = core.robust_covariance_mest(X)

        # Check shape and properties
        assert cov_matrix.shape == (shape[1], shape[1])
        assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
        assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite

### Eigen decomposition of coordinates

# Verify eigenvalues and eigenvectors are correctly computed for standard input matrix
def test_standard_input_eigendecomposition():
    # Create a simple 2D dataset with known covariance structure
    coords = jnp.array([[1.0, 0.0], 
                        [-1.0, 0.0],
                        [0.0, 0.5],
                        [0.0, -0.5]])

    evals, evecs = coord_eig_decomp(coords, robust=False, center=True)

    # Expected values (normalized)
    expected_evals = jnp.array([0.8, 0.2])  # Larger variance in x direction
    expected_evecs = jnp.array([[1.0, 0.0],  # First component along x
                                [0.0, 1.0]])   # Second component along y

    jnp.testing.assert_allclose(evals, expected_evals, rtol=1e-5)
    jnp.testing.assert_allclose(jnp.abs(evecs), jnp.abs(expected_evecs), rtol=1e-5)
