import jax.numpy as jnp
import jax.random as jr
import jax
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

    assert jnp.isclose(result, pi / 2, rtol=1e-7)


# test pairwise comparison
def test_angle_between_2d_arrays_of_vectors_in_radians():
    from math import pi

    v1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    v2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    result = core.angle(v1, v2)

    expected = jnp.array([pi / 2, pi / 2])
    assert jnp.allclose(result, expected, rtol=1e-7)


# test plane projection
def test_angle_with_plane_projection():
    from math import pi

    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    plane_normal = jnp.array([0.0, 0.0, 1.0])

    result = core.angle(v1, v2, plane_normal=plane_normal)

    assert jnp.isclose(result, pi / 2, rtol=1e-7)


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

    assert jnp.isclose(result, pi / 2, rtol=1e-7)


# Handle broadcasting when one input is single vector and other is array
def test_angle_broadcasting_single_vector_with_array():
    from math import pi

    v1 = jnp.array([1.0, 0.0])  # Single vector
    v2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # Array of vectors

    result = core.angle(v1, v2)

    expected = jnp.array([pi / 2, 0.0])  # Expected angles in radians

    assert jnp.allclose(result, expected, rtol=1e-7)


### angle (signed)


# Compute signed angle between two 1D vectors with plane normal in radians
def test_signed_angle_1d_vectors_radians():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=False)

    assert jnp.allclose(angle, jnp.pi / 2)


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

    expected_angles = jnp.array([jnp.pi / 2, -jnp.pi / 2])
    assert jnp.allclose(angles, expected_angles)


# Project vectors onto plane defined by normal vector before angle calculation
def test_signed_angle_projection_on_plane():
    v1 = jnp.array([1.0, 1.0, 0.0])
    v2 = jnp.array([1.0, -1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=False)

    # Expected angle is pi/2 since v1 and v2 are orthogonal on the plane defined by the normal
    assert jnp.allclose(angle, -jnp.pi / 2)


# Maintain correct sign based on cross product with plane normal
def test_signed_angle_sign_with_plane_normal():
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    angle = core.signed_angle(v1, v2, normal, to_degree=False)

    assert jnp.allclose(
        angle, jnp.pi / 2
    ), "Angle should be positive when normal is [0, 0, 1]"

    # Test with opposite normal direction
    normal_opposite = jnp.array([0.0, 0.0, -1.0])
    angle_opposite = core.signed_angle(v1, v2, normal_opposite, to_degree=False)

    assert jnp.allclose(
        angle_opposite, -jnp.pi / 2
    ), "Angle should be negative when normal is [0, 0, -1]"


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
    X = jnp.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [3.0, 4.0, 0.0], [4.0, 5.0, 0.0]])

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
    X = jnp.concatenate(
        [
            jr.normal(rng1, shape=(95, 3)),  # Well-behaved data
            10.0 + 5.0 * jr.normal(rng2, shape=(5, 3)),  # Outliers
        ]
    )

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
    coords = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5], [0.0, -0.5]])

    evals, evecs = core.coord_eig_decomp(coords, robust=False, center=True)

    # Expected values (normalized)
    expected_evals = jnp.array([0.8, 0.2])  # Larger variance in x direction
    expected_evecs = jnp.array(
        [[1.0, 0.0], [0.0, 1.0]]  # First component along x
    )  # Second component along y

    assert jnp.allclose(evals, expected_evals, rtol=1e-5)
    assert jnp.allclose(jnp.abs(evecs), jnp.abs(expected_evecs), rtol=1e-5)


# Handle input matrix with zero variance in some dimensions
def test_zero_variance_dimension():
    # Create dataset with zero variance in y direction
    coords = jnp.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])

    evals, evecs = core.coord_eig_decomp(coords, robust=False, center=True)

    # First eigenvalue should be 1.0 (all variance), second should be 0
    expected_evals = jnp.array([1.0, 0.0])
    # First eigenvector should be along x, second along y
    expected_evecs = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    assert jnp.allclose(evals, expected_evals, rtol=1e-5)
    assert jnp.allclose(jnp.abs(evecs), jnp.abs(expected_evecs), rtol=1e-5)


# Check robust covariance estimation produces valid results when robust=True
def test_robust_covariance_estimation():
    # Create a dataset with an outlier to test robust covariance estimation
    coords = jnp.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5], [0.0, -0.5], [10.0, 10.0]]  # Outlier
    )

    evals, evecs = core.coord_eig_decomp(coords, robust=True, center=True)

    # Instead of comparing directly to hardcoded eigenvalues/eigenvectors,
    # we check for expected properties.

    # 1. The eigenvalues should be positive and sum to 1 (if PCA=True).
    assert jnp.all(evals > 0)
    assert jnp.allclose(jnp.sum(evals), 1.0, atol=1e-5)

    # 2. The eigenvector matrix should be orthonormal.
    ortho = jnp.allclose(jnp.matmul(evecs, evecs.T), jnp.eye(2), atol=0.2)
    assert ortho, "Eigenvectors are not orthonormal within tolerance."

    # 3. The principal eigenvector (first row) should roughly align with [1, 0]
    # (up to a sign). We check that the absolute dot product is near 1.
    principal_alignment = jnp.abs(jnp.dot(evecs[0], jnp.array([1.0, 0.0])))
    assert jnp.allclose(
        principal_alignment, 1.0, atol=0.2
    ), f"Principal eigenvector does not align with [1,0]: {principal_alignment}"


### minimum theta


# Compute minimal angle between vector and line in 3D space
def test_minimal_angle_computation():
    # Test vectors and normal
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    # Expected angle should be pi/2 radians (90 degrees)
    # But minimum_theta returns pi/2 or -pi/2 for perpendicular vectors
    result = core.minimum_theta(v1, v2, normal)

    # Check that absolute value is pi/2
    assert jnp.abs(result) == jnp.pi / 2


# Handle parallel vectors (0 degree angle)
def test_parallel_vectors():
    # Test with parallel vectors
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([2.0, 0.0, 0.0])  # Same direction, different magnitude
    normal = jnp.array([0.0, 0.0, 1.0])

    result = core.minimum_theta(v1, v2, normal)

    # For parallel vectors, angle should be 0
    assert jnp.allclose(result, 0.0, atol=1e-7)


# Return angle in degrees when to_degree is True
def test_return_angle_in_degrees():
    # Test vectors and normal
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    # Expected angle should be 90 degrees
    result = core.minimum_theta(v1, v2, normal, to_degree=True)

    # Check that the result is 90 degrees
    assert jnp.isclose(result, 90.0)


# Handle perpendicular vectors correctly (90 degree angle)
def test_perpendicular_vectors():
    # Test vectors and normal
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    # Expected angle should be pi/2 radians (90 degrees)
    result = core.minimum_theta(v1, v2, normal)

    # Check that absolute value is pi/2
    assert jnp.abs(result) == jnp.pi / 2


# Project vectors onto plane before angle calculation
def test_projection_before_angle_calculation():
    # Test vectors and normal
    v1 = jnp.array([1.0, 2.0, 3.0])
    v2 = jnp.array([4.0, 5.0, 6.0])
    normal = jnp.array([0.0, 0.0, 1.0])

    # Manually perform rejection (project onto the plane)
    def manual_reject(v, n):
        # Compute the scalar projection
        proj = jnp.dot(v, n)
        return v - proj * n

    v1_proj = manual_reject(v1, normal)
    v2_proj = manual_reject(v2, normal)

    # Compute expected angle using the projected vectors.
    # First, normalize the projected vectors.
    v1_norm = core.normalise(v1_proj)
    v2_norm = core.normalise(v2_proj)
    # Then compute the dot product and the (unsigned) angle.
    dot_val = jnp.dot(v1_norm, v2_norm)
    expected_angle = jnp.arccos(jnp.clip(dot_val, -1.0, 1.0))
    # Note: minimum_theta computes a signed angle, so we compare absolute values.

    # Call the function under test.
    result = core.minimum_theta(v1, v2, normal, to_degree=False)

    # Check that the absolute value of the result equals the expected angle.
    assert jnp.allclose(jnp.abs(result), expected_angle, atol=1e-5)


### rotation matrix from rotation vector


# Rotation matrix is correctly computed for a non-zero rotation vector using Rodrigues formula
def test_nonzero_rotation_vector():

    # Create a rotation vector of pi/2 around z-axis
    rot_vec = jnp.array([0.0, 0.0, jnp.pi / 2])

    # Expected rotation matrix for 90 degree rotation around z-axis
    expected = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if result matches expected with small tolerance
    assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-6)


# Rotation vector with magnitude very close to zero (near 1e-8)
def test_near_zero_rotation_vector():

    # Create a very small rotation vector
    rot_vec = jnp.array([1e-9, 1e-9, 1e-9])

    # For near-zero rotation, should return identity matrix
    expected = jnp.eye(3)

    result = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if result is identity matrix
    assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-6)


# Identity matrix is returned for zero rotation vector
def test_zero_rotation_vector():

    # Create a zero rotation vector
    rot_vec = jnp.array([0.0, 0.0, 0.0])

    # Expected identity matrix
    expected = jnp.eye(3)

    result = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if result matches expected identity matrix
    assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-6)


# Function preserves orthogonality property of rotation matrices
def test_orthogonality_of_rotation_matrix():

    # Create a random rotation vector
    rot_vec = jnp.array([0.5, -0.5, 0.5])

    # Compute the rotation matrix
    R = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if R is orthogonal: R.T * R should be close to identity matrix
    identity = jnp.eye(3)
    result = jnp.dot(R.T, R)

    # Assert that the result is close to the identity matrix
    assert jnp.allclose(result, identity, rtol=1e-7, atol=1e-6)


# Function correctly handles rotation around each principal axis (x, y, z)
def test_rotation_around_principal_axes():

    # Test rotation around x-axis by 90 degrees
    rot_vec_x = jnp.array([jnp.pi / 2, 0.0, 0.0])
    expected_x = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    result_x = core.rotation_matrix_from_rotvec(rot_vec_x)
    assert jnp.allclose(result_x, expected_x, rtol=1e-7, atol=1e-6)

    # Test rotation around y-axis by 90 degrees
    rot_vec_y = jnp.array([0.0, jnp.pi / 2, 0.0])
    expected_y = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    result_y = core.rotation_matrix_from_rotvec(rot_vec_y)
    assert jnp.allclose(result_y, expected_y, rtol=1e-7, atol=1e-6)

    # Test rotation around z-axis by 90 degrees
    rot_vec_z = jnp.array([0.0, 0.0, jnp.pi / 2])
    expected_z = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result_z = core.rotation_matrix_from_rotvec(rot_vec_z)
    assert jnp.allclose(result_z, expected_z, rtol=1e-7, atol=1e-6)


# Function handles standard rotation vectors with different magnitudes
def test_standard_rotation_vectors():

    # Test with a rotation vector of pi/4 around x-axis
    rot_vec_x = jnp.array([jnp.pi / 4, 0.0, 0.0])
    expected_x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(jnp.pi / 4), -jnp.sin(jnp.pi / 4)],
            [0.0, jnp.sin(jnp.pi / 4), jnp.cos(jnp.pi / 4)],
        ]
    )
    result_x = core.rotation_matrix_from_rotvec(rot_vec_x)
    assert jnp.allclose(result_x, expected_x, rtol=1e-7, atol=1e-6)

    # Test with a rotation vector of pi/3 around y-axis
    rot_vec_y = jnp.array([0.0, jnp.pi / 3, 0.0])
    expected_y = jnp.array(
        [
            [jnp.cos(jnp.pi / 3), 0.0, jnp.sin(jnp.pi / 3)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(jnp.pi / 3), 0.0, jnp.cos(jnp.pi / 3)],
        ]
    )
    result_y = core.rotation_matrix_from_rotvec(rot_vec_y)
    assert jnp.allclose(result_y, expected_y, rtol=1e-7, atol=1e-6)

    # Test with a rotation vector of pi/6 around z-axis
    rot_vec_z = jnp.array([0.0, 0.0, jnp.pi / 6])
    expected_z = jnp.array(
        [
            [jnp.cos(jnp.pi / 6), -jnp.sin(jnp.pi / 6), 0.0],
            [jnp.sin(jnp.pi / 6), jnp.cos(jnp.pi / 6), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result_z = core.rotation_matrix_from_rotvec(rot_vec_z)
    assert jnp.allclose(result_z, expected_z, rtol=1e-7, atol=1e-6)


# Output matrix determinant should equal 1 (proper rotation)
def test_rotation_matrix_determinant():

    # Create a rotation vector
    rot_vec = jnp.array([1.0, 0.0, 0.0])

    # Compute the rotation matrix
    result = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if the determinant of the result is close to 1
    determinant = jnp.linalg.det(result)
    assert jnp.isclose(determinant, 1.0, atol=1e-7)


# Rotation by 2π should return to identity
def test_rotation_by_2pi_returns_identity():

    # Create a rotation vector of 2π around z-axis
    rot_vec = jnp.array([0.0, 0.0, 2 * jnp.pi])

    # Expected identity matrix
    expected = jnp.eye(3)

    result = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if result matches expected with small tolerance
    assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-6)


# Function handles negative rotation angles correctly
def test_negative_rotation_angle():

    # Create a rotation vector of -pi/2 around z-axis
    rot_vec = jnp.array([0.0, 0.0, -jnp.pi / 2])

    # Expected rotation matrix for -90 degree rotation around z-axis
    expected = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = core.rotation_matrix_from_rotvec(rot_vec)

    # Check if result matches expected with small tolerance
    assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-6)


### rotate around axis


# Rotate 3D coordinates around arbitrary axis by positive angle
def test_rotate_3d_coords_positive_angle():
    # Test rotation of 3D points around y-axis by pi/2
    coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    axis = jnp.array([0.0, 1.0, 0.0])  # y-axis
    theta = jnp.pi / 2

    rotated = core.rotate_around_axis(coords, theta, axis)

    expected = jnp.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    assert jnp.allclose(rotated, expected, rtol=1e-7, atol=1e-6)


# Rotate 2D coordinates around arbitrary axis by positive angle
def test_rotate_2d_coords_positive_angle():
    # Test rotation of 2D points around z-axis by pi/2
    coords = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    axis = jnp.array([0.0, 0.0, 1.0])  # z-axis
    theta = jnp.pi / 2

    rotated = core.rotate_around_axis(coords, theta, axis)

    expected = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    assert jnp.allclose(rotated, expected, rtol=1e-7, atol=1e-6)


# Rotation around zero vector axis
def test_rotate_around_zero_axis():
    coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    axis = jnp.array([0.0, 0.0, 0.0])
    theta = jnp.pi / 4

    rotated = core.rotate_around_axis(coords, theta, axis)

    # Rotation around zero vector should return original coordinates
    assert jnp.allclose(rotated, coords, rtol=1e-7, atol=1e-6)


# Rotate coordinates by zero angle returns original coordinates
def test_rotate_coords_zero_angle():
    # Test rotation of 3D points around any axis by 0 angle
    coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    axis = jnp.array([1.0, 0.0, 0.0])  # arbitrary axis
    theta = 0.0  # zero angle

    rotated = core.rotate_around_axis(coords, theta, axis)

    expected = coords  # Expecting the original coordinates
    assert jnp.allclose(rotated, expected, rtol=1e-7, atol=1e-6)


# Rotate coordinates by 2π returns (approximately) original coordinates
def test_rotate_around_axis_full_rotation():
    # Test rotation of 3D points around z-axis by 2π
    coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    axis = jnp.array([0.0, 0.0, 1.0])  # z-axis
    theta = 2 * jnp.pi

    rotated = core.rotate_around_axis(coords, theta, axis)

    expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert jnp.allclose(rotated, expected, rtol=1e-7, atol=1e-6)


# Rotation by negative angle is inverse of positive angle rotation
def test_rotate_3d_coords_negative_angle_inverse():
    # Test rotation of 3D points around y-axis by pi/2 and -pi/2
    coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    axis = jnp.array([0.0, 1.0, 0.0])  # y-axis
    theta = jnp.pi / 2

    # Rotate by positive angle
    rotated_positive = core.rotate_around_axis(coords, theta, axis)

    # Rotate back by negative angle
    rotated_back = core.rotate_around_axis(rotated_positive, -theta, axis)

    # The result should be the original coordinates
    assert jnp.allclose(rotated_back, coords, rtol=1e-7, atol=1e-6)


# Single point rotation (array shape (1,2) or (1,3))
def test_single_point_rotation_2d():
    # Test rotation of a single 2D point around z-axis by pi/2
    coords = jnp.array([[1.0, 0.0]])  # Single 2D point
    axis = jnp.array([0.0, 0.0, 1.0])  # z-axis
    theta = jnp.pi / 2

    rotated = core.rotate_around_axis(coords, theta, axis)

    expected = jnp.array([[0.0, 1.0]])  # Expected result after rotation
    assert jnp.allclose(rotated, expected, rtol=1e-7, atol=1e-6)


# Rotation preserves distances between points
def test_rotation_preserves_distances():
    # Test that rotation preserves distances between points
    coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    axis = jnp.array([0.0, 0.0, 1.0])  # z-axis
    theta = jnp.pi / 4  # 45 degrees

    rotated_coords = core.rotate_around_axis(coords, theta, axis)

    # Calculate distances before and after rotation
    original_distance = jnp.linalg.norm(coords[0] - coords[1])
    rotated_distance = jnp.linalg.norm(rotated_coords[0] - rotated_coords[1])

    # Assert that the distances are preserved
    assert jnp.allclose(rotated_distance, original_distance, rtol=1e-7, atol=1e-6)


# Rotation preserves handedness/orientation
def test_rotation_preserves_handedness():
    # Test that rotating a right-handed coordinate system preserves its handedness
    coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    axis = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3)  # arbitrary axis
    theta = jnp.pi / 3  # 60 degrees

    rotated = core.rotate_around_axis(coords, theta, axis)

    # Calculate the volume of the parallelepiped formed by the vectors
    original_volume = jnp.linalg.det(coords)
    rotated_volume = jnp.linalg.det(rotated)

    # The volume should be preserved, indicating the orientation is preserved
    assert jnp.allclose(rotated_volume, original_volume, rtol=1e-7, atol=1e-6)


# Sequential rotations are composable
def test_sequential_rotations_composable():
    # Initial coordinates
    coords = jnp.array([[1.0, 0.0, 0.0]])
    # First rotation around z-axis by pi/2
    axis1 = jnp.array([0.0, 0.0, 1.0])
    theta1 = jnp.pi / 2
    rotated_once = core.rotate_around_axis(coords, theta1, axis1)

    # Second rotation around y-axis by pi/2
    axis2 = jnp.array([0.0, 1.0, 0.0])
    theta2 = jnp.pi / 2
    rotated_twice = core.rotate_around_axis(rotated_once, theta2, axis2)

    # Expected result after sequential rotations
    expected = jnp.array([[0.0, 1.0, 0.0]])

    assert jnp.allclose(rotated_twice, expected, rtol=1e-7, atol=1e-6)


# Rotation around standard basis vectors (x,y,z axes)
def test_rotate_around_standard_basis_vectors():
    # Test rotation of 3D points around x-axis by pi/2
    coords_x = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    axis_x = jnp.array([1.0, 0.0, 0.0])  # x-axis
    theta_x = jnp.pi / 2

    rotated_x = core.rotate_around_axis(coords_x, theta_x, axis_x)

    expected_x = jnp.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    assert jnp.allclose(rotated_x, expected_x, rtol=1e-7, atol=1e-6)

    # Test rotation of 3D points around y-axis by pi/2
    coords_y = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    axis_y = jnp.array([0.0, 1.0, 0.0])  # y-axis
    theta_y = jnp.pi / 2

    rotated_y = core.rotate_around_axis(coords_y, theta_y, axis_y)

    expected_y = jnp.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    assert jnp.allclose(rotated_y, expected_y, rtol=1e-7, atol=1e-6)

    # Test rotation of 3D points around z-axis by pi/2
    coords_z = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    axis_z = jnp.array([0.0, 0.0, 1.0])  # z-axis
    theta_z = jnp.pi / 2

    rotated_z = core.rotate_around_axis(coords_z, theta_z, axis_z)

    expected_z = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert jnp.allclose(rotated_z, expected_z, rtol=1e-7, atol=1e-6)


### align_point_cloud

def gen_coords():
    # Create a full-dimensional 3D point cloud
    rng = jr.PRNGKey(0)
    N = 100
    # Generate points with variability in all dimensions:
    # - Large spread along x
    # - Moderate spread along y
    # - Small spread along z
    rng, key_x, key_y, key_z = jr.split(rng, 4)
    x = jr.normal(key_x, (N, 1)) * 5.0
    y = jr.normal(key_y, (N, 1)) * 2.0
    z = jr.normal(key_z, (N, 1)) * 1.0
    coords = jnp.concatenate([x, y, z], axis=1)

    return coords

# Point cloud alignment with mean centering and robust eigen decomposition
def test_align_point_cloud_with_mean_centering():

    coords = gen_coords()

    # Define order and target basis
    order = jnp.array([0, 1, 2])
    target_basis = jnp.eye(3)

    # Align point cloud with mean centering and robust decomposition
    aligned = core.align_point_cloud(
        coords, order, target_basis, robust=True, center=True
    )

    # Verify shape is preserved
    assert aligned.shape == coords.shape

    # Verify mean centering
    assert jnp.allclose(jnp.mean(aligned, axis=0), jnp.zeros(3), atol=1e-6)

def test_align_point_cloud_custom_centering_no_rotation():

    rng = jr.PRNGKey(0)
    N = 1000
    # Generate an isotropic 3D point cloud (mean ~0, covariance ~identity)
    centered_data = jr.normal(rng, (N, 3))
    
    # Define a custom center offset.
    custom_center = jnp.array([1.0, 1.0, 1.0])
    
    # Shift the point cloud by the custom center.
    coords = centered_data + custom_center
    
    # Use an identity target basis and natural order.
    order = jnp.array([0, 1, 2])
    target_basis = jnp.eye(3)
    
    # Call align_point_cloud with robust=False and custom center.
    # With isotropic data, the classical covariance of the centered data is nearly I,
    # so the eigenvectors should be nearly the identity basis.
    aligned = core.align_point_cloud(coords, order, target_basis, robust=False, center=False, center_coord=custom_center)
    
    # The expected result is that the mean of the aligned point cloud should be custom center
    # Allow some tolerance due to sampling noise.
    assert jnp.allclose(aligned.mean(axis = 0), custom_center, atol=1e-2)

# # Reordering eigenvectors based on provided order indices
# def test_reorder_eigenvectors_based_on_order_indices(self, mocker):
#     import jax.numpy as jnp
#     from NeuGeometry.core import align_point_cloud

#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 2.0, 3.0]),  # eigenvalues (not used in this test)
#         jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eigenvectors
#     ))

#     # Define input parameters
#     coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     order = jnp.array([2, 0, 1])  # Reorder indices
#     target_basis = jnp.eye(3)  # Identity matrix as target basis
#     robust = True
#     center = False

#     # Call the function under test
#     result = align_point_cloud(coords, order, target_basis, robust, center)

#     # Expected eigenvectors after reordering and sign correction
#     expected_eigenvectors = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

#     # Check if the eigenvectors were reordered correctly
#     assert jnp.allclose(result, expected_eigenvectors), "Eigenvectors were not reordered correctly"

# # Sign correction of eigenvectors against target basis
# def test_sign_correction_of_eigenvectors(self, mocker):
#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 2.0, 3.0]),  # eigenvalues (not used in this test)
#         jnp.array([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])  # eigenvectors
#     ))

#     # Define input parameters
#     coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     order = jnp.array([2, 1, 0])
#     target_basis = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#     robust = True
#     center = False

#     # Call the function under test
#     aligned_coords = align_point_cloud(coords, order, target_basis, robust, center)

#     # Check if the eigenvectors were sign-corrected against the target basis
#     expected_eigenvectors = jnp.array([[0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5]])
#     expected_dots = jnp.sum(expected_eigenvectors * target_basis, axis=1)
#     assert jnp.all(expected_dots >= 0), "Eigenvectors were not correctly sign-corrected"

# # Non-orthogonal target basis
# def test_align_with_non_orthogonal_target_basis(self, mocker):
#     import jax.numpy as jnp
#     from NeuGeometry.core import align_point_cloud

#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 2.0, 3.0]),  # eigenvalues (not used in this test)
#         jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eigenvectors
#     ))

#     # Define input parameters
#     coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     order = jnp.array([2, 0, 1])  # Reorder indices
#     target_basis = jnp.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])  # Non-orthogonal target basis
#     robust = True
#     center = False

#     # Call the function under test
#     result = align_point_cloud(coords, order, target_basis, robust, center)

#     # Expected result after alignment with non-orthogonal target basis
#     expected_result = jnp.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])

#     # Check if the point cloud was aligned correctly
#     assert jnp.allclose(result, expected_result), "Point cloud was not aligned correctly with non-orthogonal target basis"

# # Preservation of point cloud scale after transformation
# def test_preservation_of_point_cloud_scale(self, mocker):
#     import jax.numpy as jnp
#     from NeuGeometry.core import align_point_cloud

#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 2.0, 3.0]),  # eigenvalues (not used in this test)
#         jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eigenvectors
#     ))

#     # Define input parameters
#     coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     order = jnp.array([2, 0, 1])  # Reorder indices
#     target_basis = jnp.eye(3)  # Identity matrix as target basis
#     robust = True
#     center = False

#     # Call the function under test
#     result = align_point_cloud(coords, order, target_basis, robust, center)

#     # Calculate the scale (norm) of the original and transformed point clouds
#     original_scale = jnp.linalg.norm(coords - jnp.mean(coords, axis=0))
#     transformed_scale = jnp.linalg.norm(result - jnp.mean(result, axis=0))

#     # Check if the scale is preserved after transformation
#     assert jnp.isclose(original_scale, transformed_scale), "Point cloud scale was not preserved after transformation"

# # Degenerate point clouds (coplanar or collinear points)
# def test_align_point_cloud_with_degenerate_points(self, mocker):
#     import jax.numpy as jnp
#     from NeuGeometry.core import align_point_cloud

#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 1.0, 0.0]),  # eigenvalues for degenerate case
#         jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # eigenvectors for coplanar points
#     ))

#     # Define input parameters for a degenerate point cloud (coplanar points)
#     coords = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
#     order = jnp.array([0, 1, 2])  # No reordering needed
#     target_basis = jnp.eye(3)  # Identity matrix as target basis
#     robust = True
#     center = False

#     # Call the function under test
#     result = align_point_cloud(coords, order, target_basis, robust, center)

#     # Expected result should handle degenerate case gracefully
#     expected_result = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])

#     # Check if the function handles degenerate point clouds correctly
#     assert jnp.allclose(result, expected_result), "Function did not handle degenerate point cloud correctly"

# # test that Rotation preserves handedness/orientation
# def test_rotation_preserves_handedness(self, mocker):
#     import jax.numpy as jnp
#     from NeuGeometry.core import align_point_cloud

#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 2.0, 3.0]),  # eigenvalues (not used in this test)
#         jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eigenvectors
#     ))

#     # Define input parameters
#     coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     order = jnp.array([2, 0, 1])  # Reorder indices
#     target_basis = jnp.eye(3)  # Identity matrix as target basis
#     robust = True
#     center = False

#     # Call the function under test
#     result = align_point_cloud(coords, order, target_basis, robust, center)

#     # Check if the determinant of the rotation matrix is close to 1 (preserving orientation)
#     R_total = jnp.matmul(target_basis.T, jnp.take(jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), order, axis=0))
#     assert jnp.isclose(jnp.linalg.det(R_total), 1.0), "Rotation does not preserve handedness/orientation"

# # test that Rotation preserves handedness/orientation
# def test_rotation_preserves_handedness(self, mocker):
#     import jax.numpy as jnp
#     from NeuGeometry.core import align_point_cloud

#     # Mock the coord_eig_decomp function to control its output
#     mocker.patch('NeuGeometry.core.coord_eig_decomp', return_value=(
#         jnp.array([1.0, 2.0, 3.0]),  # eigenvalues (not used in this test)
#         jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eigenvectors
#     ))

#     # Define input parameters
#     coords = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     order = jnp.array([2, 0, 1])  # Reorder indices
#     target_basis = jnp.eye(3)  # Identity matrix as target basis
#     robust = True
#     center = False

#     # Call the function under test
#     result = align_point_cloud(coords, order, target_basis, robust, center)

#     # Check if the rotation preserves handedness/orientation
#     det_target_basis = jnp.linalg.det(target_basis)
#     det_result_basis = jnp.linalg.det(result)
#     assert jnp.isclose(det_target_basis, det_result_basis), "Rotation did not preserve handedness/orientation"

### circ_mean

# Calculate circular mean for array of angles in default [0, 2π] range
def test_default_range_mean():
    angles = jnp.array([0.0, jnp.pi/2, jnp.pi])
    expected = jnp.pi/2
    result = core.circ_mean(angles)
    assert jnp.allclose(result, expected, rtol=1e-7)

# Compute weighted circular mean with valid weights array
def test_weighted_circular_mean():
    angles = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
    weights = jnp.array([1, 2, 1, 2])
    expected = jnp.pi
    result = core.circ_mean(angles, weights=weights)
    assert jnp.allclose(result, expected, rtol=1e-7)

    # Calculate mean for angles in custom interval range [low, high]
    def test_custom_interval_mean(self):
        import jax.numpy as jnp
        angles = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
        low = -jnp.pi
        high = jnp.pi
        expected = 0.0
        result = circ_mean(angles, high=high, low=low)
        jnp.testing.assert_allclose(result, expected, rtol=1e-7)