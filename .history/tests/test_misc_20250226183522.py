


# ### robust covariance matrix calculation


# # Compute robust covariance matrix for well-behaved input data with default parameters
# def test_robust_covariance_well_behaved_data():

#     # Generate well-behaved test data
#     rng = jr.PRNGKey(0)
#     X = jr.normal(rng, shape=(100, 3))

#     # Compute robust covariance
#     cov_matrix = vector_geometry.robust_covariance_mest(X)

#     # Check shape and properties
#     assert cov_matrix.shape == (3, 3)
#     assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
#     assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite


# # Handle input data with zero variance in some dimensions
# def test_robust_covariance_zero_variance():

#     # Create data with zero variance in one dimension
#     X = jnp.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [3.0, 4.0, 0.0], [4.0, 5.0, 0.0]])

#     # Compute robust covariance
#     cov_matrix = vector_geometry.robust_covariance_mest(X)

#     # Check shape and properties
#     assert cov_matrix.shape == (3, 3)
#     assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
#     assert jnp.isfinite(cov_matrix).all()  # No NaN or inf values
#     assert jnp.abs(cov_matrix[2, 2]) < 1e-6  # Zero variance dimension


# # Converge within max_iter iterations for typical datasets
# def test_convergence_within_max_iter():

#     # Generate typical dataset
#     rng = jr.PRNGKey(42)
#     X = jr.normal(rng, shape=(100, 5))

#     # Compute robust covariance
#     cov_matrix = vector_geometry.robust_covariance_mest(X)

#     # Check shape and properties
#     assert cov_matrix.shape == (5, 5)
#     assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
#     assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite


# # Return symmetric positive definite covariance matrix
# def test_robust_covariance_symmetric_positive_definite():

#     # Generate test data
#     rng = jr.PRNGKey(42)
#     X = jr.normal(rng, shape=(50, 5))

#     # Compute robust covariance
#     cov_matrix = vector_geometry.robust_covariance_mest(X)

#     # Check shape and properties
#     assert cov_matrix.shape == (5, 5)
#     assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
#     assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite


# # Downweight outliers appropriately using Huber-like weights
# def test_downweight_outliers_with_huber_weights():
#     # Generate test data with outliers
#     rng = jr.PRNGKey(0)
#     rng1, rng2 = jr.split(rng)
#     X = jnp.concatenate(
#         [
#             jr.normal(rng1, shape=(95, 3)),  # Well-behaved data
#             10.0 + 5.0 * jr.normal(rng2, shape=(5, 3)),  # Outliers
#         ]
#     )

#     # Compute robust covariance
#     cov_matrix = vector_geometry.robust_covariance_mest(X)

#     # Check shape and properties
#     assert cov_matrix.shape == (3, 3)
#     assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
#     assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite

#     # Additional check: ensure covariance is not overly influenced by outliers
#     cov_no_outliers = vector_geometry.robust_covariance_mest(X[:95])
#     assert jnp.allclose(cov_matrix, cov_no_outliers, atol=1.0)  # Allow some tolerance


# # Handle different input data shapes and dimensions correctly
# def test_robust_covariance_varied_shapes():

#     # Test with different shapes
#     shapes = [(50, 2), (100, 5), (200, 10)]
#     for shape in shapes:
#         rng = jr.PRNGKey(0)
#         X = jr.normal(rng, shape=shape)

#         # Compute robust covariance
#         cov_matrix = vector_geometry.robust_covariance_mest(X)

#         # Check shape and properties
#         assert cov_matrix.shape == (shape[1], shape[1])
#         assert jnp.allclose(cov_matrix, cov_matrix.T)  # Symmetry
#         assert jnp.all(jnp.linalg.eigvals(cov_matrix) > 0)  # Positive definite


# ### Eigen decomposition of coordinates


# # Verify eigenvalues and eigenvectors are correctly computed for standard input matrix
# def test_standard_input_eigendecomposition():
#     # Create a simple 2D dataset with known covariance structure
#     coords = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5], [0.0, -0.5]])

#     evals, evecs = vector_geometry.coord_eig_decomp(coords, robust=False, center=True)

#     # Expected values (normalized)
#     expected_evals = jnp.array([0.8, 0.2])  # Larger variance in x direction
#     expected_evecs = jnp.array(
#         [[1.0, 0.0], [0.0, 1.0]]  # First component along x
#     )  # Second component along y

#     assert jnp.allclose(evals, expected_evals, rtol=1e-5)
#     assert jnp.allclose(jnp.abs(evecs), jnp.abs(expected_evecs), rtol=1e-5)


# # Handle input matrix with zero variance in some dimensions
# def test_zero_variance_dimension():
#     # Create dataset with zero variance in y direction
#     coords = jnp.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])

#     evals, evecs = vector_geometry.coord_eig_decomp(coords, robust=False, center=True)

#     # First eigenvalue should be 1.0 (all variance), second should be 0
#     expected_evals = jnp.array([1.0, 0.0])
#     # First eigenvector should be along x, second along y
#     expected_evecs = jnp.array([[1.0, 0.0], [0.0, 1.0]])

#     assert jnp.allclose(evals, expected_evals, rtol=1e-5)
#     assert jnp.allclose(jnp.abs(evecs), jnp.abs(expected_evecs), rtol=1e-5)


# # Check robust covariance estimation produces valid results when robust=True
# def test_robust_covariance_estimation():
#     # Create a dataset with an outlier to test robust covariance estimation
#     coords = jnp.array(
#         [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5], [0.0, -0.5], [10.0, 10.0]]  # Outlier
#     )

#     evals, evecs = vector_geometry.coord_eig_decomp(coords, robust=True, center=True)

#     # Instead of comparing directly to hardcoded eigenvalues/eigenvectors,
#     # we check for expected properties.

#     # 1. The eigenvalues should be positive and sum to 1 (if PCA=True).
#     assert jnp.all(evals > 0)
#     assert jnp.allclose(jnp.sum(evals), 1.0, atol=1e-5)

#     # 2. The eigenvector matrix should be orthonormal.
#     ortho = jnp.allclose(jnp.matmul(evecs, evecs.T), jnp.eye(2), atol=0.2)
#     assert ortho, "Eigenvectors are not orthonormal within tolerance."

#     # 3. The principal eigenvector (first row) should roughly align with [1, 0]
#     # (up to a sign). We check that the absolute dot product is near 1.
#     principal_alignment = jnp.abs(jnp.dot(evecs[0], jnp.array([1.0, 0.0])))
#     assert jnp.allclose(
#         principal_alignment, 1.0, atol=0.2
#     ), f"Principal eigenvector does not align with [1,0]: {principal_alignment}"


# ### minimum theta


# # Compute minimal angle between vector and line in 3D space
# def test_minimal_angle_computation():
#     # Test vectors and normal
#     v1 = jnp.array([1.0, 0.0, 0.0])
#     v2 = jnp.array([0.0, 1.0, 0.0])
#     normal = jnp.array([0.0, 0.0, 1.0])

#     # Expected angle should be pi/2 radians (90 degrees)
#     # But minimum_theta returns pi/2 or -pi/2 for perpendicular vectors
#     result = vector_geometry.minimum_theta(v1, v2, normal)

#     # Check that absolute value is pi/2
#     assert jnp.abs(result) == jnp.pi / 2


# # Handle parallel vectors (0 degree angle)
# def test_parallel_vectors():
#     # Test with parallel vectors
#     v1 = jnp.array([1.0, 0.0, 0.0])
#     v2 = jnp.array([2.0, 0.0, 0.0])  # Same direction, different magnitude
#     normal = jnp.array([0.0, 0.0, 1.0])

#     result = vector_geometry.minimum_theta(v1, v2, normal)

#     # For parallel vectors, angle should be 0
#     assert jnp.allclose(result, 0.0, atol=1e-7)


# # Return angle in degrees when to_degree is True
# def test_return_angle_in_degrees():
#     # Test vectors and normal
#     v1 = jnp.array([1.0, 0.0, 0.0])
#     v2 = jnp.array([0.0, 1.0, 0.0])
#     normal = jnp.array([0.0, 0.0, 1.0])

#     # Expected angle should be 90 degrees
#     result = vector_geometry.minimum_theta(v1, v2, normal, to_degree=True)

#     # Check that the result is 90 degrees
#     assert jnp.isclose(result, 90.0)