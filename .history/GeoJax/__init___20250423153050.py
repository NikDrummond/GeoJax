# __init__.py

# Core vector operations

from .core import (
    magnitude,
    normalize,
    dot,
    cross,
    reject,
    project_to_vector,
    scalar_triple,
    reflect,
    gram_schmidt,
    apply_affine,
    point_to_plane_distance,
    ray_plane_intersect,
    tetrahedron_volume,
)

# Alignment and decomposition
from .alignment import (
    robust_covariance_mest,
    coord_eig_decomp,
    align_point_cloud,
    minimum_theta,
    alignment_matrix,
)

# Rotation
from .rotation import (
    rotation_matrix_from_rotvec,
    rotate_around_axis,
    rotation_matrix_between_vectors,
    angle_between_rotations,
)


# Projection utilities
from .projection import (
    reject_axis,
    project_to_sphere,
    project_to_vector,
    project_to_plane,
)


# Circular statistics
from .circstats import (
    circmean,
    circstd,
)

# Bounding shapes
from .bounds import (
    aabb_bounds,
    bounding_sphere,
    oriented_bounding_box,
)

# Statistical analysis
from .analysis import (
    mahalanobis_distance,
    detect_outliers_mahalanobis,
    ellipsoid_axes_from_covariance,
)

# Geometric checks and validation
from .checks import (
    is_unit_vector,
    is_collinear,
    is_orthogonal,
    angle_between_planes,
    orthonormal_basis_from_vector,
)

# Basis vectors
from .basis import (
    basis_vectors,
)

# Utility functions
from .utils import (
    normalize_angle_array,
    center_points,
    scale_coords,
    origin_flip,
)

__all__ = [
    # Core
    "magnitude", "normalize", "dot", "cross", "reject", "reflect", "scalar_triple",
    "project_to_vector", "gram_schmidt", "apply_affine", "point_to_plane_distance",
    "ray_plane_intersect", "tetrahedron_volume",

    # Alignment
    "align_point_cloud", "coord_eig_decomp", "minimum_theta",

    # Rotation
    "rotation_matrix_from_rotvec", "rotate_around_axis",

    # Projection
    "project_to_plane", "project_to_sphere",

    # Circular stats
    "circmean", "circstd",

    # Bounding volumes
    "aabb_bounds", "bounding_sphere", "oriented_bounding_box",

    # Analysis
    "mahalanobis_distance", "detect_outliers_mahalanobis", "ellipsoid_axes_from_covariance",

    # Checks
    "is_unit_vector", "is_collinear", "is_orthogonal", "angle_between_planes", "orthonormal_basis_from_vector",

    # Basis
    "basis_vectors",

    # Utils
    "normalize_angle_array", "center_points", "scale_coords", "origin_flip",
]