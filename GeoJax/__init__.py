__version__ = "0.1.1"

# Core
from .core import (
    magnitude,
    normalize,
    dot,
    cross,
    reject,
    scalar_triple,
    reflect,
    gram_schmidt,
    apply_affine,
    scale_along_basis,
    point_to_plane_distance,
    ray_plane_intersect,
    tetrahedron_volume,
    scale_point_cloud_by_robust_axis_extent,
)

# Angles & rotation
from .angles import angle, signed_angle, angle_between_planes, minimum_signed_angle
from .rotation import (
    rotation_matrix_from_rotvec,
    rotate_around_axis,
    rotation_between_vectors,
    rotation_matrix_between_vectors,
    angle_between_rotations,
)

# Projection
from .projection import (
    reject_axis,
    project_to_sphere,
    project_to_plane,
    project_to_xy_plane,
    project_to_yz_plane,
    project_to_xz_plane,
    orthographic_projection,
    stereographic_projection,
    equirectangular_projection,
    mercator_projection,
    lambert_azimuthal_projection,
    project_to_2d,
    project_to_vector,
)

# Alignment
from .alignment import (
    coord_eig_decomp,
    align_point_cloud,
    robust_covariance_mest,
    alignment_matrix,
    minimum_theta,
)

# Circ stats
from .circstats import circmean, circstd, circvar

# Bounds
from .bounds import (
    aabb_bounds,
    bounding_sphere,
    oriented_bounding_box,
    bounding_cylinder,
    tight_aabb_in_frame,
)

# Analysis
from .analysis import (
    mahalanobis_distance,
    mahalanobis_squared,
    detect_outliers_mahalanobis,
    ellipsoid_axes_from_covariance,
    robust_proportional_dispersion,
)

# Checks
from .checks import (
    is_unit_vector,
    is_collinear,
    is_orthogonal,
    orthonormal_basis_from_vector,
)

# Distance
from .distance import (
    euclidean,
    manhattan,
    chebyshev,
    minkowski,
    cosine,
    haversine,
    compute_distance,
)

# Basis
from .basis import basis, Basis

# Utils
from .utils import normalize_angle_array, origin_flip


__all__ = [
    # core
    "magnitude",
    "normalize",
    "dot",
    "cross",
    "reject",
    "scalar_triple",
    "reflect",
    "gram_schmidt",
    "apply_affine",
    "scale_along_basis",
    "point_to_plane_distance",
    "ray_plane_intersect",
    "tetrahedron_volume",
    "scale_point_cloud_by_robust_axis_extent",
    # angles
    "angle",
    "signed_angle",
    "angle_between_planes",
    "minimum_signed_angle",
    # rotation
    "rotation_matrix_from_rotvec",
    "rotate_around_axis",
    "rotation_between_vectors",
    "rotation_matrix_between_vectors",
    "angle_between_rotations",
    # projection
    "reject_axis",
    "project_to_sphere",
    "project_to_plane",
    "project_to_xy_plane",
    "project_to_yz_plane",
    "project_to_xz_plane",
    "orthographic_projection",
    "stereographic_projection",
    "equirectangular_projection",
    "mercator_projection",
    "lambert_azimuthal_projection",
    "project_to_2d",
    "project_to_vector",
    # alignment
    "coord_eig_decomp",
    "align_point_cloud",
    "robust_covariance_mest",
    "alignment_matrix",
    "minimum_theta",
    # circstats
    "circmean",
    "circstd",
    "circvar",
    # bounds
    "aabb_bounds",
    "bounding_sphere",
    "oriented_bounding_box",
    "bounding_cylinder",
    "tight_aabb_in_frame",
    # analysis
    "mahalanobis_distance",
    "mahalanobis_squared",
    "detect_outliers_mahalanobis",
    "ellipsoid_axes_from_covariance",
    "robust_proportional_dispersion",
    # checks
    "is_unit_vector",
    "is_collinear",
    "is_orthogonal",
    "orthonormal_basis_from_vector",
    # distance
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "cosine",
    "haversine",
    "compute_distance",
    # basis
    "basis",
    "Basis",
    # utils
    "normalize_angle_array",
    "origin_flip",
]
