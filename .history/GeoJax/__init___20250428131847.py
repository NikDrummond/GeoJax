from .core import (
    magnitude, normalize, dot, cross, reject, project_to_vector,
    scalar_triple, reflect, gram_schmidt, apply_affine,
    point_to_plane_distance, ray_plane_intersect, tetrahedron_volume, scale_along_basis,
    scale_point_cloud_by_robust_axis_extent,
)

from .angles import angle, signed_angle
from .rotation import rotation_matrix_from_rotvec, rotate_around_axis, rotation_between_vectors
from .projection import (
    reject_axis, project_to_sphere,
    project_to_vector, project_to_plane, project_to_xy_plane, project_to_yz_plane, project_to_xz_plane, orthographic_projection, stereographic_projection, equirectangular_projection, mercator_projection
)
from .alignment import (
    coord_eig_decomp, align_point_cloud,
    robust_covariance_mest, alignment_matrix, minimum_theta
)
from .circstats import (
    circmean, circstd, circvar
)
from .bounds import (
    aabb_bounds, bounding_sphere, oriented_bounding_box
)
from .analysis import (
    mahalanobis_distance, detect_outliers_mahalanobis,
    ellipsoid_axes_from_covariance,
    robust_proportional_dispersion,
)
from .checks import (
    is_unit_vector, is_collinear, is_orthogonal,
    angle_between_planes, orthonormal_basis_from_vector
)
from .basis import basis
from .utils import normalize_angle_array, origin_flip

__all__ = [
    # core
    "magnitude", "normalize", "dot", "cross", "reject", "project_to_vector",
    "scalar_triple", "reflect", "gram_schmidt", "apply_affine", "scale_along_basis",
    "point_to_plane_distance", "ray_plane_intersect", "tetrahedron_volume", "scale_point_cloud_by_robust_axis_extent",

    # angles
    "angle", "signed_angle",

    # rotation
    "rotation_matrix_from_rotvec", "rotate_around_axis", "rotation_between_vectors",

    # projection
    "reject_axis", "project_to_sphere", "project_to_vector", "project_to_plane",

    # alignment
    "coord_eig_decomp", "align_point_cloud", "robust_covariance_mest",
    "alignment_matrix", "minimum_theta",

    # circstats
    "circmean", "circstd", "circvar",

    # bounds
    "aabb_bounds", "bounding_sphere", "oriented_bounding_box",

    # analysis
    "mahalanobis_distance", "detect_outliers_mahalanobis",
    "ellipsoid_axes_from_covariance", "robust_proportional_dispersion",

    # checks
    "is_unit_vector", "is_collinear", "is_orthogonal",
    "angle_between_planes", "orthonormal_basis_from_vector",

    # basis
    "basis",

    # utils
    "normalize_angle_array", "origin_flip",
]
