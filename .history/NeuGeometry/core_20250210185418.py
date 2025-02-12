import numpy as np
import math

def _check_value(arr, shape, name=None):
    """
    Check that the given argument has the expected shape. Shape dimensions can
    be ints or -1 for a wildcard. The wildcard dimensions are returned, which
    allows them to be used for subsequent validation or elsewhere in the
    function.

    Args:
        arr (np.arraylike): An array-like input.
        shape (list): Shape to validate. To require an array with 3 elements,
            pass `(3,)`. To require n by 3, pass `(-1, 3)`.
        name (str): Variable name to embed in the error message.

    Returns:
        object: The wildcard dimension (if one) or a tuple of wildcard
        dimensions (if more than one).

    Example:
        >>> vg.shape.check_value(np.zeros((4, 3)), (-1, 3))
        >>> # Proceed with confidence that `points` is a k x 3 array.

    Example:
        >>> k = vg.shape.check_value(np.zeros((4, 3)), (-1, 3))
        >>> k
        4
    """

    def is_wildcard(dim):
        return dim == -1

    if any(not isinstance(dim, int) and not is_wildcard(dim) for dim in shape):
        raise ValueError("Expected shape dimensions to be int")

    if name is None:
        preamble = "Expected an array"
    else:
        preamble = f"{name} must be an array"

    if arr is None:
        raise ValueError(f"{preamble} with shape {shape}; got None")
    try:
        len(arr.shape)
    except (AttributeError, TypeError):
        raise ValueError(f"{preamble} with shape {shape}; got {arr.__class__.__name__}")

    # Check non-wildcard dimensions.
    if len(arr.shape) != len(shape) or any(
        actual != expected
        for actual, expected in zip(arr.shape, shape)
        if not is_wildcard(expected)
    ):
        raise ValueError(f"{preamble} with shape {shape}; got {arr.shape}")

    wildcard_dims = [
        actual for actual, expected in zip(arr.shape, shape) if is_wildcard(expected)
    ]
    if len(wildcard_dims) == 0:
        return None
    elif len(wildcard_dims) == 1:
        return wildcard_dims[0]
    else:
        return tuple(wildcard_dims)

def _check_value_any(arr, *shapes, name=None):
    """
    Check that the given argument has any of the expected shapes. Shape dimensons
    can be ints or -1 for a wildcard.

    Args:
        arr (np.arraylike): An array-like input.
        shape (list): Shape candidates to validate. To require an array with 3
            elements, pass `(3,)`. To require n by 3, pass `(-1, 3)`.
        name (str): Variable name to embed in the error message.

    Returns:
        object: The wildcard dimension of the matched shape (if one) or a tuple
        of wildcard dimensions (if more than one). If the matched shape has no
        wildcard dimensions, returns `None`.

    Example:
        >>> k = check_shape_any(points, (3,), (-1, 3), name="points")
        >>> check_shape_any(
                reference_points_of_lines,
                (3,),
                (-1 if k is None else k, 3),
                name="reference_points_of_lines",
            )
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required")
    for shape in shapes:
        try:
            return _check_value(arr, shape, name=name or "arr")
        except ValueError:
            pass

    if name is None:
        preamble = "Expected an array"
    else:
        preamble = f"Expected {name} to be an array"

    if len(shapes) == 1:
        (shape_choices,) = shapes
    else:
        shape_choices = ", ".join(
            shapes[:-2] + (" or ".join([str(shapes[-2]), str(shapes[-1])]),)
        )

    if arr is None:
        raise ValueError(f"{preamble} with shape {shape_choices}; got None")
    else:
        try:
            len(arr.shape)
        except (AttributeError, TypeError):
            raise ValueError(
                f"{preamble} with shape {shape_choices}; got {arr.__class__.__name__}"
            )
        raise ValueError(f"{preamble} with shape {shape_choices}; got {arr.shape}")


def _check(locals_namespace, name, shape):
    """
    Convenience function for invoking `vg.shape.check_value()` with a
    `locals()` dict.

    Args:
        namespace (dict): A subscriptable object, typically `locals()`.
        name (str): Key to pull from `namespace`.
        shape (list): Shape to validate. To require 3 by 1, pass `(3,)`. To
            require n by 3, pass `(-1, 3)`.

    Returns:
        object: The wildcard dimension (if one) or a tuple of wildcard
        dimensions (if more than one).

    Example:
        >>> def my_fun_function(points):
        ...     vg.shape.check(locals(), 'points', (-1, 3))
        ...     # Proceed with confidence that `points` is a k x 3 array.

    Example:
        >>> def my_fun_function(points):
        ...     k = vg.shape.check(locals(), 'points', (-1, 3))
        ...     print(f"my_fun_function invoked with {k} points")

    """
    return _check_value(locals_namespace[name], shape, name=name)


def broadcast_and_tile(v1, v2):
    if v1.ndim == 1 and v2.ndim == 2:
        _check(locals(), "v1", (3,))
        k = _check(locals(), "v2", (-1, 3))
        return np.tile(v1, (k, 1)), v2
    elif v1.ndim == 2 and v2.ndim == 1:
        k = _check(locals(), "v1", (-1, 3))
        _check(locals(), "v2", (3,))
        return v1, np.tile(v2, (k, 1))
    elif v1.ndim == 2 and v2.ndim == 2:
        k = _check(locals(), "v1", (-1, 3))
        _check(locals(), "v2", (k, 3))
        return v1, v2
    else:
        raise ValueError('arr must be a single vector, or a stack of them!')

def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise a given vector (or stack of vectors) to be unit vectors

    Parameters
    ----------
    arr : np.ndarray
        Either a single vector, or a stacked array of vectors

    Returns
    -------
    np.ndarray
        Normalised version of the input array. Either the single vector given, or the normalised version of the stack of vectors.

    Raises
    ------
    ValueError
        If the input is not either a single vector, or a stack of vectors.
    """
    if arr.ndim == 1:
        return arr / np.linalg.norm(arr)
    elif arr.ndim == 2:
        return arr / np.linalg.norm(arr, axis = 1)[:, np.newaxis]
    else:
        raise ValueError('arr must be a single vector, or a stack of them!')
    
def magnitude(arr: np.ndarray) -> np.ndarray:
    """Get the length of a given vector, or stack of vectors

    Parameters
    ----------
    arr : np.ndarray
        Either a single vector, or a stacked array of vectors

    Returns
    -------
    np.ndarray
        Magnitude of the given vector(s)

    Raises
    ------
    ValueError
        If the input is not either a single vector, or a stack of vectors.
    """

    if arr.ndim == 1:
        return np.linalg.norm(arr)
    elif arr.ndim == 2:
        return np.linalg.norm(arr, axis = 1)
    else:
        raise ValueError('arr must be a single vector, or a stack of them!')
    
def euclidean_distance(v1:np.ndarray,v2:np.ndarray) -> np.ndarray | float:
    """Compute the Euclidean Distance between either two points. 
    If two single points are given, this will simply be the distance between the two points.

    Two sets of stacked vectors can also be given, in which case the must both have the same number 
    of vectors! So v1 must have the same number of rows as v2!

    Also, input must be 3 dimensional - so x,y,z!

    Parameters
    ----------
    v1 : np.ndarray
        The first vector, or stack of vectors
    v2 : np.ndarray
        The second vector or stack of vectors

    Returns
    -------
    np.ndarray | float
        Array of pairwise distances between points, or single distance between two points if two points are given.
    
    ToDo: Extend for one to many calculations?
    
    """

    # make sure we have 3D vectors and if multiple, that they are the same shape
    k = _check_value_any(v1, (3,), (-1, 3), name="v1")
    _check_value_any(
        v2,
        (3,),
        (-1 if k is None else k, 3),
        name="v2",
    )

    if (v1.ndim == 1) and (v2.ndim == 1):
        return np.sqrt(np.sum(np.square(v2-v1)))
    else:
        return np.sqrt(np.sum(np.square(v2 - v1), axis=1))
    
def dot(v1, v2):
    """
    Compute individual or pairwise dot products.

    Args:
        v1 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A `(3,)` vector or a `kx3` stack of vectors. If
            stacks are provided for both `v1` and `v2` they must have the
            same shape.
    """
    if v1.ndim == 1 and v2.ndim == 1:
        _check(locals(), "v1", (3,))
        _check(locals(), "v2", (3,))
        return np.dot(v1, v2)
    else:
        v1, v2 = broadcast_and_tile(v1, v2)
        return np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))


def scalar_projection(vector, onto):
    """
    Compute the scalar projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if vector.ndim == 1:
        _check(locals(), "vector", (3,))
        _check(locals(), "onto", (3,))
    else:
        k = _check(locals(), "vector", (-1, 3))
        if onto.ndim == 1:
            _check(locals(), "onto", (3,))
        else:
            _check(locals(), "onto", (k, 3))

    return dot(vector, normalise(onto))
    
def project(vector, onto):
    """
    Compute the vector projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if vector.ndim == 1:
        return scalar_projection(vector, onto=onto) * normalise(onto)
    elif vector.ndim == 2:
        return scalar_projection(vector, onto=onto)[:, np.newaxis] * normalize(onto)
    else:
        raise ValueError('arr must be a single vector, or a stack of them!')

def reject(vector, from_v):
    """
    Compute the vector rejection of `vector` from `from_v` -- i.e.
    the vector component of `vector` perpendicular to `from_v`.

    `from_v` need not be normalized.

    """
    return vector - project(vector, onto=from_v)
    
def angle(v1: np.ndarray, v2: np.ndarray, plane_normal: np.ndarray = None, assume_normalized: bool = False, units: str = "deg") -> np.ndarray | float:

    # check returned units are ok
    if units not in ["deg", "rad"]:
        raise ValueError(f"Unrecognized units {units}; expected deg or rad")


    if plane_normal is not None:
        # This is a simple approach. Since this is working in two dimensions,
        # a smarter approach could reduce the amount of computation needed.
        v1, v2 = [reject(v, from_v=plane_normal) for v in (v1, v2)]

    dot_products = np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))

    if assume_normalized:
        cosines = dot_products
    else:
        cosines = dot_products / magnitude(v1) / magnitude(v2)

    # Clip, because the dot product can slip past 1 or -1 due to rounding and
    # we can't compute arccos(-1.00001).
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    if units == "deg":
        angles = np.degrees(angles)

    return angles[0] if v1.ndim == 1 and v2.ndim == 1 else angles