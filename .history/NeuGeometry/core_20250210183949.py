import numpy as np
import math

def check_value_any(arr, *shapes, name=None):
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
            return check_value(arr, shape, name=name or "arr")
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
    
def euclidean_distance(v1:np.ndarray,v2:np.ndarray) -> np.ndarray:




    if (v1.ndim == 1) and (v2.ndim == 1):
        return np.sqrt(np.sum(np.square(v2-v1)))
    else:
        return np.sqrt(np.sum(np.square(v2 - v1), axis=1))