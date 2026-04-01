"""
Module for creating various types of matrices.

Provides functions for creating matrices of given size,
identity, zero, random matrices, and matrices from existing lists.
Supports both int and float data types.
"""

import random
from typing import List, Any

from .types import Matrix, MatrixElement
from .validation import is_valid_matrix


def create_matrix(rows: int, cols: int, default_value: MatrixElement = 0) -> Matrix:
    """
    Creates a matrix of specified size with the given default value.

    Args:
        rows: Number of rows (must be positive)
        cols: Number of columns (must be positive)
        default_value: Value to fill the matrix with (int or float)

    Returns:
        Matrix: Matrix of size rows x cols filled with default_value

    Raises:
        ValueError: If rows or cols are not positive
        TypeError: If rows or cols are not integers
        TypeError: If default_value is not int or float

    Examples:
        >>> create_matrix(2, 2, 5)
        [[5, 5], [5, 5]]
        >>> create_matrix(2, 2, 3.14)
        [[3.14, 3.14], [3.14, 3.14]]
    """
    # Type validation for dimensions
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError(f"Rows and cols must be integers, got rows={type(rows).__name__}, cols={type(cols).__name__}")

    # Validation for positive dimensions
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Matrix size must be positive, got rows={rows}, cols={cols}")

    # Validate default_value type
    if not isinstance(default_value, (int, float)):
        raise TypeError(f"Default value must be int or float, got {type(default_value).__name__}")

    # Create matrix
    return [[default_value for _ in range(cols)] for _ in range(rows)]


def create_identity_matrix(n: int) -> Matrix:
    """
    Creates an identity matrix of size n x n.

    Args:
        n: Size of the matrix (must be positive)

    Returns:
        Matrix: Identity matrix of size n x n

    Raises:
        ValueError: If n is not positive
        TypeError: If n is not an integer

    Examples:
        >>> create_identity_matrix(3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    if not isinstance(n, int):
        raise TypeError(f"Matrix size must be integer, got {type(n).__name__}")

    if n <= 0:
        raise ValueError(f"Matrix size must be positive, got {n}")

    matrix = create_matrix(n, n, 0)
    for i in range(n):
        matrix[i][i] = 1
    return matrix


def create_zero_matrix(rows: int, cols: int) -> Matrix:
    """
    Creates a zero matrix of specified size.

    Args:
        rows: Number of rows (must be positive)
        cols: Number of columns (must be positive)

    Returns:
        Matrix: Zero matrix of size rows x cols

    Raises:
        ValueError: If rows or cols are not positive
        TypeError: If rows or cols are not integers

    Examples:
        >>> create_zero_matrix(2, 3)
        [[0, 0, 0], [0, 0, 0]]
    """
    return create_matrix(rows, cols, 0)


def create_random_matrix(
    rows: int,
    cols: int,
    min_val: MatrixElement,
    max_val: MatrixElement,
) -> Matrix:
    """
    Creates a matrix of specified size with random values in the given range.

    Args:
        rows: Number of rows (must be positive)
        cols: Number of columns (must be positive)
        min_val: Minimum value (int or float)
        max_val: Maximum value (int or float)

    Returns:
        Matrix: Matrix of size rows x cols with random values

    Raises:
        ValueError: If rows or cols are not positive
        ValueError: If min_val > max_val
        TypeError: If rows or cols are not integers
        TypeError: If min_val or max_val are not int or float

    Examples:
        >>> create_random_matrix(2, 2, 0, 10)  # random 2x2 matrix with numbers from 0 to 10
        [[3, 7], [2, 9]]
    """
    # Type validation for dimensions
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError(f"Rows and cols must be integers, got rows={type(rows).__name__}, cols={type(cols).__name__}")

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Matrix size must be positive, got rows={rows}, cols={cols}")

    # Type validation for bounds
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise TypeError(f"min_val and max_val must be int or float, got min_val={type(min_val).__name__}, max_val={type(max_val).__name__}")

    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) can't be greater than max_val ({max_val})")

    # Support float via random.uniform
    if isinstance(min_val, int) and isinstance(max_val, int):
        # Both are integers, use randint (includes max_val)
        rand_func = lambda: random.randint(min_val, max_val)
    else:
        # At least one is float, use uniform
        rand_func = lambda: random.uniform(min_val, max_val)

    return [[rand_func() for _ in range(cols)] for _ in range(rows)]


def create_matrix_from_list(data: List[List[Any]]) -> Matrix:
    """
    Creates a matrix from an existing 2D list.

    Args:
        data: 2D list to create the matrix from

    Returns:
        Matrix: Matrix created from the input data (returns the same object)

    Raises:
        ValueError: If data is empty or rows have different lengths
        ValueError: If data contains non-numeric or non-finite values
        TypeError: If data is not a list

    Examples:
        >>> create_matrix_from_list([[1, 2], [3, 4]])
        [[1, 2], [3, 4]]
        >>> create_matrix_from_list([[1.5, 2.5], [3.5, 4.5]])
        [[1.5, 2.5], [3.5, 4.5]]
    """
    # Validate input type
    if not isinstance(data, list):
        raise TypeError(f"Input data must be a list, got {type(data).__name__}")

    if not data:
        raise ValueError("A list can't be empty")

    # Check that data is a 2D list
    if not isinstance(data[0], list):
        raise TypeError(f"Data must be a 2D list, got {type(data[0]).__name__} at row 0")

    # Use the common validation (which checks structure, types, finiteness)
    is_valid_matrix(data, allow_empty=False)
    return data


def create_matrix_from_list_copy(data: List[List[Any]]) -> Matrix:
    """
    Creates a copy of a matrix from an existing 2D list.
    Useful when you need to avoid side effects from modifying the original data.

    Args:
        data: 2D list to create the matrix from

    Returns:
        Matrix: Copy of the matrix created from the input data

    Raises:
        ValueError: If data is empty or rows have different lengths
        ValueError: If data contains non-numeric or non-finite values
        TypeError: If data is not a list

    Examples:
        >>> original = [[1, 2], [3, 4]]
        >>> copy = create_matrix_from_list_copy(original)
        >>> copy[0][0] = 999
        >>> original[0][0]  # Original unchanged
        1
    """
    # Validate and return a deep copy (list comprehension of rows)
    matrix = create_matrix_from_list(data)
    return [row[:] for row in matrix]