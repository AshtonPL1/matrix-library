"""
Module for advanced matrix operations.

Provides functions for checking matrix properties:
- Check if matrix is square
- Calculate trace (sum of diagonal elements)
- Check if matrix is symmetric (A == A^T)
"""

from typing import Any

from .types import Matrix, MatrixElement
from .validation import is_valid_matrix


def is_square(matrix: Any) -> bool:
    """
    Checks if the matrix is square (number of rows equals number of columns).

    This function never raises exceptions; it returns False for invalid input.

    Args:
        matrix: Input matrix (any type)

    Returns:
        bool: True if the matrix is square, False otherwise

    Examples:
        >>> is_square([[1, 2], [3, 4]])
        True
        >>> is_square([[1, 2, 3], [4, 5, 6]])
        False
        >>> is_square([])
        False
        >>> is_square("not a matrix")
        False
    """
    if not isinstance(matrix, list):
        return False
    if not matrix:
        return False
    try:
        rows = len(matrix)
        cols = len(matrix[0])
        # Check all rows are lists and have same length
        for row in matrix:
            if not isinstance(row, list) or len(row) != cols:
                return False
        return rows == cols
    except (IndexError, TypeError):
        return False


def trace(matrix: Matrix) -> MatrixElement:
    """
    Calculates the trace of a square matrix (sum of diagonal elements).

    Args:
        matrix: Square matrix

    Returns:
        MatrixElement: Sum of diagonal elements (int or float)

    Raises:
        TypeError: If matrix is not a list of lists or contains non-numeric elements
        ValueError: If matrix is empty, not square, or contains non-finite values

    Examples:
        >>> trace([[1, 2], [3, 4]])
        5
        >>> trace([[1.5, 2.5], [3.5, 4.5]])
        6.0
    """
    # Validate matrix (checks structure, numeric types, finiteness)
    is_valid_matrix(matrix, allow_empty=False)

    if not is_square(matrix):
        raise ValueError("Trace requires a square matrix")

    n = len(matrix)
    total = 0
    for i in range(n):
        total += matrix[i][i]
    return total


def is_symmetric(matrix: Matrix) -> bool:
    """
    Checks if the matrix is symmetric (A = A^T).

    Args:
        matrix: Square matrix

    Returns:
        bool: True if symmetric, False otherwise

    Raises:
        TypeError: If matrix is not a list of lists or contains non-numeric elements
        ValueError: If matrix is empty, not square, or contains non-finite values

    Examples:
        >>> is_symmetric([[1, 2], [2, 3]])
        True
        >>> is_symmetric([[1, 2], [3, 4]])
        False
    """
    # Validate matrix (checks structure, numeric types, finiteness)
    is_valid_matrix(matrix, allow_empty=False)

    if not is_square(matrix):
        raise ValueError("Symmetric check requires a square matrix")

    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True