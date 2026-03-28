"""
Module for basic matrix operations.

Provides functions for accessing and modifying matrix elements,
getting matrix dimensions, copying matrices, and printing them.
"""

import copy
from typing import Any

from matrix_operations.types import Matrix, MatrixElement
from matrix_operations.validation import is_valid_matrix


def get_element(matrix: Matrix, row: int, col: int) -> MatrixElement:
    """
    Retrieves an element from the matrix at the specified position.

    Args:
        matrix: Input matrix
        row: Row index (0-based)
        col: Column index (0-based)

    Returns:
        MatrixElement: Value at the specified position

    Raises:
        TypeError: If matrix is invalid or indices are not integers
        ValueError: If matrix is empty
        IndexError: If row or col index is out of range

    Examples:
        >>> matrix = [[1, 2], [3, 4]]
        >>> get_element(matrix, 0, 1)
        2
        >>> get_element(matrix, 1, 0)
        3
    """
    # Validate matrix
    is_valid_matrix(matrix, allow_empty=False)

    # Validate index types
    if not isinstance(row, int):
        raise TypeError(f"Row index must be integer, got {type(row).__name__}")
    if not isinstance(col, int):
        raise TypeError(f"Column index must be integer, got {type(col).__name__}")

    # Validate indices range
    if row < 0 or row >= len(matrix):
        raise IndexError(f"Row index {row} out of range [0, {len(matrix)-1}]")
    if col < 0 or col >= len(matrix[0]):
        raise IndexError(f"Column index {col} out of range [0, {len(matrix[0])-1}]")

    return matrix[row][col]


def set_element(matrix: Matrix, row: int, col: int, value: MatrixElement) -> None:
    """
    Sets an element in the matrix at the specified position.

    Args:
        matrix: Input matrix (modified in place)
        row: Row index (0-based)
        col: Column index (0-based)
        value: New value to set (must be int or float)

    Raises:
        TypeError: If matrix is invalid, indices are not integers,
                  or value is not a number
        ValueError: If matrix is empty
        IndexError: If row or col index is out of range

    Examples:
        >>> matrix = [[1, 2], [3, 4]]
        >>> set_element(matrix, 0, 1, 10)
        >>> matrix
        [[1, 10], [3, 4]]
    """
    # Validate matrix
    is_valid_matrix(matrix, allow_empty=False)

    # Validate index types
    if not isinstance(row, int):
        raise TypeError(f"Row index must be integer, got {type(row).__name__}")
    if not isinstance(col, int):
        raise TypeError(f"Column index must be integer, got {type(col).__name__}")

    # Validate value type
    if not isinstance(value, (int, float)):
        raise TypeError(f"Value must be int or float, got {type(value).__name__}")

    # Validate indices range
    if row < 0 or row >= len(matrix):
        raise IndexError(f"Row index {row} out of range [0, {len(matrix)-1}]")
    if col < 0 or col >= len(matrix[0]):
        raise IndexError(f"Column index {col} out of range [0, {len(matrix[0])-1}]")

    matrix[row][col] = value


def get_rows(matrix: Matrix) -> int:
    """
    Returns the number of rows in the matrix.

    Args:
        matrix: Input matrix

    Returns:
        int: Number of rows

    Raises:
        TypeError: If matrix is invalid
        ValueError: If matrix is empty

    Examples:
        >>> matrix = [[1, 2], [3, 4], [5, 6]]
        >>> get_rows(matrix)
        3
    """
    is_valid_matrix(matrix, allow_empty=False)
    return len(matrix)


def get_cols(matrix: Matrix) -> int:
    """
    Returns the number of columns in the matrix.

    Args:
        matrix: Input matrix

    Returns:
        int: Number of columns

    Raises:
        TypeError: If matrix is invalid
        ValueError: If matrix is empty

    Examples:
        >>> matrix = [[1, 2, 3], [4, 5, 6]]
        >>> get_cols(matrix)
        3
    """
    is_valid_matrix(matrix, allow_empty=False)
    return len(matrix[0])


def matrix_to_list(matrix: Matrix, deep_copy: bool = False) -> Matrix:
    """
    Creates a copy of the matrix.

    Args:
        matrix: Input matrix to copy
        deep_copy: If True, creates a deep copy using copy.deepcopy()
                  If False, creates a shallow copy using list comprehension

    Returns:
        Matrix: A copy of the original matrix

    Raises:
        TypeError: If matrix is not a list or rows are not lists
        ValueError: If matrix is empty

    Examples:
        >>> original = [[1, 2], [3, 4]]
        >>> copy = matrix_to_list(original)
        >>> copy[0][0] = 999
        >>> original[0][0]  # Original unchanged
        1

        >>> # With deep copy for nested mutable objects
        >>> nested = [[1, [2, 3]], [4, [5, 6]]]
        >>> copy = matrix_to_list(nested, deep_copy=True)
    """
    # Validate matrix (allow empty? No, but if empty, we return empty list)
    if not isinstance(matrix, list):
        raise TypeError(f"Matrix must be a list, got {type(matrix).__name__}")

    if not matrix:
        raise ValueError("Matrix cannot be empty")

    # Validate structure (rows are lists, same lengths) but skip element type checks
    # because we just need structure for copying.
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("All rows must be lists")
    cols = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError(
                f"All rows must have the same length. "
                f"Row 0 has {cols} columns, row {i} has {len(row)} columns"
            )

    if deep_copy:
        return copy.deepcopy(matrix)
    else:
        return [row[:] for row in matrix]


def print_matrix(matrix: Matrix) -> None:
    """
    Prints the matrix in a readable format row by row.

    Args:
        matrix: Input matrix to print

    Raises:
        TypeError: If matrix is invalid
        ValueError: If matrix is empty

    Examples:
        >>> matrix = [[1, 2, 3], [4, 5, 6]]
        >>> print_matrix(matrix)
        [1, 2, 3]
        [4, 5, 6]
    """
    is_valid_matrix(matrix, allow_empty=False)

    for row in matrix:
        formatted_row = []
        for element in row:
            if isinstance(element, float):
                # Format floats to 2 decimal places if they have decimals
                if element.is_integer():
                    formatted_row.append(str(int(element)))
                else:
                    formatted_row.append(f"{element:.2f}")
            else:
                formatted_row.append(str(element))

        print(f"[{', '.join(formatted_row)}]")


def is_square_matrix(matrix: Matrix) -> bool:
    """
    Checks if the matrix is square (number of rows equals number of columns).

    Args:
        matrix: Input matrix

    Returns:
        bool: True if the matrix is square, False otherwise

    Raises:
        TypeError: If matrix is invalid
        ValueError: If matrix is empty

    Examples:
        >>> is_square_matrix([[1, 2], [3, 4]])
        True
        >>> is_square_matrix([[1, 2, 3], [4, 5, 6]])
        False
    """
    is_valid_matrix(matrix, allow_empty=False)
    return len(matrix) == len(matrix[0])