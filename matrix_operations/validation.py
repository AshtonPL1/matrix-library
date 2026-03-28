"""
Matrix validation module.

Provides a unified function to validate matrix structure and content.
"""

import math
from typing import Any

from .types import Matrix, MatrixElement


def is_valid_matrix(matrix: Any, allow_empty: bool = False) -> bool:
    """
    Validates that the input is a proper matrix.

    Checks:
    - Matrix is a list of lists
    - All rows have the same length
    - All elements are numbers (int or float) and finite (not inf, not nan)
    - Matrix is not empty (unless allow_empty is True)
    - Rows cannot be empty (each row must have at least one element)

    Args:
        matrix: Object to validate as a matrix
        allow_empty: If True, allows empty matrices (0 rows)

    Returns:
        bool: True if the matrix is valid

    Raises:
        TypeError: If matrix is not a list or rows are not lists
        ValueError: If matrix is empty (and allow_empty=False),
                   rows have different lengths, rows are empty,
                   or elements are not finite numbers
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError(f"Matrix must be a list, got {type(matrix).__name__}")

    # Check if matrix is empty
    if not matrix:
        if allow_empty:
            return True
        raise ValueError("Matrix cannot be empty")

    # Check if all rows are lists and not empty
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"Row {i} must be a list, got {type(row).__name__}")
        if not row and not allow_empty:
            raise ValueError(f"Row {i} cannot be empty")

    # Check if all rows have the same length
    cols_count = len(matrix[0])
    if cols_count == 0:
        raise ValueError("Matrix cannot have zero columns")

    for i, row in enumerate(matrix):
        if len(row) != cols_count:
            raise ValueError(
                f"All rows must have the same length. "
                f"Row 0 has {cols_count} columns, row {i} has {len(row)} columns"
            )

    # Check if all elements are finite numbers (int or float)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Matrix elements must be int or float, "
                    f"got {type(value).__name__} at position [{i}][{j}]"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"Matrix elements must be finite numbers, "
                    f"got {value} at position [{i}][{j}]"
                )

    return True