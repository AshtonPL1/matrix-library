"""
Module for arithmetic operations on matrices.

Provides functions for matrix addition, subtraction, scalar multiplication,
matrix multiplication, transposition, and exponentiation.
"""

from typing import Callable

from .types import Matrix, MatrixElement
from .validation import is_valid_matrix
from .basic_operations import matrix_to_list


def _apply_elementwise_operation(
    matrix1: Matrix,
    matrix2: Matrix,
    operation: Callable[[MatrixElement, MatrixElement], MatrixElement],
    operation_name: str,
) -> Matrix:
    """
    Helper function to apply element-wise operations on two matrices.

    Args:
        matrix1: First matrix
        matrix2: Second matrix
        operation: Function that takes two elements and returns the result
        operation_name: Name of the operation for error messages

    Returns:
        Matrix: Result of applying the operation element-wise

    Raises:
        ValueError: If matrices have incompatible dimensions
        TypeError: If matrices are invalid
    """
    # Validate matrices (allow_empty=False)
    is_valid_matrix(matrix1, allow_empty=False)
    is_valid_matrix(matrix2, allow_empty=False)

    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    if rows1 != rows2:
        raise ValueError(
            f"Cannot {operation_name} matrices: "
            f"different number of rows ({rows1} vs {rows2})"
        )
    if cols1 != cols2:
        raise ValueError(
            f"Cannot {operation_name} matrices: "
            f"different number of columns ({cols1} vs {cols2})"
        )

    # Perform operation
    return [
        [operation(matrix1[i][j], matrix2[i][j]) for j in range(cols1)]
        for i in range(rows1)
    ]


def add_matrices(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Adds two matrices element-wise.

    Args:
        matrix1: First matrix
        matrix2: Second matrix

    Returns:
        Matrix: Element-wise sum of the matrices

    Raises:
        ValueError: If matrices have different dimensions or are invalid
        TypeError: If matrices contain non-numeric elements

    Examples:
        >>> add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[6, 8], [10, 12]]
        >>> add_matrices([[1.5, 2.5]], [[0.5, 1.5]])
        [[2.0, 4.0]]
    """
    return _apply_elementwise_operation(matrix1, matrix2, lambda x, y: x + y, "add")


def subtract_matrices(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Subtracts the second matrix from the first matrix element-wise.

    Args:
        matrix1: First matrix (minuend)
        matrix2: Second matrix (subtrahend)

    Returns:
        Matrix: Element-wise difference of the matrices (matrix1 - matrix2)

    Raises:
        ValueError: If matrices have different dimensions or are invalid
        TypeError: If matrices contain non-numeric elements

    Examples:
        >>> subtract_matrices([[5, 6], [7, 8]], [[1, 2], [3, 4]])
        [[4, 4], [4, 4]]
        >>> subtract_matrices([[3.5, 4.5]], [[1.5, 2.5]])
        [[2.0, 2.0]]
    """
    return _apply_elementwise_operation(matrix1, matrix2, lambda x, y: x - y, "subtract")


def multiply_by_scalar(matrix: Matrix, scalar: MatrixElement) -> Matrix:
    """
    Multiplies each element of the matrix by a scalar value.

    Args:
        matrix: Input matrix
        scalar: Scalar multiplier (must be int or float)

    Returns:
        Matrix: Matrix with each element multiplied by the scalar

    Raises:
        TypeError: If matrix is invalid or scalar is not a number
        ValueError: If matrix is empty

    Examples:
        >>> multiply_by_scalar([[1, 2], [3, 4]], 2)
        [[2, 4], [6, 8]]
        >>> multiply_by_scalar([[1.5, 2.5]], 0.5)
        [[0.75, 1.25]]
    """
    is_valid_matrix(matrix, allow_empty=False)

    if not isinstance(scalar, (int, float)):
        raise TypeError(f"Scalar must be int or float, got {type(scalar).__name__}")

    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]


def multiply_matrices(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    Multiplies two matrices using matrix multiplication.

    For matrices A (m x n) and B (n x p), the result is a matrix C (m x p)
    where C[i][j] = sum_{k=0}^{n-1} A[i][k] * B[k][j].

    Args:
        matrix1: First matrix (left operand)
        matrix2: Second matrix (right operand)

    Returns:
        Matrix: Product of the matrices (matrix1 @ matrix2)

    Raises:
        ValueError: If matrices have incompatible dimensions or are invalid
        TypeError: If matrices contain non-numeric elements

    Examples:
        >>> multiply_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
        >>> multiply_matrices([[1, 2, 3]], [[4], [5], [6]])
        [[32]]
    """
    is_valid_matrix(matrix1, allow_empty=False)
    is_valid_matrix(matrix2, allow_empty=False)

    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    if cols1 != rows2:
        raise ValueError(
            f"Cannot multiply matrices: "
            f"number of columns of first matrix ({cols1}) must equal "
            f"number of rows of second matrix ({rows2})"
        )

    # Perform matrix multiplication
    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            total = 0
            for k in range(cols1):
                total += matrix1[i][k] * matrix2[k][j]
            row.append(total)
        result.append(row)
    return result


def transpose(matrix: Matrix) -> Matrix:
    """
    Transposes a matrix (swaps rows and columns).

    For a matrix A of size m x n, the transpose is a matrix of size n x m
    where result[j][i] = A[i][j].

    Args:
        matrix: Input matrix

    Returns:
        Matrix: Transposed matrix

    Raises:
        TypeError: If matrix is invalid
        ValueError: If matrix is empty

    Examples:
        >>> transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> transpose([[1], [2], [3]])
        [[1, 2, 3]]
    """
    is_valid_matrix(matrix, allow_empty=False)

    rows, cols = len(matrix), len(matrix[0])
    # Build transposed matrix
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def matrix_power(matrix: Matrix, exponent: int) -> Matrix:
    """
    Raises a square matrix to the given power.

    Uses exponentiation by squaring for efficiency.

    Args:
        matrix: Square matrix to raise to power
        exponent: Non-negative integer exponent

    Returns:
        Matrix: Matrix raised to the given power

    Raises:
        ValueError: If matrix is not square or exponent is negative
        TypeError: If matrix is invalid or exponent is not integer

    Examples:
        >>> matrix_power([[1, 2], [3, 4]], 2)
        [[7, 10], [15, 22]]
        >>> matrix_power([[2, 0], [0, 2]], 3)
        [[8, 0], [0, 8]]
    """
    is_valid_matrix(matrix, allow_empty=False)

    if len(matrix) != len(matrix[0]):
        raise ValueError(
            f"Cannot raise to power: matrix must be square, "
            f"got {len(matrix)}x{len(matrix[0])}"
        )

    if not isinstance(exponent, int):
        raise TypeError(f"Exponent must be integer, got {type(exponent).__name__}")

    if exponent < 0:
        raise ValueError(f"Exponent must be non-negative, got {exponent}")

    n = len(matrix)

    # Special cases
    if exponent == 0:
        # Return identity matrix
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    if exponent == 1:
        return matrix_to_list(matrix)

    # Exponentiation by squaring
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = matrix_to_list(matrix)
    exp = exponent

    while exp > 0:
        if exp % 2 == 1:
            result = multiply_matrices(result, base)
        base = multiply_matrices(base, base)
        exp //= 2

    return result