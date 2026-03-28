"""
Test module for arithmetic matrix operations.

Comprehensive tests using pytest parametrization, covering:
- element-wise addition/subtraction
- scalar multiplication
- matrix multiplication
- transpose
- matrix power (including exponentiation by squaring)
- interaction between functions (e.g., transpose and multiplication)
- edge cases and error handling
"""

import math
import pytest

from matrix_operations.arithmetic import (
    add_matrices,
    subtract_matrices,
    multiply_by_scalar,
    multiply_matrices,
    transpose,
    matrix_power,
)
from matrix_operations.basic_operations import matrix_to_list, is_square_matrix
from matrix_operations.creation import create_identity_matrix


# ---------- Helper fixtures ----------
@pytest.fixture
def matrix_2x2():
    return [[1, 2], [3, 4]]


@pytest.fixture
def matrix_2x2_alt():
    return [[5, 6], [7, 8]]


@pytest.fixture
def matrix_3x3():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


# ---------- Test add_matrices ----------
class TestAddMatrices:
    @pytest.mark.parametrize(
        "m1, m2, expected",
        [
            ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[6, 8], [10, 12]]),
            ([[1.5, 2.5], [3.5, 4.5]], [[0.5, 1.5], [2.5, 3.5]], [[2.0, 4.0], [6.0, 8.0]]),
            ([[5]], [[3]], [[8]]),
            ([[1, 2, 3]], [[4, 5, 6]], [[5, 7, 9]]),
            ([[1], [2], [3]], [[4], [5], [6]], [[5], [7], [9]]),
        ],
    )
    def test_add_positive(self, m1, m2, expected):
        result = add_matrices(m1, m2)
        assert result == expected

    @pytest.mark.parametrize(
        "m1, m2",
        [
            ([[1, 2], [3, 4]], [[5, 6]]),                      # different rows
            ([[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]),      # different cols
            ([[1, 2], [3]], [[5, 6], [7, 8]]),                # invalid matrix
        ],
    )
    def test_add_invalid_dimensions(self, m1, m2):
        with pytest.raises((ValueError, TypeError)):
            add_matrices(m1, m2)


# ---------- Test subtract_matrices ----------
class TestSubtractMatrices:
    @pytest.mark.parametrize(
        "m1, m2, expected",
        [
            ([[5, 6], [7, 8]], [[1, 2], [3, 4]], [[4, 4], [4, 4]]),
            ([[2.5, 3.5], [4.5, 5.5]], [[0.5, 1.5], [2.5, 3.5]], [[2.0, 2.0], [2.0, 2.0]]),
            ([[8]], [[3]], [[5]]),
            ([[5, 7, 9]], [[1, 2, 3]], [[4, 5, 6]]),
            ([[5], [7], [9]], [[1], [2], [3]], [[4], [5], [6]]),
        ],
    )
    def test_subtract_positive(self, m1, m2, expected):
        result = subtract_matrices(m1, m2)
        assert result == expected

    @pytest.mark.parametrize(
        "m1, m2",
        [
            ([[1, 2], [3, 4]], [[5, 6]]),
            ([[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]),
        ],
    )
    def test_subtract_invalid_dimensions(self, m1, m2):
        with pytest.raises(ValueError):
            subtract_matrices(m1, m2)


# ---------- Test multiply_by_scalar ----------
class TestMultiplyByScalar:
    @pytest.mark.parametrize(
        "matrix, scalar, expected",
        [
            ([[1, 2], [3, 4]], 2, [[2, 4], [6, 8]]),
            ([[1, 2], [3, 4]], 2.5, [[2.5, 5.0], [7.5, 10.0]]),
            ([[1, 2], [3, 4]], 0, [[0, 0], [0, 0]]),
            ([[1, 2], [3, 4]], -1, [[-1, -2], [-3, -4]]),
            ([[5]], 3, [[15]]),
            ([[1, 2, 3, 4]], 2, [[2, 4, 6, 8]]),
            ([[1], [2], [3], [4]], 2, [[2], [4], [6], [8]]),
        ],
    )
    def test_scalar_positive(self, matrix, scalar, expected):
        result = multiply_by_scalar(matrix, scalar)
        assert result == expected

    def test_scalar_invalid_type(self):
        with pytest.raises(TypeError, match="Scalar must be int or float"):
            multiply_by_scalar([[1, 2], [3, 4]], "2")

    def test_scalar_empty_matrix(self):
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            multiply_by_scalar([], 2)


# ---------- Test multiply_matrices ----------
class TestMultiplyMatrices:
    @pytest.mark.parametrize(
        "m1, m2, expected",
        [
            ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[19, 22], [43, 50]]),
            ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]),
            ([[1, 2, 3]], [[4], [5], [6]], [[32]]),
            ([[5]], [[3]], [[15]]),
            ([[1, 2], [3, 4], [5, 6]], [[7, 8, 9], [10, 11, 12]], [[27, 30, 33], [61, 68, 75], [95, 106, 117]]),
            ([[1.5, 2.5], [3.5, 4.5]], [[0.5, 1.5], [2.5, 3.5]], [[1.5*0.5+2.5*2.5, 1.5*1.5+2.5*3.5], [3.5*0.5+4.5*2.5, 3.5*1.5+4.5*3.5]]),
        ],
    )
    def test_multiply_positive(self, m1, m2, expected):
        result = multiply_matrices(m1, m2)
        assert result == expected

    def test_multiply_incompatible_dimensions(self):
        m1 = [[1, 2, 3], [4, 5, 6]]
        m2 = [[7, 8], [9, 10]]
        with pytest.raises(ValueError, match="number of columns of first matrix.*must equal"):
            multiply_matrices(m1, m2)

    def test_multiply_identity(self, matrix_3x3):
        identity = create_identity_matrix(3)
        result = multiply_matrices(matrix_3x3, identity)
        assert result == matrix_3x3

    def test_multiply_with_invalid_matrix(self):
        with pytest.raises(ValueError):
            multiply_matrices([[1, 2], [3, 4]], [[5, 6], [7]])


# ---------- Test transpose ----------
class TestTranspose:
    @pytest.mark.parametrize(
        "matrix, expected",
        [
            ([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]),
            ([[1, 2], [3, 4], [5, 6]], [[1, 3, 5], [2, 4, 6]]),
            ([[1, 2, 3]], [[1], [2], [3]]),
            ([[1], [2], [3]], [[1, 2, 3]]),
            ([[42]], [[42]]),
        ],
    )
    def test_transpose_positive(self, matrix, expected):
        result = transpose(matrix)
        assert result == expected

    def test_transpose_twice(self, matrix_2x2):
        t = transpose(matrix_2x2)
        tt = transpose(t)
        assert tt == matrix_2x2

    def test_transpose_empty(self):
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            transpose([])


# ---------- Test matrix_power ----------
class TestMatrixPower:
    @pytest.mark.parametrize(
        "matrix, exponent, expected",
        [
            ([[1, 2], [3, 4]], 0, [[1, 0], [0, 1]]),
            ([[1, 2], [3, 4]], 1, [[1, 2], [3, 4]]),
            ([[1, 2], [3, 4]], 2, [[7, 10], [15, 22]]),
            ([[1, 2], [3, 4]], 3, [[37, 54], [81, 118]]),
            ([[2, 0], [0, 3]], 3, [[8, 0], [0, 27]]),
            ([[5]], 4, [[625]]),
            ([[1, 1], [0, 1]], 1000, [[1, 1000], [0, 1]]),
        ],
    )
    def test_power_positive(self, matrix, exponent, expected):
        result = matrix_power(matrix, exponent)
        assert result == expected

    def test_power_returns_copy(self, matrix_2x2):
        result = matrix_power(matrix_2x2, 1)
        assert result == matrix_2x2
        assert result is not matrix_2x2  # Should be a copy

    def test_power_non_square(self):
        with pytest.raises(ValueError, match="matrix must be square"):
            matrix_power([[1, 2, 3], [4, 5, 6]], 2)

    def test_power_negative_exponent(self, matrix_2x2):
        with pytest.raises(ValueError, match="Exponent must be non-negative"):
            matrix_power(matrix_2x2, -1)

    def test_power_non_integer_exponent(self, matrix_2x2):
        with pytest.raises(TypeError, match="Exponent must be integer"):
            matrix_power(matrix_2x2, 2.5)

    def test_power_large_exponent_optimized(self):
        # Test that exponentiation by squaring works correctly for large exponent
        m = [[1, 1], [1, 0]]  # Fibonacci matrix
        result = matrix_power(m, 10)
        # F(11)=89, F(10)=55
        expected = [[89, 55], [55, 34]]
        assert result == expected


# ---------- Interaction tests ----------
class TestArithmeticInteractions:
    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 2], [3, 4]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[5]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ],
    )
    def test_transpose_twice_equals_original(self, matrix):
        assert transpose(transpose(matrix)) == matrix

    def test_add_then_transpose(self, matrix_2x2, matrix_2x2_alt):
        sum_m = add_matrices(matrix_2x2, matrix_2x2_alt)
        sum_t = transpose(sum_m)
        expected = add_matrices(transpose(matrix_2x2), transpose(matrix_2x2_alt))
        assert sum_t == expected

    def test_scalar_mult_then_transpose(self, matrix_2x2):
        scalar = 3
        result = transpose(multiply_by_scalar(matrix_2x2, scalar))
        expected = multiply_by_scalar(transpose(matrix_2x2), scalar)
        assert result == expected

    def test_matrix_mult_then_transpose(self, matrix_2x2, matrix_2x2_alt):
        prod = multiply_matrices(matrix_2x2, matrix_2x2_alt)
        prod_t = transpose(prod)
        expected = multiply_matrices(transpose(matrix_2x2_alt), transpose(matrix_2x2))
        assert prod_t == expected

    def test_copy_and_set_independence(self, matrix_2x2):
        from matrix_operations.basic_operations import set_element

        copy = matrix_to_list(matrix_2x2)
        set_element(copy, 0, 0, 999)
        assert copy[0][0] == 999
        assert matrix_2x2[0][0] == 1

    def test_power_via_repeated_multiplication(self):
        m = [[1, 2], [3, 4]]
        power3 = matrix_power(m, 3)
        m2 = multiply_matrices(m, m)
        m3 = multiply_matrices(m2, m)
        assert power3 == m3

    def test_zero_matrix_power(self):
        zero = [[0, 0], [0, 0]]
        result = matrix_power(zero, 2)
        expected = [[0, 0], [0, 0]]
        assert result == expected

    def test_identity_matrix_power(self):
        identity = create_identity_matrix(3)
        result = matrix_power(identity, 5)
        assert result == identity


# ---------- Edge cases and validation ----------
class TestArithmeticEdgeCases:
    def test_add_large_float(self):
        m1 = [[1e100, 2e100], [3e100, 4e100]]
        m2 = [[1e100, 2e100], [3e100, 4e100]]
        result = add_matrices(m1, m2)
        assert math.isfinite(result[0][0])
        assert result[0][0] == 2e100

    def test_multiply_by_scalar_large(self):
        m = [[1e100, 2e100], [3e100, 4e100]]
        result = multiply_by_scalar(m, 2)
        assert result[0][0] == 2e100

    def test_matrix_power_with_one_identity(self):
        m = [[1, 0], [0, 1]]
        result = matrix_power(m, 100)
        assert result == m

    def test_add_matrices_preserve_original(self, matrix_2x2, matrix_2x2_alt):
        orig1 = matrix_to_list(matrix_2x2)
        orig2 = matrix_to_list(matrix_2x2_alt)
        _ = add_matrices(matrix_2x2, matrix_2x2_alt)
        assert matrix_2x2 == orig1
        assert matrix_2x2_alt == orig2