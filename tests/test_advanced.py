"""
Test module for advanced matrix operations.

Comprehensive tests using pytest parametrization, covering:
- is_square (safe, returns bool)
- trace (square matrix sum of diagonal)
- is_symmetric (checks symmetry)
- Interaction between functions (e.g., trace of transpose equals trace)
- Edge cases and error handling
"""

import pytest

from matrix_operations.advanced import is_square, trace, is_symmetric
from matrix_operations.arithmetic import transpose
from matrix_operations.creation import create_identity_matrix


# ---------- Test is_square ----------
class TestIsSquare:
    @pytest.mark.parametrize(
        "matrix, expected",
        [
            ([[1, 2], [3, 4]], True),
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], True),
            ([[5]], True),
            ([[1, 2, 3], [4, 5, 6]], False),          # 2x3
            ([[1, 2], [3, 4], [5, 6]], False),        # 3x2
            ([], False),                               # empty list
            ([[1, 2], [3, 4, 5]], False),             # jagged
            ([[1, 2], [3, 4], 5], False),             # row not list
            ("not a matrix", False),
            (None, False),
            (42, False),
        ],
    )
    def test_is_square(self, matrix, expected):
        assert is_square(matrix) == expected


# ---------- Test trace ----------
class TestTrace:
    @pytest.mark.parametrize(
        "matrix, expected",
        [
            ([[1, 2], [3, 4]], 5),
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 15),
            ([[42]], 42),
            ([[1.5, 2.5], [3.5, 4.5]], 6.0),
            ([[-1, -2], [-3, -4]], -5),
            ([[0, 0], [0, 0]], 0),
            (create_identity_matrix(5), 5),
        ],
    )
    def test_trace_positive(self, matrix, expected):
        assert trace(matrix) == expected

    def test_trace_non_square(self):
        with pytest.raises(ValueError, match="Trace requires a square matrix"):
            trace([[1, 2, 3], [4, 5, 6]])

    def test_trace_empty(self):
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            trace([])

    def test_trace_invalid_type(self):
        with pytest.raises(TypeError, match="Matrix must be a list"):
            trace("not a matrix")

    def test_trace_jagged(self):
        with pytest.raises(ValueError, match="All rows must have the same length"):
            trace([[1, 2], [3, 4, 5]])

    def test_trace_with_inf(self):
        with pytest.raises(ValueError, match="finite numbers"):
            trace([[1, float('inf')], [3, 4]])

    def test_trace_with_nan(self):
        with pytest.raises(ValueError, match="finite numbers"):
            trace([[1, float('nan')], [3, 4]])


# ---------- Test is_symmetric ----------
class TestIsSymmetric:
    @pytest.mark.parametrize(
        "matrix, expected",
        [
            ([[1, 2], [2, 3]], True),
            ([[1, 2, 3], [2, 4, 5], [3, 5, 6]], True),
            ([[5]], True),
            ([[1, 2], [3, 4]], False),
            (create_identity_matrix(4), True),
            ([[0, 0], [0, 0]], True),
            ([[1, 2.5], [2.5, 3]], True),
            ([[1, 2.5], [3.5, 4]], False),
            ([[1, 2], [2, 3], [4, 5]], False),   # not square, will raise
        ],
    )
    def test_is_symmetric_positive(self, matrix, expected):
        # For non-square, it raises ValueError, so we handle separately
        if not is_square(matrix):
            with pytest.raises(ValueError):
                is_symmetric(matrix)
        else:
            assert is_symmetric(matrix) == expected

    def test_is_symmetric_non_square(self):
        with pytest.raises(ValueError, match="Symmetric check requires a square matrix"):
            is_symmetric([[1, 2, 3], [4, 5, 6]])

    def test_is_symmetric_empty(self):
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            is_symmetric([])

    def test_is_symmetric_with_inf(self):
        with pytest.raises(ValueError, match="finite numbers"):
            is_symmetric([[1, float('inf')], [float('inf'), 4]])

    def test_is_symmetric_jagged(self):
        with pytest.raises(ValueError, match="All rows must have the same length"):
            is_symmetric([[1, 2], [3, 4, 5]])


# ---------- Interaction tests ----------
class TestAdvancedInteractions:
    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 2], [3, 4]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[5]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ],
    )
    def test_trace_of_transpose_equals_trace(self, matrix):
        """Mathematical property: trace(transpose(A)) == trace(A)"""
        if not is_square(matrix):
            pytest.skip("Matrix must be square for trace")
        t = transpose(matrix)
        assert trace(t) == trace(matrix)

    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 2], [2, 3]],
            [[1, 2.5], [2.5, 3]],
            create_identity_matrix(5),
            [[0, 0], [0, 0]],
        ],
    )
    def test_symmetric_implies_transpose_equals_original(self, matrix):
        """If matrix is symmetric, transpose(matrix) == matrix"""
        assert is_symmetric(matrix) is True
        assert transpose(matrix) == matrix

    def test_trace_of_identity_matrix(self):
        n = 5
        identity = create_identity_matrix(n)
        assert trace(identity) == n

    def test_symmetric_after_transpose(self):
        """Any matrix A: transpose(transpose(A)) = A, but not necessarily symmetric."""
        m = [[1, 2], [3, 4]]
        t = transpose(m)
        assert is_symmetric(m) is False
        assert is_symmetric(t) is False
        # But transpose of transpose is original
        assert transpose(t) == m

    def test_trace_of_sum_equals_sum_of_traces(self):
        from matrix_operations.arithmetic import add_matrices
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        sum_matrix = add_matrices(A, B)
        assert trace(sum_matrix) == trace(A) + trace(B)

    def test_scalar_mult_trace(self):
        from matrix_operations.arithmetic import multiply_by_scalar
        A = [[1, 2], [3, 4]]
        scalar = 3
        scaled = multiply_by_scalar(A, scalar)
        assert trace(scaled) == scalar * trace(A)


# ---------- Edge cases ----------
class TestAdvancedEdgeCases:
    def test_is_square_with_empty_row(self):
        # is_square should return False for matrix with empty row
        matrix = [[], []]
        assert is_square(matrix) is False

    def test_trace_large_numbers(self):
        m = [[1e100, 2e100], [3e100, 4e100]]
        # trace = 1e100 + 4e100 = 5e100
        assert trace(m) == 5e100

    def test_trace_float_precision(self):
        m = [[0.1, 0.2], [0.3, 0.4]]
        # trace = 0.1 + 0.4 = 0.5
        assert abs(trace(m) - 0.5) < 1e-12

    def test_is_symmetric_with_float_comparison(self):
        # Should use exact equality; slight differences cause False
        m = [[1.0, 2.0], [2.0000000001, 3.0]]
        assert is_symmetric(m) is False

    def test_is_symmetric_with_float_exact(self):
        m = [[1.0, 2.0], [2.0, 3.0]]
        assert is_symmetric(m) is True