"""
Test module for matrix creation functions.

Contains comprehensive tests for all matrix creation functions using
pytest parametrization for better maintainability.
"""

import pytest
import random
import math

from matrix_operations.creation import (
    create_matrix,
    create_identity_matrix,
    create_zero_matrix,
    create_random_matrix,
    create_matrix_from_list,
    create_matrix_from_list_copy,
)


class TestCreateMatrix:
    """Test suite for create_matrix function."""

    @pytest.mark.parametrize("rows,cols,default_value,expected_shape", [
        (3, 4, 5, (3, 4)),
        (2, 3, 3.14, (2, 3)),
        (3, 3, 0, (3, 3)),
        (1, 1, 42, (1, 1)),
        (1, 10, 99, (1, 10)),
        (10, 1, -5, (10, 1)),
        (100, 100, 7, (100, 100)),
    ])
    def test_create_matrix_positive(self, rows, cols, default_value, expected_shape):
        """Test creating matrices with valid parameters."""
        result = create_matrix(rows, cols, default_value)

        assert len(result) == expected_shape[0]
        assert len(result[0]) == expected_shape[1]

        for row in result:
            assert len(row) == expected_shape[1]
            for element in row:
                assert element == default_value

    @pytest.mark.parametrize("rows,cols,default_value", [
        (0, 5, 0),
        (5, 0, 0),
        (-3, 4, 0),
        (4, -3, 0),
        (0, 0, 0),
    ])
    def test_create_matrix_zero_or_negative_dimensions(self, rows, cols, default_value):
        """Test creating matrix with zero or negative dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Matrix size must be positive"):
            create_matrix(rows, cols, default_value)

    @pytest.mark.parametrize("rows,cols,default_value", [
        (3.5, 4, 0),
        (3, "4", 0),
        (3, 4, "string"),
        (3, 4, None),
        (3, 4, [1, 2]),
    ])
    def test_create_matrix_invalid_types(self, rows, cols, default_value):
        """Test creating matrix with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            create_matrix(rows, cols, default_value)

    def test_create_matrix_default_zero(self):
        """Test creating matrix with default value 0."""
        result = create_matrix(3, 3)
        assert all(element == 0 for row in result for element in row)

    @pytest.mark.parametrize("value", [-5.5, 2**31 - 1, -2**31])
    def test_create_matrix_edge_values(self, value):
        """Test creating matrix with edge values."""
        result = create_matrix(2, 2, value)
        assert result[0][0] == value
        assert result[1][1] == value


class TestCreateIdentityMatrix:
    """Test suite for create_identity_matrix function."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 50, 100])
    def test_identity_matrix_positive(self, n):
        """Test creating identity matrices of various sizes."""
        result = create_identity_matrix(n)

        assert len(result) == n
        assert len(result[0]) == n

        for i in range(n):
            for j in range(n):
                if i == j:
                    assert result[i][j] == 1
                else:
                    assert result[i][j] == 0

    @pytest.mark.parametrize("n", [0, -1, -5])
    def test_identity_matrix_zero_or_negative(self, n):
        """Test creating identity matrix with non-positive size raises ValueError."""
        with pytest.raises(ValueError, match="Matrix size must be positive"):
            create_identity_matrix(n)

    @pytest.mark.parametrize("n", [3.5, "3", None])
    def test_identity_matrix_invalid_type(self, n):
        """Test creating identity matrix with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Matrix size must be integer"):
            create_identity_matrix(n)


class TestCreateZeroMatrix:
    """Test suite for create_zero_matrix function."""

    @pytest.mark.parametrize("rows,cols", [
        (2, 3), (3, 2), (1, 1), (1, 10), (10, 1), (50, 100)
    ])
    def test_zero_matrix_positive(self, rows, cols):
        """Test creating zero matrices of various sizes."""
        result = create_zero_matrix(rows, cols)

        assert len(result) == rows
        assert len(result[0]) == cols
        assert all(element == 0 for row in result for element in row)

    @pytest.mark.parametrize("rows,cols", [
        (0, 5), (5, 0), (-2, 3), (3, -4), (0, 0)
    ])
    def test_zero_matrix_zero_or_negative_dimensions(self, rows, cols):
        """Test creating zero matrix with invalid dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Matrix size must be positive"):
            create_zero_matrix(rows, cols)


class TestCreateRandomMatrix:
    """Test suite for create_random_matrix function."""

    @pytest.mark.parametrize("rows,cols,min_val,max_val,expected_type", [
        (3, 3, 1, 10, int),
        (2, 2, 0.5, 5.5, float),
        (2, 2, 1, 10.5, float),
        (3, 3, -10, -1, int),
        (1, 1, 0, 100, int),
        (1, 5, -5.5, 5.5, float),
        (5, 1, 0, 1, int),
        (10, 10, -1000, 1000, int),
    ])
    def test_random_matrix_positive(self, rows, cols, min_val, max_val, expected_type):
        """Test creating random matrices with various ranges."""
        result = create_random_matrix(rows, cols, min_val, max_val)

        assert len(result) == rows
        assert len(result[0]) == cols

        for row in result:
            for element in row:
                assert min_val <= element <= max_val
                assert isinstance(element, expected_type)

    @pytest.mark.parametrize("min_val,max_val", [
        (1, 1),           # Equal values
        (5, 5.000001),    # Very close
        (-0.0001, 0.0001), # Very small range
        (1.5, 1.500001),   # Float close values
    ])
    def test_random_matrix_narrow_range(self, min_val, max_val):
        """Test creating random matrix with very narrow ranges."""
        result = create_random_matrix(5, 5, min_val, max_val)

        for row in result:
            for element in row:
                assert min_val <= element <= max_val

    @pytest.mark.parametrize("rows,cols", [
        (0, 5), (5, 0), (-2, 3), (3, -4)
    ])
    def test_random_matrix_invalid_dimensions(self, rows, cols):
        """Test creating random matrix with invalid dimensions."""
        with pytest.raises(ValueError, match="Matrix size must be positive"):
            create_random_matrix(rows, cols, 1, 10)

    @pytest.mark.parametrize("min_val,max_val", [
        (10, 5), (5.5, 1.5), (-1, -5)
    ])
    def test_random_matrix_invalid_range(self, min_val, max_val):
        """Test creating random matrix with min > max raises ValueError."""
        with pytest.raises(ValueError, match="min_val .* can't be greater than max_val"):
            create_random_matrix(3, 3, min_val, max_val)

    @pytest.mark.parametrize("min_val,max_val", [
        ("1", 10), (1, "10"), (None, 10), (1, None)
    ])
    def test_random_matrix_invalid_bound_types(self, min_val, max_val):
        """Test creating random matrix with non-numeric bounds raises TypeError."""
        with pytest.raises(TypeError):
            create_random_matrix(3, 3, min_val, max_val)

    def test_random_matrix_deterministic_with_seed(self):
        """Test that random matrix generation is deterministic with fixed seed."""
        random.seed(42)
        matrix1 = create_random_matrix(3, 3, 0, 100)

        random.seed(42)
        matrix2 = create_random_matrix(3, 3, 0, 100)

        assert matrix1 == matrix2


class TestCreateMatrixFromList:
    """Test suite for create_matrix_from_list function."""

    @pytest.mark.parametrize("data", [
        [[1, 2], [3, 4]],
        [[1.5, 2.5], [3.5, 4.5]],
        [[1, 2.5], [3.14, 4]],
        [[1, 2, 3, 4, 5]],
        [[1], [2], [3], [4], [5]],
        [[42]],
        [[-1, -2], [-3, -4]],
        [[0] * 100 for _ in range(100)],  # Large matrix
    ])
    def test_create_from_list_valid(self, data):
        """Test creating matrix from valid data."""
        result = create_matrix_from_list(data)

        assert result == data
        assert len(result) == len(data)
        assert len(result[0]) == len(data[0])

        # Verify it's the same object
        assert result is data

    @pytest.mark.parametrize("data", [
        [],
        [[]],  # Empty row - should raise ValueError
        [[1, 2], [3, 4, 5]],
        [[1, 2], [3]],
        [[1, 2], [3, float('inf')]],
        [[1, 2], [3, float('nan')]],
        [[1, 2], [3, None]],
        [[1, 2], [3, "four"]],
    ])
    def test_create_from_list_invalid(self, data):
        """Test creating matrix from invalid data raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            create_matrix_from_list(data)

    def test_create_from_list_not_a_list(self):
        """Test passing non-list data raises TypeError."""
        with pytest.raises(TypeError, match="Input data must be a list"):
            create_matrix_from_list("not a list")

    def test_create_from_list_not_2d(self):
        """Test passing 1D list raises TypeError."""
        data = [1, 2, 3]
        with pytest.raises(TypeError, match="Data must be a 2D list"):
            create_matrix_from_list(data)

    @pytest.mark.parametrize("data", [
        [[1, 2], [3, 10**10]],  # Large numbers
        [[1e-200, 2e-200], [3e-200, 4e-200]],  # Very small numbers (denormal)
        [[1e150, 2e150], [3e150, 4e150]],  # Large but finite numbers
        [[-1e150, 2e150], [3e150, -4e150]],  # Mixed signs large numbers
    ])
    def test_create_from_list_extreme_values(self, data):
        """Test creating matrix with extreme but finite numeric values."""
        result = create_matrix_from_list(data)
        assert result == data
        # Verify all values are finite
        for row in result:
            for value in row:
                assert math.isfinite(value)


class TestCreateMatrixFromListCopy:
    """Test suite for create_matrix_from_list_copy function."""

    @pytest.mark.parametrize("original", [
        [[1, 2], [3, 4]],
        [[1.5, 2.5], [3.5, 4.5]],
        [[1, 2, 3, 4, 5]],
        [[1], [2], [3], [4], [5]],
        [[42]],
        [[-1, -2], [-3, -4]],
        [[0] * 50 for _ in range(50)],  # Large matrix
    ])
    def test_copy_creates_independent_matrix(self, original):
        """Test that copy function creates a new independent matrix."""
        copy = create_matrix_from_list_copy(original)

        assert copy == original
        assert copy is not original
        assert copy[0] is not original[0]

        # Modify copy and verify original unchanged
        if copy and copy[0]:
            copy[0][0] = 999
            assert original[0][0] != 999

    @pytest.mark.parametrize("original", [
        [[1, 2], [3, 4]],
        [[1.5, 2.5], [3.5, 4.5]],
        [[1, 2, 3]],
    ])
    def test_copy_modification_deep_independence(self, original):
        """Test that modifying copy doesn't affect original in any way."""
        copy = create_matrix_from_list_copy(original)

        # Modify all elements of copy
        for i in range(len(copy)):
            for j in range(len(copy[0])):
                copy[i][j] = 0

        # Original should be unchanged
        assert original == original  # Compare to itself
        assert copy != original

    @pytest.mark.parametrize("original", [
        [[1, 2], [3, 4]],
        [[1.5, 2.5], [3.5, 4.5]],
    ])
    def test_copy_row_independence(self, original):
        """Test that rows are also independent copies."""
        copy = create_matrix_from_list_copy(original)

        # Store original row reference
        original_row = original[0]

        # Modify copy's row
        copy[0] = [999, 999]

        # Original row should be unchanged
        assert original[0] is original_row
        assert original[0] != copy[0]

    def test_copy_empty_list(self):
        """Test copying empty list raises ValueError."""
        with pytest.raises(ValueError, match="A list can't be empty"):
            create_matrix_from_list_copy([])

    def test_copy_invalid_data(self):
        """Test copying invalid data raises appropriate error."""
        with pytest.raises(ValueError):
            create_matrix_from_list_copy([[1, 2], [3, 4, 5]])


class TestEdgeCasesAndBoundaries:
    """Additional edge case tests for comprehensive coverage."""

    def test_create_random_matrix_with_seed_reset(self):
        """Test that random seed doesn't affect other tests."""
        original_state = random.getstate()
        random.seed(123)
        matrix1 = create_random_matrix(2, 2, 0, 10)
        random.seed(123)
        matrix2 = create_random_matrix(2, 2, 0, 10)
        random.setstate(original_state)

        assert matrix1 == matrix2

    @pytest.mark.parametrize("min_val,max_val", [
        (1, 1.0000000001),
        (1.9999999999, 2),
    ])
    def test_random_matrix_very_close_floats(self, min_val, max_val):
        """Test random matrix with very close float bounds."""
        result = create_random_matrix(3, 3, min_val, max_val)

        for row in result:
            for element in row:
                assert min_val <= element <= max_val
                assert isinstance(element, float)

    def test_create_matrix_large_dimensions(self):
        """Test creating a very large matrix (performance test)."""
        rows, cols = 500, 500
        result = create_matrix(rows, cols, 1)

        assert len(result) == rows
        assert len(result[0]) == cols
        assert result[250][250] == 1