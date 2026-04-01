"""
Test module for basic matrix operations.

Contains comprehensive tests for get_element, set_element, get_rows, get_cols,
matrix_to_list, print_matrix, and is_square_matrix using pytest parametrization.
"""

import pytest
from matrix_operations.basic_operations import (
    get_element,
    set_element,
    get_rows,
    get_cols,
    matrix_to_list,
    print_matrix,
    is_square_matrix,
)


class TestGetElement:
    """Test suite for get_element function."""

    @pytest.mark.parametrize("matrix,row,col,expected", [
        ([[1, 2], [3, 4]], 0, 0, 1),
        ([[1, 2], [3, 4]], 0, 1, 2),
        ([[1, 2], [3, 4]], 1, 0, 3),
        ([[1, 2], [3, 4]], 1, 1, 4),
        ([[42]], 0, 0, 42),
        ([[1, 2, 3, 4, 5]], 0, 4, 5),
        ([[1], [2], [3], [4], [5]], 4, 0, 5),
        ([[-5, -10], [-15, -20]], 1, 1, -20),
        ([[1.5, 2.5], [3.5, 4.5]], 0, 1, 2.5),
    ])
    def test_get_element_positive(self, matrix, row, col, expected):
        """Test getting elements from valid matrices."""
        assert get_element(matrix, row, col) == expected

    @pytest.mark.parametrize("matrix,row,col", [
        ([[1, 2], [3, 4]], -1, 0),
        ([[1, 2], [3, 4]], 2, 0),
        ([[1, 2], [3, 4]], 0, -1),
        ([[1, 2], [3, 4]], 0, 2),
        ([[42]], 0, 1),
        ([[1, 2, 3]], 0, 3),
        ([[1], [2]], 2, 0),
    ])
    def test_get_element_index_error(self, matrix, row, col):
        """Test getting element with out-of-range indices raises IndexError."""
        with pytest.raises(IndexError):
            get_element(matrix, row, col)

    @pytest.mark.parametrize("matrix,row,col", [
        ([[1, 2], [3, 4]], "0", 0),
        ([[1, 2], [3, 4]], 0, "1"),
        ([[1, 2], [3, 4]], 3.5, 0),
        ([[1, 2], [3, 4]], 0, 3.5),
    ])
    def test_get_element_invalid_index_type(self, matrix, row, col):
        """Test getting element with non-integer indices raises TypeError."""
        with pytest.raises(TypeError, match="index must be integer"):
            get_element(matrix, row, col)

    def test_get_element_invalid_matrix(self):
        """Test getting element from invalid matrix raises error."""
        with pytest.raises(ValueError):
            get_element([[1, 2], [3]], 0, 0)


class TestSetElement:
    """Test suite for set_element function."""

    @pytest.mark.parametrize("initial,row,col,value,expected", [
        ([[1, 2], [3, 4]], 0, 0, 99, [[99, 2], [3, 4]]),
        ([[1, 2], [3, 4]], 0, 1, 99, [[1, 99], [3, 4]]),
        ([[1, 2], [3, 4]], 1, 0, 99, [[1, 2], [99, 4]]),
        ([[1, 2], [3, 4]], 1, 1, 99, [[1, 2], [3, 99]]),
        ([[42]], 0, 0, 99, [[99]]),
        ([[1, 2, 3]], 0, 2, 99, [[1, 2, 99]]),
        ([[1], [2], [3]], 2, 0, 99, [[1], [2], [99]]),
        ([[1.5, 2.5], [3.5, 4.5]], 0, 0, 3.14, [[3.14, 2.5], [3.5, 4.5]]),
    ])
    def test_set_element_positive(self, initial, row, col, value, expected):
        """Test setting elements in valid matrices."""
        set_element(initial, row, col, value)
        assert initial == expected

    @pytest.mark.parametrize("matrix,row,col,value", [
        ([[1, 2], [3, 4]], -1, 0, 99),
        ([[1, 2], [3, 4]], 2, 0, 99),
        ([[1, 2], [3, 4]], 0, -1, 99),
        ([[1, 2], [3, 4]], 0, 2, 99),
        ([[42]], 0, 1, 99),
    ])
    def test_set_element_index_error(self, matrix, row, col, value):
        """Test setting element with out-of-range indices raises IndexError."""
        with pytest.raises(IndexError):
            set_element(matrix, row, col, value)

    @pytest.mark.parametrize("matrix,row,col,value", [
        ([[1, 2], [3, 4]], "0", 0, 99),
        ([[1, 2], [3, 4]], 0, "1", 99),
        ([[1, 2], [3, 4]], 0, 0, "string"),
        ([[1, 2], [3, 4]], 0, 0, None),
    ])
    def test_set_element_invalid_types(self, matrix, row, col, value):
        """Test setting element with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            set_element(matrix, row, col, value)


class TestGetRows:
    """Test suite for get_rows function."""

    @pytest.mark.parametrize("matrix,expected", [
        ([[1, 2], [3, 4]], 2),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3),
        ([[1, 2, 3, 4, 5]], 1),
        ([[1], [2], [3], [4], [5]], 5),
        ([[0] * 100 for _ in range(50)], 50),
    ])
    def test_get_rows_positive(self, matrix, expected):
        """Test getting number of rows."""
        assert get_rows(matrix) == expected

    def test_get_rows_empty_matrix(self):
        """Test get_rows on empty matrix raises ValueError."""
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            get_rows([])


class TestGetCols:
    """Test suite for get_cols function."""

    @pytest.mark.parametrize("matrix,expected", [
        ([[1, 2], [3, 4]], 2),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3),
        ([[1, 2, 3, 4, 5]], 5),
        ([[1], [2], [3], [4], [5]], 1),
        ([[0] * 100 for _ in range(50)], 100),
    ])
    def test_get_cols_positive(self, matrix, expected):
        """Test getting number of columns."""
        assert get_cols(matrix) == expected

    def test_get_cols_empty_matrix(self):
        """Test get_cols on empty matrix raises ValueError."""
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            get_cols([])


class TestMatrixToList:
    """Test suite for matrix_to_list function."""

    @pytest.mark.parametrize("original", [
        [[1, 2], [3, 4]],
        [[1.5, 2.5], [3.5, 4.5]],
        [[1, 2, 3, 4, 5]],
        [[1], [2], [3], [4], [5]],
        [[42]],
        [[-1, -2], [-3, -4]],
        [[0] * 50 for _ in range(50)],  # Large matrix
    ])
    def test_copy_shallow_independence(self, original):
        """Test that shallow copy creates independent copy."""
        copy = matrix_to_list(original, deep_copy=False)

        assert copy == original
        assert copy is not original
        assert copy[0] is not original[0]

        # Modify copy and verify original unchanged
        if copy and copy[0]:
            copy[0][0] = 999
            assert original[0][0] != 999

    def test_deep_copy_nested(self):
        """Test deep copy for nested mutable objects."""
        original = [[1, [2, 3]], [4, [5, 6]]]
        copy = matrix_to_list(original, deep_copy=True)

        assert copy == original
        assert copy[0][1] is not original[0][1]

        # Modify nested list in copy
        copy[0][1][0] = 999
        assert original[0][1][0] == 2

    @pytest.mark.parametrize("original", [
        [[1, 2], [3, 4]],
        [[1.5, 2.5], [3.5, 4.5]],
    ])
    def test_copy_modification_deep_independence(self, original):
        """Test that modifying copy doesn't affect original."""
        copy = matrix_to_list(original)

        # Modify all elements of copy
        for i in range(len(copy)):
            for j in range(len(copy[0])):
                copy[i][j] = 0

        assert copy != original
        assert original != copy

    @pytest.mark.parametrize("invalid_matrix", [
        [],
        [[1, 2], [3]],
        [[1, 2], "not a list"],
        [1, 2, 3],
    ])
    def test_copy_invalid_matrix(self, invalid_matrix):
        """Test copying invalid matrix raises appropriate error."""
        with pytest.raises((TypeError, ValueError)):
            matrix_to_list(invalid_matrix)


class TestPrintMatrix:
    """Test suite for print_matrix function using capsys fixture."""

    @pytest.mark.parametrize("matrix,expected_lines", [
        ([[1, 2], [3, 4]], ["[1, 2]", "[3, 4]"]),
        ([[1.5, 2.5], [3.5, 4.5]], ["[1.50, 2.50]", "[3.50, 4.50]"]),
        ([[1.0, 2.0], [3.0, 4.0]], ["[1, 2]", "[3, 4]"]),
        ([[1, 2.5], [3.14, 4]], ["[1, 2.50]", "[3.14, 4]"]),
        ([[42]], ["[42]"]),
        ([[1, 2, 3, 4, 5]], ["[1, 2, 3, 4, 5]"]),
        ([[1], [2], [3], [4], [5]], ["[1]", "[2]", "[3]", "[4]", "[5]"]),
        ([[-1, -2], [-3, -4]], ["[-1, -2]", "[-3, -4]"]),
    ])
    def test_print_matrix_positive(self, matrix, expected_lines, capsys):
        """Test printing various matrices."""
        print_matrix(matrix)
        captured = capsys.readouterr()
        expected_output = "\n".join(expected_lines) + "\n"
        assert captured.out == expected_output

    def test_print_matrix_empty(self):
        """Test printing empty matrix raises ValueError."""
        with pytest.raises(ValueError, match="Matrix cannot be empty"):
            print_matrix([])


class TestIsSquareMatrix:
    """Test suite for is_square_matrix function."""

    @pytest.mark.parametrize("matrix,expected", [
        ([[1, 2], [3, 4]], True),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], True),
        ([[42]], True),
        ([[1, 2, 3], [4, 5, 6]], False),
        ([[1, 2], [3, 4], [5, 6]], False),
        ([[1, 2, 3, 4]], False),
        ([[1], [2], [3], [4]], False),
        ([[0] * 50 for _ in range(50)], True),
        ([[0] * 100 for _ in range(50)], False),
    ])
    def test_is_square_matrix(self, matrix, expected):
        """Test square matrix detection."""
        assert is_square_matrix(matrix) == expected

    def test_is_square_matrix_invalid(self):
        """Test is_square_matrix on invalid matrix raises error."""
        with pytest.raises(ValueError):
            is_square_matrix([[1, 2], [3]])


class TestInteraction:
    """Tests for interaction between functions."""

    def test_copy_and_set_element_independence(self):
        """Test that modifying a copy via set_element doesn't affect original."""
        original = [[1, 2], [3, 4]]
        copy = matrix_to_list(original)

        set_element(copy, 0, 0, 999)

        assert copy[0][0] == 999
        assert original[0][0] == 1

    def test_multiple_copies_independence(self):
        """Test that two copies are independent from each other."""
        original = [[1, 2], [3, 4]]
        copy1 = matrix_to_list(original)
        copy2 = matrix_to_list(original)

        set_element(copy1, 0, 0, 999)
        set_element(copy2, 1, 1, 888)

        assert copy1[0][0] == 999
        assert copy1[1][1] == 4
        assert copy2[0][0] == 1
        assert copy2[1][1] == 888
        assert original[0][0] == 1
        assert original[1][1] == 4

    def test_get_element_after_set_element(self):
        """Test that get_element reflects changes made by set_element."""
        matrix = [[1, 2], [3, 4]]
        set_element(matrix, 0, 1, 10)
        assert get_element(matrix, 0, 1) == 10

    def test_rows_and_cols_after_modification(self):
        """Test that get_rows and get_cols remain correct after modifications."""
        matrix = [[1, 2], [3, 4]]
        set_element(matrix, 0, 0, 99)
        assert get_rows(matrix) == 2
        assert get_cols(matrix) == 2