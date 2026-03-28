"""
Usage examples for the matrix library.

This file demonstrates how to use the main functions of the library.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matrix_operations import *


def example_creation():
    """Examples of creating matrices."""
    print("=== Creation examples ===\n")

    # Zero matrix
    zero = create_zero_matrix(2, 3)
    print("Zero matrix 2x3:")
    print_matrix(zero)
    print()

    # Identity matrix
    identity = create_identity_matrix(3)
    print("Identity matrix 3x3:")
    print_matrix(identity)
    print()

    # Random matrix (integers 0-10)
    random_mat = create_random_matrix(3, 3, 0, 10)
    print("Random matrix 3x3 (0..10):")
    print_matrix(random_mat)
    print()

    # From list
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mat = create_matrix_from_list(data)
    print("Matrix from list:")
    print_matrix(mat)
    print()

    # Copy from list
    original = [[1, 2], [3, 4]]
    copy = create_matrix_from_list_copy(original)
    print("Original:")
    print_matrix(original)
    print("Copy:")
    print_matrix(copy)
    print("Modifying copy[0][0] = 99...")
    copy[0][0] = 99
    print("Original (unchanged):")
    print_matrix(original)
    print("Copy (modified):")
    print_matrix(copy)
    print()


def example_basic_operations():
    """Examples of basic matrix operations."""
    print("=== Basic operations examples ===\n")

    mat = create_matrix_from_list([[1, 2, 3], [4, 5, 6]])
    print("Original matrix:")
    print_matrix(mat)
    print()

    # Get dimensions
    rows = get_rows(mat)
    cols = get_cols(mat)
    print(f"Rows: {rows}, Columns: {cols}")

    # Get element
    elem = get_element(mat, 0, 1)
    print(f"Element at (0,1): {elem}")

    # Set element
    set_element(mat, 1, 2, 99)
    print("After setting (1,2) to 99:")
    print_matrix(mat)

    # Copy
    mat_copy = matrix_to_list(mat)
    print("Copy of matrix:")
    print_matrix(mat_copy)

    # Check if square
    print(f"Is matrix square? {is_square_matrix(mat)}")
    print()


def example_arithmetic():
    """Examples of arithmetic operations."""
    print("=== Arithmetic examples ===\n")

    A = create_matrix_from_list([[1, 2], [3, 4]])
    B = create_matrix_from_list([[5, 6], [7, 8]])

    print("Matrix A:")
    print_matrix(A)
    print("Matrix B:")
    print_matrix(B)
    print()

    # Addition
    C = add_matrices(A, B)
    print("A + B:")
    print_matrix(C)
    print()

    # Subtraction
    D = subtract_matrices(A, B)
    print("A - B:")
    print_matrix(D)
    print()

    # Scalar multiplication
    E = multiply_by_scalar(A, 2)
    print("2 * A:")
    print_matrix(E)
    print()

    # Matrix multiplication
    F = multiply_matrices(A, B)
    print("A * B:")
    print_matrix(F)
    print()

    # Transpose
    G = transpose(A)
    print("Transpose of A:")
    print_matrix(G)
    print()

    # Matrix power
    H = matrix_power(A, 3)
    print("A^3:")
    print_matrix(H)
    print()


def example_advanced():
    """Examples of advanced operations."""
    print("=== Advanced examples ===\n")

    # Square matrix
    A = create_matrix_from_list([[1, 2], [2, 3]])
    print("Matrix A:")
    print_matrix(A)
    print()

    # is_square (safe)
    print(f"Is A square? {is_square(A)}")

    # Trace
    print(f"Trace of A: {trace(A)}")

    # Symmetric check
    print(f"Is A symmetric? {is_symmetric(A)}")

    B = create_matrix_from_list([[1, 2], [3, 4]])
    print("\nMatrix B:")
    print_matrix(B)
    print(f"Is B symmetric? {is_symmetric(B)}")

    # Trace of transpose
    print(f"Trace(transpose(B)) = {trace(transpose(B))} (same as trace(B) = {trace(B)})")
    print()


def example_interaction():
    """Examples showing interaction between different operations."""
    print("=== Interaction examples ===\n")

    A = create_matrix_from_list([[1, 2], [3, 4]])
    print("Matrix A:")
    print_matrix(A)
    print()

    # Copy and modify
    A_copy = matrix_to_list(A)
    set_element(A_copy, 0, 0, 999)
    print("After copying and setting element (0,0) to 999 in copy:")
    print("Original A (unchanged):")
    print_matrix(A)
    print("Modified copy:")
    print_matrix(A_copy)
    print()

    # Trace and transpose
    print(f"trace(A) = {trace(A)}")
    print(f"trace(transpose(A)) = {trace(transpose(A))}")
    print("They are equal!")
    print()


if __name__ == "__main__":
    example_creation()
    example_basic_operations()
    example_arithmetic()
    example_advanced()
    example_interaction()