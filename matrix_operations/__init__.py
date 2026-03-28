"""
Matrix Library - a simple library for matrix operations.

This package provides functions for creating matrices,
basic operations (access, modification), arithmetic operations
(addition, multiplication, transpose, power), and advanced checks
(square, trace, symmetric).
"""

# Import creation functions
from .creation import (
    create_matrix,
    create_identity_matrix,
    create_zero_matrix,
    create_random_matrix,
    create_matrix_from_list,
    create_matrix_from_list_copy,
)

# Import basic operations
from .basic_operations import (
    is_valid_matrix,
    get_element,
    set_element,
    get_rows,
    get_cols,
    matrix_to_list,
    print_matrix,
    is_square_matrix,
)

# Import arithmetic operations
from .arithmetic import (
    add_matrices,
    subtract_matrices,
    multiply_by_scalar,
    multiply_matrices,
    transpose,
    matrix_power,
)

# Import advanced operations
from .advanced import (
    is_square,
    trace,
    is_symmetric,
)

# Define what is exported when using "from matrix_operations import *"
__all__ = [
    # Creation
    'create_matrix',
    'create_identity_matrix',
    'create_zero_matrix',
    'create_random_matrix',
    'create_matrix_from_list',
    'create_matrix_from_list_copy',
    # Basic operations
    'is_valid_matrix',
    'get_element',
    'set_element',
    'get_rows',
    'get_cols',
    'matrix_to_list',
    'print_matrix',
    'is_square_matrix',
    # Arithmetic
    'add_matrices',
    'subtract_matrices',
    'multiply_by_scalar',
    'multiply_matrices',
    'transpose',
    'matrix_power',
    # Advanced
    'is_square',
    'trace',
    'is_symmetric',
]