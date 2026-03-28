"""
Type definitions for matrix library.

Defines common types used across all modules.
"""

from typing import List, Union

# Type for matrix elements - supports int and float
MatrixElement = Union[int, float]
Matrix = List[List[MatrixElement]]