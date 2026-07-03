---

# Matrix Library

A lightweight pure‑Python library for matrix creation, manipulation, and property checking.  
Designed for educational use, with strict validation, clear error diagnostics, and full test coverage.

---

## Features

- **Creation**: zero, identity, constant, random (int/float), from list (with/without copy)  
- **Basic**: element access/modification, dimensions, copying (shallow/deep), formatted printing  
- **Arithmetic**: element‑wise addition/subtraction, scalar multiplication, matrix multiplication, transpose, power (non‑negative integer, exponentiation by squaring)  
- **Advanced**: safe square test, trace, symmetry check  
- **Types**: supports `int` and `float`, mixed types handled automatically  
- **Validation**: centralised checks for structure, numeric types, and finiteness

---

## Installation

```bash
git clone https://github.com/AshtonPL1/matrix-library
cd matrix-library
pip install -r requirements.txt
pytest tests/   # optional, verify installation
```

**Dependencies:** Python 3.8+, `pytest` (for tests only).

---

## Project Structure

```
matrix_operations/          # Main package
├── __init__.py             # Public API exports
├── types.py                # Type aliases: Matrix, MatrixElement
├── validation.py           # is_valid_matrix – core validator
├── creation.py             # Matrix factory functions
├── basic_operations.py     # Access, modification, dims, copy, print, square check
├── arithmetic.py           # Add, sub, scalar mul, matmul, transpose, power
└── advanced.py             # Safe is_square, trace, is_symmetric

tests/                      # Pytest suite
├── test_creation.py
├── test_basic_operations.py
├── test_arithmetic.py
└── test_advanced.py

examples/usage_examples.py  # Usage demonstrations
requirements.txt
README.md
LICENSE.txt
```

---

## API Reference

### Creation (`creation.py`)

| Function | Description |
|----------|-------------|
| `create_matrix(rows, cols, default_value=0)` | Matrix filled with `default_value`. |
| `create_identity_matrix(n)` | Identity matrix of size `n`. |
| `create_zero_matrix(rows, cols)` | All‑zero matrix. |
| `create_random_matrix(rows, cols, min_val, max_val)` | Random values in `[min, max]`; `int` if both bounds are int, else `float`. |
| `create_matrix_from_list(data)` | Returns input after validation (same object). |
| `create_matrix_from_list_copy(data)` | Returns a deep copy. |

All raise `TypeError` or `ValueError` on invalid input.

### Basic Operations (`basic_operations.py`)

| Function | Description |
|----------|-------------|
| `get_element(matrix, row, col)` | Access element; raises `IndexError`. |
| `set_element(matrix, row, col, value)` | In‑place modification. |
| `get_rows(matrix)` / `get_cols(matrix)` | Dimension queries. |
| `matrix_to_list(matrix, deep_copy=False)` | Copy; `deep_copy=True` uses `copy.deepcopy`. |
| `print_matrix(matrix)` | Formatted output (floats with 2 decimals). |
| `is_square_matrix(matrix)` | Square check; raises if invalid. |

### Arithmetic (`arithmetic.py`)

| Function | Description |
|----------|-------------|
| `add_matrices(A, B)` / `subtract_matrices(A, B)` | Element‑wise; requires same shape. |
| `multiply_by_scalar(matrix, scalar)` | Scalar multiplication. |
| `multiply_matrices(A, B)` | Matrix product; checks dimensions. |
| `transpose(matrix)` | Transpose. |
| `matrix_power(matrix, exponent)` | Fast exponentiation for non‑negative integer exponents. |

### Advanced (`advanced.py`)

| Function | Description |
|----------|-------------|
| `is_square(matrix)` | **Safe** – returns `False` for any invalid input (never raises). |
| `trace(matrix)` | Sum of diagonal; requires square. |
| `is_symmetric(matrix)` | Checks `A == A^T`; requires square. |

---

## Core Principles

- **Validation first** – Every public function validates input via `is_valid_matrix`, ensuring rectangular structure, numeric types, and finite values.  
- **Explicit errors** – Exceptions include context (e.g., mismatched dimensions, index bounds) to simplify debugging.  
- **Immutability by default** – Arithmetic functions return new matrices; in‑place mutation is available only via `set_element`.  
- **Copy safety** – `matrix_to_list` and `create_matrix_from_list_copy` provide independent copies, preventing accidental alias effects.

---

## Testing

The test suite uses `pytest` with extensive parametrisation, covering:

- All creation functions (including edge cases: zero/negative sizes, `inf`/`nan`, huge dimensions).  
- Element access, modification, copying, and printing.  
- Arithmetic operations and their mathematical properties (e.g., `trace(transpose(A)) == trace(A)`, `A * I == A`).  
- Advanced checks (safe `is_square`, `trace`, `is_symmetric`) and interaction between modules.

Run all tests:

```bash
pytest tests/
```

All tests must pass. The suite ensures high reliability even with non‑standard inputs.

---

## Usage Example

```python
from matrix_operations import *

A = create_matrix_from_list([[1, 2], [3, 4]])
B = create_zero_matrix(2, 2)

C = add_matrices(A, B)                 # [[1,2],[3,4]]
D = multiply_matrices(A, A)            # [[7,10],[15,22]]
E = matrix_power(A, 3)                 # [[37,54],[81,118]]
t = trace(A)                           # 5
print(is_symmetric(A))                 # False

print_matrix(E)
```

More examples are in `examples/usage_examples.py`.

---

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/AshtonPL1/matrix-library/blob/main/LICENSE.txt) for details.

## Contact

Author: **Borovoy Nikita**  
Email: nurmag00@bk.ru  
GitHub: [AshtonPL1](https://github.com/AshtonPL1)
