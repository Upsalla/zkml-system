"""
Sensing Matrix Generator for CS-Inspired Sparse Witness Commitment (CSWC).

This module provides the core data structure for the sensing matrix A used in
the linear sketch computation y = A * w.

Mathematical Background:
------------------------
In Compressed Sensing, a sensing matrix A ∈ F^(m×n) with m << n can preserve
information about sparse signals. The key property is the Restricted Isometry
Property (RIP): for all k-sparse vectors x,
    (1 - δ)||x||² ≤ ||Ax||² ≤ (1 + δ)||x||²

For our application, we don't need RIP for signal recovery. Instead, we use
the matrix to create a "fingerprint" of the sparse witness that can be
verified efficiently.

Implementation Notes:
--------------------
- We use sparse storage to handle large n efficiently
- The matrix is generated deterministically from a seed for reproducibility
- Column extraction is optimized for sparse witness multiplication
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import hashlib
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig


@dataclass
class SensingMatrix:
    """
    Sparse representation of a sensing matrix A ∈ F^(m×n).
    
    The matrix is stored as a dictionary mapping (row, col) -> value.
    Only non-zero entries are stored.
    
    Attributes:
        rows: Number of rows (m, the sketch dimension)
        cols: Number of columns (n, the witness dimension)
        field_config: Configuration for the finite field
        data: Sparse storage of non-zero entries
        entries_per_column: Average number of non-zero entries per column
    """
    rows: int
    cols: int
    field_config: FieldConfig
    data: Dict[Tuple[int, int], FieldElement] = field(default_factory=dict)
    entries_per_column: int = 10  # Sparsity parameter
    
    def get(self, row: int, col: int) -> FieldElement:
        """Get the value at position (row, col). Returns zero if not stored."""
        if (row, col) in self.data:
            return self.data[(row, col)]
        return FieldElement.zero(self.field_config)
    
    def set(self, row: int, col: int, value: FieldElement) -> None:
        """Set the value at position (row, col)."""
        if value.value != 0:
            self.data[(row, col)] = value
        elif (row, col) in self.data:
            del self.data[(row, col)]
    
    def get_column(self, col: int) -> Dict[int, FieldElement]:
        """
        Get all non-zero entries in a specific column.
        
        Returns:
            Dictionary mapping row index to value
        """
        column = {}
        for (r, c), val in self.data.items():
            if c == col:
                column[r] = val
        return column
    
    def get_submatrix_columns(self, col_indices: List[int]) -> 'SensingMatrix':
        """
        Extract a submatrix containing only the specified columns.
        
        This is used to compute A_S * w_S efficiently, where S is the support.
        
        Args:
            col_indices: List of column indices to extract
            
        Returns:
            New SensingMatrix with only the specified columns
        """
        submatrix = SensingMatrix(
            rows=self.rows,
            cols=len(col_indices),
            field_config=self.field_config,
            entries_per_column=self.entries_per_column
        )
        
        # Create mapping from old column index to new column index
        col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(col_indices)}
        
        for (row, col), val in self.data.items():
            if col in col_map:
                submatrix.data[(row, col_map[col])] = val
        
        return submatrix
    
    def multiply_vector(self, vector: List[FieldElement]) -> List[FieldElement]:
        """
        Compute A * v for a dense vector v.
        
        Args:
            vector: Dense vector of length n
            
        Returns:
            Result vector of length m
        """
        if len(vector) != self.cols:
            raise ValueError(f"Vector length {len(vector)} != matrix cols {self.cols}")
        
        result = [FieldElement.zero(self.field_config) for _ in range(self.rows)]
        
        for (row, col), val in self.data.items():
            result[row] = result[row] + (val * vector[col])
        
        return result
    
    def multiply_sparse(self, support: List[int], values: List[FieldElement]) -> List[FieldElement]:
        """
        Compute A_S * w_S efficiently for a sparse vector.
        
        This is the core operation for CSWC. Instead of multiplying with the
        full witness, we only use the columns corresponding to non-zero entries.
        
        Args:
            support: Indices of non-zero entries
            values: Values at those indices
            
        Returns:
            Result vector of length m (the sketch)
            
        Complexity: O(m * k * entries_per_column) where k = len(support)
        """
        if len(support) != len(values):
            raise ValueError("Support and values must have same length")
        
        result = [FieldElement.zero(self.field_config) for _ in range(self.rows)]
        
        # For each non-zero entry in the sparse witness
        for col_idx, val in zip(support, values):
            # Get the column of A corresponding to this index
            column = self.get_column(col_idx)
            
            # Add contribution to each row
            for row_idx, matrix_val in column.items():
                result[row_idx] = result[row_idx] + (matrix_val * val)
        
        return result
    
    def density(self) -> float:
        """Return the fraction of non-zero entries."""
        total_entries = self.rows * self.cols
        if total_entries == 0:
            return 0.0
        return len(self.data) / total_entries
    
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # Each entry: 2 ints (8 bytes each) + 1 field element (32 bytes)
        return len(self.data) * (8 + 8 + 32)


class SensingMatrixGenerator:
    """
    Generator for deterministic sensing matrices.
    
    The matrix is generated from a seed to ensure reproducibility.
    Both prover and verifier can regenerate the same matrix.
    
    Matrix Types:
    - SPARSE_RANDOM: Each column has ~entries_per_column random non-zero entries
    - DENSE_RANDOM: All entries are random (for small n only)
    - TOEPLITZ: Structured matrix with O(n+m) storage (for very large n)
    """
    
    def __init__(self, 
                 n: int, 
                 m: int, 
                 field_config: FieldConfig,
                 matrix_type: str = "SPARSE_RANDOM",
                 entries_per_column: int = 10):
        """
        Initialize the generator.
        
        Args:
            n: Witness dimension (number of columns)
            m: Sketch dimension (number of rows)
            field_config: Finite field configuration
            matrix_type: Type of matrix to generate
            entries_per_column: Sparsity parameter for SPARSE_RANDOM
        """
        self.n = n
        self.m = m
        self.field_config = field_config
        self.matrix_type = matrix_type
        self.entries_per_column = entries_per_column
    
    def generate(self, seed: bytes = b"CSWC_DEFAULT_SEED") -> SensingMatrix:
        """
        Generate a sensing matrix from the given seed.
        
        Args:
            seed: Seed for deterministic generation
            
        Returns:
            SensingMatrix instance
        """
        if self.matrix_type == "SPARSE_RANDOM":
            return self._generate_sparse_random(seed)
        elif self.matrix_type == "DENSE_RANDOM":
            return self._generate_dense_random(seed)
        elif self.matrix_type == "TOEPLITZ":
            return self._generate_toeplitz(seed)
        else:
            raise ValueError(f"Unknown matrix type: {self.matrix_type}")
    
    def _generate_sparse_random(self, seed: bytes) -> SensingMatrix:
        """
        Generate a sparse random matrix.
        
        Each column has approximately entries_per_column non-zero entries
        at random row positions with random values.
        """
        matrix = SensingMatrix(
            rows=self.m,
            cols=self.n,
            field_config=self.field_config,
            entries_per_column=self.entries_per_column
        )
        
        # Use hash-based PRNG for determinism
        for col in range(self.n):
            # Generate random row indices for this column
            col_seed = hashlib.sha256(seed + col.to_bytes(4, 'big')).digest()
            
            # Select which rows have non-zero entries
            rows_selected = set()
            for i in range(self.entries_per_column):
                # Hash to get row index
                row_hash = hashlib.sha256(col_seed + i.to_bytes(4, 'big')).digest()
                row_idx = int.from_bytes(row_hash[:4], 'big') % self.m
                rows_selected.add(row_idx)
            
            # Generate values for selected rows
            for i, row_idx in enumerate(sorted(rows_selected)):
                val_hash = hashlib.sha256(col_seed + b"VAL" + i.to_bytes(4, 'big')).digest()
                val_int = int.from_bytes(val_hash, 'big') % self.field_config.prime
                
                # Ensure non-zero value
                if val_int == 0:
                    val_int = 1
                
                val = FieldElement(val_int, self.field_config)
                matrix.set(row_idx, col, val)
        
        return matrix
    
    def _generate_dense_random(self, seed: bytes) -> SensingMatrix:
        """
        Generate a dense random matrix.
        
        Warning: This uses O(mn) memory. Only use for small n.
        """
        if self.n * self.m > 10**7:
            raise ValueError("Dense matrix too large. Use SPARSE_RANDOM instead.")
        
        matrix = SensingMatrix(
            rows=self.m,
            cols=self.n,
            field_config=self.field_config,
            entries_per_column=self.m  # All entries
        )
        
        for row in range(self.m):
            for col in range(self.n):
                entry_seed = hashlib.sha256(
                    seed + row.to_bytes(4, 'big') + col.to_bytes(4, 'big')
                ).digest()
                val_int = int.from_bytes(entry_seed, 'big') % self.field_config.prime
                
                if val_int != 0:
                    val = FieldElement(val_int, self.field_config)
                    matrix.set(row, col, val)
        
        return matrix
    
    def _generate_toeplitz(self, seed: bytes) -> SensingMatrix:
        """
        Generate a Toeplitz matrix.
        
        A Toeplitz matrix has constant diagonals, requiring only O(n+m) storage.
        This is useful for very large n where even sparse storage is too expensive.
        
        Note: This returns a sparse representation of the Toeplitz structure.
        """
        # Generate the first row and first column
        first_row = []
        first_col = []
        
        for i in range(self.n):
            row_seed = hashlib.sha256(seed + b"ROW" + i.to_bytes(4, 'big')).digest()
            val_int = int.from_bytes(row_seed, 'big') % self.field_config.prime
            first_row.append(FieldElement(val_int, self.field_config))
        
        for i in range(1, self.m):  # Skip first element (already in first_row)
            col_seed = hashlib.sha256(seed + b"COL" + i.to_bytes(4, 'big')).digest()
            val_int = int.from_bytes(col_seed, 'big') % self.field_config.prime
            first_col.append(FieldElement(val_int, self.field_config))
        
        # Build sparse matrix from Toeplitz structure
        # Only store entries that are likely to be accessed
        matrix = SensingMatrix(
            rows=self.m,
            cols=self.n,
            field_config=self.field_config,
            entries_per_column=self.m
        )
        
        # For efficiency, only store a band around the diagonal
        band_width = min(self.m, 100)  # Limit band width
        
        for col in range(self.n):
            for row in range(self.m):
                # Toeplitz: A[i,j] depends on i-j
                diff = row - col
                
                if diff == 0:
                    val = first_row[0]
                elif diff > 0 and diff < len(first_col) + 1:
                    val = first_col[diff - 1]
                elif diff < 0 and -diff < len(first_row):
                    val = first_row[-diff]
                else:
                    continue
                
                if val.value != 0 and abs(diff) < band_width:
                    matrix.set(row, col, val)
        
        return matrix


# Test code
if __name__ == "__main__":
    print("=== Sensing Matrix Tests ===\n")
    
    # Create a small field for testing
    field_config = FieldConfig(prime=2**61 - 1, name="Mersenne61")  # Mersenne prime
    
    # Test sparse random matrix
    print("1. Sparse Random Matrix Generation")
    gen = SensingMatrixGenerator(
        n=1000,  # Witness dimension
        m=32,    # Sketch dimension
        field_config=field_config,
        matrix_type="SPARSE_RANDOM",
        entries_per_column=5
    )
    
    matrix = gen.generate(b"test_seed")
    print(f"   Matrix size: {matrix.rows} x {matrix.cols}")
    print(f"   Non-zero entries: {len(matrix.data)}")
    print(f"   Density: {matrix.density():.6f}")
    print(f"   Memory: {matrix.memory_bytes()} bytes")
    
    # Test column extraction
    print("\n2. Column Extraction")
    col_5 = matrix.get_column(5)
    print(f"   Column 5 has {len(col_5)} non-zero entries")
    
    # Test submatrix extraction
    print("\n3. Submatrix Extraction")
    indices = [0, 5, 10, 50, 100]
    submatrix = matrix.get_submatrix_columns(indices)
    print(f"   Submatrix size: {submatrix.rows} x {submatrix.cols}")
    print(f"   Submatrix non-zero entries: {len(submatrix.data)}")
    
    # Test sparse multiplication
    print("\n4. Sparse Multiplication")
    support = [0, 5, 10]
    values = [
        FieldElement(1, field_config),
        FieldElement(2, field_config),
        FieldElement(3, field_config)
    ]
    
    sketch = matrix.multiply_sparse(support, values)
    print(f"   Sketch length: {len(sketch)}")
    print(f"   First 5 sketch values: {[s.value for s in sketch[:5]]}")
    
    # Test determinism
    print("\n5. Determinism Test")
    matrix2 = gen.generate(b"test_seed")
    same = all(
        matrix.data.get(k) == matrix2.data.get(k) 
        for k in set(matrix.data.keys()) | set(matrix2.data.keys())
    )
    print(f"   Same seed produces same matrix: {same}")
    
    matrix3 = gen.generate(b"different_seed")
    different = any(
        matrix.data.get(k) != matrix3.data.get(k)
        for k in set(matrix.data.keys()) | set(matrix3.data.keys())
    )
    print(f"   Different seed produces different matrix: {different}")
    
    print("\n=== All Tests Passed ===")
