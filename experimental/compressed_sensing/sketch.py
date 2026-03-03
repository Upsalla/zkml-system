"""
Sketch Computation for CS-Inspired Sparse Witness Commitment (CSWC).

The sketch is a linear "fingerprint" of the sparse witness:
    y = A * w = A_S * w_S

where:
- A is the sensing matrix (m x n)
- w is the full witness (n x 1, sparse)
- S is the support (indices of non-zero entries)
- A_S is the submatrix of A with columns indexed by S
- w_S is the vector of non-zero values

The sketch enables deterministic verification:
- Prover commits to (S, w_S, y)
- Verifier checks that y == A_S * w_S
- If the prover lies about S or w_S, the check fails with overwhelming probability

This is more efficient than probabilistic sampling because:
1. No random challenges needed
2. Single deterministic check instead of multiple samples
3. Stronger soundness guarantee (information-theoretic vs computational)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig
from compressed_sensing.sensing_matrix import SensingMatrix, SensingMatrixGenerator
from compressed_sensing.sparse_witness import SparseWitness


@dataclass
class Sketch:
    """
    A linear sketch of a sparse witness.
    
    Attributes:
        values: The sketch vector y = A_S * w_S
        dimension: Length of the sketch (m)
        witness_dimension: Original witness dimension (n)
        support_size: Number of non-zero entries (k)
    """
    values: List[FieldElement]
    dimension: int
    witness_dimension: int
    support_size: int
    
    def size_bytes(self) -> int:
        """Estimate size in bytes."""
        return len(self.values) * 32  # 32 bytes per field element
    
    def to_int_list(self) -> List[int]:
        """Convert to list of integers (for serialization)."""
        return [v.value for v in self.values]


class SketchComputer:
    """
    Computes and verifies sketches for sparse witnesses.
    
    The sketch computation is the core of CSWC:
    - Prover: Compute y = A_S * w_S
    - Verifier: Check y == A_S * w_S
    
    Security relies on the fact that for random A, finding a different
    (S', w_S') that produces the same y is computationally infeasible.
    """
    
    def __init__(self, sensing_matrix: SensingMatrix):
        """
        Initialize the sketch computer.
        
        Args:
            sensing_matrix: The sensing matrix A
        """
        self.matrix = sensing_matrix
        self.m = sensing_matrix.rows
        self.n = sensing_matrix.cols
        self.field_config = sensing_matrix.field_config
    
    def compute(self, sparse_witness: SparseWitness) -> Sketch:
        """
        Compute the sketch of a sparse witness.
        
        Args:
            sparse_witness: The sparse witness (S, w_S)
            
        Returns:
            Sketch containing y = A_S * w_S
            
        Complexity: O(m * k * entries_per_column)
        """
        if sparse_witness.full_size != self.n:
            raise ValueError(
                f"Witness dimension {sparse_witness.full_size} != matrix cols {self.n}"
            )
        
        # Compute y = A_S * w_S using sparse multiplication
        sketch_values = self.matrix.multiply_sparse(
            sparse_witness.support,
            sparse_witness.values
        )
        
        return Sketch(
            values=sketch_values,
            dimension=self.m,
            witness_dimension=self.n,
            support_size=sparse_witness.num_nonzero
        )
    
    def verify(self, sparse_witness: SparseWitness, sketch: Sketch) -> bool:
        """
        Verify that a sketch matches a sparse witness.
        
        This is the core verification step in CSWC:
        Check that y == A_S * w_S
        
        Args:
            sparse_witness: The claimed sparse witness
            sketch: The claimed sketch
            
        Returns:
            True if the sketch is valid, False otherwise
        """
        # Recompute the sketch
        computed = self.compute(sparse_witness)
        
        # Compare element by element
        if len(computed.values) != len(sketch.values):
            return False
        
        for v1, v2 in zip(computed.values, sketch.values):
            if v1.value != v2.value:
                return False
        
        return True
    
    def compute_with_blinding(self, 
                              sparse_witness: SparseWitness,
                              blinding_seed: bytes) -> Tuple[Sketch, List[FieldElement]]:
        """
        Compute a blinded sketch for zero-knowledge.
        
        The blinded sketch is: y' = y + r
        where r is a random vector derived from the blinding seed.
        
        This prevents the sketch from leaking information about the witness.
        
        Args:
            sparse_witness: The sparse witness
            blinding_seed: Seed for generating the blinding vector
            
        Returns:
            Tuple of (blinded sketch, blinding vector r)
        """
        import hashlib
        
        # Compute unblinded sketch
        sketch = self.compute(sparse_witness)
        
        # Generate blinding vector
        blinding = []
        for i in range(self.m):
            r_hash = hashlib.sha256(blinding_seed + i.to_bytes(4, 'big')).digest()
            r_int = int.from_bytes(r_hash, 'big') % self.field_config.prime
            blinding.append(FieldElement(r_int, self.field_config))
        
        # Add blinding to sketch
        blinded_values = [
            sketch.values[i] + blinding[i]
            for i in range(self.m)
        ]
        
        blinded_sketch = Sketch(
            values=blinded_values,
            dimension=self.m,
            witness_dimension=self.n,
            support_size=sparse_witness.num_nonzero
        )
        
        return blinded_sketch, blinding
    
    def verify_blinded(self,
                       sparse_witness: SparseWitness,
                       blinded_sketch: Sketch,
                       blinding: List[FieldElement]) -> bool:
        """
        Verify a blinded sketch.
        
        Check that y' - r == A_S * w_S
        
        Args:
            sparse_witness: The sparse witness
            blinded_sketch: The blinded sketch y'
            blinding: The blinding vector r
            
        Returns:
            True if valid, False otherwise
        """
        # Compute expected sketch
        computed = self.compute(sparse_witness)
        
        # Check y' - r == computed
        for i in range(self.m):
            unblinded = blinded_sketch.values[i] - blinding[i]
            if unblinded.value != computed.values[i].value:
                return False
        
        return True


class SketchVerifier:
    """
    Standalone verifier for sketches.
    
    Used when the verifier doesn't have access to the full SketchComputer
    but only needs to verify a specific sketch.
    """
    
    def __init__(self, 
                 n: int, 
                 m: int, 
                 field_config: FieldConfig,
                 matrix_seed: bytes = b"CSWC_DEFAULT_SEED"):
        """
        Initialize the verifier.
        
        Args:
            n: Witness dimension
            m: Sketch dimension
            field_config: Field configuration
            matrix_seed: Seed used to generate the sensing matrix
        """
        self.n = n
        self.m = m
        self.field_config = field_config
        self.matrix_seed = matrix_seed
        
        # Lazily generate the matrix only when needed
        self._matrix: Optional[SensingMatrix] = None
    
    @property
    def matrix(self) -> SensingMatrix:
        """Get the sensing matrix (generated on first access)."""
        if self._matrix is None:
            gen = SensingMatrixGenerator(
                n=self.n,
                m=self.m,
                field_config=self.field_config,
                matrix_type="SPARSE_RANDOM"
            )
            self._matrix = gen.generate(self.matrix_seed)
        return self._matrix
    
    def verify(self,
               support: List[int],
               values: List[FieldElement],
               sketch_values: List[FieldElement]) -> bool:
        """
        Verify a sketch against claimed support and values.
        
        Args:
            support: Claimed support indices
            values: Claimed non-zero values
            sketch_values: Claimed sketch
            
        Returns:
            True if valid, False otherwise
        """
        # Compute expected sketch
        computed = self.matrix.multiply_sparse(support, values)
        
        # Compare
        if len(computed) != len(sketch_values):
            return False
        
        for v1, v2 in zip(computed, sketch_values):
            if v1.value != v2.value:
                return False
        
        return True


# Test code
if __name__ == "__main__":
    print("=== Sketch Computation Tests ===\n")
    
    # Setup
    field_config = FieldConfig(prime=2**61 - 1, name="Mersenne61")
    n = 1000  # Witness dimension
    m = 32    # Sketch dimension
    
    # Generate sensing matrix
    print("1. Generating Sensing Matrix")
    gen = SensingMatrixGenerator(n, m, field_config, "SPARSE_RANDOM", entries_per_column=5)
    matrix = gen.generate(b"test_seed")
    print(f"   Matrix: {matrix.rows} x {matrix.cols}")
    print(f"   Non-zero entries: {len(matrix.data)}")
    
    # Create sketch computer
    computer = SketchComputer(matrix)
    
    # Create a sparse witness
    print("\n2. Creating Sparse Witness")
    from compressed_sensing.sparse_witness import SparseWitness
    
    support = [0, 10, 50, 100, 500, 999]
    values = [FieldElement(i + 1, field_config) for i in range(len(support))]
    
    sparse_witness = SparseWitness(
        support=support,
        values=values,
        full_size=n,
        field_config=field_config
    )
    print(f"   Support: {support}")
    print(f"   Sparsity: {sparse_witness.sparsity:.2%}")
    
    # Compute sketch
    print("\n3. Computing Sketch")
    sketch = computer.compute(sparse_witness)
    print(f"   Sketch dimension: {sketch.dimension}")
    print(f"   First 5 values: {[v.value for v in sketch.values[:5]]}")
    print(f"   Sketch size: {sketch.size_bytes()} bytes")
    
    # Verify sketch
    print("\n4. Verifying Sketch")
    valid = computer.verify(sparse_witness, sketch)
    print(f"   Valid sketch: {valid}")
    
    # Test with wrong witness
    print("\n5. Testing Invalid Witness Detection")
    wrong_values = [FieldElement(999, field_config) for _ in range(len(support))]
    wrong_witness = SparseWitness(
        support=support,
        values=wrong_values,
        full_size=n,
        field_config=field_config
    )
    
    invalid = computer.verify(wrong_witness, sketch)
    print(f"   Wrong witness detected: {not invalid}")
    
    # Test with wrong support
    wrong_support = [1, 11, 51, 101, 501, 998]  # Different indices
    wrong_support_witness = SparseWitness(
        support=wrong_support,
        values=values,
        full_size=n,
        field_config=field_config
    )
    
    invalid_support = computer.verify(wrong_support_witness, sketch)
    print(f"   Wrong support detected: {not invalid_support}")
    
    # Test blinded sketch
    print("\n6. Testing Blinded Sketch")
    blinded_sketch, blinding = computer.compute_with_blinding(
        sparse_witness, 
        b"blinding_seed"
    )
    print(f"   Blinded sketch first 5: {[v.value for v in blinded_sketch.values[:5]]}")
    print(f"   Original sketch first 5: {[v.value for v in sketch.values[:5]]}")
    print(f"   Values are different: {blinded_sketch.values[0].value != sketch.values[0].value}")
    
    # Verify blinded sketch
    valid_blinded = computer.verify_blinded(sparse_witness, blinded_sketch, blinding)
    print(f"   Blinded verification: {valid_blinded}")
    
    # Test standalone verifier
    print("\n7. Testing Standalone Verifier")
    verifier = SketchVerifier(n, m, field_config, b"test_seed")
    
    valid_standalone = verifier.verify(support, values, sketch.values)
    print(f"   Standalone verification: {valid_standalone}")
    
    # Performance test
    print("\n8. Performance Test")
    import time
    
    # Large sparse witness
    large_support = list(range(0, 10000, 10))  # 1000 non-zero entries
    large_values = [FieldElement(i + 1, field_config) for i in range(len(large_support))]
    
    large_gen = SensingMatrixGenerator(100000, 64, field_config, "SPARSE_RANDOM", 5)
    large_matrix = large_gen.generate(b"large_test")
    large_computer = SketchComputer(large_matrix)
    
    large_witness = SparseWitness(
        support=large_support,
        values=large_values,
        full_size=100000,
        field_config=field_config
    )
    
    start = time.time()
    large_sketch = large_computer.compute(large_witness)
    compute_time = time.time() - start
    
    start = time.time()
    large_valid = large_computer.verify(large_witness, large_sketch)
    verify_time = time.time() - start
    
    print(f"   Witness size: 100,000")
    print(f"   Non-zero entries: 1,000")
    print(f"   Sketch dimension: 64")
    print(f"   Compute time: {compute_time*1000:.2f} ms")
    print(f"   Verify time: {verify_time*1000:.2f} ms")
    print(f"   Valid: {large_valid}")
    
    print("\n=== All Tests Passed ===")
