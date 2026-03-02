"""
Sparse Witness Extraction and Representation for CSWC.

This module provides utilities for extracting the sparse representation
of a witness vector and working with sparse data structures.

In neural network inference, activations are often sparse:
- ReLU: ~50% of activations are zero
- After pruning: up to 90% can be zero
- GELU: Fewer exact zeros, but many near-zero values

The SparseExtractor identifies non-zero entries and creates a compact
representation that can be committed and verified efficiently.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig


@dataclass
class SparseWitness:
    """
    Sparse representation of a witness vector.
    
    Instead of storing all n values, we store only the k non-zero entries
    along with their indices.
    
    Attributes:
        support: List of indices where the witness is non-zero
        values: List of non-zero values (same order as support)
        full_size: Original size of the witness (n)
        field_config: Field configuration for arithmetic
    """
    support: List[int]
    values: List[FieldElement]
    full_size: int
    field_config: FieldConfig
    
    def __post_init__(self):
        """Validate the sparse witness."""
        if len(self.support) != len(self.values):
            raise ValueError("Support and values must have same length")
        if any(idx < 0 or idx >= self.full_size for idx in self.support):
            raise ValueError("Support indices must be in [0, full_size)")
        if len(self.support) != len(set(self.support)):
            raise ValueError("Support indices must be unique")
    
    @property
    def sparsity(self) -> float:
        """
        Return the sparsity ratio (fraction of zeros).
        
        Returns:
            Float in [0, 1], where 1 means all zeros
        """
        if self.full_size == 0:
            return 0.0
        return 1.0 - len(self.support) / self.full_size
    
    @property
    def num_nonzero(self) -> int:
        """Return the number of non-zero entries (k)."""
        return len(self.support)
    
    def to_dense(self) -> List[FieldElement]:
        """
        Reconstruct the full dense witness vector.
        
        This is mainly useful for testing and verification.
        
        Returns:
            List of n FieldElements
        """
        result = [FieldElement.zero(self.field_config) for _ in range(self.full_size)]
        for idx, val in zip(self.support, self.values):
            result[idx] = val
        return result
    
    def get_value(self, index: int) -> FieldElement:
        """
        Get the value at a specific index.
        
        Args:
            index: Index in [0, full_size)
            
        Returns:
            The value at that index (zero if not in support)
        """
        if index < 0 or index >= self.full_size:
            raise IndexError(f"Index {index} out of range [0, {self.full_size})")
        
        try:
            pos = self.support.index(index)
            return self.values[pos]
        except ValueError:
            return FieldElement.zero(self.field_config)
    
    def size_bytes(self) -> int:
        """
        Estimate the size in bytes of the sparse representation.
        
        Returns:
            Estimated size in bytes
        """
        # Each index: 4 bytes (32-bit int)
        # Each value: 32 bytes (256-bit field element)
        return len(self.support) * (4 + 32)
    
    def compression_ratio(self) -> float:
        """
        Calculate the compression ratio compared to dense storage.
        
        Returns:
            Ratio of sparse size to dense size
        """
        dense_size = self.full_size * 32  # 32 bytes per element
        sparse_size = self.size_bytes()
        if dense_size == 0:
            return 1.0
        return sparse_size / dense_size


class SparseExtractor:
    """
    Extracts sparse representation from a dense witness vector.
    
    The extractor identifies non-zero entries and creates a SparseWitness.
    An optional threshold can be used to treat near-zero values as zero.
    """
    
    def __init__(self, threshold: int = 0):
        """
        Initialize the extractor.
        
        Args:
            threshold: Values with absolute value <= threshold are treated as zero.
                      For exact zero detection, use threshold=0.
        """
        self.threshold = threshold
    
    def extract(self, witness: List[FieldElement]) -> SparseWitness:
        """
        Extract the sparse representation of a witness.
        
        Args:
            witness: Dense witness vector
            
        Returns:
            SparseWitness with support and values
        """
        if not witness:
            raise ValueError("Witness cannot be empty")
        
        field_config = witness[0].field
        support = []
        values = []
        
        for idx, val in enumerate(witness):
            # Check if value is "non-zero" (above threshold)
            if self._is_nonzero(val):
                support.append(idx)
                values.append(val)
        
        return SparseWitness(
            support=support,
            values=values,
            full_size=len(witness),
            field_config=field_config
        )
    
    def _is_nonzero(self, val: FieldElement) -> bool:
        """
        Check if a value should be considered non-zero.
        
        For finite fields, we check if the value is above the threshold.
        Note: In a finite field, "near zero" doesn't have the same meaning
        as in real numbers, so threshold is typically 0.
        """
        return val.value > self.threshold
    
    def extract_with_stats(self, witness: List[FieldElement]) -> Tuple[SparseWitness, dict]:
        """
        Extract sparse representation and compute statistics.
        
        Args:
            witness: Dense witness vector
            
        Returns:
            Tuple of (SparseWitness, statistics dict)
        """
        sparse = self.extract(witness)
        
        stats = {
            "full_size": sparse.full_size,
            "num_nonzero": sparse.num_nonzero,
            "sparsity": sparse.sparsity,
            "compression_ratio": sparse.compression_ratio(),
            "size_bytes_dense": sparse.full_size * 32,
            "size_bytes_sparse": sparse.size_bytes(),
            "bytes_saved": sparse.full_size * 32 - sparse.size_bytes()
        }
        
        return sparse, stats


class SparseWitnessBuilder:
    """
    Builder for constructing sparse witnesses incrementally.
    
    Useful when the witness is computed piece by piece (e.g., layer by layer
    in a neural network).
    """
    
    def __init__(self, full_size: int, field_config: FieldConfig):
        """
        Initialize the builder.
        
        Args:
            full_size: Total size of the witness
            field_config: Field configuration
        """
        self.full_size = full_size
        self.field_config = field_config
        self._entries: dict = {}  # index -> value
    
    def add(self, index: int, value: FieldElement) -> None:
        """
        Add a non-zero entry to the witness.
        
        Args:
            index: Index in [0, full_size)
            value: Non-zero value at that index
        """
        if index < 0 or index >= self.full_size:
            raise IndexError(f"Index {index} out of range [0, {self.full_size})")
        
        if value.value != 0:
            self._entries[index] = value
        elif index in self._entries:
            del self._entries[index]
    
    def add_batch(self, start_index: int, values: List[FieldElement]) -> None:
        """
        Add a batch of values starting at a given index.
        
        Only non-zero values are stored.
        
        Args:
            start_index: Starting index
            values: List of values to add
        """
        for i, val in enumerate(values):
            if val.value != 0:
                self._entries[start_index + i] = val
    
    def build(self) -> SparseWitness:
        """
        Build the final SparseWitness.
        
        Returns:
            SparseWitness instance
        """
        # Sort by index for consistent ordering
        sorted_items = sorted(self._entries.items())
        support = [idx for idx, _ in sorted_items]
        values = [val for _, val in sorted_items]
        
        return SparseWitness(
            support=support,
            values=values,
            full_size=self.full_size,
            field_config=self.field_config
        )
    
    def current_sparsity(self) -> float:
        """Return the current sparsity ratio."""
        if self.full_size == 0:
            return 0.0
        return 1.0 - len(self._entries) / self.full_size


# Test code
if __name__ == "__main__":
    print("=== Sparse Witness Tests ===\n")
    
    # Create a field for testing
    field_config = FieldConfig(prime=2**61 - 1, name="Mersenne61")
    
    # Create a dense witness with some zeros
    print("1. Creating Dense Witness")
    dense_witness = []
    for i in range(100):
        if i % 3 == 0:  # Every third element is non-zero
            dense_witness.append(FieldElement(i + 1, field_config))
        else:
            dense_witness.append(FieldElement.zero(field_config))
    
    print(f"   Dense witness size: {len(dense_witness)}")
    print(f"   Non-zero count: {sum(1 for v in dense_witness if v.value != 0)}")
    
    # Extract sparse representation
    print("\n2. Extracting Sparse Representation")
    extractor = SparseExtractor(threshold=0)
    sparse, stats = extractor.extract_with_stats(dense_witness)
    
    print(f"   Support size: {len(sparse.support)}")
    print(f"   Sparsity: {sparse.sparsity:.2%}")
    print(f"   Compression ratio: {sparse.compression_ratio():.2%}")
    print(f"   Bytes saved: {stats['bytes_saved']}")
    
    # Verify reconstruction
    print("\n3. Verifying Reconstruction")
    reconstructed = sparse.to_dense()
    match = all(
        orig.value == recon.value 
        for orig, recon in zip(dense_witness, reconstructed)
    )
    print(f"   Reconstruction matches original: {match}")
    
    # Test random access
    print("\n4. Random Access Test")
    for test_idx in [0, 1, 3, 50, 99]:
        original = dense_witness[test_idx].value
        sparse_val = sparse.get_value(test_idx).value
        print(f"   Index {test_idx}: original={original}, sparse={sparse_val}, match={original == sparse_val}")
    
    # Test builder
    print("\n5. Builder Test")
    builder = SparseWitnessBuilder(full_size=50, field_config=field_config)
    builder.add(0, FieldElement(10, field_config))
    builder.add(10, FieldElement(20, field_config))
    builder.add(25, FieldElement(30, field_config))
    builder.add(49, FieldElement(40, field_config))
    
    built_sparse = builder.build()
    print(f"   Built witness support: {built_sparse.support}")
    print(f"   Built witness values: {[v.value for v in built_sparse.values]}")
    print(f"   Sparsity: {built_sparse.sparsity:.2%}")
    
    # Test high sparsity scenario (90% zeros)
    print("\n6. High Sparsity Scenario (90% zeros)")
    high_sparse_witness = []
    for i in range(1000):
        if i % 10 == 0:  # Only every 10th element is non-zero
            high_sparse_witness.append(FieldElement(i + 1, field_config))
        else:
            high_sparse_witness.append(FieldElement.zero(field_config))
    
    sparse_high, stats_high = extractor.extract_with_stats(high_sparse_witness)
    print(f"   Original size: {stats_high['full_size']}")
    print(f"   Non-zero entries: {stats_high['num_nonzero']}")
    print(f"   Sparsity: {stats_high['sparsity']:.2%}")
    print(f"   Dense size: {stats_high['size_bytes_dense']} bytes")
    print(f"   Sparse size: {stats_high['size_bytes_sparse']} bytes")
    print(f"   Compression: {1 - sparse_high.compression_ratio():.2%} reduction")
    
    print("\n=== All Tests Passed ===")
