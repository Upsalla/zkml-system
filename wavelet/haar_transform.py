"""
Haar Wavelet Witness Batching (HWWB) Implementation.

This module implements a Haar wavelet-based approach to witness compression
that exploits correlation between adjacent witness values.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig


@dataclass
class HaarCoefficients:
    """
    Haar-transformed representation of a witness vector.
    
    For a witness w of length n, this stores:
    - sums[k] = w[2k] + w[2k+1]  (approximation coefficients)
    - diffs[k] = w[2k] - w[2k+1] (detail coefficients)
    
    The original witness can be reconstructed as:
    - w[2k] = (sums[k] + diffs[k]) * inv2
    - w[2k+1] = (sums[k] - diffs[k]) * inv2
    """
    sums: List[FieldElement]
    diffs: List[FieldElement]
    field_config: FieldConfig
    original_length: int
    
    # Cached inverse of 2 for reconstruction
    _inv2: Optional[FieldElement] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compute inverse of 2 in the field."""
        two = FieldElement(2, self.field_config)
        self._inv2 = two.inverse()
    
    @property
    def inv2(self) -> FieldElement:
        """Get the multiplicative inverse of 2."""
        if self._inv2 is None:
            two = FieldElement(2, self.field_config)
            self._inv2 = two.inverse()
        return self._inv2
    
    def reconstruct(self) -> List[FieldElement]:
        """
        Reconstruct the original witness from Haar coefficients.
        
        Returns:
            List of FieldElements representing the original witness
        """
        result = []
        for i in range(len(self.sums)):
            # w[2k] = (sum + diff) / 2
            w_even = (self.sums[i] + self.diffs[i]) * self.inv2
            # w[2k+1] = (sum - diff) / 2
            w_odd = (self.sums[i] - self.diffs[i]) * self.inv2
            result.append(w_even)
            result.append(w_odd)
        
        # Handle odd-length original (last element stored separately)
        if self.original_length % 2 == 1:
            result = result[:self.original_length]
        
        return result


class HaarTransformer:
    """
    Performs Haar wavelet transformation on witness vectors.
    """
    
    def __init__(self, field_config: FieldConfig):
        self.field_config = field_config
    
    def transform(self, witness: List[FieldElement]) -> HaarCoefficients:
        """
        Transform a witness vector using Haar wavelets.
        
        Args:
            witness: List of FieldElements
            
        Returns:
            HaarCoefficients containing sums and differences
        """
        n = len(witness)
        sums = []
        diffs = []
        
        # Process pairs
        for i in range(0, n - 1, 2):
            sums.append(witness[i] + witness[i + 1])
            diffs.append(witness[i] - witness[i + 1])
        
        # Handle odd length: treat last element as sum with diff=0
        if n % 2 == 1:
            sums.append(witness[-1] + witness[-1])  # 2 * last
            diffs.append(FieldElement(0, self.field_config))
        
        return HaarCoefficients(
            sums=sums,
            diffs=diffs,
            field_config=self.field_config,
            original_length=n
        )
    
    def analyze_correlation(self, witness: List[FieldElement], 
                           threshold: int = 100) -> Dict[str, Any]:
        """
        Analyze the correlation structure of a witness.
        
        Args:
            witness: List of FieldElements
            threshold: Maximum absolute difference to consider "small"
            
        Returns:
            Dictionary with correlation statistics
        """
        coeffs = self.transform(witness)
        
        # Count small differences
        small_diffs = 0
        zero_diffs = 0
        total_diffs = len(coeffs.diffs)
        
        for diff in coeffs.diffs:
            val = diff.value
            # Handle field wrap-around for "small" negative values
            if val > self.field_config.prime // 2:
                val = self.field_config.prime - val
            
            if val == 0:
                zero_diffs += 1
                small_diffs += 1
            elif val <= threshold:
                small_diffs += 1
        
        return {
            "total_pairs": total_diffs,
            "zero_diffs": zero_diffs,
            "small_diffs": small_diffs,
            "zero_ratio": zero_diffs / total_diffs if total_diffs > 0 else 0,
            "small_ratio": small_diffs / total_diffs if total_diffs > 0 else 0,
            "correlation_score": small_diffs / total_diffs if total_diffs > 0 else 0
        }


@dataclass
class HWWBProof:
    """
    Proof generated using Haar Wavelet Witness Batching.
    """
    # Commitments
    sums_commitment: bytes
    diffs_commitment: bytes
    batched_small_diffs_commitment: bytes
    
    # Revealed values for verification
    sums: List[int]
    large_diffs: List[Tuple[int, int]]  # (index, value) for large diffs
    small_diff_indices: List[int]
    small_diff_bound: int
    
    # Metadata
    original_length: int
    field_modulus: int
    
    def size_bytes(self) -> int:
        """Calculate the proof size in bytes."""
        # Commitments: 3 * 32 bytes
        size = 96
        # Sums: 32 bytes each
        size += len(self.sums) * 32
        # Large diffs: 4 bytes index + 32 bytes value each
        size += len(self.large_diffs) * 36
        # Small diff indices: 4 bytes each
        size += len(self.small_diff_indices) * 4
        # Metadata
        size += 16
        return size


class HWWBProver:
    """
    Prover for Haar Wavelet Witness Batching.
    """
    
    def __init__(self, field_config: FieldConfig, small_diff_threshold: int = 1000):
        self.field_config = field_config
        self.transformer = HaarTransformer(field_config)
        self.small_diff_threshold = small_diff_threshold
    
    def _is_small_diff(self, diff: FieldElement) -> bool:
        """Check if a difference is considered 'small'."""
        val = diff.value
        # Handle field wrap-around
        if val > self.field_config.prime // 2:
            val = self.field_config.prime - val
        return val <= self.small_diff_threshold
    
    def _commit(self, data: bytes) -> bytes:
        """Create a cryptographic commitment."""
        return hashlib.sha256(data).digest()
    
    def prove(self, witness: List[FieldElement]) -> HWWBProof:
        """
        Generate an HWWB proof for a witness.
        
        Args:
            witness: List of FieldElements
            
        Returns:
            HWWBProof containing commitments and revealed values
        """
        # Transform
        coeffs = self.transformer.transform(witness)
        
        # Separate large and small diffs
        large_diffs = []
        small_diff_indices = []
        
        for i, diff in enumerate(coeffs.diffs):
            if self._is_small_diff(diff):
                small_diff_indices.append(i)
            else:
                large_diffs.append((i, diff.value))
        
        # Create commitments
        sums_data = b''.join(s.value.to_bytes(32, 'big') for s in coeffs.sums)
        sums_commitment = self._commit(sums_data)
        
        diffs_data = b''.join(d.value.to_bytes(32, 'big') for d in coeffs.diffs)
        diffs_commitment = self._commit(diffs_data)
        
        # Batch commitment for small diffs (just commit to their indices and bound)
        small_diffs_data = bytes(small_diff_indices) + self.small_diff_threshold.to_bytes(8, 'big')
        batched_small_diffs_commitment = self._commit(small_diffs_data)
        
        return HWWBProof(
            sums_commitment=sums_commitment,
            diffs_commitment=diffs_commitment,
            batched_small_diffs_commitment=batched_small_diffs_commitment,
            sums=[s.value for s in coeffs.sums],
            large_diffs=large_diffs,
            small_diff_indices=small_diff_indices,
            small_diff_bound=self.small_diff_threshold,
            original_length=coeffs.original_length,
            field_modulus=self.field_config.prime
        )


class HWWBVerifier:
    """
    Verifier for Haar Wavelet Witness Batching.
    """
    
    def __init__(self, field_config: FieldConfig):
        self.field_config = field_config
    
    def _commit(self, data: bytes) -> bytes:
        """Create a cryptographic commitment."""
        return hashlib.sha256(data).digest()
    
    def verify(self, proof: HWWBProof) -> Tuple[bool, str]:
        """
        Verify an HWWB proof.
        
        Args:
            proof: The HWWBProof to verify
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Verify sums commitment
        sums_data = b''.join(s.to_bytes(32, 'big') for s in proof.sums)
        expected_sums_commitment = self._commit(sums_data)
        
        if proof.sums_commitment != expected_sums_commitment:
            return False, "Sums commitment mismatch"
        
        # Verify small diffs commitment
        small_diffs_data = bytes(proof.small_diff_indices) + proof.small_diff_bound.to_bytes(8, 'big')
        expected_small_commitment = self._commit(small_diffs_data)
        
        if proof.batched_small_diffs_commitment != expected_small_commitment:
            return False, "Small diffs commitment mismatch"
        
        # Verify that large diffs are indeed large
        for idx, val in proof.large_diffs:
            # Check if value is actually large
            if val > proof.field_modulus // 2:
                val = proof.field_modulus - val
            if val <= proof.small_diff_bound:
                return False, f"Diff at index {idx} claimed large but is small"
        
        # Verify completeness: all indices should be covered
        num_diffs = (proof.original_length + 1) // 2
        covered_indices = set(proof.small_diff_indices) | set(idx for idx, _ in proof.large_diffs)
        
        if len(covered_indices) != num_diffs:
            return False, "Not all diff indices are covered"
        
        return True, "Valid"
    
    def reconstruct_witness(self, proof: HWWBProof, 
                           small_diffs: List[int]) -> List[FieldElement]:
        """
        Reconstruct the witness from a proof and revealed small diffs.
        
        Args:
            proof: The HWWBProof
            small_diffs: The actual values of small differences
            
        Returns:
            Reconstructed witness
        """
        # Build full diffs array
        num_diffs = (proof.original_length + 1) // 2
        diffs = [0] * num_diffs
        
        # Fill in large diffs
        for idx, val in proof.large_diffs:
            diffs[idx] = val
        
        # Fill in small diffs
        for i, idx in enumerate(proof.small_diff_indices):
            diffs[idx] = small_diffs[i]
        
        # Reconstruct
        sums = [FieldElement(s, self.field_config) for s in proof.sums]
        diffs_fe = [FieldElement(d, self.field_config) for d in diffs]
        
        coeffs = HaarCoefficients(
            sums=sums,
            diffs=diffs_fe,
            field_config=self.field_config,
            original_length=proof.original_length
        )
        
        return coeffs.reconstruct()


class HWWBSystem:
    """
    Complete HWWB system combining prover and verifier.
    """
    
    def __init__(self, field_config: FieldConfig, small_diff_threshold: int = 1000):
        self.field_config = field_config
        self.prover = HWWBProver(field_config, small_diff_threshold)
        self.verifier = HWWBVerifier(field_config)
        self.transformer = HaarTransformer(field_config)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Haar Wavelet Witness Batching (HWWB) Tests")
    print("=" * 60)
    
    # Setup
    field_config = FieldConfig(prime=2**61 - 1, name="Test")
    
    # Test 1: Basic transformation and reconstruction
    print("\n1. Testing Haar Transform")
    transformer = HaarTransformer(field_config)
    
    # Create a witness with correlated values
    witness = [FieldElement(100 + i, field_config) for i in range(10)]
    print(f"   Original witness: {[w.value for w in witness]}")
    
    coeffs = transformer.transform(witness)
    print(f"   Sums: {[s.value for s in coeffs.sums]}")
    print(f"   Diffs: {[d.value for d in coeffs.diffs]}")
    
    reconstructed = coeffs.reconstruct()
    print(f"   Reconstructed: {[w.value for w in reconstructed]}")
    
    # Verify reconstruction
    match = all(w1.value == w2.value for w1, w2 in zip(witness, reconstructed))
    print(f"   Reconstruction correct: {match}")
    
    # Test 2: Correlation analysis
    print("\n2. Testing Correlation Analysis")
    
    # Highly correlated witness (adjacent values similar)
    correlated = [FieldElement(1000 + (i // 2), field_config) for i in range(100)]
    analysis = transformer.analyze_correlation(correlated, threshold=10)
    print(f"   Correlated witness: {analysis['small_ratio']*100:.1f}% small diffs")
    
    # Uncorrelated witness (random-ish values)
    uncorrelated = [FieldElement((i * 12345) % 10000, field_config) for i in range(100)]
    analysis = transformer.analyze_correlation(uncorrelated, threshold=10)
    print(f"   Uncorrelated witness: {analysis['small_ratio']*100:.1f}% small diffs")
    
    # Test 3: Full HWWB proof
    print("\n3. Testing HWWB Proof Generation and Verification")
    
    system = HWWBSystem(field_config, small_diff_threshold=100)
    
    # Create witness with mixed correlation
    mixed_witness = []
    for i in range(50):
        if i % 2 == 0:
            mixed_witness.append(FieldElement(1000 + i, field_config))
        else:
            mixed_witness.append(FieldElement(1000 + i - 1 + (i % 5), field_config))  # Similar to previous
    
    # Generate proof
    start = time.time()
    proof = system.prover.prove(mixed_witness)
    prove_time = (time.time() - start) * 1000
    
    print(f"   Witness size: {len(mixed_witness)}")
    print(f"   Small diffs: {len(proof.small_diff_indices)}")
    print(f"   Large diffs: {len(proof.large_diffs)}")
    print(f"   Proof size: {proof.size_bytes()} bytes")
    print(f"   Prove time: {prove_time:.2f} ms")
    
    # Verify
    start = time.time()
    valid, reason = system.verifier.verify(proof)
    verify_time = (time.time() - start) * 1000
    
    print(f"   Valid: {valid} ({reason})")
    print(f"   Verify time: {verify_time:.2f} ms")
    
    # Test 4: Comparison with standard approach
    print("\n4. Comparison with Standard Commitment")
    
    standard_size = 32 + len(mixed_witness) * 32  # hash + all values
    hwwb_size = proof.size_bytes()
    
    print(f"   Standard size: {standard_size} bytes")
    print(f"   HWWB size: {hwwb_size} bytes")
    print(f"   Reduction: {(1 - hwwb_size/standard_size)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
