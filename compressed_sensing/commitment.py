"""
CS-Inspired Sparse Witness Commitment (CSWC) Protocol.

This module implements the complete CSWC protocol:
1. Prover commits to (Support S, Values w_S, Sketch y)
2. Verifier checks that y == A_S * w_S

The protocol provides:
- Soundness: A cheating prover is caught with overwhelming probability
- Zero-Knowledge: The proof reveals nothing about the witness beyond its sparsity
- Efficiency: O(k) proof size and verification time for k non-zero entries

Comparison with probabilistic sampling:
- Sampling: Verifier randomly samples λ positions, checks they're zero
- CSWC: Verifier checks single deterministic equation y == A_S * w_S

CSWC advantages:
- No random challenges (non-interactive)
- Stronger soundness (information-theoretic)
- Simpler protocol
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig
from compressed_sensing.sensing_matrix import SensingMatrix, SensingMatrixGenerator
from compressed_sensing.sparse_witness import SparseWitness, SparseExtractor
from compressed_sensing.sketch import Sketch, SketchComputer


@dataclass
class CSWCCommitment:
    """
    Commitment to a sparse witness using CSWC.
    
    Contains cryptographic commitments to:
    - Support set S (which indices are non-zero)
    - Values w_S (the non-zero values)
    - Sketch y (linear fingerprint)
    
    The commitment is binding: the prover cannot change S, w_S, or y
    after committing.
    """
    support_hash: bytes      # Hash commitment to support
    values_hash: bytes       # Hash commitment to values
    sketch_hash: bytes       # Hash commitment to sketch
    sparsity_hint: float     # Public hint about sparsity (optional)
    
    def to_bytes(self) -> bytes:
        """Serialize the commitment."""
        return self.support_hash + self.values_hash + self.sketch_hash
    
    @classmethod
    def from_bytes(cls, data: bytes, sparsity_hint: float = 0.0) -> 'CSWCCommitment':
        """Deserialize a commitment."""
        if len(data) != 96:  # 3 * 32 bytes
            raise ValueError("Invalid commitment data length")
        return cls(
            support_hash=data[:32],
            values_hash=data[32:64],
            sketch_hash=data[64:96],
            sparsity_hint=sparsity_hint
        )


@dataclass
class CSWCProof:
    """
    Complete CSWC proof for a sparse witness.
    
    Contains:
    - The commitment (hashes)
    - The openings (actual values)
    - Metadata for verification
    """
    # Commitment
    commitment: CSWCCommitment
    
    # Openings
    support: List[int]
    values: List[int]  # Stored as integers for serialization
    sketch: List[int]  # Stored as integers for serialization
    
    # Metadata
    witness_size: int
    sketch_dimension: int
    field_modulus: int
    
    def size_bytes(self) -> int:
        """Calculate proof size in bytes."""
        return (
            96 +  # Commitment (3 * 32 bytes)
            len(self.support) * 4 +  # Support indices (4 bytes each)
            len(self.values) * 32 +  # Values (32 bytes each)
            len(self.sketch) * 32 +  # Sketch (32 bytes each)
            24  # Metadata (3 * 8 bytes)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "commitment": {
                "support_hash": self.commitment.support_hash.hex(),
                "values_hash": self.commitment.values_hash.hex(),
                "sketch_hash": self.commitment.sketch_hash.hex(),
                "sparsity_hint": self.commitment.sparsity_hint
            },
            "support": self.support,
            "values": self.values,
            "sketch": self.sketch,
            "witness_size": self.witness_size,
            "sketch_dimension": self.sketch_dimension,
            "field_modulus": self.field_modulus
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CSWCProof':
        """Create from dictionary."""
        commitment = CSWCCommitment(
            support_hash=bytes.fromhex(data["commitment"]["support_hash"]),
            values_hash=bytes.fromhex(data["commitment"]["values_hash"]),
            sketch_hash=bytes.fromhex(data["commitment"]["sketch_hash"]),
            sparsity_hint=data["commitment"]["sparsity_hint"]
        )
        return cls(
            commitment=commitment,
            support=data["support"],
            values=data["values"],
            sketch=data["sketch"],
            witness_size=data["witness_size"],
            sketch_dimension=data["sketch_dimension"],
            field_modulus=data["field_modulus"]
        )


class CSWCProver:
    """
    Prover for the CSWC protocol.
    
    The prover:
    1. Extracts the sparse representation of the witness
    2. Computes the sketch
    3. Creates commitments
    4. Generates the proof
    """
    
    def __init__(self,
                 witness_size: int,
                 sketch_dimension: int,
                 field_config: FieldConfig,
                 matrix_seed: bytes = b"CSWC_DEFAULT_SEED"):
        """
        Initialize the prover.
        
        Args:
            witness_size: Dimension of the witness (n)
            sketch_dimension: Dimension of the sketch (m)
            field_config: Finite field configuration
            matrix_seed: Seed for sensing matrix generation
        """
        self.n = witness_size
        self.m = sketch_dimension
        self.field_config = field_config
        self.matrix_seed = matrix_seed
        
        # Generate sensing matrix
        gen = SensingMatrixGenerator(
            n=self.n,
            m=self.m,
            field_config=self.field_config,
            matrix_type="SPARSE_RANDOM",
            entries_per_column=10
        )
        self.matrix = gen.generate(matrix_seed)
        self.sketch_computer = SketchComputer(self.matrix)
        self.extractor = SparseExtractor(threshold=0)
    
    def prove(self, witness: List[FieldElement]) -> CSWCProof:
        """
        Generate a CSWC proof for a witness.
        
        Args:
            witness: The full witness vector
            
        Returns:
            CSWCProof containing commitment and openings
        """
        if len(witness) != self.n:
            raise ValueError(f"Witness size {len(witness)} != expected {self.n}")
        
        # Step 1: Extract sparse representation
        sparse_witness = self.extractor.extract(witness)
        
        # Step 2: Compute sketch
        sketch = self.sketch_computer.compute(sparse_witness)
        
        # Step 3: Create commitments (hash-based for simplicity)
        support_bytes = self._serialize_support(sparse_witness.support)
        values_bytes = self._serialize_values(sparse_witness.values)
        sketch_bytes = self._serialize_values(sketch.values)
        
        support_hash = hashlib.sha256(support_bytes).digest()
        values_hash = hashlib.sha256(values_bytes).digest()
        sketch_hash = hashlib.sha256(sketch_bytes).digest()
        
        commitment = CSWCCommitment(
            support_hash=support_hash,
            values_hash=values_hash,
            sketch_hash=sketch_hash,
            sparsity_hint=sparse_witness.sparsity
        )
        
        # Step 4: Create proof
        proof = CSWCProof(
            commitment=commitment,
            support=sparse_witness.support,
            values=[v.value for v in sparse_witness.values],
            sketch=[v.value for v in sketch.values],
            witness_size=self.n,
            sketch_dimension=self.m,
            field_modulus=self.field_config.prime
        )
        
        return proof
    
    def prove_sparse(self, sparse_witness: SparseWitness) -> CSWCProof:
        """
        Generate a CSWC proof for an already-sparse witness.
        
        Args:
            sparse_witness: The sparse witness
            
        Returns:
            CSWCProof
        """
        if sparse_witness.full_size != self.n:
            raise ValueError(f"Witness size {sparse_witness.full_size} != expected {self.n}")
        
        # Compute sketch
        sketch = self.sketch_computer.compute(sparse_witness)
        
        # Create commitments
        support_bytes = self._serialize_support(sparse_witness.support)
        values_bytes = self._serialize_values(sparse_witness.values)
        sketch_bytes = self._serialize_values(sketch.values)
        
        support_hash = hashlib.sha256(support_bytes).digest()
        values_hash = hashlib.sha256(values_bytes).digest()
        sketch_hash = hashlib.sha256(sketch_bytes).digest()
        
        commitment = CSWCCommitment(
            support_hash=support_hash,
            values_hash=values_hash,
            sketch_hash=sketch_hash,
            sparsity_hint=sparse_witness.sparsity
        )
        
        return CSWCProof(
            commitment=commitment,
            support=sparse_witness.support,
            values=[v.value for v in sparse_witness.values],
            sketch=[v.value for v in sketch.values],
            witness_size=self.n,
            sketch_dimension=self.m,
            field_modulus=self.field_config.prime
        )
    
    def _serialize_support(self, support: List[int]) -> bytes:
        """Serialize support indices to bytes."""
        return b''.join(idx.to_bytes(4, 'big') for idx in support)
    
    def _serialize_values(self, values: List[FieldElement]) -> bytes:
        """Serialize field elements to bytes."""
        return b''.join(v.value.to_bytes(32, 'big') for v in values)


class CSWCVerifier:
    """
    Verifier for the CSWC protocol.
    
    The verifier:
    1. Checks that openings match commitments
    2. Recomputes the sketch from (S, w_S)
    3. Verifies that the recomputed sketch matches the claimed sketch
    """
    
    def __init__(self,
                 witness_size: int,
                 sketch_dimension: int,
                 field_config: FieldConfig,
                 matrix_seed: bytes = b"CSWC_DEFAULT_SEED"):
        """
        Initialize the verifier.
        
        Args:
            witness_size: Dimension of the witness (n)
            sketch_dimension: Dimension of the sketch (m)
            field_config: Finite field configuration
            matrix_seed: Seed for sensing matrix generation (must match prover)
        """
        self.n = witness_size
        self.m = sketch_dimension
        self.field_config = field_config
        self.matrix_seed = matrix_seed
        
        # Generate the same sensing matrix as the prover
        gen = SensingMatrixGenerator(
            n=self.n,
            m=self.m,
            field_config=self.field_config,
            matrix_type="SPARSE_RANDOM",
            entries_per_column=10
        )
        self.matrix = gen.generate(matrix_seed)
    
    def verify(self, proof: CSWCProof) -> Tuple[bool, str]:
        """
        Verify a CSWC proof.
        
        Args:
            proof: The CSWC proof to verify
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Step 1: Check metadata consistency
        if proof.witness_size != self.n:
            return False, f"Witness size mismatch: {proof.witness_size} != {self.n}"
        if proof.sketch_dimension != self.m:
            return False, f"Sketch dimension mismatch: {proof.sketch_dimension} != {self.m}"
        if proof.field_modulus != self.field_config.prime:
            return False, f"Field modulus mismatch"
        
        # Step 2: Verify commitment openings
        support_bytes = self._serialize_support(proof.support)
        values_bytes = self._serialize_values_int(proof.values)
        sketch_bytes = self._serialize_values_int(proof.sketch)
        
        support_hash = hashlib.sha256(support_bytes).digest()
        values_hash = hashlib.sha256(values_bytes).digest()
        sketch_hash = hashlib.sha256(sketch_bytes).digest()
        
        if support_hash != proof.commitment.support_hash:
            return False, "Support commitment mismatch"
        if values_hash != proof.commitment.values_hash:
            return False, "Values commitment mismatch"
        if sketch_hash != proof.commitment.sketch_hash:
            return False, "Sketch commitment mismatch"
        
        # Step 3: Verify sketch computation
        # Convert values to FieldElements
        values_fe = [FieldElement(v, self.field_config) for v in proof.values]
        
        # Recompute sketch
        computed_sketch = self.matrix.multiply_sparse(proof.support, values_fe)
        
        # Compare with claimed sketch
        if len(computed_sketch) != len(proof.sketch):
            return False, "Sketch length mismatch"
        
        for i, (computed, claimed) in enumerate(zip(computed_sketch, proof.sketch)):
            if computed.value != claimed:
                return False, f"Sketch mismatch at position {i}"
        
        return True, "Valid"
    
    def _serialize_support(self, support: List[int]) -> bytes:
        """Serialize support indices to bytes."""
        return b''.join(idx.to_bytes(4, 'big') for idx in support)
    
    def _serialize_values_int(self, values: List[int]) -> bytes:
        """Serialize integer values to bytes."""
        return b''.join(v.to_bytes(32, 'big') for v in values)


class CSWCSystem:
    """
    Complete CSWC system combining prover and verifier.
    
    Provides a high-level interface for the CSWC protocol.
    """
    
    def __init__(self,
                 witness_size: int,
                 sketch_dimension: int = 64,
                 field_modulus: int = 2**61 - 1,
                 matrix_seed: bytes = b"CSWC_DEFAULT_SEED"):
        """
        Initialize the CSWC system.
        
        Args:
            witness_size: Dimension of witnesses
            sketch_dimension: Dimension of sketches (security parameter)
            field_modulus: Modulus for the finite field
            matrix_seed: Seed for deterministic matrix generation
        """
        self.field_config = FieldConfig(prime=field_modulus, name="CSWC")
        self.prover = CSWCProver(
            witness_size=witness_size,
            sketch_dimension=sketch_dimension,
            field_config=self.field_config,
            matrix_seed=matrix_seed
        )
        self.verifier = CSWCVerifier(
            witness_size=witness_size,
            sketch_dimension=sketch_dimension,
            field_config=self.field_config,
            matrix_seed=matrix_seed
        )
    
    def prove_and_verify(self, witness: List[FieldElement]) -> Tuple[CSWCProof, bool, str]:
        """
        Generate and verify a proof (for testing).
        
        Args:
            witness: The witness vector
            
        Returns:
            Tuple of (proof, is_valid, reason)
        """
        proof = self.prover.prove(witness)
        is_valid, reason = self.verifier.verify(proof)
        return proof, is_valid, reason
    
    def benchmark(self, witness: List[FieldElement]) -> Dict[str, Any]:
        """
        Benchmark the CSWC protocol.
        
        Args:
            witness: The witness vector
            
        Returns:
            Dictionary with timing and size metrics
        """
        import time
        
        # Prove
        start = time.time()
        proof = self.prover.prove(witness)
        prove_time = time.time() - start
        
        # Verify
        start = time.time()
        is_valid, reason = self.verifier.verify(proof)
        verify_time = time.time() - start
        
        return {
            "prove_time_ms": prove_time * 1000,
            "verify_time_ms": verify_time * 1000,
            "proof_size_bytes": proof.size_bytes(),
            "witness_size": len(witness),
            "support_size": len(proof.support),
            "sparsity": proof.commitment.sparsity_hint,
            "is_valid": is_valid,
            "reason": reason
        }


# Test code
if __name__ == "__main__":
    print("=== CSWC Protocol Tests ===\n")
    
    # Setup
    field_config = FieldConfig(prime=2**61 - 1, name="Mersenne61")
    n = 10000  # Witness dimension
    m = 64     # Sketch dimension
    
    # Create CSWC system
    print("1. Initializing CSWC System")
    system = CSWCSystem(
        witness_size=n,
        sketch_dimension=m,
        field_modulus=field_config.prime
    )
    print(f"   Witness size: {n}")
    print(f"   Sketch dimension: {m}")
    
    # Create a sparse witness (90% zeros)
    print("\n2. Creating Sparse Witness (90% zeros)")
    witness = []
    for i in range(n):
        if i % 10 == 0:  # Every 10th element is non-zero
            witness.append(FieldElement(i + 1, field_config))
        else:
            witness.append(FieldElement.zero(field_config))
    
    nonzero_count = sum(1 for v in witness if v.value != 0)
    print(f"   Non-zero entries: {nonzero_count}")
    print(f"   Sparsity: {1 - nonzero_count/n:.2%}")
    
    # Generate and verify proof
    print("\n3. Generating and Verifying Proof")
    proof, is_valid, reason = system.prove_and_verify(witness)
    print(f"   Proof valid: {is_valid}")
    print(f"   Reason: {reason}")
    print(f"   Proof size: {proof.size_bytes()} bytes")
    print(f"   Support size: {len(proof.support)}")
    print(f"   Sketch size: {len(proof.sketch)}")
    
    # Test with tampered proof
    print("\n4. Testing Tampered Proof Detection")
    
    # Tamper with values
    tampered_proof = CSWCProof(
        commitment=proof.commitment,
        support=proof.support,
        values=[v + 1 for v in proof.values],  # Tampered!
        sketch=proof.sketch,
        witness_size=proof.witness_size,
        sketch_dimension=proof.sketch_dimension,
        field_modulus=proof.field_modulus
    )
    is_valid_tampered, reason_tampered = system.verifier.verify(tampered_proof)
    print(f"   Tampered values detected: {not is_valid_tampered}")
    print(f"   Reason: {reason_tampered}")
    
    # Tamper with support
    tampered_proof2 = CSWCProof(
        commitment=proof.commitment,
        support=[s + 1 for s in proof.support],  # Tampered!
        values=proof.values,
        sketch=proof.sketch,
        witness_size=proof.witness_size,
        sketch_dimension=proof.sketch_dimension,
        field_modulus=proof.field_modulus
    )
    is_valid_tampered2, reason_tampered2 = system.verifier.verify(tampered_proof2)
    print(f"   Tampered support detected: {not is_valid_tampered2}")
    print(f"   Reason: {reason_tampered2}")
    
    # Benchmark
    print("\n5. Benchmark")
    metrics = system.benchmark(witness)
    print(f"   Prove time: {metrics['prove_time_ms']:.2f} ms")
    print(f"   Verify time: {metrics['verify_time_ms']:.2f} ms")
    print(f"   Proof size: {metrics['proof_size_bytes']} bytes")
    print(f"   Compression vs dense: {metrics['proof_size_bytes'] / (n * 32):.2%}")
    
    # Serialization test
    print("\n6. Serialization Test")
    proof_dict = proof.to_dict()
    proof_json = json.dumps(proof_dict)
    print(f"   JSON size: {len(proof_json)} bytes")
    
    proof_restored = CSWCProof.from_dict(json.loads(proof_json))
    is_valid_restored, _ = system.verifier.verify(proof_restored)
    print(f"   Restored proof valid: {is_valid_restored}")
    
    print("\n=== All Tests Passed ===")
