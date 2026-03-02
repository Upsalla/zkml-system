"""
Batch Proof Aggregation for zkML

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

This module implements batch proof aggregation techniques to reduce
verification costs when multiple proofs need to be verified.

Performance Characteristics:
----------------------------
- Single proof: 2 pairings
- n proofs naive: 2n pairings
- n proofs batched: 2 pairings + n scalar multiplications

For n=10 proofs, this is approximately 5x faster.
For n=100 proofs, this is approximately 50x faster.

Security:
---------
Uses random linear combinations with cryptographic randomness.
Soundness error is negligible (2^-128 for 128-bit challenges).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import hashlib
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.bn254.field import Fr
from crypto.bn254.curve import G1Point, G2Point


@dataclass
class BatchVerificationResult:
    """Result of batch verification."""
    is_valid: bool
    num_proofs: int
    verification_time_ms: float
    pairings_saved: int
    reason: str


class BatchAggregator:
    """
    Aggregates multiple KZG proofs for efficient batch verification.
    
    Uses random linear combinations to reduce the number of pairing
    operations from 2n to 2, where n is the number of proofs.
    """
    
    def __init__(self, tau_g2: G2Point):
        """
        Initialize the batch aggregator.
        
        Args:
            tau_g2: The [τ]₂ element from the SRS
        """
        self.tau_g2 = tau_g2
        self.g2 = G2Point.generator()
    
    def _generate_challenges(
        self,
        proofs: List[Tuple[G1Point, Fr, Fr, G1Point]],
        seed: Optional[bytes] = None
    ) -> List[Fr]:
        """
        Generate random challenges for batch verification.
        
        Uses Fiat-Shamir to derive challenges from the proofs themselves,
        ensuring the verifier cannot choose challenges after seeing proofs.
        
        Args:
            proofs: List of (commitment, point, value, proof) tuples
            seed: Optional additional randomness
        
        Returns:
            List of random challenges, one per proof
        """
        hasher = hashlib.sha256()
        
        # Add seed if provided
        if seed:
            hasher.update(seed)
        
        # Hash all proof data
        for commitment, point, value, proof in proofs:
            if not commitment.is_identity():
                cx, cy = commitment.to_affine()
                hasher.update(cx.to_int().to_bytes(32, 'big'))
                hasher.update(cy.to_int().to_bytes(32, 'big'))
            hasher.update(point.to_int().to_bytes(32, 'big'))
            hasher.update(value.to_int().to_bytes(32, 'big'))
            if not proof.is_identity():
                px, py = proof.to_affine()
                hasher.update(px.to_int().to_bytes(32, 'big'))
                hasher.update(py.to_int().to_bytes(32, 'big'))
        
        # Generate challenges
        challenges = []
        base_hash = hasher.digest()
        
        for i in range(len(proofs)):
            h = hashlib.sha256(base_hash + i.to_bytes(4, 'big')).digest()
            challenge = Fr(int.from_bytes(h, 'big') % Fr.MODULUS)
            challenges.append(challenge)
        
        return challenges
    
    def aggregate(
        self,
        proofs: List[Tuple[G1Point, Fr, Fr, G1Point]]
    ) -> Tuple[G1Point, G1Point, List[Fr]]:
        """
        Aggregate multiple proofs into a single verification instance.
        
        Given proofs [(C_i, z_i, v_i, π_i)], computes:
        - Aggregated proof: Π = Σ r_i · π_i
        - Aggregated commitment: C' = Σ r_i · (z_i·π_i + C_i - v_i·G₁)
        
        Args:
            proofs: List of (commitment, point, value, proof) tuples
        
        Returns:
            (aggregated_proof, aggregated_commitment, challenges)
        """
        if not proofs:
            return G1Point.identity(), G1Point.identity(), []
        
        challenges = self._generate_challenges(proofs)
        g1 = G1Point.generator()
        
        # Compute aggregated values
        agg_proof = G1Point.identity()
        agg_commitment = G1Point.identity()
        
        for i, (commitment, point, value, proof) in enumerate(proofs):
            r = challenges[i]
            
            # Aggregated proof: Σ r_i · π_i
            agg_proof = agg_proof + (proof * r)
            
            # Aggregated commitment: Σ r_i · (z_i·π_i + C_i - v_i·G₁)
            term = (proof * point) + commitment - (g1 * value)
            agg_commitment = agg_commitment + (term * r)
        
        return agg_proof, agg_commitment, challenges
    
    def verify_batch(
        self,
        proofs: List[Tuple[G1Point, Fr, Fr, G1Point]]
    ) -> BatchVerificationResult:
        """
        Verify multiple proofs in a single batch.
        
        Uses only 2 pairing operations regardless of the number of proofs.
        
        Args:
            proofs: List of (commitment, point, value, proof) tuples
        
        Returns:
            BatchVerificationResult with detailed information
        """
        start_time = time.time()
        
        if not proofs:
            return BatchVerificationResult(
                is_valid=True,
                num_proofs=0,
                verification_time_ms=0,
                pairings_saved=0,
                reason="Empty batch trivially verified"
            )
        
        n = len(proofs)
        
        # Aggregate proofs
        agg_proof, agg_commitment, challenges = self.aggregate(proofs)
        
        # Verify: e(Π, [τ]₂) = e(C', G₂)
        from crypto.bn254.pairing_pyecc import pairing
        
        lhs = pairing(agg_proof, self.tau_g2)
        rhs = pairing(agg_commitment, self.g2)
        
        is_valid = (lhs == rhs)
        
        elapsed_ms = (time.time() - start_time) * 1000
        pairings_saved = 2 * n - 2  # Would have needed 2n, used 2
        
        return BatchVerificationResult(
            is_valid=is_valid,
            num_proofs=n,
            verification_time_ms=elapsed_ms,
            pairings_saved=pairings_saved,
            reason="Batch verification " + ("succeeded" if is_valid else "failed")
        )


class ProofAggregator:
    """
    High-level interface for proof aggregation in zkML.
    
    Supports both KZG proofs and full PLONK proofs.
    """
    
    def __init__(self, srs):
        """
        Initialize with SRS.
        
        Args:
            srs: The structured reference string
        """
        self.srs = srs
        self.batch_verifier = BatchAggregator(srs.g2_powers[1])
    
    def aggregate_kzg_proofs(
        self,
        proofs: List
    ) -> BatchVerificationResult:
        """
        Aggregate and verify multiple KZG proofs.
        
        Args:
            proofs: List of KZGProof objects
        
        Returns:
            BatchVerificationResult
        """
        # Convert KZGProof objects to tuples
        proof_tuples = [
            (p.commitment.point, p.point, p.value, p.proof)
            for p in proofs
        ]
        
        return self.batch_verifier.verify_batch(proof_tuples)


def benchmark_batch_verification():
    """Benchmark batch verification vs individual verification."""
    from plonk.core import SRS, Polynomial, KZG
    
    print("=" * 60)
    print("Batch Verification Benchmark")
    print("=" * 60)
    
    # Setup
    tau = Fr(12345)
    srs = SRS.generate_insecure(32, tau=tau)
    kzg = KZG(srs)
    aggregator = ProofAggregator(srs)
    
    # Generate test proofs
    batch_sizes = [1, 2, 5, 10]
    
    for n in batch_sizes:
        print(f"\nBatch size: {n}")
        
        # Generate proofs
        proofs = []
        for i in range(n):
            poly = Polynomial.from_ints([i+1, i+2, i+3])
            point = Fr(i + 10)
            proof = kzg.create_proof(poly, point)
            proofs.append(proof)
        
        # Individual verification
        start = time.time()
        for p in proofs:
            kzg.verify(p)
        individual_time = (time.time() - start) * 1000
        
        # Batch verification
        result = aggregator.aggregate_kzg_proofs(proofs)
        
        print(f"  Individual: {individual_time:.1f}ms")
        print(f"  Batch:      {result.verification_time_ms:.1f}ms")
        print(f"  Speedup:    {individual_time/result.verification_time_ms:.1f}x")
        print(f"  Pairings saved: {result.pairings_saved}")
        print(f"  Valid: {result.is_valid}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Batch Aggregation Self-Test")
    print("=" * 40)
    
    from plonk.core import SRS, Polynomial, KZG
    
    # Setup
    tau = Fr(12345)
    srs = SRS.generate_insecure(16, tau=tau)
    kzg = KZG(srs)
    
    # Create test proofs
    proofs = []
    for i in range(3):
        poly = Polynomial.from_ints([i+1, i+2, i+3])
        point = Fr(i + 5)
        proof = kzg.create_proof(poly, point)
        proofs.append(proof)
        print(f"Proof {i+1}: p({point.to_int()}) = {proof.value.to_int()}")
    
    # Batch verify
    aggregator = ProofAggregator(srs)
    result = aggregator.aggregate_kzg_proofs(proofs)
    
    print(f"\nBatch verification:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Time: {result.verification_time_ms:.1f}ms")
    print(f"  Pairings saved: {result.pairings_saved}")
    
    # Test with invalid proof
    print("\nTesting with invalid proof...")
    from plonk.core import KZGProof
    invalid_proof = KZGProof(
        commitment=proofs[0].commitment,
        point=proofs[0].point,
        value=proofs[0].value + Fr.one(),  # Wrong value
        proof=proofs[0].proof
    )
    proofs_with_invalid = proofs + [invalid_proof]
    result_invalid = aggregator.aggregate_kzg_proofs(proofs_with_invalid)
    print(f"  Valid: {result_invalid.is_valid} (expected: False)")
    
    print("\n" + "=" * 40)
    print("BATCH AGGREGATION TEST COMPLETE!")
