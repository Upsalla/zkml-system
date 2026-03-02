"""
KZG Polynomial Commitment Scheme - Production Verifier

Author: David Weyhe
Date: 27. Januar 2026
Version: 2.0

This module implements the complete KZG verification with proper pairing checks
using the py_ecc library for cryptographically correct BN254 pairings.

Mathematical Foundation:
------------------------
KZG commitments use the following verification equation:

    e(C - v·G₁, G₂) = e(π, [τ]₂ - z·G₂)

Where:
- C = Commitment to polynomial p(X) = Σ cᵢ·[τⁱ]₁
- v = p(z), the claimed evaluation
- π = Commitment to quotient q(X) = (p(X) - v) / (X - z)
- z = The evaluation point
- [τ]₂ = τ·G₂ from the SRS (tau is unknown)
- G₁, G₂ = Generators of the respective groups

The equation can be rewritten for efficient verification as:
    e(π, [τ]₂) = e(z·π + C - v·G₁, G₂)

Security Considerations:
------------------------
1. All inputs must be validated to be on the curve
2. The SRS must come from a trusted setup ceremony
3. Batch verification should use random linear combinations
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkml_system.crypto.bn254.fr_adapter import Fr
from crypto.bn254.field import Fp
from crypto.bn254.curve import G1Point, G2Point
from crypto.bn254.pairing_pyecc import (
    pairing,
    verify_kzg_opening,
    verify_kzg_batch,
    verify_pairing_equation
)


@dataclass
class VerificationResult:
    """Result of a KZG verification."""
    is_valid: bool
    reason: str
    computation_time_ms: Optional[float] = None


class KZGVerifier:
    """
    Production-grade KZG polynomial commitment verifier.
    
    This class provides cryptographically sound verification of KZG opening proofs
    using the BN254 optimal Ate pairing via py_ecc.
    
    Attributes:
        g1: Generator of G1
        g2: Generator of G2
        tau_g2: [τ]₂ from the SRS (second element of g2_powers)
    """
    
    def __init__(self, g2_powers: List[G2Point]):
        """
        Initialize the verifier with SRS elements.
        
        Args:
            g2_powers: At least [G₂, τ·G₂] from the SRS
        """
        if len(g2_powers) < 2:
            raise ValueError("g2_powers must contain at least [G2, tau*G2]")
        
        self.g1 = G1Point.generator()
        self.g2 = g2_powers[0]
        self.tau_g2 = g2_powers[1]
    
    def verify_opening(
        self,
        commitment: G1Point,
        point: Fr,
        value: Fr,
        proof: G1Point
    ) -> VerificationResult:
        """
        Verify a KZG opening proof.
        
        Verifies that commitment C opens to value v at point z, given proof π.
        
        The verification equation is:
            e(π, [τ]₂) = e(z·π + C - v·G₁, G₂)
        
        Args:
            commitment: The polynomial commitment C
            point: The evaluation point z
            value: The claimed value v = p(z)
            proof: The opening proof π
        
        Returns:
            VerificationResult with validity and reason
        """
        start_time = time.time()
        
        # Input validation
        if commitment.is_identity():
            # Zero polynomial case - value must be zero
            if value != Fr.zero():
                return VerificationResult(
                    is_valid=False,
                    reason="Zero commitment but non-zero value claimed"
                )
            return VerificationResult(
                is_valid=True,
                reason="Zero polynomial verified"
            )
        
        # Use py_ecc for verification
        is_valid = verify_kzg_opening(
            commitment=commitment,
            point=point,
            value=value,
            proof=proof,
            tau_g2=self.tau_g2
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if is_valid:
            return VerificationResult(
                is_valid=True,
                reason="Pairing equation verified successfully",
                computation_time_ms=elapsed_ms
            )
        else:
            return VerificationResult(
                is_valid=False,
                reason="Pairing equation failed: e(π, [τ]₂) ≠ e(z·π + C - v·G₁, G₂)",
                computation_time_ms=elapsed_ms
            )
    
    def verify_batch(
        self,
        openings: List[Tuple[G1Point, Fr, Fr, G1Point]],
        random_challenge: Optional[Fr] = None
    ) -> VerificationResult:
        """
        Batch verify multiple KZG opening proofs.
        
        Uses random linear combinations to verify multiple proofs with fewer
        pairing computations.
        
        Args:
            openings: List of (commitment, point, value, proof) tuples
            random_challenge: Optional random scalar for linear combination
        
        Returns:
            VerificationResult with validity and reason
        """
        start_time = time.time()
        
        if not openings:
            return VerificationResult(
                is_valid=True,
                reason="Empty batch trivially verified"
            )
        
        if len(openings) == 1:
            c, z, v, pi = openings[0]
            return self.verify_opening(c, z, v, pi)
        
        is_valid = verify_kzg_batch(openings, self.tau_g2, random_challenge)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if is_valid:
            return VerificationResult(
                is_valid=True,
                reason=f"Batch verification of {len(openings)} proofs succeeded",
                computation_time_ms=elapsed_ms
            )
        else:
            return VerificationResult(
                is_valid=False,
                reason=f"Batch verification of {len(openings)} proofs failed",
                computation_time_ms=elapsed_ms
            )


# =============================================================================
# Integration with core.py
# =============================================================================

def patch_kzg_class():
    """
    Patch the KZG class in core.py to use production verification.
    
    This function modifies the KZG class to use the KZGVerifier for
    cryptographically sound verification.
    """
    from plonk.core import KZG, KZGProof, SRS
    
    # Store original __init__
    original_init = KZG.__init__
    
    def new_init(self, srs: SRS):
        original_init(self, srs)
        self._verifier = KZGVerifier(srs.g2_powers)
    
    def new_verify(self, proof: KZGProof) -> bool:
        """
        Verify a KZG opening proof using pairing check.
        
        This is the production-grade verification that uses the BN254
        optimal Ate pairing to verify the opening proof.
        """
        result = self._verifier.verify_opening(
            commitment=proof.commitment.point,
            point=proof.point,
            value=proof.value,
            proof=proof.proof
        )
        return result.is_valid
    
    def new_verify_with_reason(self, proof: KZGProof) -> VerificationResult:
        """
        Verify with detailed result including reason and timing.
        """
        return self._verifier.verify_opening(
            commitment=proof.commitment.point,
            point=proof.point,
            value=proof.value,
            proof=proof.proof
        )
    
    # Apply patches
    KZG.__init__ = new_init
    KZG.verify = new_verify
    KZG.verify_with_reason = new_verify_with_reason
    
    return KZG


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KZG Verifier Self-Test (py_ecc backend)")
    print("=" * 70)
    
    from plonk.core import SRS, Polynomial, KZG, Field
    
    # Generate test SRS
    print("\n1. Generating test SRS...")
    tau = Fr(12345)
    srs = SRS.generate_insecure(16, tau=tau)
    print(f"   SRS max degree: {srs.max_degree}")
    
    # Create verifier
    print("\n2. Initializing KZG Verifier...")
    verifier = KZGVerifier(srs.g2_powers)
    print("   Verifier initialized successfully")
    
    # Create KZG instance
    kzg = KZG(srs)
    
    # Test polynomial
    print("\n3. Creating test polynomial...")
    poly = Polynomial.from_ints([1, 2, 3, 4])  # 1 + 2x + 3x² + 4x³
    print(f"   Polynomial: 1 + 2x + 3x² + 4x³")
    
    # Create opening proof
    print("\n4. Creating opening proof at z=5...")
    point = Fr(5)
    proof = kzg.create_proof(poly, point)
    expected_value = poly.evaluate(point)
    print(f"   Value: p(5) = {expected_value.to_int()}")
    
    # Verify with production verifier
    print("\n5. Verifying with production verifier...")
    result = verifier.verify_opening(
        commitment=proof.commitment.point,
        point=proof.point,
        value=proof.value,
        proof=proof.proof
    )
    print(f"   Valid: {result.is_valid}")
    print(f"   Reason: {result.reason}")
    if result.computation_time_ms:
        print(f"   Time: {result.computation_time_ms:.2f}ms")
    
    # Test with wrong value
    print("\n6. Testing with wrong value (should fail)...")
    wrong_result = verifier.verify_opening(
        commitment=proof.commitment.point,
        point=proof.point,
        value=proof.value + Fr.one(),
        proof=proof.proof
    )
    print(f"   Valid: {wrong_result.is_valid}")
    print(f"   Reason: {wrong_result.reason}")
    
    # Batch verification
    print("\n7. Testing batch verification...")
    openings = []
    for i in range(3):
        z = Fr(i + 1)
        p = kzg.create_proof(poly, z)
        openings.append((p.commitment.point, p.point, p.value, p.proof))
    
    batch_result = verifier.verify_batch(openings)
    print(f"   Batch size: {len(openings)}")
    print(f"   Valid: {batch_result.is_valid}")
    if batch_result.computation_time_ms:
        print(f"   Time: {batch_result.computation_time_ms:.2f}ms")
    
    # Summary
    print("\n" + "=" * 70)
    all_passed = result.is_valid and not wrong_result.is_valid and batch_result.is_valid
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)
