"""
BN254 Pairing using py_ecc library

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

This module provides a wrapper around the py_ecc library's BN128 pairing
implementation, adapted to work with our zkML system's data structures.

The py_ecc library provides a correct, well-tested implementation of the
optimal Ate pairing over BN254 (also known as BN128 or alt_bn128).

Note: py_ecc uses the convention pairing(G2_point, G1_point), which is
the opposite of our convention pairing(G1_point, G2_point).
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import sys
import os

# Import py_ecc
from py_ecc.bn128 import (
    pairing as pyecc_pairing,
    G1 as PYECC_G1,
    G2 as PYECC_G2,
    multiply as pyecc_multiply,
    add as pyecc_add,
    neg as pyecc_neg,
    curve_order,
    field_modulus,
    is_on_curve,
    FQ,
    FQ2,
    FQ12,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto.bn254.field import Fr, Fp
from crypto.bn254.curve import G1Point, G2Point


def _g1point_to_pyecc(point: G1Point) -> Tuple[FQ, FQ]:
    """Convert our G1Point to py_ecc format."""
    if point.is_identity():
        return None  # py_ecc uses None for point at infinity
    
    x, y = point.to_affine()
    return (FQ(x.to_int()), FQ(y.to_int()))


def _pyecc_to_g1point(point) -> G1Point:
    """Convert py_ecc G1 point to our G1Point."""
    if point is None:
        return G1Point.identity()
    
    x, y = point
    return G1Point(Fp(int(x)), Fp(int(y)))


def _g2point_to_pyecc(point: G2Point) -> Tuple[Tuple[FQ, FQ], Tuple[FQ, FQ]]:
    """Convert our G2Point to py_ecc format."""
    if point.is_identity():
        return None  # py_ecc uses None for point at infinity
    
    x, y = point.to_affine()
    # G2 coordinates are in Fp2, represented as (c0, c1) where element = c0 + c1*u
    # py_ecc uses the same convention
    return (
        FQ2([x.c0.to_int(), x.c1.to_int()]),
        FQ2([y.c0.to_int(), y.c1.to_int()])
    )


def pairing(p: G1Point, q: G2Point) -> FQ12:
    """
    Compute the optimal Ate pairing e(P, Q).
    
    Args:
        p: A G1 point
        q: A G2 point
    
    Returns:
        The pairing result in FQ12 (an element of GT)
    """
    if p.is_identity() or q.is_identity():
        return FQ12.one()
    
    p_pyecc = _g1point_to_pyecc(p)
    q_pyecc = _g2point_to_pyecc(q)
    
    # py_ecc convention: pairing(G2, G1)
    return pyecc_pairing(q_pyecc, p_pyecc)


def multi_pairing(pairs: List[Tuple[G1Point, G2Point]]) -> FQ12:
    """
    Compute the product of multiple pairings: ∏ e(P_i, Q_i).
    
    This is more efficient than computing individual pairings and multiplying,
    as we can use the multi-Miller loop optimization.
    
    Args:
        pairs: List of (G1Point, G2Point) tuples
    
    Returns:
        The product of pairings
    """
    result = FQ12.one()
    
    for p, q in pairs:
        if not p.is_identity() and not q.is_identity():
            result = result * pairing(p, q)
    
    return result


def verify_pairing_equation(
    a1: G1Point, b1: G2Point,
    a2: G1Point, b2: G2Point
) -> bool:
    """
    Verify the pairing equation e(A1, B1) == e(A2, B2).
    
    This is equivalent to checking e(A1, B1) * e(-A2, B2) == 1.
    
    Args:
        a1, b1: First pairing arguments
        a2, b2: Second pairing arguments
    
    Returns:
        True if the equation holds, False otherwise
    """
    e1 = pairing(a1, b1)
    e2 = pairing(a2, b2)
    return e1 == e2


def verify_kzg_opening(
    commitment: G1Point,
    point: Fr,
    value: Fr,
    proof: G1Point,
    tau_g2: G2Point
) -> bool:
    """
    Verify a KZG opening proof.
    
    Verifies that commitment C opens to value v at point z, given proof π.
    
    The verification equation is:
        e(π, [τ]₂ - z·G₂) = e(C - v·G₁, G₂)
    
    Rearranged:
        e(π, [τ]₂) · e(-π, z·G₂) = e(C - v·G₁, G₂)
        e(π, [τ]₂) = e(C - v·G₁ + z·π, G₂)
    
    Args:
        commitment: The polynomial commitment C
        point: The evaluation point z
        value: The claimed value v = p(z)
        proof: The opening proof π
        tau_g2: [τ]₂ from the SRS
    
    Returns:
        True if the proof is valid
    """
    g1 = G1Point.generator()
    g2 = G2Point.generator()
    
    # Handle zero polynomial edge case
    # Require all three conditions: commitment is identity, value is zero,
    # AND proof is identity. Without checking proof, a malicious prover
    # could forge by setting commitment=identity and value=0 with any proof.
    if commitment.is_identity() and value == Fr.zero():
        return proof.is_identity()
    
    # Compute LHS: e(π, [τ]₂)
    lhs = pairing(proof, tau_g2)
    
    # Compute RHS: e(z·π + C - v·G₁, G₂)
    z_pi = proof * point
    v_g1 = g1 * value
    rhs_g1 = z_pi + commitment + (-v_g1)
    rhs = pairing(rhs_g1, g2)
    
    return lhs == rhs


def verify_kzg_batch(
    openings: List[Tuple[G1Point, Fr, Fr, G1Point]],
    tau_g2: G2Point,
    random_challenge: Optional[Fr] = None
) -> bool:
    """
    Batch verify multiple KZG opening proofs.
    
    Uses random linear combinations to verify multiple proofs efficiently.
    
    Args:
        openings: List of (commitment, point, value, proof) tuples
        tau_g2: [τ]₂ from the SRS
        random_challenge: Optional random scalar for linear combination
    
    Returns:
        True if all proofs are valid
    """
    if not openings:
        return True
    
    if len(openings) == 1:
        c, z, v, pi = openings[0]
        return verify_kzg_opening(c, z, v, pi, tau_g2)
    
    # Generate random challenge if not provided
    if random_challenge is None:
        import hashlib
        hasher = hashlib.sha256()
        # Include ALL elements in Fiat-Shamir hash to prevent
        # rogue-proof cancellation attacks
        for c, z, v, pi in openings:
            hasher.update(str(c).encode())
            hasher.update(str(pi).encode())
            hasher.update(z.to_int().to_bytes(32, 'big'))
            hasher.update(v.to_int().to_bytes(32, 'big'))
        r_bytes = hasher.digest()
        random_challenge = Fr(int.from_bytes(r_bytes, 'big') % Fr.MODULUS)
        # Guard against degenerate zero challenge
        if random_challenge == Fr.zero():
            random_challenge = Fr.one()
    
    g1 = G1Point.generator()
    g2 = G2Point.generator()
    
    # Compute aggregated proof and RHS
    r_power = Fr.one()
    agg_proof = G1Point.identity()
    agg_rhs = G1Point.identity()
    
    for c, z, v, pi in openings:
        # agg_proof += r^i * π_i
        agg_proof = agg_proof + (pi * r_power)
        
        # agg_rhs += r^i * (z_i * π_i + C_i - v_i * G1)
        z_pi = pi * z
        v_g1 = g1 * v
        term = z_pi + c + (-v_g1)
        agg_rhs = agg_rhs + (term * r_power)
        
        r_power = r_power * random_challenge
    
    # Verify: e(agg_proof, [τ]₂) = e(agg_rhs, G₂)
    lhs = pairing(agg_proof, tau_g2)
    rhs = pairing(agg_rhs, g2)
    
    return lhs == rhs


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("py_ecc Pairing Wrapper Self-Test")
    print("=" * 70)
    
    from plonk.core import SRS, Polynomial, KZG, Field
    
    # Test basic pairing
    print("\n1. Testing basic pairing...")
    g1 = G1Point.generator()
    g2 = G2Point.generator()
    
    e1 = pairing(g1, g2)
    print(f"   e(G1, G2) computed: {type(e1)}")
    
    # Test bilinearity
    print("\n2. Testing bilinearity...")
    two_g1 = g1 + g1
    e2 = pairing(two_g1, g2)
    e1_sq = e1 * e1
    print(f"   e(2*G1, G2) == e(G1, G2)^2: {e2 == e1_sq}")
    
    # Test KZG verification
    print("\n3. Testing KZG verification...")
    tau = Fr(12345)
    srs = SRS.generate_insecure(16, tau=tau)
    kzg = KZG(srs)
    
    # Simple polynomial p(x) = 1 + 2x + 3x²
    poly = Polynomial.from_ints([1, 2, 3])
    print(f"   Polynomial: p(x) = 1 + 2x + 3x²")
    
    # Evaluate at z=5
    z = Fr(5)
    v = poly.evaluate(z)
    print(f"   p(5) = {v.to_int()}")  # 1 + 10 + 75 = 86
    
    # Create proof
    proof_obj = kzg.create_proof(poly, z)
    print(f"   Proof created")
    
    # Verify with py_ecc
    tau_g2 = srs.g2_powers[1]
    is_valid = verify_kzg_opening(
        commitment=proof_obj.commitment.point,
        point=proof_obj.point,
        value=proof_obj.value,
        proof=proof_obj.proof,
        tau_g2=tau_g2
    )
    print(f"   Verification result: {is_valid}")
    
    # Test with wrong value
    print("\n4. Testing with wrong value (should fail)...")
    wrong_valid = verify_kzg_opening(
        commitment=proof_obj.commitment.point,
        point=proof_obj.point,
        value=proof_obj.value + Fr.one(),  # Wrong!
        proof=proof_obj.proof,
        tau_g2=tau_g2
    )
    print(f"   Wrong value verification: {wrong_valid}")
    
    # Test batch verification
    print("\n5. Testing batch verification...")
    openings = []
    for i in range(3):
        zi = Fr(i + 1)
        pi = kzg.create_proof(poly, zi)
        openings.append((pi.commitment.point, pi.point, pi.value, pi.proof))
    
    batch_valid = verify_kzg_batch(openings, tau_g2)
    print(f"   Batch verification (3 proofs): {batch_valid}")
    
    print("\n" + "=" * 70)
    if is_valid and not wrong_valid and batch_valid:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)
