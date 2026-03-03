"""
KZG Polynomial Commitment Scheme over BN254.

Provides:
    - TrustedSetup: generates SRS (structured reference string)
    - commit(poly, srs) → G1 commitment
    - create_proof(poly, point, srs) → opening proof (G1 witness)
    - verify_opening(commitment, point, value, proof, srs) → bool

Uses the existing BN254 primitives: G1Point, G2Point, pairing.

Security: 128-bit discrete-log security on BN254.
Performance: Pure Python — ~1ms per scalar mul on typical hardware.
             Acceptable for correctness verification, not production.

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point, G2Point
from zkml_system.crypto.bn254.pairing import verify_pairing_equation
from zkml_system.plonk.polynomial import Polynomial


@dataclass
class TrustedSetup:
    """
    Structured Reference String (SRS) for KZG.

    Generated via a trusted setup ceremony with toxic waste τ.
    In production, this would use a multi-party computation (MPC).
    Currently uses cryptographically random τ (secure for single-party).

    Contents:
        g1_powers: [G₁, τ·G₁, τ²·G₁, ..., τⁿ·G₁]
        g2_tau:    τ·G₂ (for pairing verification)
        g1:        G₁ generator
        g2:        G₂ generator
        max_degree: maximum polynomial degree supported
    """
    g1_powers: List[G1Point]
    g2_tau: G2Point
    g1: G1Point
    g2: G2Point
    max_degree: int

    @classmethod
    def generate(cls, max_degree: int, tau: Fr = None) -> TrustedSetup:
        """
        Generate a trusted setup with toxic waste τ.

        Args:
            max_degree: Maximum polynomial degree to support.
            tau: The secret evaluation point. If None, generates
                 deterministically for testing (NOT secure).

        Returns:
            TrustedSetup with SRS for KZG commitments.
        """
        if tau is None:
            # Cryptographically random tau — secure for non-MPC usage
            import os
            tau = Fr(int.from_bytes(os.urandom(32), 'big'))

        g1 = G1Point.generator()
        g2 = G2Point.generator()

        # Compute [τ⁰·G₁, τ¹·G₁, ..., τⁿ·G₁]
        g1_powers = []
        tau_power = Fr.one()
        for i in range(max_degree + 1):
            g1_powers.append(g1 * tau_power)
            tau_power = tau_power * tau

        # Compute τ·G₂
        g2_tau = g2 * tau

        return cls(
            g1_powers=g1_powers,
            g2_tau=g2_tau,
            g1=g1,
            g2=g2,
            max_degree=max_degree,
        )


def commit(poly: Polynomial, srs: TrustedSetup) -> G1Point:
    """
    Commit to a polynomial: C = Σ cᵢ · [τⁱ]₁.

    Uses Rust MSM (Pippenger) when available for ~10-50x speedup,
    falls back to Python scalar-mul loop.

    Args:
        poly: The polynomial to commit to.
        srs: The trusted setup.

    Returns:
        G1 commitment point.

    Raises:
        ValueError: If polynomial degree exceeds SRS capacity.
    """
    if poly.degree() > srs.max_degree:
        raise ValueError(
            f"Polynomial degree {poly.degree()} exceeds SRS max {srs.max_degree}"
        )

    # Try Rust MSM path
    try:
        return _commit_rust_msm(poly, srs)
    except (ImportError, RuntimeError):
        pass

    # Python fallback: MSM via loop
    result = G1Point.identity()
    for i, coeff in enumerate(poly.coeffs):
        if not coeff.is_zero():
            result = result + srs.g1_powers[i] * coeff
    return result


def _commit_rust_msm(poly: Polynomial, srs: TrustedSetup) -> G1Point:
    """Rust MSM path for commit(). Converts at type boundaries."""
    from zkml_rust import RustG1Point, RustFr

    # Convert non-zero coefficients and matching SRS points to Rust types
    rust_points = []
    rust_scalars = []
    for i, coeff in enumerate(poly.coeffs):
        if not coeff.is_zero():
            # Convert Python G1Point → RustG1Point via affine coords
            py_pt = srs.g1_powers[i]
            aff = py_pt.to_affine()  # returns (Fp, Fp)
            rust_pt = RustG1Point.from_affine(aff[0].to_int(), aff[1].to_int())
            rust_points.append(rust_pt)
            # Convert Fr → RustFr
            rust_scalars.append(RustFr(coeff.to_int()))

    if not rust_points:
        return G1Point.identity()

    # MSM in Rust (Pippenger)
    rust_result = RustG1Point.msm(rust_points, rust_scalars)

    # Convert RustG1Point back to Python G1Point
    aff = rust_result.to_affine()  # returns (int, int) or None
    if aff is None:
        return G1Point.identity()
    return G1Point.from_affine(aff[0], aff[1])


def create_proof(
    poly: Polynomial, point: Fr, srs: TrustedSetup
) -> Tuple[G1Point, Fr]:
    """
    Create a KZG opening proof at a point.

    Given polynomial p(X) and evaluation point z:
    1. Compute value y = p(z)
    2. Compute quotient q(X) = (p(X) - y) / (X - z)
    3. Compute witness W = commit(q, srs)

    The quotient exists iff p(z) = y (polynomial remainder theorem).

    Args:
        poly: The committed polynomial.
        point: The evaluation point z.
        srs: The trusted setup.

    Returns:
        Tuple of (witness W, evaluation y).
    """
    # Evaluate
    value = poly.evaluate(point)

    # Compute numerator: p(X) - y
    numerator = poly - Polynomial([value])

    # Divide by (X - z)
    quotient, remainder = numerator.divide_by_linear(point)

    # Sanity check: remainder should be zero
    if not remainder.is_zero():
        raise RuntimeError(
            f"KZG proof failed: non-zero remainder {remainder}. "
            f"This indicates a bug in polynomial arithmetic."
        )

    # Commit to quotient
    witness = commit(quotient, srs)

    return witness, value


def verify_opening(
    commitment: G1Point,
    point: Fr,
    value: Fr,
    witness: G1Point,
    srs: TrustedSetup,
) -> bool:
    """
    Verify a KZG opening proof.

    Check the pairing equation:
        e(C - [y]₁, G₂) == e(W, [τ]₂ - [z]₂)

    Which is equivalent to:
        e(C - [y]₁, G₂) · e(-W, [τ]₂ - [z]₂) == 1

    Rearranged for a single multi-pairing check:
        e(C - [y]₁ + z·W, G₂) == e(W, [τ]₂)

    Args:
        commitment: The polynomial commitment C.
        point: The evaluation point z.
        value: The claimed evaluation y = p(z).
        witness: The opening proof W.
        srs: The trusted setup.

    Returns:
        True if the proof is valid.
    """
    # LHS pair: C - y·G₁
    lhs = commitment - srs.g1 * value

    # RHS pair: [τ]₂ - z·G₂
    rhs = srs.g2_tau - srs.g2 * point

    # Pairing check: e(C - [y]₁, G₂) == e(W, [τ]₂ - [z]₂)
    return verify_pairing_equation(lhs, srs.g2, witness, rhs)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("KZG Polynomial Commitment Self-Test")
    print("=" * 60)

    # Generate SRS
    t0 = time.time()
    srs = TrustedSetup.generate(max_degree=16)
    t_setup = (time.time() - t0) * 1000
    print(f"\n  SRS generated (degree=16): {t_setup:.0f}ms")

    # Test polynomial: p(X) = 3X² + 2X + 1
    poly = Polynomial([Fr(1), Fr(2), Fr(3)])

    # Commit
    t0 = time.time()
    C = commit(poly, srs)
    t_commit = (time.time() - t0) * 1000
    print(f"  Commitment: {t_commit:.0f}ms")

    # Open at z = 5
    z = Fr(5)
    t0 = time.time()
    W, y = create_proof(poly, z, srs)
    t_proof = (time.time() - t0) * 1000
    print(f"  Opening proof: {t_proof:.0f}ms")
    print(f"  p(5) = {y.to_int()} (expected: {3*25 + 2*5 + 1} = 86)")

    # Verify
    t0 = time.time()
    valid = verify_opening(C, z, y, W, srs)
    t_verify = (time.time() - t0) * 1000
    print(f"  Verification: {'✅' if valid else '❌'} ({t_verify:.0f}ms)")

    # Test with wrong value (should fail)
    wrong_y = Fr(999)
    invalid = verify_opening(C, z, wrong_y, W, srs)
    print(f"  Wrong value rejected: {'✅' if not invalid else '❌'}")

    # Test with different point
    z2 = Fr(10)
    W2, y2 = create_proof(poly, z2, srs)
    valid2 = verify_opening(C, z2, y2, W2, srs)
    print(f"  Different point: {'✅' if valid2 else '❌'}")
    print(f"  p(10) = {y2.to_int()} (expected: {3*100 + 2*10 + 1} = 321)")

    print("\n" + "=" * 60)
    print("KZG Self-Test Complete!")
    print("=" * 60)
