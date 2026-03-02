"""
Unit tests for BN254 pairing operations.

These tests verify the bilinearity and non-degeneracy properties of the pairing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fp, Fr
from zkml_system.crypto.bn254.extension_field import Fp12
from zkml_system.crypto.bn254.curve import G1Point, G2Point
from zkml_system.crypto.bn254.pairing import pairing, multi_pairing, verify_pairing_equation


def test_pairing_identity():
    """Test that pairing with identity returns 1."""
    print("Testing pairing with identity...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()
    o1 = G1Point.identity()
    o2 = G2Point.identity()

    # e(O, Q) = 1
    result = pairing(o1, g2)
    assert result.is_one(), "e(O, Q) != 1"

    # e(P, O) = 1
    result = pairing(g1, o2)
    assert result.is_one(), "e(P, O) != 1"

    print("  Pairing identity tests passed!")


def test_pairing_non_degeneracy():
    """Test that pairing of generators is not 1."""
    print("Testing pairing non-degeneracy...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()

    # e(G1, G2) != 1
    result = pairing(g1, g2)
    assert not result.is_one(), "e(G1, G2) == 1 (degenerate)"

    print("  Pairing non-degeneracy test passed!")


def test_pairing_bilinearity_g1():
    """Test bilinearity: e(aP, Q) = e(P, Q)^a."""
    print("Testing pairing bilinearity in G1...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()

    a = 7  # Small scalar for faster testing

    # Compute e(aG1, G2)
    ag1 = g1 * a
    lhs = pairing(ag1, g2)

    # Compute e(G1, G2)^a
    e_g1_g2 = pairing(g1, g2)
    rhs = e_g1_g2 ** a

    assert lhs == rhs, f"e(aP, Q) != e(P, Q)^a"

    print("  Bilinearity in G1 test passed!")


def test_pairing_bilinearity_g2():
    """Test bilinearity: e(P, bQ) = e(P, Q)^b."""
    print("Testing pairing bilinearity in G2...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()

    b = 5  # Small scalar for faster testing

    # Compute e(G1, bG2)
    bg2 = g2 * b
    lhs = pairing(g1, bg2)

    # Compute e(G1, G2)^b
    e_g1_g2 = pairing(g1, g2)
    rhs = e_g1_g2 ** b

    assert lhs == rhs, f"e(P, bQ) != e(P, Q)^b"

    print("  Bilinearity in G2 test passed!")


def test_pairing_bilinearity_both():
    """Test bilinearity: e(aP, bQ) = e(P, Q)^(ab)."""
    print("Testing pairing bilinearity in both groups...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()

    a = 3
    b = 5

    # Compute e(aG1, bG2)
    ag1 = g1 * a
    bg2 = g2 * b
    lhs = pairing(ag1, bg2)

    # Compute e(G1, G2)^(ab)
    e_g1_g2 = pairing(g1, g2)
    rhs = e_g1_g2 ** (a * b)

    assert lhs == rhs, f"e(aP, bQ) != e(P, Q)^(ab)"

    print("  Bilinearity in both groups test passed!")


def test_pairing_equation():
    """Test the pairing equation verification."""
    print("Testing pairing equation verification...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()

    a = 3
    b = 5

    # e(aG1, G2) == e(G1, aG2)
    ag1 = g1 * a
    ag2 = g2 * a

    result = verify_pairing_equation(ag1, g2, g1, ag2)
    assert result, "e(aG1, G2) != e(G1, aG2)"

    # e(abG1, G2) == e(aG1, bG2)
    abg1 = g1 * (a * b)
    bg2 = g2 * b

    result = verify_pairing_equation(abg1, g2, ag1, bg2)
    assert result, "e(abG1, G2) != e(aG1, bG2)"

    print("  Pairing equation verification tests passed!")


def test_multi_pairing():
    """Test multi-pairing computation."""
    print("Testing multi-pairing...")

    g1 = G1Point.generator()
    g2 = G2Point.generator()

    # e(G1, G2) * e(2G1, G2) = e(3G1, G2)
    g1_2 = g1 * 2
    g1_3 = g1 * 3

    lhs = multi_pairing([(g1, g2), (g1_2, g2)])
    rhs = pairing(g1_3, g2)

    assert lhs == rhs, "Multi-pairing failed"

    print("  Multi-pairing test passed!")


def run_all_tests():
    """Run all pairing tests."""
    print("=" * 60)
    print("BN254 Pairing Tests")
    print("=" * 60)
    print("NOTE: Pairing tests are computationally intensive.")
    print("      This may take several minutes...")
    print("=" * 60)

    test_pairing_identity()
    test_pairing_non_degeneracy()
    test_pairing_bilinearity_g1()
    test_pairing_bilinearity_g2()
    test_pairing_bilinearity_both()
    test_pairing_equation()
    test_multi_pairing()

    print("=" * 60)
    print("ALL PAIRING TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
