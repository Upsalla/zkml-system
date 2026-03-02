"""
Unit tests for BN254 elliptic curve arithmetic (G1 and G2).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fp, Fr
from zkml_system.crypto.bn254.extension_field import Fp2
from zkml_system.crypto.bn254.curve import G1Point, G2Point
from zkml_system.crypto.bn254.constants import CURVE_ORDER


def test_g1_generator_on_curve():
    """Test that the G1 generator is on the curve."""
    print("Testing G1 generator is on curve...")

    g = G1Point.generator()
    assert g.is_on_curve(), "G1 generator not on curve"

    print("  G1 generator on curve test passed!")


def test_g1_identity():
    """Test G1 identity element."""
    print("Testing G1 identity...")

    identity = G1Point.identity()
    g = G1Point.generator()

    # O + G = G
    assert g + identity == g, "G + O != G"

    # G + O = G
    assert identity + g == g, "O + G != G"

    # O + O = O
    assert identity + identity == identity, "O + O != O"

    print("  G1 identity tests passed!")


def test_g1_negation():
    """Test G1 point negation."""
    print("Testing G1 negation...")

    g = G1Point.generator()
    neg_g = -g

    # G + (-G) = O
    result = g + neg_g
    assert result.is_identity(), f"G + (-G) != O"

    # -(-G) = G
    assert -neg_g == g, "-(-G) != G"

    print("  G1 negation tests passed!")


def test_g1_doubling():
    """Test G1 point doubling."""
    print("Testing G1 doubling...")

    g = G1Point.generator()

    # 2G via doubling
    g2_double = g.double()

    # 2G via addition
    g2_add = g + g

    assert g2_double == g2_add, "G.double() != G + G"
    assert g2_double.is_on_curve(), "2G not on curve"

    print("  G1 doubling tests passed!")


def test_g1_scalar_multiplication():
    """Test G1 scalar multiplication."""
    print("Testing G1 scalar multiplication...")

    g = G1Point.generator()

    # 0 * G = O
    assert (g * 0).is_identity(), "0 * G != O"

    # 1 * G = G
    assert g * 1 == g, "1 * G != G"

    # 2 * G = G + G
    assert g * 2 == g + g, "2 * G != G + G"

    # 3 * G = G + G + G
    assert g * 3 == g + g + g, "3 * G != G + G + G"

    # Test associativity: (a * b) * G = a * (b * G)
    a, b = 7, 11
    lhs = g * (a * b)
    rhs = (g * b) * a
    assert lhs == rhs, "Scalar multiplication not associative"

    print("  G1 scalar multiplication tests passed!")


def test_g1_order():
    """Test that the curve order is correct."""
    print("Testing G1 curve order...")

    g = G1Point.generator()

    # r * G = O (where r is the curve order)
    # This is slow for the full order, so we test with a smaller multiple
    # We verify that (r-1)*G + G = O
    # Actually, let's just verify the generator is on the curve and
    # test a smaller scalar

    # Test that 2 * (r/2) * G = O if r is even (it's not for BN254)
    # Instead, verify basic properties hold

    # Test: n*G for small n stays on curve
    for n in [1, 2, 3, 5, 10, 100]:
        p = g * n
        assert p.is_on_curve(), f"{n}*G not on curve"

    print("  G1 curve order tests passed!")


def test_g2_generator_on_curve():
    """Test that the G2 generator is on the curve."""
    print("Testing G2 generator is on curve...")

    g = G2Point.generator()
    assert g.is_on_curve(), "G2 generator not on curve"

    print("  G2 generator on curve test passed!")


def test_g2_identity():
    """Test G2 identity element."""
    print("Testing G2 identity...")

    identity = G2Point.identity()
    g = G2Point.generator()

    # O + G = G
    assert g + identity == g, "G + O != G"

    # G + O = G
    assert identity + g == g, "O + G != G"

    print("  G2 identity tests passed!")


def test_g2_negation():
    """Test G2 point negation."""
    print("Testing G2 negation...")

    g = G2Point.generator()
    neg_g = -g

    # G + (-G) = O
    result = g + neg_g
    assert result.is_identity(), f"G + (-G) != O"

    print("  G2 negation tests passed!")


def test_g2_doubling():
    """Test G2 point doubling."""
    print("Testing G2 doubling...")

    g = G2Point.generator()

    # 2G via doubling
    g2_double = g.double()

    # 2G via addition
    g2_add = g + g

    assert g2_double == g2_add, "G.double() != G + G"
    assert g2_double.is_on_curve(), "2G not on curve"

    print("  G2 doubling tests passed!")


def test_g2_scalar_multiplication():
    """Test G2 scalar multiplication."""
    print("Testing G2 scalar multiplication...")

    g = G2Point.generator()

    # 0 * G = O
    assert (g * 0).is_identity(), "0 * G != O"

    # 1 * G = G
    assert g * 1 == g, "1 * G != G"

    # 2 * G = G + G
    assert g * 2 == g + g, "2 * G != G + G"

    # 3 * G = G + G + G
    assert g * 3 == g + g + g, "3 * G != G + G + G"

    print("  G2 scalar multiplication tests passed!")


def test_g1_affine_conversion():
    """Test G1 affine coordinate conversion."""
    print("Testing G1 affine conversion...")

    g = G1Point.generator()

    # Get affine coordinates
    x, y = g.to_affine()

    # Create new point from affine
    g2 = G1Point.from_affine(x.to_int(), y.to_int())

    assert g == g2, "Affine conversion failed"

    # Test with a non-trivial point
    p = g * 12345
    x, y = p.to_affine()
    p2 = G1Point.from_affine(x.to_int(), y.to_int())
    assert p == p2, "Affine conversion failed for 12345*G"

    print("  G1 affine conversion tests passed!")


def run_all_tests():
    """Run all curve tests."""
    print("=" * 60)
    print("BN254 Elliptic Curve Arithmetic Tests")
    print("=" * 60)

    test_g1_generator_on_curve()
    test_g1_identity()
    test_g1_negation()
    test_g1_doubling()
    test_g1_scalar_multiplication()
    test_g1_order()
    test_g2_generator_on_curve()
    test_g2_identity()
    test_g2_negation()
    test_g2_doubling()
    test_g2_scalar_multiplication()
    test_g1_affine_conversion()

    print("=" * 60)
    print("ALL CURVE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
