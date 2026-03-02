"""
Unit tests for BN254 Fp and Fr field arithmetic.

These tests verify the correctness of the Montgomery-based field implementations
against known test vectors and algebraic properties.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fp, Fr
from zkml_system.crypto.bn254.constants import FIELD_MODULUS, CURVE_ORDER


def test_fp_basic_arithmetic():
    """Test basic Fp arithmetic operations."""
    print("Testing Fp basic arithmetic...")

    # Test zero and one
    assert Fp.zero().to_int() == 0, "Zero failed"
    assert Fp.one().to_int() == 1, "One failed"

    # Test construction and conversion
    a = Fp(123456789)
    assert a.to_int() == 123456789, "Construction/conversion failed"

    # Test addition
    a = Fp(100)
    b = Fp(200)
    c = a + b
    assert c.to_int() == 300, f"Addition failed: {c.to_int()} != 300"

    # Test subtraction
    c = b - a
    assert c.to_int() == 100, f"Subtraction failed: {c.to_int()} != 100"

    # Test negation
    neg_a = -a
    assert (a + neg_a).is_zero(), "Negation failed"

    # Test multiplication
    a = Fp(12345)
    b = Fp(67890)
    c = a * b
    expected = (12345 * 67890) % FIELD_MODULUS
    assert c.to_int() == expected, f"Multiplication failed: {c.to_int()} != {expected}"

    # Test squaring
    a = Fp(999)
    c = a.square()
    expected = (999 * 999) % FIELD_MODULUS
    assert c.to_int() == expected, f"Squaring failed: {c.to_int()} != {expected}"

    print("  All basic arithmetic tests passed!")


def test_fp_inverse():
    """Test Fp multiplicative inverse."""
    print("Testing Fp inverse...")

    a = Fp(12345)
    a_inv = a.inverse()
    product = a * a_inv
    assert product.is_one(), f"Inverse failed: {a} * {a_inv} = {product.to_int()}"

    # Test with larger number
    a = Fp(98765432109876543210)
    a_inv = a.inverse()
    product = a * a_inv
    assert product.is_one(), f"Inverse failed for large number"

    print("  Inverse tests passed!")


def test_fp_exponentiation():
    """Test Fp exponentiation."""
    print("Testing Fp exponentiation...")

    a = Fp(2)
    result = a ** 10
    expected = pow(2, 10, FIELD_MODULUS)
    assert result.to_int() == expected, f"Exponentiation failed: {result.to_int()} != {expected}"

    # Test Fermat's Little Theorem: a^(p-1) = 1 for a != 0
    a = Fp(12345)
    result = a ** (FIELD_MODULUS - 1)
    assert result.is_one(), f"Fermat's Little Theorem failed"

    print("  Exponentiation tests passed!")


def test_fp_sqrt():
    """Test Fp square root."""
    print("Testing Fp square root...")

    # Test with a known square
    a = Fp(16)  # 4^2 = 16
    sqrt_a = a.sqrt()
    assert sqrt_a.square() == a, f"Square root failed: {sqrt_a}^2 != {a}"

    # Test with a random quadratic residue
    original = Fp(12345)
    squared = original.square()
    sqrt_result = squared.sqrt()
    # sqrt could be +/- original
    assert sqrt_result.square() == squared, f"Square root verification failed"

    print("  Square root tests passed!")


def test_fp_edge_cases():
    """Test Fp edge cases."""
    print("Testing Fp edge cases...")

    # Test modular reduction
    a = Fp(FIELD_MODULUS + 1)
    assert a.to_int() == 1, "Modular reduction failed"

    a = Fp(-1)
    assert a.to_int() == FIELD_MODULUS - 1, "Negative number handling failed"

    # Test zero division
    try:
        Fp.zero().inverse()
        assert False, "Should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass

    print("  Edge case tests passed!")


def test_fr_basic_arithmetic():
    """Test basic Fr arithmetic operations."""
    print("Testing Fr basic arithmetic...")

    # Test zero and one
    assert Fr.zero().to_int() == 0, "Zero failed"
    assert Fr.one().to_int() == 1, "One failed"

    # Test construction and conversion
    a = Fr(123456789)
    assert a.to_int() == 123456789, "Construction/conversion failed"

    # Test addition
    a = Fr(100)
    b = Fr(200)
    c = a + b
    assert c.to_int() == 300, f"Addition failed: {c.to_int()} != 300"

    # Test multiplication
    a = Fr(12345)
    b = Fr(67890)
    c = a * b
    expected = (12345 * 67890) % CURVE_ORDER
    assert c.to_int() == expected, f"Multiplication failed: {c.to_int()} != {expected}"

    # Test inverse
    a = Fr(12345)
    a_inv = a.inverse()
    product = a * a_inv
    assert product.is_one(), f"Inverse failed"

    print("  All Fr basic arithmetic tests passed!")


def test_fp_consistency_with_python():
    """Test that Fp operations match Python's native modular arithmetic."""
    print("Testing Fp consistency with Python...")

    import random
    random.seed(42)

    for _ in range(100):
        a_int = random.randint(0, FIELD_MODULUS - 1)
        b_int = random.randint(1, FIELD_MODULUS - 1)  # b != 0 for division

        a = Fp(a_int)
        b = Fp(b_int)

        # Addition
        assert (a + b).to_int() == (a_int + b_int) % FIELD_MODULUS, "Add consistency failed"

        # Subtraction
        assert (a - b).to_int() == (a_int - b_int) % FIELD_MODULUS, "Sub consistency failed"

        # Multiplication
        assert (a * b).to_int() == (a_int * b_int) % FIELD_MODULUS, "Mul consistency failed"

        # Division (a / b = a * b^(-1))
        b_inv_int = pow(b_int, FIELD_MODULUS - 2, FIELD_MODULUS)
        expected_div = (a_int * b_inv_int) % FIELD_MODULUS
        assert (a / b).to_int() == expected_div, "Div consistency failed"

    print("  Consistency tests passed (100 random cases)!")


def run_all_tests():
    """Run all field tests."""
    print("=" * 60)
    print("BN254 Field Arithmetic Tests")
    print("=" * 60)

    test_fp_basic_arithmetic()
    test_fp_inverse()
    test_fp_exponentiation()
    test_fp_sqrt()
    test_fp_edge_cases()
    test_fr_basic_arithmetic()
    test_fp_consistency_with_python()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
