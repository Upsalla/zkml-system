"""
Unit tests for BN254 extension field arithmetic (Fp2, Fp6, Fp12).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fp
from zkml_system.crypto.bn254.extension_field import Fp2, Fp6, Fp12


def test_fp2_basic():
    """Test basic Fp2 arithmetic."""
    print("Testing Fp2 basic arithmetic...")

    # Test zero and one
    assert Fp2.zero().is_zero(), "Zero failed"
    assert Fp2.one().is_one(), "One failed"

    # Test construction
    a = Fp2(Fp(3), Fp(4))
    assert a.c0.to_int() == 3 and a.c1.to_int() == 4, "Construction failed"

    # Test addition
    a = Fp2(Fp(1), Fp(2))
    b = Fp2(Fp(3), Fp(4))
    c = a + b
    assert c.c0.to_int() == 4 and c.c1.to_int() == 6, "Addition failed"

    # Test subtraction
    c = b - a
    assert c.c0.to_int() == 2 and c.c1.to_int() == 2, "Subtraction failed"

    # Test negation
    neg_a = -a
    assert (a + neg_a).is_zero(), "Negation failed"

    print("  Fp2 basic tests passed!")


def test_fp2_multiplication():
    """Test Fp2 multiplication."""
    print("Testing Fp2 multiplication...")

    # (1 + 2u) * (3 + 4u) = 3 + 4u + 6u + 8u^2
    #                     = 3 + 10u - 8  (since u^2 = -1)
    #                     = -5 + 10u
    a = Fp2(Fp(1), Fp(2))
    b = Fp2(Fp(3), Fp(4))
    c = a * b

    from zkml_system.crypto.bn254.constants import FIELD_MODULUS
    expected_c0 = (3 - 8) % FIELD_MODULUS  # -5 mod p
    expected_c1 = 10

    assert c.c0.to_int() == expected_c0, f"Mul c0 failed: {c.c0.to_int()} != {expected_c0}"
    assert c.c1.to_int() == expected_c1, f"Mul c1 failed: {c.c1.to_int()} != {expected_c1}"

    # Test squaring
    a = Fp2(Fp(3), Fp(4))
    c = a.square()
    c_expected = a * a
    assert c == c_expected, "Squaring failed"

    print("  Fp2 multiplication tests passed!")


def test_fp2_inverse():
    """Test Fp2 multiplicative inverse."""
    print("Testing Fp2 inverse...")

    a = Fp2(Fp(3), Fp(4))
    a_inv = a.inverse()
    product = a * a_inv

    assert product.is_one(), f"Inverse failed: {a} * {a_inv} = {product}"

    # Test with another value
    a = Fp2(Fp(12345), Fp(67890))
    a_inv = a.inverse()
    product = a * a_inv
    assert product.is_one(), "Inverse failed for larger values"

    print("  Fp2 inverse tests passed!")


def test_fp2_exponentiation():
    """Test Fp2 exponentiation."""
    print("Testing Fp2 exponentiation...")

    a = Fp2(Fp(2), Fp(3))

    # Test a^2 = a * a
    assert a ** 2 == a.square(), "a^2 failed"

    # Test a^3 = a * a * a
    assert a ** 3 == a * a * a, "a^3 failed"

    # Test a^0 = 1
    assert (a ** 0).is_one(), "a^0 failed"

    # Test a^1 = a
    assert a ** 1 == a, "a^1 failed"

    print("  Fp2 exponentiation tests passed!")


def test_fp6_basic():
    """Test basic Fp6 arithmetic."""
    print("Testing Fp6 basic arithmetic...")

    # Test zero and one
    assert Fp6.zero().is_zero(), "Zero failed"
    assert Fp6.one().is_one(), "One failed"

    # Test addition
    a = Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6)))
    b = Fp6(Fp2(Fp(7), Fp(8)), Fp2(Fp(9), Fp(10)), Fp2(Fp(11), Fp(12)))
    c = a + b

    assert c.c0.c0.to_int() == 8, "Addition c0.c0 failed"
    assert c.c1.c0.to_int() == 12, "Addition c1.c0 failed"
    assert c.c2.c0.to_int() == 16, "Addition c2.c0 failed"

    # Test negation
    neg_a = -a
    assert (a + neg_a).is_zero(), "Negation failed"

    print("  Fp6 basic tests passed!")


def test_fp6_multiplication():
    """Test Fp6 multiplication."""
    print("Testing Fp6 multiplication...")

    a = Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6)))
    b = Fp6(Fp2(Fp(7), Fp(8)), Fp2(Fp(9), Fp(10)), Fp2(Fp(11), Fp(12)))

    # Test that multiplication is consistent with squaring
    c = a * a
    c_sq = a.square()
    assert c == c_sq, "Multiplication != Squaring for same element"

    # Test associativity: (a * b) * a = a * (b * a)
    lhs = (a * b) * a
    rhs = a * (b * a)
    assert lhs == rhs, "Associativity failed"

    print("  Fp6 multiplication tests passed!")


def test_fp6_inverse():
    """Test Fp6 multiplicative inverse."""
    print("Testing Fp6 inverse...")

    a = Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6)))
    a_inv = a.inverse()
    product = a * a_inv

    assert product.is_one(), f"Inverse failed"

    print("  Fp6 inverse tests passed!")


def test_fp12_basic():
    """Test basic Fp12 arithmetic."""
    print("Testing Fp12 basic arithmetic...")

    # Test zero and one
    assert Fp12.zero().is_zero(), "Zero failed"
    assert Fp12.one().is_one(), "One failed"

    # Test addition
    a = Fp12(
        Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6))),
        Fp6(Fp2(Fp(7), Fp(8)), Fp2(Fp(9), Fp(10)), Fp2(Fp(11), Fp(12)))
    )
    b = Fp12(
        Fp6(Fp2(Fp(13), Fp(14)), Fp2(Fp(15), Fp(16)), Fp2(Fp(17), Fp(18))),
        Fp6(Fp2(Fp(19), Fp(20)), Fp2(Fp(21), Fp(22)), Fp2(Fp(23), Fp(24)))
    )
    c = a + b

    assert c.c0.c0.c0.to_int() == 14, "Addition failed"

    # Test negation
    neg_a = -a
    assert (a + neg_a).is_zero(), "Negation failed"

    print("  Fp12 basic tests passed!")


def test_fp12_multiplication():
    """Test Fp12 multiplication."""
    print("Testing Fp12 multiplication...")

    a = Fp12(
        Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6))),
        Fp6(Fp2(Fp(7), Fp(8)), Fp2(Fp(9), Fp(10)), Fp2(Fp(11), Fp(12)))
    )

    # Test that multiplication is consistent with squaring
    c = a * a
    c_sq = a.square()
    assert c == c_sq, "Multiplication != Squaring for same element"

    print("  Fp12 multiplication tests passed!")


def test_fp12_inverse():
    """Test Fp12 multiplicative inverse."""
    print("Testing Fp12 inverse...")

    a = Fp12(
        Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6))),
        Fp6(Fp2(Fp(7), Fp(8)), Fp2(Fp(9), Fp(10)), Fp2(Fp(11), Fp(12)))
    )
    a_inv = a.inverse()
    product = a * a_inv

    assert product.is_one(), f"Inverse failed"

    print("  Fp12 inverse tests passed!")


def test_fp12_conjugate():
    """Test Fp12 conjugate."""
    print("Testing Fp12 conjugate...")

    a = Fp12(
        Fp6(Fp2(Fp(1), Fp(2)), Fp2(Fp(3), Fp(4)), Fp2(Fp(5), Fp(6))),
        Fp6(Fp2(Fp(7), Fp(8)), Fp2(Fp(9), Fp(10)), Fp2(Fp(11), Fp(12)))
    )

    # For elements in the cyclotomic subgroup, a * conj(a) = 1
    # But for general elements, we just check that conj(conj(a)) = a
    conj_a = a.conjugate()
    conj_conj_a = conj_a.conjugate()

    assert conj_conj_a == a, "Double conjugate failed"

    print("  Fp12 conjugate tests passed!")


def run_all_tests():
    """Run all extension field tests."""
    print("=" * 60)
    print("BN254 Extension Field Arithmetic Tests")
    print("=" * 60)

    test_fp2_basic()
    test_fp2_multiplication()
    test_fp2_inverse()
    test_fp2_exponentiation()
    test_fp6_basic()
    test_fp6_multiplication()
    test_fp6_inverse()
    test_fp12_basic()
    test_fp12_multiplication()
    test_fp12_inverse()
    test_fp12_conjugate()

    print("=" * 60)
    print("ALL EXTENSION FIELD TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
