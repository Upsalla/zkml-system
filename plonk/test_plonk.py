"""
Unit tests for PLONK components (Polynomial, FFT, KZG).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.polynomial import Polynomial, FFT, lagrange_interpolation
from zkml_system.plonk.kzg import SRS, KZG, KZGCommitment


def test_polynomial_basic():
    """Test basic polynomial operations."""
    print("Testing Polynomial basic operations...")

    # Zero and one
    assert Polynomial.zero().is_zero(), "Zero failed"
    assert not Polynomial.one().is_zero(), "One failed"

    # Construction
    p = Polynomial.from_ints([1, 2, 3])  # 1 + 2x + 3x²
    assert p.degree() == 2, f"Degree failed: {p.degree()}"

    # Evaluation
    x = Fr(2)
    y = p.evaluate(x)
    # 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
    assert y.to_int() == 17, f"Evaluation failed: {y.to_int()}"

    print("  Polynomial basic tests passed!")


def test_polynomial_arithmetic():
    """Test polynomial arithmetic."""
    print("Testing Polynomial arithmetic...")

    p = Polynomial.from_ints([1, 2])  # 1 + 2x
    q = Polynomial.from_ints([3, 4])  # 3 + 4x

    # Addition: (1 + 2x) + (3 + 4x) = 4 + 6x
    r = p + q
    assert r.coeffs[0].to_int() == 4, "Addition c0 failed"
    assert r.coeffs[1].to_int() == 6, "Addition c1 failed"

    # Subtraction: (1 + 2x) - (3 + 4x) = -2 - 2x
    r = p - q
    assert r.coeffs[0].to_int() == Fr.MODULUS - 2, "Subtraction c0 failed"

    # Multiplication: (1 + 2x) * (3 + 4x) = 3 + 4x + 6x + 8x² = 3 + 10x + 8x²
    r = p * q
    assert r.coeffs[0].to_int() == 3, "Multiplication c0 failed"
    assert r.coeffs[1].to_int() == 10, "Multiplication c1 failed"
    assert r.coeffs[2].to_int() == 8, "Multiplication c2 failed"

    print("  Polynomial arithmetic tests passed!")


def test_polynomial_division():
    """Test polynomial division."""
    print("Testing Polynomial division...")

    # p(x) = x² - 1 = (x - 1)(x + 1)
    p = Polynomial.from_ints([-1, 0, 1])

    # Divide by (x - 1)
    q, r = p.divide_by_linear(Fr(1))

    # Quotient should be (x + 1)
    assert q.coeffs[0].to_int() == 1, f"Quotient c0 failed: {q.coeffs[0].to_int()}"
    assert q.coeffs[1].to_int() == 1, f"Quotient c1 failed: {q.coeffs[1].to_int()}"

    # Remainder should be 0 (since 1 is a root)
    assert r.is_zero(), f"Remainder failed: {r.to_int()}"

    print("  Polynomial division tests passed!")


def test_fft():
    """Test FFT operations."""
    print("Testing FFT...")

    n = 4
    fft = FFT(n)

    # Test with simple polynomial: 1 + x + x² + x³
    coeffs = [Fr(1), Fr(1), Fr(1), Fr(1)]

    # FFT
    evals = fft.fft(coeffs)
    assert len(evals) == n, "FFT output length mismatch"

    # IFFT should recover original
    recovered = fft.ifft(evals)
    for i in range(n):
        assert recovered[i].to_int() == coeffs[i].to_int(), \
            f"IFFT recovery failed at {i}: {recovered[i].to_int()} != {coeffs[i].to_int()}"

    print("  FFT tests passed!")


def test_fft_multiplication():
    """Test FFT-based polynomial multiplication."""
    print("Testing FFT multiplication...")

    fft = FFT(8)

    p = Polynomial.from_ints([1, 2, 3])  # 1 + 2x + 3x²
    q = Polynomial.from_ints([4, 5])     # 4 + 5x

    # Naive multiplication
    naive_result = p * q

    # FFT multiplication
    p_evals = fft.fft(p.coeffs)
    q_evals = fft.fft(q.coeffs)
    result_evals = [a * b for a, b in zip(p_evals, q_evals)]
    fft_coeffs = fft.ifft(result_evals)

    # Compare
    for i in range(len(naive_result.coeffs)):
        assert fft_coeffs[i].to_int() == naive_result.coeffs[i].to_int(), \
            f"FFT multiplication mismatch at {i}"

    print("  FFT multiplication tests passed!")


def test_lagrange_interpolation():
    """Test Lagrange interpolation."""
    print("Testing Lagrange interpolation...")

    # Points: (0, 1), (1, 2), (2, 5)
    # These lie on y = x² + 1
    points = [
        (Fr(0), Fr(1)),
        (Fr(1), Fr(2)),
        (Fr(2), Fr(5))
    ]

    poly = lagrange_interpolation(points)

    # Verify interpolation
    for x, y in points:
        assert poly.evaluate(x) == y, f"Interpolation failed at x={x.to_int()}"

    # Verify it's x² + 1
    assert poly.coeffs[0].to_int() == 1, "Constant term wrong"
    assert poly.coeffs[1].to_int() == 0, "Linear term wrong"
    assert poly.coeffs[2].to_int() == 1, "Quadratic term wrong"

    print("  Lagrange interpolation tests passed!")


def test_srs_generation():
    """Test SRS generation."""
    print("Testing SRS generation...")

    max_degree = 8
    tau = Fr(42)
    srs = SRS.generate(max_degree, tau)

    assert len(srs.g1_powers) == max_degree + 1, "G1 powers length mismatch"
    assert len(srs.g2_powers) == 2, "G2 powers length mismatch"

    # Verify G1 powers are on curve
    for i, p in enumerate(srs.g1_powers):
        assert p.is_on_curve(), f"G1 power {i} not on curve"

    # Verify G2 powers are on curve
    for i, p in enumerate(srs.g2_powers):
        assert p.is_on_curve(), f"G2 power {i} not on curve"

    print("  SRS generation tests passed!")


def test_kzg_commit():
    """Test KZG commitment."""
    print("Testing KZG commitment...")

    srs = SRS.generate(16)
    kzg = KZG(srs)

    # Commit to polynomial 1 + 2x + 3x²
    poly = Polynomial.from_ints([1, 2, 3])
    commitment = kzg.commit(poly)

    # Commitment should be on curve
    assert commitment.point.is_on_curve(), "Commitment not on curve"

    # Same polynomial should give same commitment
    commitment2 = kzg.commit(poly)
    assert commitment == commitment2, "Commitment not deterministic"

    # Different polynomial should give different commitment
    poly2 = Polynomial.from_ints([1, 2, 4])
    commitment3 = kzg.commit(poly2)
    assert commitment != commitment3, "Different polynomials have same commitment"

    print("  KZG commitment tests passed!")


def test_kzg_proof():
    """Test KZG evaluation proof."""
    print("Testing KZG evaluation proof...")

    srs = SRS.generate(16)
    kzg = KZG(srs)

    # Polynomial: 1 + 2x + 3x²
    poly = Polynomial.from_ints([1, 2, 3])
    commitment = kzg.commit(poly)

    # Create proof for evaluation at x = 5
    z = Fr(5)
    proof, y = kzg.create_proof(poly, z)

    # Verify y = p(5) = 1 + 10 + 75 = 86
    expected_y = poly.evaluate(z)
    assert y == expected_y, f"Evaluation mismatch: {y.to_int()} != {expected_y.to_int()}"

    # Proof should be on curve
    assert proof.point.is_on_curve(), "Proof not on curve"

    # Verify the proof
    assert kzg.verify(commitment, z, y, proof), "Proof verification failed"

    print("  KZG evaluation proof tests passed!")


def run_all_tests():
    """Run all PLONK tests."""
    print("=" * 60)
    print("PLONK Component Tests")
    print("=" * 60)

    test_polynomial_basic()
    test_polynomial_arithmetic()
    test_polynomial_division()
    test_fft()
    test_fft_multiplication()
    test_lagrange_interpolation()
    test_srs_generation()
    test_kzg_commit()
    test_kzg_proof()

    print("=" * 60)
    print("ALL PLONK TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
