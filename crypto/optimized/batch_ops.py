"""
Optimized Batch Operations for Cryptographic Primitives

This module implements:
1. Montgomery Batch Inversion - O(n) inversions with only 1 actual inversion
2. Pippenger Multi-Scalar Multiplication - O(n/log n) for batch scalar muls
3. Windowed NAF Scalar Multiplication - 3-5x faster than double-and-add

These optimizations target the critical bottlenecks identified in profiling:
- KZG commit: Uses MSM (multi-scalar multiplication)
- KZG proof: Uses polynomial division which requires inversions
- Verification: Uses batch operations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, TypeVar, Generic
from dataclasses import dataclass


T = TypeVar('T')


class BatchInversion:
    """
    Montgomery Batch Inversion Algorithm

    Computes n inversions using only 1 actual inversion + 3n multiplications.

    Algorithm:
    1. Compute cumulative products: c[i] = a[0] * a[1] * ... * a[i]
    2. Invert the final product: inv = c[n-1]^(-1)
    3. Work backwards: a[i]^(-1) = c[i-1] * inv, then inv *= a[i]

    Complexity: O(3n) multiplications + 1 inversion
    vs Naive: O(n) inversions = O(256n) multiplications

    Speedup: ~85x for large batches
    """

    @staticmethod
    def batch_inverse_fp(elements: List['Fp']) -> List['Fp']:
        """Batch inversion for Fp elements."""
        from zkml_system.crypto.bn254.field import Fp

        n = len(elements)
        if n == 0:
            return []
        if n == 1:
            return [elements[0].inverse()]

        # Step 1: Compute cumulative products
        cumulative = [Fp.zero()] * n
        cumulative[0] = elements[0]
        for i in range(1, n):
            cumulative[i] = cumulative[i-1] * elements[i]

        # Step 2: Invert the final product (single inversion)
        inv = cumulative[n-1].inverse()

        # Step 3: Work backwards to compute all inverses
        result = [Fp.zero()] * n
        for i in range(n-1, 0, -1):
            result[i] = cumulative[i-1] * inv
            inv = inv * elements[i]
        result[0] = inv

        return result

    @staticmethod
    def batch_inverse_fr(elements: List['Fr']) -> List['Fr']:
        """Batch inversion for Fr elements."""
        from zkml_system.crypto.bn254.field import Fr

        n = len(elements)
        if n == 0:
            return []
        if n == 1:
            return [elements[0].inverse()]

        # Step 1: Compute cumulative products
        cumulative = [Fr.zero()] * n
        cumulative[0] = elements[0]
        for i in range(1, n):
            cumulative[i] = cumulative[i-1] * elements[i]

        # Step 2: Invert the final product
        inv = cumulative[n-1].inverse()

        # Step 3: Work backwards
        result = [Fr.zero()] * n
        for i in range(n-1, 0, -1):
            result[i] = cumulative[i-1] * inv
            inv = inv * elements[i]
        result[0] = inv

        return result


class WindowedNAF:
    """
    Windowed Non-Adjacent Form (wNAF) Scalar Multiplication

    Instead of processing 1 bit at a time, process w bits at a time.
    Uses precomputation to reduce the number of point additions.

    For w=4:
    - Precompute: P, 3P, 5P, 7P, 9P, 11P, 13P, 15P (8 points)
    - Process 4 bits at a time
    - Expected additions: 256/4 = 64 (vs 128 for double-and-add)

    Speedup: ~2x for single scalar mul
    """

    @staticmethod
    def scalar_to_wnaf(scalar: int, w: int = 4) -> List[int]:
        """
        Convert scalar to windowed NAF representation.

        Returns a list of digits in {-(2^(w-1)-1), ..., -1, 0, 1, ..., 2^(w-1)-1}
        """
        result = []
        while scalar > 0:
            if scalar & 1:  # Odd
                # Get the w-bit window
                window = scalar & ((1 << w) - 1)
                if window >= (1 << (w - 1)):
                    # Make it negative
                    window = window - (1 << w)
                result.append(window)
                scalar = scalar - window
            else:
                result.append(0)
            scalar >>= 1
        return result

    @staticmethod
    def mul_g1(point: 'G1Point', scalar: int, w: int = 4) -> 'G1Point':
        """Windowed NAF scalar multiplication for G1."""
        from zkml_system.crypto.bn254.curve import G1Point

        if scalar == 0:
            return G1Point.identity()
        if scalar < 0:
            return WindowedNAF.mul_g1(-point, -scalar, w)

        # Precompute odd multiples: P, 3P, 5P, ..., (2^(w-1)-1)P
        precomp_size = 1 << (w - 1)
        precomp = [G1Point.identity()] * precomp_size
        precomp[0] = point
        double_point = point.double()
        for i in range(1, precomp_size):
            precomp[i] = precomp[i-1] + double_point

        # Convert scalar to wNAF
        wnaf = WindowedNAF.scalar_to_wnaf(scalar, w)

        # Process from most significant to least significant
        result = G1Point.identity()
        for i in range(len(wnaf) - 1, -1, -1):
            result = result.double()
            digit = wnaf[i]
            if digit > 0:
                result = result + precomp[(digit - 1) // 2]
            elif digit < 0:
                result = result + (-precomp[(-digit - 1) // 2])

        return result


class PippengerMSM:
    """
    Pippenger's Multi-Scalar Multiplication Algorithm

    For computing: sum(s[i] * P[i]) for i in 0..n-1

    Instead of n separate scalar multiplications, this algorithm:
    1. Groups scalars by their bit patterns
    2. Processes all points with the same bit pattern together
    3. Combines results using a clever accumulation scheme

    Complexity: O(n / log n) point additions
    vs Naive: O(n * 256) point additions

    Speedup: 10-50x for large batches (n > 100)
    """

    @staticmethod
    def msm_g1(points: List['G1Point'], scalars: List[int], c: int = None) -> 'G1Point':
        """
        Multi-scalar multiplication using Pippenger's algorithm.

        Args:
            points: List of G1 points
            scalars: List of scalars (integers)
            c: Window size (auto-computed if None)

        Returns:
            sum(scalars[i] * points[i])
        """
        from zkml_system.crypto.bn254.curve import G1Point

        n = len(points)
        if n == 0:
            return G1Point.identity()
        if n != len(scalars):
            raise ValueError("Points and scalars must have same length")

        # Auto-compute optimal window size
        # Optimal c ≈ log2(n) for large n
        if c is None:
            c = max(1, (n.bit_length() + 1) // 2)
            c = min(c, 16)  # Cap at 16 for memory reasons

        num_buckets = 1 << c
        num_windows = (256 + c - 1) // c

        result = G1Point.identity()

        # Process each window
        for window_idx in range(num_windows - 1, -1, -1):
            # Double the result c times (except for first window)
            if window_idx < num_windows - 1:
                for _ in range(c):
                    result = result.double()

            # Initialize buckets
            buckets = [G1Point.identity() for _ in range(num_buckets)]

            # Distribute points to buckets based on scalar bits
            for i in range(n):
                scalar = scalars[i]
                # Extract c bits from position window_idx * c
                bucket_idx = (scalar >> (window_idx * c)) & (num_buckets - 1)
                if bucket_idx > 0:
                    buckets[bucket_idx] = buckets[bucket_idx] + points[i]

            # Accumulate buckets: sum(i * buckets[i])
            # Using the trick: sum = B[k] + (B[k] + B[k-1]) + (B[k] + B[k-1] + B[k-2]) + ...
            running_sum = G1Point.identity()
            window_sum = G1Point.identity()
            for i in range(num_buckets - 1, 0, -1):
                running_sum = running_sum + buckets[i]
                window_sum = window_sum + running_sum

            result = result + window_sum

        return result

    @staticmethod
    def msm_g1_simple(points: List['G1Point'], scalars: List[int]) -> 'G1Point':
        """
        Simple (naive) MSM for comparison and correctness testing.

        Complexity: O(n * 256) point operations
        """
        from zkml_system.crypto.bn254.curve import G1Point

        result = G1Point.identity()
        for point, scalar in zip(points, scalars):
            result = result + (point * scalar)
        return result


def test_batch_inversion():
    """Test batch inversion correctness and performance."""
    print("Testing Batch Inversion...")

    from zkml_system.crypto.bn254.field import Fp, Fr
    import time

    # Test correctness
    elements = [Fp(i + 1) for i in range(100)]
    batch_result = BatchInversion.batch_inverse_fp(elements)
    for i, (elem, inv) in enumerate(zip(elements, batch_result)):
        product = elem * inv
        assert product == Fp.one(), f"Batch inversion failed at index {i}"
    print("  Correctness: OK")

    # Test performance
    n = 100
    elements = [Fp(i + 1) for i in range(n)]

    # Naive
    start = time.perf_counter()
    naive_result = [e.inverse() for e in elements]
    naive_time = time.perf_counter() - start

    # Batch
    start = time.perf_counter()
    batch_result = BatchInversion.batch_inverse_fp(elements)
    batch_time = time.perf_counter() - start

    speedup = naive_time / batch_time
    print(f"  Naive time: {naive_time*1000:.2f}ms")
    print(f"  Batch time: {batch_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")


def test_windowed_naf():
    """Test windowed NAF scalar multiplication."""
    print("\nTesting Windowed NAF Scalar Multiplication...")

    from zkml_system.crypto.bn254.curve import G1Point
    import time

    g1 = G1Point.generator()
    scalar = 12345678901234567890123456789012345678901234567890

    # Test correctness
    naive_result = g1 * scalar
    wnaf_result = WindowedNAF.mul_g1(g1, scalar)
    assert naive_result == wnaf_result, "wNAF result differs from naive"
    print("  Correctness: OK")

    # Test performance
    start = time.perf_counter()
    for _ in range(10):
        naive_result = g1 * scalar
    naive_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):
        wnaf_result = WindowedNAF.mul_g1(g1, scalar)
    wnaf_time = time.perf_counter() - start

    speedup = naive_time / wnaf_time
    print(f"  Naive time: {naive_time*1000:.2f}ms (10 ops)")
    print(f"  wNAF time: {wnaf_time*1000:.2f}ms (10 ops)")
    print(f"  Speedup: {speedup:.1f}x")


def test_pippenger_msm():
    """Test Pippenger MSM."""
    print("\nTesting Pippenger MSM...")

    from zkml_system.crypto.bn254.curve import G1Point
    import time
    import random

    # Small test for correctness
    n = 16
    g1 = G1Point.generator()
    points = [g1 * (i + 1) for i in range(n)]
    scalars = [random.randint(1, 1000) for _ in range(n)]

    # Test correctness
    naive_result = PippengerMSM.msm_g1_simple(points, scalars)
    pippenger_result = PippengerMSM.msm_g1(points, scalars)
    assert naive_result == pippenger_result, "Pippenger result differs from naive"
    print("  Correctness: OK")

    # Test performance (larger batch)
    n = 32
    points = [g1 * (i + 1) for i in range(n)]
    scalars = [random.randint(1, 2**64) for _ in range(n)]

    start = time.perf_counter()
    naive_result = PippengerMSM.msm_g1_simple(points, scalars)
    naive_time = time.perf_counter() - start

    start = time.perf_counter()
    pippenger_result = PippengerMSM.msm_g1(points, scalars)
    pippenger_time = time.perf_counter() - start

    speedup = naive_time / pippenger_time
    print(f"  Naive time (n={n}): {naive_time*1000:.2f}ms")
    print(f"  Pippenger time (n={n}): {pippenger_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED BATCH OPERATIONS TESTS")
    print("=" * 60)

    test_batch_inversion()
    test_windowed_naf()
    test_pippenger_msm()

    print("\n" + "=" * 60)
    print("ALL OPTIMIZATION TESTS PASSED")
    print("=" * 60)
