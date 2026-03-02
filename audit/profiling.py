"""
Performance Profiling for zkML System

This script identifies the exact bottlenecks in the cryptographic operations.
It measures wall-clock time and operation counts for each component.

Key metrics:
- Time per operation (µs)
- Operations per second
- Percentage of total time
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import statistics
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ProfileResult:
    """Result of a profiling run."""
    name: str
    total_time_ms: float
    ops_count: int
    time_per_op_us: float
    ops_per_second: float


def profile_operation(name: str, func, iterations: int = 100) -> ProfileResult:
    """Profile a single operation."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_time = sum(times) * 1000  # ms
    avg_time = statistics.mean(times) * 1_000_000  # µs
    ops_per_sec = iterations / sum(times)

    return ProfileResult(
        name=name,
        total_time_ms=total_time,
        ops_count=iterations,
        time_per_op_us=avg_time,
        ops_per_second=ops_per_sec
    )


def profile_field_operations():
    """Profile field arithmetic operations."""
    print("\n" + "=" * 70)
    print("FIELD ARITHMETIC PROFILING")
    print("=" * 70)

    from zkml_system.crypto.bn254.field import Fp, Fr

    # Setup
    a_fp = Fp(123456789012345678901234567890)
    b_fp = Fp(987654321098765432109876543210)
    a_fr = Fr(123456789012345678901234567890)
    b_fr = Fr(987654321098765432109876543210)

    results = []

    # Fp operations
    results.append(profile_operation(
        "Fp multiplication",
        lambda: a_fp * b_fp,
        iterations=10000
    ))

    results.append(profile_operation(
        "Fp addition",
        lambda: a_fp + b_fp,
        iterations=10000
    ))

    results.append(profile_operation(
        "Fp inversion",
        lambda: a_fp.inverse(),
        iterations=100
    ))

    results.append(profile_operation(
        "Fp exponentiation (256-bit)",
        lambda: a_fp ** 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
        iterations=10
    ))

    # Fr operations
    results.append(profile_operation(
        "Fr multiplication",
        lambda: a_fr * b_fr,
        iterations=10000
    ))

    results.append(profile_operation(
        "Fr inversion",
        lambda: a_fr.inverse(),
        iterations=100
    ))

    # Print results
    print(f"\n{'Operation':<30} {'Time/Op (µs)':<15} {'Ops/sec':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<30} {r.time_per_op_us:<15.2f} {r.ops_per_second:<15.0f}")

    return results


def profile_curve_operations():
    """Profile elliptic curve operations."""
    print("\n" + "=" * 70)
    print("CURVE OPERATIONS PROFILING")
    print("=" * 70)

    from zkml_system.crypto.bn254.field import Fr
    from zkml_system.crypto.bn254.curve import G1Point, G2Point

    # Setup
    g1 = G1Point.generator()
    g2 = G2Point.generator()
    scalar = Fr(12345678901234567890).to_int()
    g1_2 = g1 + g1

    results = []

    # G1 operations
    results.append(profile_operation(
        "G1 addition",
        lambda: g1 + g1_2,
        iterations=1000
    ))

    results.append(profile_operation(
        "G1 doubling",
        lambda: g1.double(),
        iterations=1000
    ))

    results.append(profile_operation(
        "G1 scalar mul (small)",
        lambda: g1 * 1000,
        iterations=100
    ))

    results.append(profile_operation(
        "G1 scalar mul (256-bit)",
        lambda: g1 * scalar,
        iterations=10
    ))

    # G2 operations
    results.append(profile_operation(
        "G2 addition",
        lambda: g2 + g2,
        iterations=100
    ))

    results.append(profile_operation(
        "G2 scalar mul (small)",
        lambda: g2 * 1000,
        iterations=10
    ))

    # Print results
    print(f"\n{'Operation':<30} {'Time/Op (µs)':<15} {'Ops/sec':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<30} {r.time_per_op_us:<15.2f} {r.ops_per_second:<15.0f}")

    return results


def profile_polynomial_operations():
    """Profile polynomial operations."""
    print("\n" + "=" * 70)
    print("POLYNOMIAL OPERATIONS PROFILING")
    print("=" * 70)

    from zkml_system.crypto.bn254.field import Fr
    from zkml_system.plonk.polynomial import Polynomial, FFT

    # Setup
    n = 64
    coeffs = [Fr(i) for i in range(n)]
    poly = Polynomial(coeffs)
    fft = FFT(n)

    results = []

    # Polynomial operations
    results.append(profile_operation(
        f"Polynomial eval (deg {n-1})",
        lambda: poly.evaluate(Fr(12345)),
        iterations=100
    ))

    results.append(profile_operation(
        f"Polynomial mul (deg {n-1})",
        lambda: poly * poly,
        iterations=10
    ))

    results.append(profile_operation(
        f"FFT (n={n})",
        lambda: fft.fft(coeffs),
        iterations=10
    ))

    results.append(profile_operation(
        f"IFFT (n={n})",
        lambda: fft.ifft(coeffs),
        iterations=10
    ))

    # Print results
    print(f"\n{'Operation':<30} {'Time/Op (µs)':<15} {'Ops/sec':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<30} {r.time_per_op_us:<15.2f} {r.ops_per_second:<15.0f}")

    return results


def profile_kzg_operations():
    """Profile KZG commitment operations."""
    print("\n" + "=" * 70)
    print("KZG COMMITMENT PROFILING")
    print("=" * 70)

    from zkml_system.crypto.bn254.field import Fr
    from zkml_system.plonk.polynomial import Polynomial
    from zkml_system.plonk.kzg import SRS, KZG

    # Setup
    print("Generating SRS (this takes time)...")
    srs = SRS.generate(32)
    kzg = KZG(srs)
    poly = Polynomial([Fr(i) for i in range(16)])
    z = Fr(12345)

    results = []

    # KZG operations
    results.append(profile_operation(
        "KZG commit (deg 15)",
        lambda: kzg.commit(poly),
        iterations=10
    ))

    commitment = kzg.commit(poly)

    results.append(profile_operation(
        "KZG create proof",
        lambda: kzg.create_proof(poly, z),
        iterations=10
    ))

    proof, y = kzg.create_proof(poly, z)

    results.append(profile_operation(
        "KZG verify",
        lambda: kzg.verify(commitment, z, y, proof),
        iterations=10
    ))

    # Print results
    print(f"\n{'Operation':<30} {'Time/Op (µs)':<15} {'Ops/sec':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<30} {r.time_per_op_us:<15.2f} {r.ops_per_second:<15.0f}")

    return results


def analyze_bottlenecks(all_results: List[ProfileResult]):
    """Analyze and rank bottlenecks."""
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    # Sort by time per operation (descending)
    sorted_results = sorted(all_results, key=lambda r: r.time_per_op_us, reverse=True)

    print("\nRanked by time per operation (slowest first):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Operation':<35} {'Time/Op (µs)':<15} {'Impact':<15}")
    print("-" * 70)

    total_time = sum(r.time_per_op_us for r in sorted_results)

    for i, r in enumerate(sorted_results[:10], 1):
        impact = (r.time_per_op_us / total_time) * 100
        print(f"{i:<6} {r.name:<35} {r.time_per_op_us:<15.2f} {impact:<15.1f}%")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Identify critical bottlenecks
    critical = [r for r in sorted_results if r.time_per_op_us > 10000]  # > 10ms
    high = [r for r in sorted_results if 1000 < r.time_per_op_us <= 10000]  # 1-10ms
    medium = [r for r in sorted_results if 100 < r.time_per_op_us <= 1000]  # 0.1-1ms

    if critical:
        print("\n🔴 CRITICAL (>10ms per op):")
        for r in critical:
            print(f"   - {r.name}: {r.time_per_op_us/1000:.1f}ms")

    if high:
        print("\n🟠 HIGH (1-10ms per op):")
        for r in high:
            print(f"   - {r.name}: {r.time_per_op_us/1000:.2f}ms")

    if medium:
        print("\n🟡 MEDIUM (0.1-1ms per op):")
        for r in medium:
            print(f"   - {r.name}: {r.time_per_op_us:.0f}µs")

    return sorted_results


def main():
    """Run all profiling."""
    print("=" * 70)
    print("zkML SYSTEM PERFORMANCE PROFILING")
    print("=" * 70)
    print("\nThis will measure the performance of all cryptographic operations.")
    print("Results will identify the exact bottlenecks for optimization.\n")

    all_results = []

    # Profile each component
    all_results.extend(profile_field_operations())
    all_results.extend(profile_curve_operations())
    all_results.extend(profile_polynomial_operations())
    all_results.extend(profile_kzg_operations())

    # Analyze bottlenecks
    analyze_bottlenecks(all_results)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. G1/G2 Scalar Multiplication:
   - Current: Double-and-add (O(256) doublings + additions)
   - Optimization: Windowed NAF or Pippenger for batch operations
   - Expected speedup: 3-5x for single, 10-50x for batch

2. Field Inversion:
   - Current: Fermat's little theorem (O(256) multiplications)
   - Optimization: Montgomery batch inversion for multiple inversions
   - Expected speedup: 10-100x for batch operations

3. FFT:
   - Current: Recursive Cooley-Tukey
   - Optimization: Iterative with bit-reversal, cache-friendly access
   - Expected speedup: 2-3x

4. Polynomial Multiplication:
   - Current: Naive O(n²)
   - Optimization: FFT-based O(n log n) - already implemented
   - Note: Ensure FFT is used for large polynomials

5. KZG Commit (Multi-Scalar Multiplication):
   - Current: Naive sum of scalar muls
   - Optimization: Pippenger's algorithm
   - Expected speedup: 5-20x depending on size
""")


if __name__ == "__main__":
    main()
