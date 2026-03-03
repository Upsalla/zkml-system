"""
Fuzzing Tests for zkML System

This module performs randomized testing to find edge cases and crashes.
It tests:
1. Field arithmetic with boundary values
2. Curve operations with edge cases
3. Polynomial operations with extreme inputs
4. Proof system with malformed inputs

Fuzzing Strategy:
- Boundary values (0, 1, p-1, p, p+1)
- Random values in valid range
- Invalid values (negative, overflow)
- Special structures (all zeros, all ones)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import traceback
from typing import List, Callable, Any, Tuple
from dataclasses import dataclass


@dataclass
class FuzzResult:
    """Result of a fuzzing test."""
    test_name: str
    iterations: int
    crashes: int
    failures: int
    crash_inputs: List[Any]


class Fuzzer:
    """Fuzzer for cryptographic operations."""

    def __init__(self, seed: int = 42):
        """Initialize with a seed for reproducibility."""
        random.seed(seed)
        self.results: List[FuzzResult] = []

    def fuzz_field_arithmetic(self, iterations: int = 1000) -> FuzzResult:
        """Fuzz field arithmetic operations."""
        print(f"\nFuzzing Field Arithmetic ({iterations} iterations)...")

        from zkml_system.crypto.bn254.field import Fp, Fr

        crashes = 0
        failures = 0
        crash_inputs = []

        # Boundary values
        boundary_values = [
            0, 1, 2,
            Fp.MODULUS - 1, Fp.MODULUS, Fp.MODULUS + 1,
            Fp.MODULUS // 2,
            2**128, 2**256 - 1
        ]

        for i in range(iterations):
            try:
                # Generate random or boundary value
                if i < len(boundary_values):
                    val = boundary_values[i]
                else:
                    val = random.randint(0, 2**256)

                # Test Fp
                a = Fp(val)
                b = Fp(random.randint(1, Fp.MODULUS - 1))

                # Operations that should not crash
                _ = a + b
                _ = a - b
                _ = a * b
                _ = -a

                # Division by non-zero
                if b.to_int() != 0:
                    _ = a / b

                # Inversion of non-zero
                if a.to_int() != 0:
                    inv = a.inverse()
                    # Verify: a * a^(-1) = 1
                    if a * inv != Fp.one():
                        failures += 1
                        crash_inputs.append(('Fp_inverse_verify', val))

                # Test Fr similarly
                a_fr = Fr(val % Fr.MODULUS)
                b_fr = Fr(random.randint(1, Fr.MODULUS - 1))
                _ = a_fr + b_fr
                _ = a_fr * b_fr

            except ZeroDivisionError:
                pass  # Expected for zero inversion
            except Exception as e:
                crashes += 1
                crash_inputs.append(('field', val, str(e)))
                if crashes <= 3:
                    print(f"  Crash at iteration {i}: {e}")

        result = FuzzResult(
            test_name="Field Arithmetic",
            iterations=iterations,
            crashes=crashes,
            failures=failures,
            crash_inputs=crash_inputs[:10]
        )
        self.results.append(result)

        status = "PASS" if crashes == 0 and failures == 0 else "ISSUES FOUND"
        print(f"  Result: {status} (crashes={crashes}, failures={failures})")
        return result

    def fuzz_curve_operations(self, iterations: int = 100) -> FuzzResult:
        """Fuzz elliptic curve operations."""
        print(f"\nFuzzing Curve Operations ({iterations} iterations)...")

        from zkml_system.crypto.bn254.curve import G1Point, G2Point
        from zkml_system.crypto.bn254.field import Fr

        crashes = 0
        failures = 0
        crash_inputs = []

        g1 = G1Point.generator()
        identity = G1Point.identity()

        # Boundary scalars
        boundary_scalars = [
            0, 1, 2,
            Fr.MODULUS - 1, Fr.MODULUS, Fr.MODULUS + 1,
            2**128, 2**256 - 1
        ]

        for i in range(iterations):
            try:
                # Generate scalar
                if i < len(boundary_scalars):
                    scalar = boundary_scalars[i]
                else:
                    scalar = random.randint(0, 2**256)

                # Scalar multiplication
                result = g1 * scalar

                # Result should be on curve
                if not result.is_on_curve() and not result.is_identity():
                    failures += 1
                    crash_inputs.append(('not_on_curve', scalar))

                # Test addition
                p1 = g1 * random.randint(1, 1000)
                p2 = g1 * random.randint(1, 1000)
                p3 = p1 + p2

                if not p3.is_on_curve() and not p3.is_identity():
                    failures += 1
                    crash_inputs.append(('addition_not_on_curve', i))

                # Test identity properties
                if p1 + identity != p1:
                    failures += 1
                    crash_inputs.append(('identity_add', i))

                # Test negation
                neg_p1 = -p1
                if not neg_p1.is_on_curve() and not neg_p1.is_identity():
                    failures += 1
                    crash_inputs.append(('negation_not_on_curve', i))

                # p + (-p) should be identity
                sum_result = p1 + neg_p1
                if not sum_result.is_identity():
                    failures += 1
                    crash_inputs.append(('negation_sum', i))

            except Exception as e:
                crashes += 1
                crash_inputs.append(('curve', i, str(e)))
                if crashes <= 3:
                    print(f"  Crash at iteration {i}: {e}")

        result = FuzzResult(
            test_name="Curve Operations",
            iterations=iterations,
            crashes=crashes,
            failures=failures,
            crash_inputs=crash_inputs[:10]
        )
        self.results.append(result)

        status = "PASS" if crashes == 0 and failures == 0 else "ISSUES FOUND"
        print(f"  Result: {status} (crashes={crashes}, failures={failures})")
        return result

    def fuzz_polynomial_operations(self, iterations: int = 100) -> FuzzResult:
        """Fuzz polynomial operations."""
        print(f"\nFuzzing Polynomial Operations ({iterations} iterations)...")

        from zkml_system.crypto.bn254.field import Fr
        from zkml_system.plonk.polynomial import Polynomial

        crashes = 0
        failures = 0
        crash_inputs = []

        for i in range(iterations):
            try:
                # Generate random polynomial
                degree = random.randint(0, 64)
                coeffs = [Fr(random.randint(0, Fr.MODULUS - 1)) for _ in range(degree + 1)]
                poly = Polynomial(coeffs)

                # Evaluation at random point
                z = Fr(random.randint(0, Fr.MODULUS - 1))
                y = poly.evaluate(z)

                # Verify evaluation is in field
                if y.to_int() >= Fr.MODULUS:
                    failures += 1
                    crash_inputs.append(('eval_overflow', i))

                # Test polynomial arithmetic
                poly2 = Polynomial([Fr(random.randint(1, 100)) for _ in range(random.randint(1, 10))])

                # Addition
                sum_poly = poly + poly2
                # Multiplication (only for small degrees to avoid timeout)
                if degree < 16:
                    prod_poly = poly * poly2

                # Test zero polynomial
                zero_poly = Polynomial([Fr.zero()])
                sum_with_zero = poly + zero_poly

                # Test edge cases
                empty_poly = Polynomial([])
                single_poly = Polynomial([Fr.one()])

            except Exception as e:
                crashes += 1
                crash_inputs.append(('polynomial', i, str(e)))
                if crashes <= 3:
                    print(f"  Crash at iteration {i}: {e}")

        result = FuzzResult(
            test_name="Polynomial Operations",
            iterations=iterations,
            crashes=crashes,
            failures=failures,
            crash_inputs=crash_inputs[:10]
        )
        self.results.append(result)

        status = "PASS" if crashes == 0 and failures == 0 else "ISSUES FOUND"
        print(f"  Result: {status} (crashes={crashes}, failures={failures})")
        return result

    def fuzz_kzg_operations(self, iterations: int = 20) -> FuzzResult:
        """Fuzz KZG commitment operations."""
        print(f"\nFuzzing KZG Operations ({iterations} iterations)...")

        from zkml_system.crypto.bn254.field import Fr
        from zkml_system.plonk.polynomial import Polynomial
        from zkml_system.plonk.kzg import SRS, KZG

        crashes = 0
        failures = 0
        crash_inputs = []

        # Generate SRS once
        srs = SRS.generate(32)
        kzg = KZG(srs)

        for i in range(iterations):
            try:
                # Generate random polynomial
                degree = random.randint(1, 16)
                coeffs = [Fr(random.randint(0, Fr.MODULUS - 1)) for _ in range(degree)]
                poly = Polynomial(coeffs)

                # Commit
                commitment = kzg.commit(poly)

                # Verify commitment is on curve
                if not commitment.point.is_on_curve():
                    failures += 1
                    crash_inputs.append(('commit_not_on_curve', i))

                # Create and verify proof
                z = Fr(random.randint(1, Fr.MODULUS - 1))
                proof, y = kzg.create_proof(poly, z)

                # Verify should pass
                if not kzg.verify(commitment, z, y, proof):
                    failures += 1
                    crash_inputs.append(('valid_proof_rejected', i))

                # Verify with wrong y should fail
                wrong_y = y + Fr.one()
                if kzg.verify(commitment, z, wrong_y, proof):
                    failures += 1
                    crash_inputs.append(('invalid_proof_accepted', i))

            except Exception as e:
                crashes += 1
                crash_inputs.append(('kzg', i, str(e)))
                if crashes <= 3:
                    print(f"  Crash at iteration {i}: {e}")
                    traceback.print_exc()

        result = FuzzResult(
            test_name="KZG Operations",
            iterations=iterations,
            crashes=crashes,
            failures=failures,
            crash_inputs=crash_inputs[:10]
        )
        self.results.append(result)

        status = "PASS" if crashes == 0 and failures == 0 else "ISSUES FOUND"
        print(f"  Result: {status} (crashes={crashes}, failures={failures})")
        return result

    def generate_report(self):
        """Generate a summary report."""
        print("\n" + "=" * 60)
        print("FUZZING SUMMARY")
        print("=" * 60)

        total_crashes = sum(r.crashes for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_iterations = sum(r.iterations for r in self.results)

        print(f"\nTotal iterations: {total_iterations}")
        print(f"Total crashes: {total_crashes}")
        print(f"Total failures: {total_failures}")

        print("\nPer-component results:")
        for r in self.results:
            status = "PASS" if r.crashes == 0 and r.failures == 0 else "FAIL"
            print(f"  {r.test_name}: {status} ({r.crashes} crashes, {r.failures} failures)")
            if r.crash_inputs:
                print(f"    Sample crash inputs: {r.crash_inputs[:3]}")

        if total_crashes > 0 or total_failures > 0:
            print("\n🔴 FUZZING FOUND ISSUES - Review crash inputs above")
        else:
            print("\n✅ FUZZING PASSED - No crashes or failures detected")


def main():
    """Run all fuzzing tests."""
    print("=" * 60)
    print("zkML SYSTEM FUZZING TESTS")
    print("=" * 60)
    print("\nThis will run randomized tests to find edge cases and crashes.")
    print("Using seed=42 for reproducibility.\n")

    fuzzer = Fuzzer(seed=42)

    fuzzer.fuzz_field_arithmetic(iterations=500)
    fuzzer.fuzz_curve_operations(iterations=50)
    fuzzer.fuzz_polynomial_operations(iterations=50)
    fuzzer.fuzz_kzg_operations(iterations=10)

    fuzzer.generate_report()


if __name__ == "__main__":
    main()
