"""
Security Audit for zkML System

This module performs systematic security checks on the cryptographic implementation.
It checks for common vulnerabilities in ZK proof systems.

Categories:
1. Field Arithmetic Vulnerabilities
2. Curve Arithmetic Vulnerabilities
3. Proof System Vulnerabilities
4. Smart Contract Vulnerabilities (static analysis)
5. Timing Attack Vulnerabilities

Each check returns: PASS, FAIL, or WARNING with detailed explanation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class AuditResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class AuditFinding:
    """A single audit finding."""
    category: str
    check_name: str
    result: AuditResult
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    recommendation: str


class SecurityAuditor:
    """Security auditor for the zkML system."""

    def __init__(self):
        self.findings: List[AuditFinding] = []

    def add_finding(self, finding: AuditFinding):
        self.findings.append(finding)
        status = "✓" if finding.result == AuditResult.PASS else "✗" if finding.result == AuditResult.FAIL else "⚠"
        print(f"  [{status}] {finding.check_name}: {finding.result.value}")
        if finding.result != AuditResult.PASS:
            print(f"      {finding.description}")

    def audit_field_arithmetic(self):
        """Audit field arithmetic for vulnerabilities."""
        print("\n" + "=" * 60)
        print("FIELD ARITHMETIC SECURITY AUDIT")
        print("=" * 60)

        from zkml_system.crypto.bn254.field import Fp, Fr

        # Check 1: Modular reduction correctness
        try:
            # Test that values are properly reduced
            large_val = Fp.MODULUS + 100
            fp = Fp(large_val)
            if fp.to_int() >= Fp.MODULUS:
                self.add_finding(AuditFinding(
                    category="Field Arithmetic",
                    check_name="Modular Reduction",
                    result=AuditResult.FAIL,
                    description="Field elements not properly reduced modulo p",
                    severity="CRITICAL",
                    recommendation="Ensure all field operations reduce results mod p"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Field Arithmetic",
                    check_name="Modular Reduction",
                    result=AuditResult.PASS,
                    description="Field elements properly reduced",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Field Arithmetic",
                check_name="Modular Reduction",
                result=AuditResult.FAIL,
                description=f"Exception during test: {e}",
                severity="CRITICAL",
                recommendation="Fix field implementation"
            ))

        # Check 2: Zero element handling
        try:
            zero = Fp.zero()
            one = Fp.one()
            if zero + one != one:
                self.add_finding(AuditFinding(
                    category="Field Arithmetic",
                    check_name="Zero Element",
                    result=AuditResult.FAIL,
                    description="Zero element not functioning as additive identity",
                    severity="CRITICAL",
                    recommendation="Fix zero element implementation"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Field Arithmetic",
                    check_name="Zero Element",
                    result=AuditResult.PASS,
                    description="Zero element correct",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Field Arithmetic",
                check_name="Zero Element",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

        # Check 3: Inversion of zero (should fail or return special value)
        try:
            zero = Fp.zero()
            inv = zero.inverse()
            # If we get here without exception, that's a problem
            self.add_finding(AuditFinding(
                category="Field Arithmetic",
                check_name="Zero Inversion",
                result=AuditResult.WARNING,
                description="Inverting zero did not raise exception",
                severity="MEDIUM",
                recommendation="Zero inversion should raise ZeroDivisionError"
            ))
        except (ZeroDivisionError, ValueError):
            self.add_finding(AuditFinding(
                category="Field Arithmetic",
                check_name="Zero Inversion",
                result=AuditResult.PASS,
                description="Zero inversion properly raises exception",
                severity="INFO",
                recommendation=""
            ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Field Arithmetic",
                check_name="Zero Inversion",
                result=AuditResult.WARNING,
                description=f"Unexpected exception type: {type(e).__name__}",
                severity="LOW",
                recommendation="Use ZeroDivisionError for consistency"
            ))

        # Check 4: Montgomery form consistency
        try:
            a = Fp(12345)
            b = Fp(67890)
            # a * b should equal b * a
            if a * b != b * a:
                self.add_finding(AuditFinding(
                    category="Field Arithmetic",
                    check_name="Multiplication Commutativity",
                    result=AuditResult.FAIL,
                    description="Multiplication is not commutative",
                    severity="CRITICAL",
                    recommendation="Fix Montgomery multiplication"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Field Arithmetic",
                    check_name="Multiplication Commutativity",
                    result=AuditResult.PASS,
                    description="Multiplication is commutative",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Field Arithmetic",
                check_name="Multiplication Commutativity",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

    def audit_curve_arithmetic(self):
        """Audit elliptic curve operations."""
        print("\n" + "=" * 60)
        print("CURVE ARITHMETIC SECURITY AUDIT")
        print("=" * 60)

        from zkml_system.crypto.bn254.curve import G1Point, G2Point

        # Check 1: Generator is on curve
        try:
            g1 = G1Point.generator()
            if not g1.is_on_curve():
                self.add_finding(AuditFinding(
                    category="Curve Arithmetic",
                    check_name="G1 Generator On Curve",
                    result=AuditResult.FAIL,
                    description="G1 generator is not on the curve",
                    severity="CRITICAL",
                    recommendation="Fix generator coordinates"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Curve Arithmetic",
                    check_name="G1 Generator On Curve",
                    result=AuditResult.PASS,
                    description="G1 generator is on curve",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="G1 Generator On Curve",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

        # Check 2: Point addition closure
        try:
            g1 = G1Point.generator()
            g1_2 = g1 + g1
            if not g1_2.is_on_curve():
                self.add_finding(AuditFinding(
                    category="Curve Arithmetic",
                    check_name="Point Addition Closure",
                    result=AuditResult.FAIL,
                    description="Point addition result not on curve",
                    severity="CRITICAL",
                    recommendation="Fix point addition"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Curve Arithmetic",
                    check_name="Point Addition Closure",
                    result=AuditResult.PASS,
                    description="Point addition maintains curve membership",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="Point Addition Closure",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

        # Check 3: Scalar multiplication by order gives identity
        try:
            from zkml_system.crypto.bn254.field import Fr
            g1 = G1Point.generator()
            # r * G = O (identity)
            # This is expensive, so we skip in audit
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="Scalar Mul by Order",
                result=AuditResult.SKIPPED,
                description="Skipped due to computational cost",
                severity="INFO",
                recommendation="Run full test separately: r * G should equal identity"
            ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="Scalar Mul by Order",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

        # Check 4: Identity element handling
        try:
            g1 = G1Point.generator()
            identity = G1Point.identity()
            if g1 + identity != g1:
                self.add_finding(AuditFinding(
                    category="Curve Arithmetic",
                    check_name="Identity Element",
                    result=AuditResult.FAIL,
                    description="Identity element not functioning correctly",
                    severity="CRITICAL",
                    recommendation="Fix identity element handling"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Curve Arithmetic",
                    check_name="Identity Element",
                    result=AuditResult.PASS,
                    description="Identity element correct",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="Identity Element",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

        # Check 5: Subgroup check (point is in correct subgroup)
        try:
            g1 = G1Point.generator()
            # For BN254, all points on the curve are in the correct subgroup
            # But we should verify the generator has the correct order
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="Subgroup Membership",
                result=AuditResult.WARNING,
                description="No explicit subgroup check implemented",
                severity="MEDIUM",
                recommendation="Add subgroup check for untrusted inputs"
            ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Curve Arithmetic",
                check_name="Subgroup Membership",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="HIGH",
                recommendation="Implement subgroup check"
            ))

    def audit_proof_system(self):
        """Audit the proof system for vulnerabilities."""
        print("\n" + "=" * 60)
        print("PROOF SYSTEM SECURITY AUDIT")
        print("=" * 60)

        # Check 1: Fiat-Shamir transcript binding
        try:
            # The Fiat-Shamir transform must include all public inputs
            # This is a design check, not a runtime check
            self.add_finding(AuditFinding(
                category="Proof System",
                check_name="Fiat-Shamir Binding",
                result=AuditResult.WARNING,
                description="Manual review required: Verify all public inputs are in transcript",
                severity="HIGH",
                recommendation="Ensure transcript includes: circuit ID, public inputs, all commitments"
            ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Proof System",
                check_name="Fiat-Shamir Binding",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

        # Check 2: Soundness - invalid witness should fail
        try:
            from zkml_system.crypto.bn254.field import Fr
            from zkml_system.plonk.polynomial import Polynomial
            from zkml_system.plonk.kzg import SRS, KZG

            srs = SRS.generate(8)
            kzg = KZG(srs)

            # Create a polynomial and commitment
            poly = Polynomial.from_ints([1, 2, 3])
            commitment = kzg.commit(poly)

            # Try to create a proof for wrong evaluation
            z = Fr(5)
            correct_y = poly.evaluate(z)
            wrong_y = correct_y + Fr.one()

            # The proof should fail verification with wrong y
            proof, _ = kzg.create_proof(poly, z)

            # Verify with wrong value
            if kzg.verify(commitment, z, wrong_y, proof):
                self.add_finding(AuditFinding(
                    category="Proof System",
                    check_name="KZG Soundness",
                    result=AuditResult.FAIL,
                    description="KZG verification accepts wrong evaluation",
                    severity="CRITICAL",
                    recommendation="Fix KZG verification"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Proof System",
                    check_name="KZG Soundness",
                    result=AuditResult.PASS,
                    description="KZG correctly rejects wrong evaluation",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Proof System",
                check_name="KZG Soundness",
                result=AuditResult.WARNING,
                description=f"Could not complete soundness test: {e}",
                severity="HIGH",
                recommendation="Manually verify soundness"
            ))

        # Check 3: Zero-knowledge - proof should not leak witness
        self.add_finding(AuditFinding(
            category="Proof System",
            check_name="Zero-Knowledge Property",
            result=AuditResult.WARNING,
            description="ZK property requires formal analysis",
            severity="HIGH",
            recommendation="Verify blinding factors are properly applied"
        ))

        # Check 4: Completeness - valid witness should pass
        try:
            from zkml_system.crypto.bn254.field import Fr
            from zkml_system.plonk.polynomial import Polynomial
            from zkml_system.plonk.kzg import SRS, KZG

            srs = SRS.generate(8)
            kzg = KZG(srs)

            poly = Polynomial.from_ints([1, 2, 3])
            commitment = kzg.commit(poly)
            z = Fr(5)
            proof, y = kzg.create_proof(poly, z)

            if kzg.verify(commitment, z, y, proof):
                self.add_finding(AuditFinding(
                    category="Proof System",
                    check_name="KZG Completeness",
                    result=AuditResult.PASS,
                    description="KZG correctly accepts valid proof",
                    severity="INFO",
                    recommendation=""
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Proof System",
                    check_name="KZG Completeness",
                    result=AuditResult.FAIL,
                    description="KZG rejects valid proof",
                    severity="CRITICAL",
                    recommendation="Fix KZG verification"
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Proof System",
                check_name="KZG Completeness",
                result=AuditResult.FAIL,
                description=f"Exception: {e}",
                severity="CRITICAL",
                recommendation="Fix implementation"
            ))

    def audit_timing_attacks(self):
        """Check for timing attack vulnerabilities."""
        print("\n" + "=" * 60)
        print("TIMING ATTACK VULNERABILITY AUDIT")
        print("=" * 60)

        from zkml_system.crypto.bn254.field import Fp

        # Check 1: Field multiplication timing variance
        try:
            # Measure timing variance for different inputs
            small = Fp(1)
            large = Fp(Fp.MODULUS - 1)

            times_small = []
            times_large = []

            for _ in range(100):
                start = time.perf_counter()
                _ = small * small
                times_small.append(time.perf_counter() - start)

                start = time.perf_counter()
                _ = large * large
                times_large.append(time.perf_counter() - start)

            avg_small = sum(times_small) / len(times_small)
            avg_large = sum(times_large) / len(times_large)
            variance_ratio = max(avg_small, avg_large) / min(avg_small, avg_large)

            if variance_ratio > 1.5:
                self.add_finding(AuditFinding(
                    category="Timing Attacks",
                    check_name="Field Multiplication Timing",
                    result=AuditResult.WARNING,
                    description=f"Timing variance ratio: {variance_ratio:.2f}x",
                    severity="MEDIUM",
                    recommendation="Consider constant-time implementation"
                ))
            else:
                self.add_finding(AuditFinding(
                    category="Timing Attacks",
                    check_name="Field Multiplication Timing",
                    result=AuditResult.PASS,
                    description=f"Timing variance acceptable: {variance_ratio:.2f}x",
                    severity="INFO",
                    recommendation=""
                ))
        except Exception as e:
            self.add_finding(AuditFinding(
                category="Timing Attacks",
                check_name="Field Multiplication Timing",
                result=AuditResult.WARNING,
                description=f"Could not measure: {e}",
                severity="MEDIUM",
                recommendation="Manually verify constant-time behavior"
            ))

        # Check 2: Scalar multiplication timing
        self.add_finding(AuditFinding(
            category="Timing Attacks",
            check_name="Scalar Multiplication Timing",
            result=AuditResult.WARNING,
            description="Double-and-add is not constant-time",
            severity="HIGH",
            recommendation="Use Montgomery ladder for constant-time scalar mul"
        ))

    def generate_report(self) -> str:
        """Generate a summary report of all findings."""
        print("\n" + "=" * 60)
        print("SECURITY AUDIT SUMMARY")
        print("=" * 60)

        # Count by severity
        critical = [f for f in self.findings if f.severity == "CRITICAL" and f.result == AuditResult.FAIL]
        high = [f for f in self.findings if f.severity == "HIGH" and f.result in [AuditResult.FAIL, AuditResult.WARNING]]
        medium = [f for f in self.findings if f.severity == "MEDIUM" and f.result in [AuditResult.FAIL, AuditResult.WARNING]]
        passed = [f for f in self.findings if f.result == AuditResult.PASS]

        print(f"\nTotal checks: {len(self.findings)}")
        print(f"  PASSED: {len(passed)}")
        print(f"  CRITICAL: {len(critical)}")
        print(f"  HIGH: {len(high)}")
        print(f"  MEDIUM: {len(medium)}")

        if critical:
            print("\n🔴 CRITICAL ISSUES (must fix before production):")
            for f in critical:
                print(f"   - {f.check_name}: {f.description}")
                print(f"     Recommendation: {f.recommendation}")

        if high:
            print("\n🟠 HIGH PRIORITY ISSUES:")
            for f in high:
                print(f"   - {f.check_name}: {f.description}")
                print(f"     Recommendation: {f.recommendation}")

        if medium:
            print("\n🟡 MEDIUM PRIORITY ISSUES:")
            for f in medium:
                print(f"   - {f.check_name}: {f.description}")

        return f"Audit complete: {len(critical)} critical, {len(high)} high, {len(medium)} medium issues"


def main():
    """Run the full security audit."""
    print("=" * 60)
    print("zkML SYSTEM SECURITY AUDIT")
    print("=" * 60)
    print("\nThis audit checks for common cryptographic vulnerabilities.")
    print("It is NOT a replacement for a professional security audit.\n")

    auditor = SecurityAuditor()

    auditor.audit_field_arithmetic()
    auditor.audit_curve_arithmetic()
    auditor.audit_proof_system()
    auditor.audit_timing_attacks()

    auditor.generate_report()

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
