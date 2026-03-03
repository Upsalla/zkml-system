"""
KZG & BN254 fuzz tests — GA Blocker 6D.

Tests the KZG polynomial commitment scheme and BN254 curve primitives
against malformed inputs, edge cases, and boundary values.

A commitment scheme that accepts proofs for wrong evaluations
or crashes on edge-case inputs is cryptographically broken.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point, G2Point
from zkml_system.plonk.polynomial import Polynomial
from zkml_system.plonk.kzg import SRS, KZG, KZGCommitment, KZGProof
from zkml_system.plonk.plonk_kzg import TrustedSetup


def _make_kzg(max_degree: int = 16) -> KZG:
    """Create a KZG instance with deterministic tau."""
    srs = SRS.generate(max_degree, tau=Fr(7777))
    return KZG(srs)


# ===========================================================================
# Test Suite 1: KZG Commitment Edge Cases
# ===========================================================================

class TestKZGCommitmentEdgeCases(unittest.TestCase):
    """KZG commitment must handle edge-case polynomials correctly."""

    def setUp(self):
        self.kzg = _make_kzg()

    def test_commit_zero_polynomial(self):
        """Commitment to the zero polynomial should be the identity point."""
        zero_poly = Polynomial([Fr.zero()])
        com = self.kzg.commit(zero_poly)
        self.assertTrue(com.point.is_identity(),
                        "Zero polynomial commitment should be identity")

    def test_commit_constant_polynomial(self):
        """Commitment to a constant polynomial p(x) = c."""
        const_poly = Polynomial([Fr(42)])
        com = self.kzg.commit(const_poly)
        # C = 42 * G1 (since SRS[0] = G1)
        expected = G1Point.generator() * Fr(42)
        self.assertEqual(com.point, expected)

    def test_commit_degree_equals_max(self):
        """Polynomial of degree = max_degree should succeed."""
        max_deg = 16
        kzg = _make_kzg(max_deg)
        coeffs = [Fr(i + 1) for i in range(max_deg + 1)]
        poly = Polynomial(coeffs)
        com = kzg.commit(poly)
        self.assertTrue(com.point.is_on_curve())

    def test_commit_degree_exceeds_max_raises(self):
        """Polynomial degree > max_degree must raise ValueError."""
        max_deg = 4
        srs = SRS.generate(max_deg, tau=Fr(111))
        kzg = KZG(srs)
        # Degree 5 > max_degree 4
        poly = Polynomial([Fr(1)] * 6)
        with self.assertRaises(ValueError):
            kzg.commit(poly)


# ===========================================================================
# Test Suite 2: KZG Opening Proof Verification
# ===========================================================================

class TestKZGVerification(unittest.TestCase):
    """KZG verify must accept correct proofs and reject incorrect ones."""

    def setUp(self):
        self.kzg = _make_kzg()
        self.poly = Polynomial([Fr(3), Fr(2), Fr(1)])  # p(x) = 3 + 2x + x²
        self.com = self.kzg.commit(self.poly)

    def test_correct_proof_accepted(self):
        """Valid opening proof must verify."""
        z = Fr(5)
        proof, y = self.kzg.create_proof(self.poly, z)
        # y = 3 + 2*5 + 25 = 38
        self.assertEqual(y, Fr(38))
        result = self.kzg.verify(self.com, z, y, proof)
        self.assertTrue(result)

    def test_wrong_evaluation_value_rejected(self):
        """Correct proof but wrong claimed y → reject."""
        z = Fr(5)
        proof, y = self.kzg.create_proof(self.poly, z)
        wrong_y = y + Fr(1)  # 39 instead of 38
        result = self.kzg.verify(self.com, z, wrong_y, proof)
        self.assertFalse(result)

    def test_wrong_evaluation_point_rejected(self):
        """Proof for z=5 verified at z=6 → reject."""
        z = Fr(5)
        proof, y = self.kzg.create_proof(self.poly, z)
        wrong_z = Fr(6)
        result = self.kzg.verify(self.com, wrong_z, y, proof)
        self.assertFalse(result)

    def test_malformed_proof_point_rejected(self):
        """Replace proof π with generator → reject."""
        z = Fr(5)
        proof, y = self.kzg.create_proof(self.poly, z)
        bad_proof = KZGProof(G1Point.generator())
        result = self.kzg.verify(self.com, z, y, bad_proof)
        self.assertFalse(result)

    def test_identity_proof_point_rejected(self):
        """Replace proof π with identity → reject."""
        z = Fr(5)
        proof, y = self.kzg.create_proof(self.poly, z)
        bad_proof = KZGProof(G1Point.identity())
        result = self.kzg.verify(self.com, z, y, bad_proof)
        self.assertFalse(result)

    def test_wrong_commitment_rejected(self):
        """Correct proof but against a different commitment → reject."""
        z = Fr(5)
        proof, y = self.kzg.create_proof(self.poly, z)
        wrong_com = KZGCommitment(G1Point.generator())
        result = self.kzg.verify(wrong_com, z, y, proof)
        self.assertFalse(result)


# ===========================================================================
# Test Suite 3: Fr Field Boundary Values
# ===========================================================================

class TestFrBoundaryValues(unittest.TestCase):
    """Fr arithmetic at boundary values must not crash or wrap incorrectly."""

    def test_fr_zero(self):
        self.assertEqual(Fr.zero().to_int(), 0)
        self.assertEqual((Fr.zero() + Fr.zero()).to_int(), 0)

    def test_fr_one(self):
        self.assertEqual(Fr.one().to_int(), 1)

    def test_fr_modulus_minus_one(self):
        """Largest valid field element."""
        max_val = Fr.MODULUS - 1
        f = Fr(max_val)
        self.assertEqual(f.to_int(), max_val)

    def test_fr_modulus_wraps_to_zero(self):
        """Fr(MODULUS) should wrap to 0."""
        f = Fr(Fr.MODULUS)
        self.assertEqual(f.to_int(), 0)

    def test_fr_negative_one(self):
        """-Fr(1) should be MODULUS - 1."""
        neg_one = -Fr.one()
        self.assertEqual(neg_one.to_int(), Fr.MODULUS - 1)

    def test_fr_inverse_of_one(self):
        """1/1 = 1."""
        inv = Fr.one() / Fr.one()
        self.assertEqual(inv, Fr.one())

    def test_fr_mul_by_zero(self):
        """Anything * 0 = 0."""
        big = Fr(Fr.MODULUS - 1)
        self.assertEqual((big * Fr.zero()).to_int(), 0)

    def test_fr_additive_inverse(self):
        """a + (-a) = 0."""
        a = Fr(12345)
        self.assertEqual((a + (-a)).to_int(), 0)


# ===========================================================================
# Test Suite 4: KZG batch_verify
# ===========================================================================

class TestKZGBatchVerify(unittest.TestCase):
    """batch_verify must accept all-valid batches and reject batches with one bad entry."""

    def setUp(self):
        self.kzg = _make_kzg()

    def test_batch_verify_all_valid(self):
        """Batch of 3 valid proofs → accept."""
        polys = [
            Polynomial([Fr(1), Fr(2)]),
            Polynomial([Fr(3), Fr(4)]),
            Polynomial([Fr(5), Fr(6)]),
        ]
        points = [Fr(10), Fr(20), Fr(30)]

        coms = [self.kzg.commit(p) for p in polys]
        proofs = []
        values = []
        for p, z in zip(polys, points):
            proof, y = self.kzg.create_proof(p, z)
            proofs.append(proof)
            values.append(y)

        result = self.kzg.batch_verify(coms, points, values, proofs)
        self.assertTrue(result)

    def test_batch_verify_one_bad_value_rejects(self):
        """One wrong y in batch → reject entire batch."""
        polys = [
            Polynomial([Fr(1), Fr(2)]),
            Polynomial([Fr(3), Fr(4)]),
        ]
        points = [Fr(10), Fr(20)]

        coms = [self.kzg.commit(p) for p in polys]
        proofs = []
        values = []
        for p, z in zip(polys, points):
            proof, y = self.kzg.create_proof(p, z)
            proofs.append(proof)
            values.append(y)

        # Corrupt second value
        values[1] = values[1] + Fr(1)
        result = self.kzg.batch_verify(coms, points, values, proofs)
        self.assertFalse(result)

    def test_batch_verify_one_bad_proof_rejects(self):
        """One malformed proof in batch → reject entire batch."""
        polys = [
            Polynomial([Fr(1), Fr(2)]),
            Polynomial([Fr(3), Fr(4)]),
        ]
        points = [Fr(10), Fr(20)]

        coms = [self.kzg.commit(p) for p in polys]
        proofs = []
        values = []
        for p, z in zip(polys, points):
            proof, y = self.kzg.create_proof(p, z)
            proofs.append(proof)
            values.append(y)

        # Replace second proof with generator
        proofs[1] = KZGProof(G1Point.generator())
        result = self.kzg.batch_verify(coms, points, values, proofs)
        self.assertFalse(result)


# ===========================================================================
# Test Suite 5: G1Point Edge Cases
# ===========================================================================

class TestG1PointEdgeCases(unittest.TestCase):
    """BN254 G1 curve point operations at boundary conditions."""

    def test_identity_is_on_curve(self):
        self.assertTrue(G1Point.identity().is_identity())

    def test_generator_is_on_curve(self):
        g = G1Point.generator()
        self.assertTrue(g.is_on_curve())
        self.assertFalse(g.is_identity())

    def test_add_identity_returns_same(self):
        """P + O = P."""
        g = G1Point.generator()
        result = g + G1Point.identity()
        self.assertEqual(result, g)

    def test_scalar_mul_zero_gives_identity(self):
        """0 * G = O."""
        result = G1Point.generator() * Fr.zero()
        self.assertTrue(result.is_identity())

    def test_scalar_mul_one_gives_generator(self):
        """1 * G = G."""
        g = G1Point.generator()
        result = g * Fr.one()
        self.assertEqual(result, g)

    def test_double_via_add_equals_scalar_mul(self):
        """G + G = 2 * G."""
        g = G1Point.generator()
        doubled = g + g
        scalar_doubled = g * Fr(2)
        self.assertEqual(doubled, scalar_doubled)


if __name__ == "__main__":
    unittest.main()
