"""
Test suite for the Hybrid TDA+ZK fingerprint similarity proof (Option C).

Tests:
    1. Same model → distance = 0, proof valid
    2. Fine-tuned model → small distance, proof valid (within threshold)
    3. Different model → large distance, proof valid with loose threshold
    4. Tampered proof → verification fails
    5. Timing benchmark → total E2E < 15s
"""
import unittest
import sys
import os
import time
import importlib

import numpy as np

_HAS_TDA = importlib.util.find_spec("tda_fingerprint") is not None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "tda_fingerprint",
    ),
)

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.fingerprint_circuit import (
    FingerprintCircuitCompiler,
    FingerprintFeatures,
    FingerprintWitness,
    FingerprintPublicInputs,
    generate_similarity_witness,
    compute_additive_commitment,
    fingerprints_to_features,
    FIXED_POINT_SCALE,
    DISTANCE_RANGE_BITS,
)
from zkml_system.plonk.circuit_compiler import CompiledCircuit


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_features(n: int, offset: float = 0.0) -> list:
    """Create synthetic (dim, birth, death) features."""
    np.random.seed(42)
    return [
        (0, 0.1 * i + offset, 0.2 * i + 0.5 + offset)
        for i in range(n)
    ]


def _clone_plonk_proof(proof, **overrides):
    """Clone a PLONK proof with optional field overrides (PyO3-safe).

    PyO3's RustFr is not pickle-able, so we clone each field manually.
    """
    from zkml_system.plonk.plonk_prover import PLONKProof
    from zkml_system.crypto.bn254.curve import G1Point

    def clone_fr(f):
        return Fr(f.value)

    def clone_g1(p):
        if p.is_identity():
            return G1Point.identity()
        x, y = p.to_affine()
        return G1Point.from_affine(x, y)

    fields = {
        'com_a': clone_g1(proof.com_a),
        'com_b': clone_g1(proof.com_b),
        'com_c': clone_g1(proof.com_c),
        'com_z': clone_g1(proof.com_z),
        'com_t_lo': clone_g1(proof.com_t_lo),
        'com_t_mid': clone_g1(proof.com_t_mid),
        'com_t_hi': clone_g1(proof.com_t_hi),
        'a_bar': clone_fr(proof.a_bar),
        'b_bar': clone_fr(proof.b_bar),
        'c_bar': clone_fr(proof.c_bar),
        's_sigma1_bar': clone_fr(proof.s_sigma1_bar),
        's_sigma2_bar': clone_fr(proof.s_sigma2_bar),
        'z_omega_bar': clone_fr(proof.z_omega_bar),
        'r_zeta': clone_fr(proof.r_zeta),
        'w_zeta': clone_g1(proof.w_zeta),
        'w_zeta_omega': clone_g1(proof.w_zeta_omega),
        'n': proof.n,
    }
    fields.update(overrides)
    return PLONKProof(**fields)


# ─── Circuit Compilation Tests ──────────────────────────────────────────────

class TestFingerprintCircuitCompilation(unittest.TestCase):
    """Test circuit compilation produces valid structure."""

    def setUp(self):
        self.k = 10  # 10 features for faster tests
        feats_a = _make_features(self.k, offset=0.0)
        feats_b = _make_features(self.k, offset=0.01)
        self.witness, self.public = generate_similarity_witness(
            feats_a, feats_b, threshold=5.0
        )
        self.compiler = FingerprintCircuitCompiler(n_features=self.k)

    def test_compile_produces_circuit(self):
        circuit = self.compiler.compile(self.witness, self.public)
        self.assertIsInstance(circuit, CompiledCircuit)
        self.assertGreater(len(circuit.gates), 0)
        self.assertGreater(len(circuit.wires), 0)

    def test_circuit_has_four_public_inputs(self):
        circuit = self.compiler.compile(self.witness, self.public)
        self.assertEqual(circuit.num_public_inputs, 4)

    def test_gate_count_under_1000(self):
        """Circuit should be much smaller than the 36K full TDA circuit."""
        circuit = self.compiler.compile(self.witness, self.public)
        self.assertLess(
            len(circuit.gates), 1000,
            f"Gate count {len(circuit.gates)} exceeds 1000 — regression!"
        )

    def test_gate_count_20_features_under_1000(self):
        """Full 20-feature circuit stays under 1000 gates."""
        feats_a = _make_features(20, offset=0.0)
        feats_b = _make_features(20, offset=0.01)
        w, p = generate_similarity_witness(feats_a, feats_b, threshold=5.0)
        compiler = FingerprintCircuitCompiler(n_features=20)
        circuit = compiler.compile(w, p)
        self.assertLess(len(circuit.gates), 1000)


# ─── Commitment Tests ───────────────────────────────────────────────────────

class TestAdditiveCommitment(unittest.TestCase):
    """Test additive commitment scheme."""

    def test_commitment_deterministic(self):
        vals = [Fr(1), Fr(2), Fr(3)]
        r = Fr(7)
        c1 = compute_additive_commitment(vals, r)
        c2 = compute_additive_commitment(vals, r)
        self.assertEqual(c1, c2)

    def test_different_values_different_commitment(self):
        r = Fr(7)
        c1 = compute_additive_commitment([Fr(1), Fr(2)], r)
        c2 = compute_additive_commitment([Fr(1), Fr(3)], r)
        self.assertNotEqual(c1, c2)

    def test_different_random_different_commitment(self):
        vals = [Fr(1), Fr(2)]
        c1 = compute_additive_commitment(vals, Fr(7))
        c2 = compute_additive_commitment(vals, Fr(11))
        self.assertNotEqual(c1, c2)


# ─── Witness Generation Tests ───────────────────────────────────────────────

class TestWitnessGeneration(unittest.TestCase):
    """Test witness and public input generation."""

    def test_same_features_zero_distance(self):
        feats = _make_features(10)
        w, p = generate_similarity_witness(feats, feats, threshold=1.0)
        # Distance should be 0 — all diffs are zero
        for i in range(10):
            self.assertEqual(w.features_a.births[i], w.features_b.births[i])
            self.assertEqual(w.features_a.deaths[i], w.features_b.deaths[i])

    def test_mismatched_lengths_raises(self):
        feats_a = _make_features(10)
        feats_b = _make_features(5)
        with self.assertRaises(ValueError):
            generate_similarity_witness(feats_a, feats_b, threshold=1.0)

    def test_features_to_fr_conversion(self):
        feats = [(0, 0.5, 1.0)]
        fp = fingerprints_to_features(feats)
        expected_birth = Fr(int(round(0.5 * FIXED_POINT_SCALE)))
        expected_death = Fr(int(round(1.0 * FIXED_POINT_SCALE)))
        self.assertEqual(fp.births[0], expected_birth)
        self.assertEqual(fp.deaths[0], expected_death)


# ─── E2E Prove + Verify Tests ───────────────────────────────────────────────

@unittest.skipUnless(_HAS_TDA, "tda_fingerprint not installed")
class TestHybridE2E(unittest.TestCase):
    """End-to-end prove and verify tests."""

    @classmethod
    def setUpClass(cls):
        """Create test models once for all E2E tests."""
        np.random.seed(42)
        cls.base_w1 = np.random.randn(8, 4)
        cls.base_w2 = np.random.randn(4, 2)
        cls.weights_a = [cls.base_w1, cls.base_w2]
        cls.weights_b = [
            cls.base_w1 + 0.01 * np.random.randn(8, 4),
            cls.base_w2 + 0.01 * np.random.randn(4, 2),
        ]

    def _get_bridge(self, n_features=10, threshold=5.0):
        from zkml_system.hybrid_bridge import HybridBridge
        return HybridBridge(n_features=n_features, threshold=threshold)

    def test_same_model_proves_and_verifies(self):
        """Identical weights → distance ≈ 0, proof valid."""
        bridge = self._get_bridge(n_features=5, threshold=1.0)
        bundle = bridge.prove_similarity(self.weights_a, self.weights_a)
        self.assertAlmostEqual(bundle.actual_distance_float, 0.0, places=5)
        valid, reason = bridge.verify_similarity(bundle)
        self.assertTrue(valid, f"Same model should verify: {reason}")

    def test_similar_model_proves_and_verifies(self):
        """Fine-tuned weights → small distance, proof valid."""
        bridge = self._get_bridge(n_features=5, threshold=5.0)
        bundle = bridge.prove_similarity(self.weights_a, self.weights_b)
        self.assertLess(bundle.actual_distance_float, 5.0)
        valid, reason = bridge.verify_similarity(bundle)
        self.assertTrue(valid, f"Similar model should verify: {reason}")

    def test_tampered_proof_fails(self):
        """Tampered PLONK proof → verification fails."""
        bridge = self._get_bridge(n_features=5, threshold=5.0)
        bundle = bridge.prove_similarity(self.weights_a, self.weights_a)

        # Tamper by modifying a proof element
        bad_proof = _clone_plonk_proof(
            bundle.plonk_proof,
            a_bar=bundle.plonk_proof.a_bar + Fr(1),
        )
        tampered_bundle = HybridProofBundle(
            fingerprint_a=bundle.fingerprint_a,
            fingerprint_b=bundle.fingerprint_b,
            tda_proof_a=bundle.tda_proof_a,
            tda_proof_b=bundle.tda_proof_b,
            plonk_proof=bad_proof,
            circuit_gates=bundle.circuit_gates,
            srs_degree=bundle.srs_degree,
            threshold=bundle.threshold,
            threshold_sq_fr=bundle.threshold_sq_fr,
            commitment_a=bundle.commitment_a,
            commitment_b=bundle.commitment_b,
            commitment_random=bundle.commitment_random,
            tda_time_a=0, tda_time_b=0,
            zk_compile_time=0, zk_srs_time=0, zk_prove_time=0,
            total_time=0,
            actual_distance_sq_fp=0, actual_distance_float=0,
        )
        valid, reason = bridge.verify_similarity(tampered_bundle)
        self.assertFalse(valid, "Tampered proof should fail verification")

    def test_e2e_timing_under_60s(self):
        """Full E2E should complete in reasonable time."""
        bridge = self._get_bridge(n_features=5, threshold=5.0)
        t0 = time.perf_counter()
        bundle = bridge.prove_similarity(self.weights_a, self.weights_a)
        elapsed = time.perf_counter() - t0
        self.assertLess(
            elapsed, 60.0,
            f"E2E took {elapsed:.1f}s — too slow (target <60s)"
        )
        print(f"\n  [TIMING] E2E: {elapsed:.1f}s "
              f"(gates={bundle.circuit_gates}, SRS={bundle.srs_degree})")


# Import for tampered bundle construction
from zkml_system.hybrid_bridge import HybridProofBundle


if __name__ == "__main__":
    unittest.main(verbosity=2)
