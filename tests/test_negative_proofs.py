"""
Negative proof tests — Soundness verification.

These tests ensure that TAMPERED, MALFORMED, or INVALID proofs are
correctly REJECTED by both verification layers:

1. PLONKVerifier (pairing-based, cryptographic soundness)
2. ZkMLPipeline.verify (structural checks)
3. verify_gate_satisfiability (algebraic gate-level checks)

A system that accepts invalid proofs is worse than no system at all.
"""
import sys
import os
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point
from zkml_system.plonk.circuit_compiler import (
    CircuitCompiler, CompiledCircuit, Gate, GateType, Wire
)
from zkml_system.plonk.plonk_prover import (
    PLONKProver, PLONKVerifier, PLONKProof,
    verify_gate_satisfiability, SatisfiabilityResult,
)
from zkml_system.plonk.plonk_kzg import TrustedSetup
from zkml_system.plonk.zkml_pipeline import ZkMLPipeline, ZkMLProof, VerificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_addition_circuit() -> CompiledCircuit:
    """Build a small circuit: a + b = c  (3 + 5 = 8)."""
    a = Wire(index=0, value=Fr(3), is_public=True)
    b = Wire(index=1, value=Fr(5), is_public=True)
    c = Wire(index=2, value=Fr(8))

    gate = Gate(
        gate_type=GateType.ADD,
        left=0, right=1, output=2,
        q_L=Fr.one(), q_R=Fr.one(), q_O=-Fr.one(),
        q_M=Fr.zero(), q_C=Fr.zero(),
    )
    return CompiledCircuit(
        gates=[gate], wires=[a, b, c],
        num_public_inputs=2, num_public_outputs=1,
    )


def _make_mul_circuit() -> CompiledCircuit:
    """Build: a * b = c  (3 * 5 = 15)."""
    a = Wire(index=0, value=Fr(3), is_public=True)
    b = Wire(index=1, value=Fr(5), is_public=True)
    c = Wire(index=2, value=Fr(15))

    gate = Gate(
        gate_type=GateType.MUL,
        left=0, right=1, output=2,
        q_L=Fr.zero(), q_R=Fr.zero(), q_O=-Fr.one(),
        q_M=Fr.one(), q_C=Fr.zero(),
    )
    return CompiledCircuit(
        gates=[gate], wires=[a, b, c],
        num_public_inputs=2, num_public_outputs=1,
    )


def _clone_plonk_proof(proof: PLONKProof, **overrides) -> PLONKProof:
    """
    Clone a PLONKProof with optional field overrides.

    RustFr (PyO3) objects are not pickle-compatible, so copy.deepcopy fails.
    This function manually recreates Fr values and G1Points.
    """
    def clone_fr(f: Fr) -> Fr:
        return Fr(f.value)

    def clone_g1(p: G1Point) -> G1Point:
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


def _make_pipeline_proof():
    """Return (pipeline, proof, circuit) for a small network."""
    layers = [{
        'type': 'dense',
        'weights': [[1, 2], [3, 4]],
        'biases': [0, 0],
        'activation': 'linear',
    }]
    inputs = [10, 20]
    pipeline = ZkMLPipeline(use_sparse=False, use_gelu=False, srs_size=64)
    circuit = pipeline.compile_network(layers, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proof = pipeline.prove(circuit)
    return pipeline, proof, circuit


def _clone_pipeline_proof(proof: ZkMLProof, **overrides) -> ZkMLProof:
    """Clone a ZkMLProof with optional field overrides."""
    fields = {
        'circuit_hash': proof.circuit_hash,
        'num_gates': proof.num_gates,
        'num_sparse_gates': proof.num_sparse_gates,
        'num_gelu_gates': proof.num_gelu_gates,
        'wire_commitments': list(proof.wire_commitments),
        'quotient_commitment': proof.quotient_commitment,
        'opening_proof': proof.opening_proof,
        'public_inputs': list(proof.public_inputs),
        'public_outputs': list(proof.public_outputs),
        'wire_evaluations': list(proof.wire_evaluations),
        'prover_time_ms': proof.prover_time_ms,
    }
    fields.update(overrides)
    return ZkMLProof(**fields)


# ===========================================================================
# Test Suite 1: Satisfiability (algebraic gate-level checks)
# ===========================================================================

class TestNegativeSatisfiability(unittest.TestCase):
    """Tests that corrupted witnesses are caught by gate-level checks."""

    def test_valid_addition_passes(self):
        circuit = _make_addition_circuit()
        result = verify_gate_satisfiability(circuit)
        self.assertTrue(result.is_satisfied, result.message)

    def test_wrong_output_value_fails(self):
        """Corrupt c from 8 to 9 — gate equation should fail."""
        circuit = _make_addition_circuit()
        circuit.wires[2].value = Fr(9)  # 3 + 5 ≠ 9
        result = verify_gate_satisfiability(circuit)
        self.assertFalse(result.is_satisfied)
        self.assertGreater(result.gates_failed, 0)

    def test_wrong_input_value_fails(self):
        """Corrupt a from 3 to 4 — 4 + 5 ≠ 8."""
        circuit = _make_addition_circuit()
        circuit.wires[0].value = Fr(4)
        result = verify_gate_satisfiability(circuit)
        self.assertFalse(result.is_satisfied)

    def test_mul_valid_passes(self):
        circuit = _make_mul_circuit()
        result = verify_gate_satisfiability(circuit)
        self.assertTrue(result.is_satisfied, result.message)

    def test_mul_wrong_product_fails(self):
        """3 * 5 ≠ 16."""
        circuit = _make_mul_circuit()
        circuit.wires[2].value = Fr(16)
        result = verify_gate_satisfiability(circuit)
        self.assertFalse(result.is_satisfied)

    def test_all_zeros_satisfies_trivially(self):
        """All-zero wires with no constant: 0·0 + 0·0 - 0 + 0·(0·0) + 0 = 0."""
        circuit = _make_addition_circuit()
        for w in circuit.wires:
            w.value = Fr.zero()
        result = verify_gate_satisfiability(circuit)
        self.assertTrue(result.is_satisfied)

    def test_diagnostics_on_failure(self):
        """Failed gates should produce diagnostic information."""
        circuit = _make_addition_circuit()
        circuit.wires[2].value = Fr(999)
        result = verify_gate_satisfiability(circuit)
        self.assertFalse(result.is_satisfied)
        self.assertEqual(len(result.failed_gate_indices), 1)
        self.assertEqual(result.failed_gate_indices[0], 0)


# ===========================================================================
# Test Suite 2: PLONK Protocol (pairing-based verification)
# ===========================================================================

class TestNegativePLONK(unittest.TestCase):
    """Tests that tampered PLONK proofs fail the pairing check."""

    @classmethod
    def setUpClass(cls):
        """Generate a valid proof once — tests tamper with clones."""
        cls.circuit = _make_addition_circuit()
        srs_size = 16  # 3n+padding for quotient poly; n=4 after pad
        cls.srs = TrustedSetup.generate(srs_size, tau=Fr(12345))
        prover = PLONKProver(cls.srs)
        cls.proof = prover.prove(cls.circuit)
        # Smoke-check: valid proof passes
        verifier = PLONKVerifier(cls.srs)
        cls.valid_result = verifier.verify(cls.proof, cls.circuit)
        assert cls.valid_result, "Baseline proof must verify for negative tests to be meaningful"

    def _verify(self, proof: PLONKProof) -> bool:
        verifier = PLONKVerifier(self.srs)
        return verifier.verify(proof, self.circuit)

    def test_valid_proof_passes(self):
        self.assertTrue(self._verify(self.proof))

    def test_tampered_a_bar_fails(self):
        """Corrupt the evaluation ā."""
        bad = _clone_plonk_proof(self.proof, a_bar=self.proof.a_bar + Fr.one())
        self.assertFalse(self._verify(bad))

    def test_tampered_b_bar_fails(self):
        bad = _clone_plonk_proof(self.proof, b_bar=self.proof.b_bar + Fr(42))
        self.assertFalse(self._verify(bad))

    def test_tampered_c_bar_fails(self):
        bad = _clone_plonk_proof(self.proof, c_bar=self.proof.c_bar + Fr(1))
        self.assertFalse(self._verify(bad))

    def test_tampered_z_omega_bar_fails(self):
        bad = _clone_plonk_proof(self.proof, z_omega_bar=self.proof.z_omega_bar + Fr(1))
        self.assertFalse(self._verify(bad))

    def test_tampered_r_zeta_fails(self):
        bad = _clone_plonk_proof(self.proof, r_zeta=self.proof.r_zeta + Fr(1))
        self.assertFalse(self._verify(bad))

    def test_tampered_commitment_fails(self):
        """Replace com_a with the generator — should break pairing."""
        bad = _clone_plonk_proof(self.proof, com_a=G1Point.generator())
        self.assertFalse(self._verify(bad))

    def test_tampered_opening_proof_fails(self):
        """Corrupt w_zeta opening proof."""
        bad = _clone_plonk_proof(self.proof, w_zeta=G1Point.generator())
        self.assertFalse(self._verify(bad))

    def test_wrong_circuit_fails(self):
        """Verify a valid proof against a different circuit."""
        different_circuit = _make_mul_circuit()
        verifier = PLONKVerifier(self.srs)
        result = verifier.verify(self.proof, different_circuit)
        self.assertFalse(result)


# ===========================================================================
# Test Suite 3: Pipeline structural checks
# ===========================================================================

class TestNegativePipeline(unittest.TestCase):
    """Tests that ZkMLPipeline.verify rejects structurally invalid proofs."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.proof, cls.circuit = _make_pipeline_proof()

    def _verify(self, proof, circuit=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.pipeline.verify(proof, circuit or self.circuit)

    def test_valid_proof_passes(self):
        result = self._verify(self.proof)
        self.assertTrue(result.is_valid, result.error_message)

    def test_wrong_circuit_hash_fails(self):
        bad = _clone_pipeline_proof(self.proof, circuit_hash="deadbeefdeadbeef")
        result = self._verify(bad)
        self.assertFalse(result.is_valid)
        self.assertIn("hash", result.error_message.lower())

    def test_wrong_gate_count_fails(self):
        bad = _clone_pipeline_proof(self.proof, num_gates=99999)
        result = self._verify(bad)
        self.assertFalse(result.is_valid)

    def test_missing_wire_commitments_fails(self):
        bad = _clone_pipeline_proof(self.proof,
                                    wire_commitments=list(self.proof.wire_commitments)[:2])
        result = self._verify(bad)
        self.assertFalse(result.is_valid)

    def test_missing_wire_evaluations_fails(self):
        bad = _clone_pipeline_proof(self.proof,
                                    wire_evaluations=list(self.proof.wire_evaluations)[:1])
        result = self._verify(bad)
        self.assertFalse(result.is_valid)

    def test_empty_wire_commitments_fails(self):
        bad = _clone_pipeline_proof(self.proof, wire_commitments=[])
        result = self._verify(bad)
        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()
