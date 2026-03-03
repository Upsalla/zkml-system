"""
Multi-gate PLONK stress tests — GA Blocker 6B.

Tests that the PLONKProver/PLONKVerifier handle multi-gate circuits
with copy constraints (shared wires across gates), fan-out, and
larger compiled networks.

Key coverage:
- Permutation argument correctness for >1 gate
- Copy constraints across gate boundaries
- Fan-out (one wire feeds multiple gates)
- Compiled dense layer (real-world circuit)
- Negative: tampered multi-gate proofs must fail
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
from zkml_system.plonk.plonk_prover import PLONKProver, PLONKVerifier, PLONKProof
from zkml_system.plonk.plonk_kzg import TrustedSetup
from zkml_system.plonk.zkml_pipeline import ZkMLPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_plonk_proof(proof: PLONKProof, **overrides) -> PLONKProof:
    """Clone a PLONKProof with optional field overrides (PyO3-safe)."""
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


def _build_chain_circuit() -> CompiledCircuit:
    """
    3-gate linear chain with copy constraints:
      Gate 0: a(w0) + b(w1) = c(w2)     → 3 + 5 = 8
      Gate 1: c(w2) + d(w3) = e(w4)     → 8 + 2 = 10   (w2 shared = copy constraint)
      Gate 2: e(w4) * f(w5) = g(w6)     → 10 * 3 = 30  (w4 shared = copy constraint)
    """
    wires = [
        Wire(index=0, value=Fr(3), is_public=True, name="a"),
        Wire(index=1, value=Fr(5), is_public=True, name="b"),
        Wire(index=2, value=Fr(8), name="c"),   # output of gate 0, input to gate 1
        Wire(index=3, value=Fr(2), name="d"),
        Wire(index=4, value=Fr(10), name="e"),  # output of gate 1, input to gate 2
        Wire(index=5, value=Fr(3), name="f"),
        Wire(index=6, value=Fr(30), name="g"),
    ]
    gates = [
        Gate(gate_type=GateType.ADD, left=0, right=1, output=2,
             q_L=Fr.one(), q_R=Fr.one(), q_O=-Fr.one(),
             q_M=Fr.zero(), q_C=Fr.zero()),
        Gate(gate_type=GateType.ADD, left=2, right=3, output=4,
             q_L=Fr.one(), q_R=Fr.one(), q_O=-Fr.one(),
             q_M=Fr.zero(), q_C=Fr.zero()),
        Gate(gate_type=GateType.MUL, left=4, right=5, output=6,
             q_L=Fr.zero(), q_R=Fr.zero(), q_O=-Fr.one(),
             q_M=Fr.one(), q_C=Fr.zero()),
    ]
    return CompiledCircuit(
        gates=gates, wires=wires,
        num_public_inputs=2, num_public_outputs=1,
    )


def _build_fanout_circuit() -> CompiledCircuit:
    """
    Fan-out: wire x feeds two gates.
      Wire 0: x = 7  (public)
      Wire 1: y = 3
      Wire 2: z = 4
      Gate 0: x + y = z1  → 7 + 3 = 10  (wire 3)
      Gate 1: x * z = z2  → 7 * 4 = 28  (wire 4)

    x (wire 0) appears as left input in both gates — copy constraint.
    """
    wires = [
        Wire(index=0, value=Fr(7), is_public=True, name="x"),
        Wire(index=1, value=Fr(3), name="y"),
        Wire(index=2, value=Fr(4), name="z"),
        Wire(index=3, value=Fr(10), name="z1"),
        Wire(index=4, value=Fr(28), name="z2"),
    ]
    gates = [
        Gate(gate_type=GateType.ADD, left=0, right=1, output=3,
             q_L=Fr.one(), q_R=Fr.one(), q_O=-Fr.one(),
             q_M=Fr.zero(), q_C=Fr.zero()),
        Gate(gate_type=GateType.MUL, left=0, right=2, output=4,
             q_L=Fr.zero(), q_R=Fr.zero(), q_O=-Fr.one(),
             q_M=Fr.one(), q_C=Fr.zero()),
    ]
    return CompiledCircuit(
        gates=gates, wires=wires,
        num_public_inputs=1, num_public_outputs=2,
    )


# ===========================================================================
# Test Suite: Multi-Gate PLONK (Positive + Negative)
# ===========================================================================

class TestMultiGatePLONK(unittest.TestCase):
    """PLONK soundness for multi-gate circuits with copy constraints."""

    def _prove_and_verify(self, circuit: CompiledCircuit, srs_size: int = 32):
        """Prove and verify a circuit. Returns (proof, srs)."""
        srs = TrustedSetup.generate(srs_size, tau=Fr(54321))
        prover = PLONKProver(srs)
        proof = prover.prove(circuit)
        verifier = PLONKVerifier(srs)
        return proof, srs, verifier.verify(proof, circuit)

    # ---- Positive: valid circuits ----

    def test_chain_circuit_proves_and_verifies(self):
        """3-gate chain with copy constraints must prove and verify."""
        circuit = _build_chain_circuit()
        proof, srs, result = self._prove_and_verify(circuit)
        self.assertTrue(result, "Chain circuit proof failed verification")

    def test_fanout_circuit_proves_and_verifies(self):
        """Fan-out circuit (shared wire) must prove and verify."""
        circuit = _build_fanout_circuit()
        proof, srs, result = self._prove_and_verify(circuit)
        self.assertTrue(result, "Fan-out circuit proof failed verification")

    def test_compiled_dense_layer_proves_and_verifies(self):
        """Real compiled dense layer (8+ gates) must prove and verify."""
        layers = [{
            'type': 'dense',
            'weights': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'biases': [1, 1, 1],
            'activation': 'relu',
        }]
        inputs = [10, 20, 30]
        pipeline = ZkMLPipeline(use_sparse=False, use_gelu=False, srs_size=128)
        circuit = pipeline.compile_network(layers, inputs)
        self.assertGreater(len(circuit.gates), 8, "Dense layer should produce >8 gates")

        srs = TrustedSetup.generate(128, tau=Fr(99999))
        prover = PLONKProver(srs)
        proof = prover.prove(circuit)
        verifier = PLONKVerifier(srs)
        self.assertTrue(verifier.verify(proof, circuit),
                        "Compiled dense layer proof failed verification")

    # ---- Negative: tampered multi-gate proofs ----

    def test_tampered_chain_proof_a_bar_fails(self):
        """Tampering ā in a multi-gate proof must fail."""
        circuit = _build_chain_circuit()
        proof, srs, valid = self._prove_and_verify(circuit)
        self.assertTrue(valid)  # baseline

        bad = _clone_plonk_proof(proof, a_bar=proof.a_bar + Fr(1))
        verifier = PLONKVerifier(srs)
        self.assertFalse(verifier.verify(bad, circuit))

    def test_tampered_chain_proof_commitment_fails(self):
        """Tampering com_a in a multi-gate proof must fail."""
        circuit = _build_chain_circuit()
        proof, srs, valid = self._prove_and_verify(circuit)
        self.assertTrue(valid)

        bad = _clone_plonk_proof(proof, com_a=G1Point.generator())
        verifier = PLONKVerifier(srs)
        self.assertFalse(verifier.verify(bad, circuit))

    def test_tampered_fanout_proof_c_bar_fails(self):
        """Tampering c̄ in a fan-out proof must fail."""
        circuit = _build_fanout_circuit()
        proof, srs, valid = self._prove_and_verify(circuit)
        self.assertTrue(valid)

        bad = _clone_plonk_proof(proof, c_bar=proof.c_bar + Fr(1))
        verifier = PLONKVerifier(srs)
        self.assertFalse(verifier.verify(bad, circuit))

    def test_chain_proof_wrong_circuit_fails(self):
        """Chain proof verified against fan-out circuit must fail."""
        chain = _build_chain_circuit()
        fanout = _build_fanout_circuit()
        chain_proof, srs, valid = self._prove_and_verify(chain)
        self.assertTrue(valid)

        verifier = PLONKVerifier(srs)
        self.assertFalse(verifier.verify(chain_proof, fanout))

    def test_tampered_opening_proof_multi_gate_fails(self):
        """Tampering w_ζ in a multi-gate proof must fail."""
        circuit = _build_chain_circuit()
        proof, srs, valid = self._prove_and_verify(circuit)
        self.assertTrue(valid)

        bad = _clone_plonk_proof(proof, w_zeta=G1Point.generator())
        verifier = PLONKVerifier(srs)
        self.assertFalse(verifier.verify(bad, circuit))


if __name__ == "__main__":
    unittest.main()
