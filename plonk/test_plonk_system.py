"""
Integration tests for the PLONK proof system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.polynomial import Polynomial
from zkml_system.plonk.kzg import SRS, KZG
from zkml_system.plonk.prover import PlonkCircuit, PlonkWitness, PlonkProver
from zkml_system.plonk.verifier import PlonkVerifier


def create_simple_circuit() -> PlonkCircuit:
    """
    Create a simple circuit: a * b = c

    This is a single multiplication gate.
    """
    n = 4  # Must be power of 2

    circuit = PlonkCircuit.empty(n)

    # Gate 0: a * b = c
    # q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0
    # We want: a * b - c = 0
    # So: q_M = 1, q_O = -1, others = 0
    circuit.q_M[0] = Fr.one()
    circuit.q_O[0] = -Fr.one()

    # Gates 1-3: dummy gates (all zeros)
    # These are needed to pad to power of 2

    return circuit


def create_simple_witness(a_val: int, b_val: int) -> PlonkWitness:
    """
    Create a witness for the simple circuit.

    Args:
        a_val: Value for wire a.
        b_val: Value for wire b.

    Returns:
        The witness with c = a * b.
    """
    n = 4

    a = [Fr(a_val)] + [Fr.zero()] * (n - 1)
    b = [Fr(b_val)] + [Fr.zero()] * (n - 1)
    c = [Fr(a_val * b_val)] + [Fr.zero()] * (n - 1)

    return PlonkWitness(a=a, b=b, c=c)


def test_plonk_simple_circuit():
    """Test PLONK with a simple multiplication circuit."""
    print("Testing PLONK with simple multiplication circuit...")

    # Setup
    srs = SRS.generate(32)
    circuit = create_simple_circuit()

    # Create prover and verifier
    prover = PlonkProver(srs, circuit)
    verifier = PlonkVerifier.from_circuit(srs, circuit)

    # Create witness: 3 * 5 = 15
    witness = create_simple_witness(3, 5)

    # Generate proof
    print("  Generating proof...")
    proof = prover.prove(witness)

    # Verify proof
    print("  Verifying proof...")
    result = verifier.verify(proof)

    print(f"  Verification result: {result}")
    assert result, "Proof verification failed"

    print("  Simple circuit test passed!")


def test_plonk_addition_circuit():
    """Test PLONK with an addition circuit."""
    print("Testing PLONK with addition circuit...")

    n = 4
    circuit = PlonkCircuit.empty(n)

    # Gate 0: a + b = c
    # q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0
    # We want: a + b - c = 0
    # So: q_L = 1, q_R = 1, q_O = -1
    circuit.q_L[0] = Fr.one()
    circuit.q_R[0] = Fr.one()
    circuit.q_O[0] = -Fr.one()

    # Setup
    srs = SRS.generate(32)
    prover = PlonkProver(srs, circuit)
    verifier = PlonkVerifier.from_circuit(srs, circuit)

    # Witness: 7 + 11 = 18
    a = [Fr(7)] + [Fr.zero()] * (n - 1)
    b = [Fr(11)] + [Fr.zero()] * (n - 1)
    c = [Fr(18)] + [Fr.zero()] * (n - 1)
    witness = PlonkWitness(a=a, b=b, c=c)

    # Generate and verify proof
    print("  Generating proof...")
    proof = prover.prove(witness)

    print("  Verifying proof...")
    result = verifier.verify(proof)

    print(f"  Verification result: {result}")
    assert result, "Proof verification failed"

    print("  Addition circuit test passed!")


def test_plonk_constant_circuit():
    """Test PLONK with a constant constraint."""
    print("Testing PLONK with constant constraint...")

    n = 4
    circuit = PlonkCircuit.empty(n)

    # Gate 0: a = 42
    # q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0
    # We want: a - 42 = 0
    # So: q_L = 1, q_C = -42
    circuit.q_L[0] = Fr.one()
    circuit.q_C[0] = Fr(-42)

    # Setup
    srs = SRS.generate(32)
    prover = PlonkProver(srs, circuit)
    verifier = PlonkVerifier.from_circuit(srs, circuit)

    # Witness: a = 42
    a = [Fr(42)] + [Fr.zero()] * (n - 1)
    b = [Fr.zero()] * n
    c = [Fr.zero()] * n
    witness = PlonkWitness(a=a, b=b, c=c)

    # Generate and verify proof
    print("  Generating proof...")
    proof = prover.prove(witness)

    print("  Verifying proof...")
    result = verifier.verify(proof)

    print(f"  Verification result: {result}")
    assert result, "Proof verification failed"

    print("  Constant constraint test passed!")


def test_plonk_proof_structure():
    """Test that the proof has the expected structure."""
    print("Testing PLONK proof structure...")

    srs = SRS.generate(32)
    circuit = create_simple_circuit()
    prover = PlonkProver(srs, circuit)
    witness = create_simple_witness(2, 3)

    proof = prover.prove(witness)

    # Check commitments are on curve
    assert proof.a_commit.point.is_on_curve(), "a_commit not on curve"
    assert proof.b_commit.point.is_on_curve(), "b_commit not on curve"
    assert proof.c_commit.point.is_on_curve(), "c_commit not on curve"
    assert proof.z_commit.point.is_on_curve(), "z_commit not on curve"
    assert proof.t_lo_commit.point.is_on_curve(), "t_lo_commit not on curve"
    assert proof.t_mid_commit.point.is_on_curve(), "t_mid_commit not on curve"
    assert proof.t_hi_commit.point.is_on_curve(), "t_hi_commit not on curve"

    # Check evaluations are in field
    assert isinstance(proof.a_eval, Fr), "a_eval not Fr"
    assert isinstance(proof.b_eval, Fr), "b_eval not Fr"
    assert isinstance(proof.c_eval, Fr), "c_eval not Fr"

    # Check opening proofs are on curve
    assert proof.w_zeta_proof.point.is_on_curve(), "w_zeta_proof not on curve"
    assert proof.w_zeta_omega_proof.point.is_on_curve(), "w_zeta_omega_proof not on curve"

    print("  Proof structure test passed!")


def run_all_tests():
    """Run all PLONK system tests."""
    print("=" * 60)
    print("PLONK System Integration Tests")
    print("=" * 60)

    test_plonk_proof_structure()
    test_plonk_simple_circuit()
    test_plonk_addition_circuit()
    test_plonk_constant_circuit()

    print("=" * 60)
    print("ALL PLONK SYSTEM TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
