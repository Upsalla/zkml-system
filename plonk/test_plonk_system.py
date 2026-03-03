"""
Integration tests for the PLONK proof system.

Tests cover:
1. PlonkProver.prove() is BLOCKED (NotImplementedError)
2. PlonkCircuit/PlonkWitness data structures remain usable
3. PlonkVerifier.from_circuit() still constructs
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.kzg import SRS
from zkml_system.plonk.prover import PlonkCircuit, PlonkWitness, PlonkProver
from zkml_system.plonk.verifier import PlonkVerifier


def create_simple_circuit() -> PlonkCircuit:
    """Create a simple circuit: a * b = c (single multiplication gate)."""
    n = 4  # Must be power of 2
    circuit = PlonkCircuit.empty(n)
    circuit.q_M[0] = Fr.one()
    circuit.q_O[0] = -Fr.one()
    return circuit


def create_simple_witness(a_val: int, b_val: int) -> PlonkWitness:
    """Create a witness for the simple circuit."""
    n = 4
    a = [Fr(a_val)] + [Fr.zero()] * (n - 1)
    b = [Fr(b_val)] + [Fr.zero()] * (n - 1)
    c = [Fr(a_val * b_val)] + [Fr.zero()] * (n - 1)
    return PlonkWitness(a=a, b=b, c=c)


def test_old_prover_is_blocked():
    """PlonkProver.prove() MUST raise NotImplementedError."""
    srs = SRS.generate(32)
    circuit = create_simple_circuit()
    prover = PlonkProver(srs, circuit)
    witness = create_simple_witness(3, 5)

    with pytest.raises(NotImplementedError, match="BLOCKED"):
        prover.prove(witness)


def test_old_prover_blocked_addition():
    """Verify block applies to all circuit types, not just multiplication."""
    n = 4
    circuit = PlonkCircuit.empty(n)
    circuit.q_L[0] = Fr.one()
    circuit.q_R[0] = Fr.one()
    circuit.q_O[0] = -Fr.one()

    srs = SRS.generate(32)
    prover = PlonkProver(srs, circuit)

    a = [Fr(7)] + [Fr.zero()] * (n - 1)
    b = [Fr(11)] + [Fr.zero()] * (n - 1)
    c = [Fr(18)] + [Fr.zero()] * (n - 1)
    witness = PlonkWitness(a=a, b=b, c=c)

    with pytest.raises(NotImplementedError, match="BLOCKED"):
        prover.prove(witness)


def test_circuit_data_structures_intact():
    """PlonkCircuit and PlonkWitness must remain usable as data containers."""
    circuit = create_simple_circuit()
    assert len(circuit.q_M) == 4
    assert circuit.q_M[0] == Fr.one()
    assert circuit.q_O[0] == -Fr.one()

    witness = create_simple_witness(3, 5)
    assert witness.a[0] == Fr(3)
    assert witness.b[0] == Fr(5)
    assert witness.c[0] == Fr(15)


def test_verifier_constructs():
    """PlonkVerifier.from_circuit() must still construct (used for type compat)."""
    srs = SRS.generate(32)
    circuit = create_simple_circuit()
    verifier = PlonkVerifier.from_circuit(srs, circuit)
    assert verifier is not None


def test_old_prover_error_message_is_actionable():
    """Error message must point users to the correct replacement."""
    srs = SRS.generate(32)
    circuit = create_simple_circuit()
    prover = PlonkProver(srs, circuit)
    witness = create_simple_witness(2, 3)

    with pytest.raises(NotImplementedError) as exc_info:
        prover.prove(witness)

    msg = str(exc_info.value)
    assert "PLONKProver" in msg, "Error must mention the replacement class"
    assert "plonk_prover" in msg, "Error must mention the correct module"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
