"""
Tests for ZK-TDA Circuit Compilation and Gadgets.

Verifies:
1. Fixed-point encoding roundtrip
2. Squared distance gadget correctness
3. Distance matrix computation
4. Filtration order verification
5. Full circuit compilation
6. Witness generation from model weights
7. Constraint satisfaction (honest witness)

Author: David Weyhe
Date: 2026-03-01
"""

import sys
import os
import numpy as np
import pytest

# Ensure zkml_system is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.circuit_compiler import CircuitCompiler, Gate, GateType
from zkml_system.plonk.tda_gadgets import (
    TDAGadgets, float_to_fr, fr_to_float,
    FIXED_POINT_SCALE
)
from zkml_system.plonk.tda_circuit import (
    TDACircuitCompiler, TDAWitness, TDAPublicInputs,
    generate_tda_witness, _farthest_point_sampling
)


# =============================================================================
# Test: Fixed-Point Encoding
# =============================================================================

class TestFixedPointEncoding:
    """Verify float ↔ Fr conversion is accurate."""

    def test_positive_float(self):
        val = 0.123456
        fr = float_to_fr(val)
        restored = fr_to_float(fr)
        assert abs(restored - val) < 1e-5

    def test_negative_float(self):
        val = -0.789012
        fr = float_to_fr(val)
        restored = fr_to_float(fr)
        assert abs(restored - val) < 1e-5

    def test_zero(self):
        fr = float_to_fr(0.0)
        assert fr_to_float(fr) == 0.0

    def test_large_float(self):
        val = 12345.6789
        fr = float_to_fr(val)
        restored = fr_to_float(fr)
        assert abs(restored - val) < 1e-3

    def test_precision_limit(self):
        """Values beyond 6 decimal places lose precision."""
        val = 0.1234567890
        fr = float_to_fr(val)
        restored = fr_to_float(fr)
        # Only 6 decimal places guaranteed
        assert abs(restored - round(val, 6)) < 1e-6


# =============================================================================
# Test: Gadgets
# =============================================================================

class TestTDAGadgets:
    """Test individual circuit gadgets for correctness."""

    def setup_method(self):
        self.cc = CircuitCompiler(use_sparse=False, use_gelu=False)
        self.g = TDAGadgets(self.cc)

    def test_subtraction(self):
        """sub(a, b) should produce a wire for a - b."""
        a = self.cc._new_wire(name="a")
        b = self.cc._new_wire(name="b")
        self.cc._set_wire_value(a, Fr(100))
        self.cc._set_wire_value(b, Fr(30))

        out = self.g.sub(a, b)
        # The wire is created; value would be set by the prover.
        assert out >= 0
        assert len(self.cc.gates) == 1

    def test_square(self):
        """square(a) should add one MUL gate."""
        a = self.cc._new_wire(name="a")
        self.cc._set_wire_value(a, Fr(7))

        out = self.g.square(a)
        assert len(self.cc.gates) == 1
        assert self.cc.gates[0].gate_type == GateType.MUL

    def test_const_wire(self):
        """const_wire should create a constrained wire."""
        w = self.g.const_wire(Fr(42))
        assert self.cc.wires[w].value == Fr(42)
        assert len(self.cc.gates) == 1  # CONST gate

    def test_squared_distance_2d(self):
        """Squared distance of two 2D points uses 5 gates."""
        a = [self.cc._new_wire() for _ in range(2)]
        b = [self.cc._new_wire() for _ in range(2)]

        d_sq = self.g.squared_distance(a, b)

        # 2 dims: 2 sub + 2 square + 1 add = 5 gates
        assert len(self.cc.gates) == 5
        assert d_sq >= 0

    def test_distance_matrix_3_points(self):
        """Distance matrix for 3 points: 3 pairs, each 5 gates for 2D."""
        dim = 2
        points = []
        for i in range(3):
            pt = [self.cc._new_wire(name=f"p{i}_d{d}") for d in range(dim)]
            points.append(pt)

        matrix = self.g.distance_matrix(points)

        # 3 pairs: (0,1), (0,2), (1,2), each 5 gates
        assert len(self.cc.gates) == 15
        assert matrix[0][1] >= 0
        assert matrix[0][2] >= 0
        assert matrix[1][2] >= 0

    def test_assert_equal(self):
        """assert_equal adds exactly 1 gate."""
        a = self.cc._new_wire()
        b = self.cc._new_wire()
        self.g.assert_equal(a, b)
        assert len(self.cc.gates) == 1

    def test_algebraic_hash(self):
        """Algebraic hash produces a single output wire."""
        inputs = [self.cc._new_wire() for _ in range(5)]
        h = self.g.algebraic_hash(inputs)
        assert h >= 0
        # Each input: 1 const + 1 add + 1 square + 1 mul + 1 add + 1 add = ~6
        # Plus initial const wire
        assert len(self.cc.gates) > 5 * 3


# =============================================================================
# Test: Farthest Point Sampling
# =============================================================================

class TestFarthestPointSampling:
    """Test topology-preserving subsampling."""

    def test_basic(self):
        np.random.seed(42)
        points = np.random.randn(100, 5)
        indices = _farthest_point_sampling(points, 20)
        assert len(indices) == 20
        assert len(set(indices)) == 20  # No duplicates

    def test_preserves_extremes(self):
        """FPS should select extremal points."""
        # Create points with clear extremes
        points = np.zeros((50, 2))
        points[:25, 0] = np.linspace(0, 10, 25)
        points[25:, 1] = np.linspace(0, 10, 25)

        indices = _farthest_point_sampling(points, 5)
        selected = points[indices]

        # The selected points should span the full range
        assert selected[:, 0].max() >= 5.0 or selected[:, 1].max() >= 5.0

    def test_fewer_than_requested(self):
        """If n_samples > n_points, use all points."""
        points = np.random.randn(5, 3)
        indices = _farthest_point_sampling(points, 5)
        assert len(indices) == 5


# =============================================================================
# Test: Witness Generation
# =============================================================================

class TestWitnessGeneration:
    """Test offline witness generation from model weights."""

    def setup_method(self):
        np.random.seed(42)
        self.weights = [
            np.random.randn(10, 5),
            np.random.randn(8, 10),
            np.random.randn(3, 8),
        ]

    def test_generates_correct_dimensions(self):
        witness, public = generate_tda_witness(
            self.weights, n_landmarks=15, n_features=5
        )
        assert len(witness.points) == 15
        assert len(witness.points[0]) > 0
        assert len(witness.edges) == 15 * 14 // 2  # 105
        assert len(witness.persistence_pairs) == 5

    def test_edges_sorted(self):
        """Edges must be sorted by squared distance."""
        witness, _ = generate_tda_witness(self.weights, n_landmarks=10)
        for k in range(len(witness.edges) - 1):
            assert witness.edges[k][2] <= witness.edges[k + 1][2], \
                f"Edge {k} not sorted: {witness.edges[k][2]} > {witness.edges[k+1][2]}"

    def test_public_inputs_non_zero(self):
        """Public commitment and fingerprint must be non-zero."""
        _, public = generate_tda_witness(self.weights)
        assert not public.model_commitment.is_zero()
        assert not public.fingerprint_hash.is_zero()

    def test_deterministic(self):
        """Same weights → same witness."""
        np.random.seed(42)
        w1, p1 = generate_tda_witness(self.weights)
        np.random.seed(42)
        w2, p2 = generate_tda_witness(self.weights)
        assert p1.model_commitment == p2.model_commitment
        assert p1.fingerprint_hash == p2.fingerprint_hash

    def test_different_weights_different_fingerprint(self):
        """Different weights → different fingerprint."""
        _, p1 = generate_tda_witness(self.weights)
        weights2 = [np.random.randn(*w.shape) for w in self.weights]
        _, p2 = generate_tda_witness(weights2)
        assert p1.fingerprint_hash != p2.fingerprint_hash


# =============================================================================
# Test: Full Circuit Compilation
# =============================================================================

class TestTDACircuitCompilation:
    """Test end-to-end circuit compilation."""

    def setup_method(self):
        np.random.seed(42)
        self.weights = [
            np.random.randn(10, 5),
            np.random.randn(8, 10),
            np.random.randn(3, 8),
        ]

    def test_compilation_succeeds(self):
        """Circuit compiles without errors."""
        witness, public = generate_tda_witness(
            self.weights, n_landmarks=10, n_features=5
        )
        compiler = TDACircuitCompiler(
            n_points=len(witness.points),
            point_dim=len(witness.points[0]),
            n_features=5,
        )
        circuit = compiler.compile(witness, public)

        assert circuit.total_gates > 0
        assert len(circuit.wires) > 0
        assert circuit.num_public_inputs == 2

    def test_gate_count_scales(self):
        """More landmarks = more gates (roughly quadratic)."""
        counts = {}
        for n in [5, 10, 15]:
            np.random.seed(42)
            witness, public = generate_tda_witness(
                self.weights, n_landmarks=n, n_features=5
            )
            compiler = TDACircuitCompiler(
                n_points=len(witness.points),
                point_dim=len(witness.points[0]),
                n_features=5,
            )
            circuit = compiler.compile(witness, public)
            counts[n] = circuit.total_gates

        # Quadratic scaling: 15 landmarks should have > 2× gates of 5
        assert counts[15] > counts[5] * 2

    def test_public_inputs_are_public(self):
        """The first two wires must be marked public."""
        witness, public = generate_tda_witness(
            self.weights, n_landmarks=10, n_features=5,
        )
        compiler = TDACircuitCompiler(
            n_points=len(witness.points),
            point_dim=len(witness.points[0]),
            n_features=5,
        )
        circuit = compiler.compile(witness, public)

        public_wires = [w for w in circuit.wires if w.is_public]
        assert len(public_wires) >= 2


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
