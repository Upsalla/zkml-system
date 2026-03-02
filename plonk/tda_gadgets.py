"""
TDA Circuit Gadgets for Zero-Knowledge Persistent Homology.

This module provides reusable arithmetic circuit building blocks for
encoding Persistent Homology verification into PLONK constraints.

Gadgets:
    - squared_distance:  d²(p, q) = Σ(pᵢ - qᵢ)²
    - comparison:        a ≤ b  (via difference + range check)
    - poseidon_hash:     ZK-friendly hash (placeholder → algebraic hash)
    - assert_equal:      a == b  (constrained equality)
    - assert_boolean:    b ∈ {0, 1}
    - conditional_select: if b then x else y

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fr
from zkml_system.plonk.circuit_compiler import (
    CircuitCompiler, Gate, GateType, Wire, CompiledCircuit
)
from zkml_system.plonk.poseidon import PoseidonHash, PoseidonGadget


# =============================================================================
# Fixed-Point Encoding
# =============================================================================

# We encode floats as integers with SCALE precision.
# float_value * SCALE -> field element
# This gives us ~6 decimal digits of precision.
FIXED_POINT_SCALE = 1_000_000  # 10^6
FIXED_POINT_SCALE_FR = Fr(FIXED_POINT_SCALE)
FIXED_POINT_SCALE_SQ = Fr(FIXED_POINT_SCALE * FIXED_POINT_SCALE)
RANGE_CHECK_BITS = 64  # Covers squared distances up to ~1.8e19


def float_to_fr(value: float) -> Fr:
    """Convert a float to a fixed-point field element."""
    # Handle negative values via modular arithmetic
    scaled = int(round(value * FIXED_POINT_SCALE))
    return Fr(scaled)


def fr_to_float(value: Fr) -> float:
    """Convert a fixed-point field element back to float."""
    v = value.to_int()
    # If value is in the upper half of the field, it's negative
    if v > Fr.MODULUS // 2:
        v -= Fr.MODULUS
    return v / FIXED_POINT_SCALE


# =============================================================================
# Core Gadgets
# =============================================================================

class TDAGadgets:
    """
    Reusable arithmetic circuit gadgets for TDA verification.

    All methods operate on a CircuitCompiler instance, adding gates
    and wires as needed, and returning wire indices for the results.
    """

    def __init__(self, compiler: CircuitCompiler):
        self.cc = compiler

    # -----------------------------------------------------------------
    # Arithmetic helpers
    # -----------------------------------------------------------------

    def _get_val(self, wire: int) -> Fr:
        """Get wire value, defaulting to zero if unset."""
        v = self.cc.wires[wire].value
        return v if v is not None else Fr.zero()

    def sub(self, a: int, b: int) -> int:
        """
        Subtract: output = a - b.

        PLONK equation: q_L*a + q_R*b + q_O*c = 0
        With q_L=1, q_R=-1, q_O=-1  →  a - b - c = 0  →  c = a - b
        """
        out = self.cc._new_wire(name="sub_out")
        # Propagate value: c = a - b
        self.cc._set_wire_value(out, self._get_val(a) - self._get_val(b))
        gate = Gate(
            gate_type=GateType.ADD,
            left=a,
            right=b,
            output=out,
            q_L=Fr.one(),
            q_R=Fr(Fr.MODULUS - 1),  # -1
            q_O=Fr(Fr.MODULUS - 1),  # -1
        )
        self.cc._add_gate(gate)
        return out

    def mul(self, a: int, b: int) -> int:
        """Multiply: output = a * b."""
        out = self.cc._new_wire(name="mul_out")
        # Propagate value: c = a * b
        self.cc._set_wire_value(out, self._get_val(a) * self._get_val(b))
        self.cc._add_mul_gate(a, b, out)
        return out

    def add(self, a: int, b: int) -> int:
        """Add: output = a + b."""
        out = self.cc._new_wire(name="add_out")
        # Propagate value: c = a + b
        self.cc._set_wire_value(out, self._get_val(a) + self._get_val(b))
        self.cc._add_add_gate(a, b, out)
        return out

    def square(self, a: int) -> int:
        """Square: output = a * a."""
        return self.mul(a, a)

    def const_wire(self, value: Fr) -> int:
        """Create a wire constrained to a constant value."""
        w = self.cc._new_wire(name=f"const_{value.to_int()}")
        self.cc._set_wire_value(w, value)
        self.cc._add_const_gate(w, value)
        return w

    # -----------------------------------------------------------------
    # Squared Euclidean Distance
    # -----------------------------------------------------------------

    def squared_distance(
        self,
        point_a_wires: List[int],
        point_b_wires: List[int]
    ) -> int:
        """
        Compute d²(a, b) = Σ(aᵢ - bᵢ)² for two point vectors.

        Args:
            point_a_wires: Wire indices for point A coordinates.
            point_b_wires: Wire indices for point B coordinates.

        Returns:
            Wire index holding d²(a, b).

        Constraints: 3 * dim (1 sub + 1 square + 1 accumulate per dimension).
        """
        assert len(point_a_wires) == len(point_b_wires), \
            "Point dimensions must match"

        dim = len(point_a_wires)

        # First dimension: diff = a[0] - b[0], sq = diff²
        diff = self.sub(point_a_wires[0], point_b_wires[0])
        acc = self.square(diff)

        # Remaining dimensions: accumulate
        for i in range(1, dim):
            diff = self.sub(point_a_wires[i], point_b_wires[i])
            sq = self.square(diff)
            acc = self.add(acc, sq)

        return acc

    def assert_leq(self, a: int, b: int) -> int:
        """
        Assert a ≤ b in the field with a range proof.

        Computes diff = b - a, then proves diff ∈ [0, 2^RANGE_CHECK_BITS)
        via bit-decomposition. This prevents a malicious prover from
        exploiting field wrap-around (e.g. claiming a huge value < a
        small value because their difference wraps to something small mod p).

        Gate cost: ~3·RANGE_CHECK_BITS + 2 ≈ 122 gates (40-bit range).

        Returns:
            Wire index of diff = b - a (proven non-negative).
        """
        diff = self.sub(b, a)
        self.range_check(diff, RANGE_CHECK_BITS)
        return diff

    def range_check(self, wire: int, n_bits: int):
        """
        Prove wire ∈ [0, 2^n_bits) by bit-decomposition.

        Decomposes the witness value into n_bits bits, asserts each bit
        is boolean (b·(1-b) = 0), then reconstructs: Σ bᵢ·2ⁱ == wire.

        Gate cost: n_bits boolean + n_bits const + n_bits mul +
                   (n_bits-1) add + 1 assert_equal ≈ 3·n_bits + 1.
        """
        val = self._get_val(wire)
        # Convert from Montgomery to raw integer for bit decomposition
        int_val = val.to_int()

        # Check prover-side that value actually fits in n_bits
        if int_val < 0 or int_val >= (1 << n_bits):
            raise ValueError(
                f"range_check: value {int_val} does not fit in {n_bits} bits"
            )

        # Decompose into bits
        bits = []
        remaining = int_val
        for i in range(n_bits):
            b = remaining & 1
            remaining >>= 1

            w_bit = self.cc._new_wire(name=f"rc_bit_{wire}_{i}")
            self.cc._set_wire_value(w_bit, Fr(b))
            self.assert_boolean(w_bit)
            bits.append(w_bit)

        # Reconstruct: acc = Σ bᵢ · 2ⁱ
        power = Fr.one()
        two = Fr(2)

        # First term: b₀ · 1
        w_pow = self.const_wire(power)
        acc = self.mul(w_pow, bits[0])

        for i in range(1, n_bits):
            power = power * two
            w_pow = self.const_wire(power)
            term = self.mul(w_pow, bits[i])
            acc = self.add(acc, term)

        # Assert reconstruction == original wire
        self.assert_equal(acc, wire)

    # -----------------------------------------------------------------
    # Assert Equal
    # -----------------------------------------------------------------

    def assert_equal(self, a: int, b: int):
        """
        Assert a == b.

        PLONK: q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0
        With q_L=1, q_R=-1, all others zero → a - b = 0.
        """
        gate = Gate(
            gate_type=GateType.ADD,
            left=a,
            right=b,
            output=a,  # dummy — output term zeroed by q_O=0
            q_L=Fr.one(),
            q_R=Fr(Fr.MODULUS - 1),  # -1
            q_O=Fr.zero(),
            q_M=Fr.zero(),
            q_C=Fr.zero(),
        )
        self.cc._add_gate(gate)

    # -----------------------------------------------------------------
    # Assert Boolean
    # -----------------------------------------------------------------

    def assert_boolean(self, b: int):
        """
        Assert b ∈ {0, 1}.

        b * (1 - b) = 0  →  b² - b = 0
        PLONK: q_M*a*b + q_L*a + q_R*b + q_O*c + q_C = 0
        With left=right=output=b, q_M=1, q_L=-1, all others zero:
            b² - b = 0  ✓
        """
        gate = Gate(
            gate_type=GateType.MUL,
            left=b,
            right=b,
            output=b,  # dummy — output term zeroed by q_O=0
            q_M=Fr.one(),
            q_L=Fr(Fr.MODULUS - 1),  # -1
            q_R=Fr.zero(),
            q_O=Fr.zero(),
            q_C=Fr.zero(),
        )
        self.cc._add_gate(gate)

    # -----------------------------------------------------------------
    # Conditional Select: if bit then a else b
    # -----------------------------------------------------------------

    def conditional_select(self, bit: int, a: int, b: int) -> int:
        """
        Return a if bit==1 else b.

        result = b + bit * (a - b)

        Constraints: 3 (sub, mul, add).
        """
        diff = self.sub(a, b)
        selected = self.mul(bit, diff)
        result = self.add(b, selected)
        return result

    # -----------------------------------------------------------------
    # Algebraic Hash (Poseidon-like placeholder)
    # -----------------------------------------------------------------

    def algebraic_hash(self, inputs: List[int]) -> int:
        """
        ZK-friendly algebraic hash using Poseidon.

        Uses the Poseidon sponge construction (t=3, α=5, R_F=8, R_P=57)
        over BN254 Fr. This is cryptographically secure with 128-bit
        security level.

        Gate cost: ~2194 per compression call × ceil(len(inputs)/2).
        For 75 inputs: ~82,275 gates.
        """
        if not hasattr(self, '_poseidon'):
            self._poseidon = PoseidonGadget(self)
        return self._poseidon.hash_many(inputs)

    # -----------------------------------------------------------------
    # Batch squared distance matrix
    # -----------------------------------------------------------------

    def distance_matrix(
        self,
        point_wires: List[List[int]]
    ) -> List[List[int]]:
        """
        Compute the upper-triangular squared distance matrix for N points.

        Args:
            point_wires: List of N points, each a list of D wire indices.

        Returns:
            N×N matrix of wire indices. dist[i][j] = d²(pᵢ, pⱼ) for i<j.
            dist[i][j] = -1 for i >= j (unused).

        Constraints: N*(N-1)/2 * 3*D.
        For N=20, D=10: 190 * 30 = 5,700 constraints.
        """
        n = len(point_wires)
        matrix = [[-1] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                d_sq = self.squared_distance(point_wires[i], point_wires[j])
                matrix[i][j] = d_sq
                matrix[j][i] = d_sq  # Symmetric

        return matrix

    # -----------------------------------------------------------------
    # Filtration ordering verification
    # -----------------------------------------------------------------

    def verify_filtration_order(
        self,
        edge_distance_wires: List[int]
    ) -> None:
        """
        Verify that edges are sorted by ascending distance.

        For each consecutive pair (d[k], d[k+1]), asserts d[k] ≤ d[k+1].

        Args:
            edge_distance_wires: Wire indices of squared distances,
                                 in claimed filtration order.

        Constraints: len(edges) - 1.
        """
        for k in range(len(edge_distance_wires) - 1):
            self.assert_leq(
                edge_distance_wires[k],
                edge_distance_wires[k + 1]
            )
