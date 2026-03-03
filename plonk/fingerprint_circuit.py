"""
Fingerprint Similarity Circuit for Zero-Knowledge TDA Fingerprint Comparison.

Compiles a PLONK circuit that proves:
    "Two TDA fingerprints are within L2 distance ε"
without revealing the fingerprint feature vectors.

Trust model (commit-and-prove):
    - Fingerprint authenticity is established out-of-band via SHA256 TDA proofs
    - This circuit proves ONLY distance computation correctness
    - Additive commitments (Σ aᵢ·rⁱ) bind witness to public inputs

Gate count: ~400 for k=20 features (vs ~36,000 for full TDA circuit).

Author: David Weyhe
Date: 2026-03-03
"""
from __future__ import annotations

import sys
import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.circuit_compiler import (
    CircuitCompiler, Gate, GateType, Wire, CompiledCircuit,
)
from zkml_system.plonk.tda_gadgets import (
    TDAGadgets, float_to_fr, fr_to_float,
    FIXED_POINT_SCALE,
)

# Range check bit width for distance values.
# TDA fingerprint distances in fixed-point squared: threshold=5.0 →
# (5.0 * 10^6)^2 = 25 * 10^12, so we need at least 45 bits.
# 48 bits (max 2.8 × 10^14) provides headroom for thresholds up to ~16.7.
# Gate cost: ~3*48 + 1 = 145 gates per range check.
DISTANCE_RANGE_BITS = 48


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FingerprintFeatures:
    """Quantized TDA fingerprint features as field elements.

    Each feature is a (birth, death) pair in fixed-point representation.
    Dimension is dropped for the distance circuit (we compare birth-to-birth,
    death-to-death across matching pairs).
    """
    births: List[Fr]
    deaths: List[Fr]

    @property
    def n_features(self) -> int:
        return len(self.births)

    def as_wire_values(self) -> List[Fr]:
        """Interleaved [b0, d0, b1, d1, ...] for commitment."""
        result = []
        for b, d in zip(self.births, self.deaths):
            result.append(b)
            result.append(d)
        return result


@dataclass
class FingerprintWitness:
    """Private witness for the fingerprint similarity circuit."""
    features_a: FingerprintFeatures
    features_b: FingerprintFeatures


@dataclass
class FingerprintPublicInputs:
    """Public inputs for the fingerprint similarity circuit."""
    threshold_sq: Fr              # Maximum allowed squared L2 distance
    commitment_a: Fr              # Additive commitment to fingerprint A
    commitment_b: Fr              # Additive commitment to fingerprint B
    commitment_random: Fr         # Public random value r for commitments


def compute_additive_commitment(values: List[Fr], r: Fr) -> Fr:
    """Compute additive commitment: Σ vᵢ · rⁱ.

    This is a simple polynomial evaluation commitment.
    Binding under discrete log in BN254 scalar field.
    """
    commitment = Fr.zero()
    r_power = Fr.one()
    for v in values:
        commitment = commitment + v * r_power
        r_power = r_power * r
    return commitment


# =============================================================================
# Circuit Compiler
# =============================================================================

class FingerprintCircuitCompiler:
    """Compiles a PLONK circuit for fingerprint similarity proofs.

    Circuit structure:
        1. Verify commitment_A = Σ features_A[i] · rⁱ
        2. Verify commitment_B = Σ features_B[i] · rⁱ
        3. Compute d² = Σ (birthA_i - birthB_i)² + (deathA_i - deathB_i)²
        4. Assert d² ≤ threshold_sq (range check)

    Gate count breakdown (k=20 features):
        - Public inputs: 4 const gates
        - Feature wires: 80 const gates (40 per fingerprint)
        - Commitment A: 40 mul + 39 add = 79 gates
        - Commitment B: 40 mul + 39 add = 79 gates
        - Commitment checks: 2 assert_equal gates
        - Distance: 40 sub + 40 square + 39 add = 119 gates
        - Range check (d² ≤ threshold): 1 sub + ~121 gates (40-bit)
        - Total: ~525 gates
    """

    def __init__(self, n_features: int = 20):
        self.n_features = n_features

    def compile(
        self,
        witness: FingerprintWitness,
        public: FingerprintPublicInputs,
    ) -> CompiledCircuit:
        """Compile the fingerprint similarity circuit.

        Returns a CompiledCircuit ready for the PLONK prover.
        """
        cc = CircuitCompiler(use_sparse=False, use_gelu=False)
        gadgets = TDAGadgets(cc)
        k = self.n_features

        # ── Step 1: Public input wires ──
        w_threshold = cc._new_wire(name="pub_threshold_sq", is_public=True)
        cc._set_wire_value(w_threshold, public.threshold_sq)
        cc._add_const_gate(w_threshold, public.threshold_sq)

        w_com_a = cc._new_wire(name="pub_commitment_a", is_public=True)
        cc._set_wire_value(w_com_a, public.commitment_a)
        cc._add_const_gate(w_com_a, public.commitment_a)

        w_com_b = cc._new_wire(name="pub_commitment_b", is_public=True)
        cc._set_wire_value(w_com_b, public.commitment_b)
        cc._add_const_gate(w_com_b, public.commitment_b)

        w_r = cc._new_wire(name="pub_random_r", is_public=True)
        cc._set_wire_value(w_r, public.commitment_random)
        cc._add_const_gate(w_r, public.commitment_random)

        # ── Step 2: Private witness wires (fingerprint features) ──
        wires_a = []  # interleaved [b0, d0, b1, d1, ...]
        for i in range(k):
            wb = cc._new_wire(name=f"a_birth_{i}")
            cc._set_wire_value(wb, witness.features_a.births[i])
            wd = cc._new_wire(name=f"a_death_{i}")
            cc._set_wire_value(wd, witness.features_a.deaths[i])
            wires_a.extend([wb, wd])

        wires_b = []
        for i in range(k):
            wb = cc._new_wire(name=f"b_birth_{i}")
            cc._set_wire_value(wb, witness.features_b.births[i])
            wd = cc._new_wire(name=f"b_death_{i}")
            cc._set_wire_value(wd, witness.features_b.deaths[i])
            wires_b.extend([wb, wd])

        # ── Step 3: Verify additive commitment A ──
        #   computed = Σ wires_a[i] · rⁱ == pub_commitment_a
        computed_com_a = self._compile_commitment_check(
            cc, gadgets, wires_a, w_r
        )
        gadgets.assert_equal(computed_com_a, w_com_a)

        # ── Step 4: Verify additive commitment B ──
        computed_com_b = self._compile_commitment_check(
            cc, gadgets, wires_b, w_r
        )
        gadgets.assert_equal(computed_com_b, w_com_b)

        # ── Step 5: Compute squared L2 distance ──
        #   d² = Σ (birthA_i - birthB_i)² + (deathA_i - deathB_i)²
        dist_sq = self._compile_distance(
            cc, gadgets, wires_a, wires_b
        )

        # ── Step 6: Assert d² ≤ threshold ──
        #   diff = threshold - d², prove diff ≥ 0 via range check
        diff = gadgets.sub(w_threshold, dist_sq)
        gadgets.range_check(diff, DISTANCE_RANGE_BITS)

        # ── Build circuit ──
        circuit = CompiledCircuit(
            gates=cc.gates,
            wires=cc.wires,
            num_public_inputs=4,  # threshold, com_a, com_b, r
            num_public_outputs=0,
        )

        return circuit

    def _compile_commitment_check(
        self,
        cc: CircuitCompiler,
        gadgets: TDAGadgets,
        value_wires: List[int],
        w_r: int,
    ) -> int:
        """Compute additive commitment in-circuit: Σ vᵢ · rⁱ."""
        n = len(value_wires)

        # First term: v₀ · r⁰ = v₀ · 1 = v₀
        # We need r^0 = 1 as a const wire
        w_one = gadgets.const_wire(Fr.one())
        acc = gadgets.mul(value_wires[0], w_one)

        # Build r powers and accumulate: acc += vᵢ · rⁱ
        r_power = w_r  # r^1
        for i in range(1, n):
            term = gadgets.mul(value_wires[i], r_power)
            acc = gadgets.add(acc, term)
            # r^(i+1) = r^i * r (skip on last iteration)
            if i < n - 1:
                r_power = gadgets.mul(r_power, w_r)

        return acc

    def _compile_distance(
        self,
        cc: CircuitCompiler,
        gadgets: TDAGadgets,
        wires_a: List[int],
        wires_b: List[int],
    ) -> int:
        """Compute d² = Σ (aᵢ - bᵢ)² in-circuit."""
        n = len(wires_a)

        # First pair
        diff0 = gadgets.sub(wires_a[0], wires_b[0])
        acc = gadgets.square(diff0)

        for i in range(1, n):
            diff_i = gadgets.sub(wires_a[i], wires_b[i])
            sq_i = gadgets.square(diff_i)
            acc = gadgets.add(acc, sq_i)

        return acc


# =============================================================================
# Witness Generation
# =============================================================================

def fingerprints_to_features(
    fingerprint_features: List[Tuple[int, float, float]],
) -> FingerprintFeatures:
    """Convert TDA fingerprint features to circuit-compatible format.

    Args:
        fingerprint_features: List of (dimension, birth, death) tuples
            from ModelFingerprint.features

    Returns:
        FingerprintFeatures with fixed-point encoded births/deaths
    """
    births = []
    deaths = []
    for dim, birth, death in fingerprint_features:
        # Quantize to fixed-point (consistent with TDA quantization)
        b_int = int(round(birth * FIXED_POINT_SCALE))
        d_int = int(round(death * FIXED_POINT_SCALE))
        births.append(Fr(b_int))
        deaths.append(Fr(d_int))
    return FingerprintFeatures(births=births, deaths=deaths)


def generate_similarity_witness(
    fingerprint_a_features: List[Tuple[int, float, float]],
    fingerprint_b_features: List[Tuple[int, float, float]],
    threshold: float,
    random_seed: int = 42,
) -> Tuple[FingerprintWitness, FingerprintPublicInputs]:
    """Generate witness and public inputs for fingerprint similarity proof.

    Args:
        fingerprint_a_features: Features from ModelFingerprint A
        fingerprint_b_features: Features from ModelFingerprint B
        threshold: Maximum L2 distance (in original float space)
        random_seed: Seed for deterministic commitment randomness

    Returns:
        (FingerprintWitness, FingerprintPublicInputs)

    Raises:
        ValueError: If fingerprints have different lengths
    """
    if len(fingerprint_a_features) != len(fingerprint_b_features):
        raise ValueError(
            f"Fingerprint lengths differ: {len(fingerprint_a_features)} vs "
            f"{len(fingerprint_b_features)}"
        )

    features_a = fingerprints_to_features(fingerprint_a_features)
    features_b = fingerprints_to_features(fingerprint_b_features)

    # Threshold in fixed-point squared space
    # L2 dist in float space → multiply by SCALE for fixed-point
    # Then square: (threshold * SCALE)²
    threshold_fp = int(round(threshold * FIXED_POINT_SCALE))
    threshold_sq = Fr(threshold_fp * threshold_fp)

    # Deterministic random for commitment
    seed_bytes = hashlib.sha256(random_seed.to_bytes(8, "big")).digest()
    r = Fr(int.from_bytes(seed_bytes[:16], "big"))

    # Compute commitments
    commitment_a = compute_additive_commitment(
        features_a.as_wire_values(), r
    )
    commitment_b = compute_additive_commitment(
        features_b.as_wire_values(), r
    )

    witness = FingerprintWitness(features_a=features_a, features_b=features_b)
    public = FingerprintPublicInputs(
        threshold_sq=threshold_sq,
        commitment_a=commitment_a,
        commitment_b=commitment_b,
        commitment_random=r,
    )

    return witness, public
