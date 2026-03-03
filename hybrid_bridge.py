"""
Hybrid TDA+ZK Bridge — Option C.

Orchestrates the full pipeline:
    1. Classical TDA fingerprinting (0.1s each model)
    2. ZK similarity proof (~5-10s) — proves distance(FP_A, FP_B) ≤ ε

The ZK circuit proves ONLY distance correctness (~500 gates).
Fingerprint authenticity is established via the existing TDA proof system
(SHA256 5-level verification, out-of-band).

Usage:
    bridge = HybridBridge(n_features=20, threshold=2.0)
    bundle = bridge.prove_similarity(weights_a, weights_b)
    valid, reason = bridge.verify_similarity(bundle)

Author: David Weyhe
Date: 2026-03-03
"""
from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "tda_fingerprint",
    ),
)

from zkml_system.crypto.bn254.fr_adapter import Fr


# =============================================================================
# Proof Bundle
# =============================================================================

@dataclass
class HybridProofBundle:
    """Complete proof artifact from the hybrid bridge."""

    # Classical TDA artifacts
    fingerprint_a: Any  # ModelFingerprint
    fingerprint_b: Any  # ModelFingerprint
    tda_proof_a: Any    # TDAProof
    tda_proof_b: Any    # TDAProof

    # ZK similarity proof
    plonk_proof: Any           # PLONK proof object
    circuit_gates: int         # number of gates in the circuit
    srs_degree: int            # SRS polynomial degree

    # Public inputs
    threshold: float           # L2 distance threshold (float)
    threshold_sq_fr: Fr        # quantized squared threshold (Fr)
    commitment_a: Fr           # commitment to fingerprint A
    commitment_b: Fr           # commitment to fingerprint B
    commitment_random: Fr      # random value r

    # Metrics
    tda_time_a: float          # seconds for TDA fingerprint A
    tda_time_b: float          # seconds for TDA fingerprint B
    zk_compile_time: float     # seconds for circuit compilation
    zk_srs_time: float         # seconds for SRS generation
    zk_prove_time: float       # seconds for PLONK proving
    total_time: float          # total end-to-end time

    # Distance info
    actual_distance_sq_fp: int  # actual squared distance (fixed-point)
    actual_distance_float: float  # actual L2 distance (float)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Hybrid TDA+ZK Proof Bundle",
            "=" * 60,
            f"  Fingerprint A: {self.fingerprint_a.hash.hex()[:16]}...",
            f"  Fingerprint B: {self.fingerprint_b.hash.hex()[:16]}...",
            f"  Features:      {len(self.fingerprint_a.features)} per fingerprint",
            f"",
            f"  Threshold:     {self.threshold:.4f} (L2 distance)",
            f"  Actual dist:   {self.actual_distance_float:.4f}",
            f"  Within threshold: {'YES' if self.actual_distance_float <= self.threshold else 'NO'}",
            f"",
            f"  Circuit gates: {self.circuit_gates}",
            f"  SRS degree:    {self.srs_degree}",
            f"",
            f"  Timing:",
            f"    TDA-A:       {self.tda_time_a:.3f}s",
            f"    TDA-B:       {self.tda_time_b:.3f}s",
            f"    ZK compile:  {self.zk_compile_time:.3f}s",
            f"    ZK SRS:      {self.zk_srs_time:.3f}s",
            f"    ZK prove:    {self.zk_prove_time:.3f}s",
            f"    TOTAL:       {self.total_time:.3f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Hybrid Bridge
# =============================================================================

class HybridBridge:
    """Hybrid TDA+ZK bridge for fingerprint similarity proofs.

    Combines:
        - Classical TDA fingerprinting (fast, 0.1s)
        - ZK similarity proof (PLONK, ~500 gates, ~5-10s)
    """

    def __init__(
        self,
        n_features: int = 20,
        threshold: float = 2.0,
        max_points: int = 200,
    ):
        self.n_features = n_features
        self.threshold = threshold
        self.max_points = max_points

        # Lazy-loaded components
        self._tda_system = None
        self._backend = None

    @property
    def tda_system(self):
        if self._tda_system is None:
            from tda_fingerprint import TDAFingerprintSystem
            self._tda_system = TDAFingerprintSystem()
        return self._tda_system

    @property
    def backend(self) -> str:
        if self._backend is None:
            try:
                from zkml_system.crypto.bn254.fr_adapter import Fr
                _ = Fr(1) + Fr(2)
                self._backend = "rust"
            except Exception:
                self._backend = "python"
        return self._backend

    def prove_similarity(
        self,
        weights_a: List[np.ndarray],
        weights_b: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> HybridProofBundle:
        """Generate a hybrid TDA+ZK similarity proof.

        Args:
            weights_a: Model A weight matrices
            weights_b: Model B weight matrices
            threshold: L2 distance threshold (overrides instance default)

        Returns:
            HybridProofBundle with all proof artifacts and metrics
        """
        threshold = threshold or self.threshold
        t_total = time.perf_counter()

        # ── Stage 1: Classical TDA fingerprinting ──
        t0 = time.perf_counter()
        fp_a = self.tda_system.fingerprint(weights_a)
        tda_proof_a = self.tda_system.prove(weights_a)
        t_tda_a = time.perf_counter() - t0

        t0 = time.perf_counter()
        fp_b = self.tda_system.fingerprint(weights_b)
        tda_proof_b = self.tda_system.prove(weights_b)
        t_tda_b = time.perf_counter() - t0

        # Ensure same number of features (pad/truncate)
        feats_a = self._normalize_features(fp_a.features, self.n_features)
        feats_b = self._normalize_features(fp_b.features, self.n_features)

        # ── Stage 2: Generate ZK witness ──
        from zkml_system.plonk.fingerprint_circuit import (
            FingerprintCircuitCompiler,
            generate_similarity_witness,
            FIXED_POINT_SCALE,
        )

        witness, public = generate_similarity_witness(
            feats_a, feats_b, threshold
        )

        # Compute actual distance
        actual_dist_sq = 0
        for i in range(self.n_features):
            db = (feats_a[i][1] - feats_b[i][1])
            dd = (feats_a[i][2] - feats_b[i][2])
            actual_dist_sq += db * db + dd * dd
        actual_dist_float = actual_dist_sq ** 0.5

        # ── Stage 3: Compile circuit ──
        t0 = time.perf_counter()
        compiler = FingerprintCircuitCompiler(n_features=self.n_features)
        circuit = compiler.compile(witness, public)
        t_compile = time.perf_counter() - t0

        # ── Stage 4: Generate SRS ──
        t0 = time.perf_counter()
        from zkml_system.plonk.plonk_kzg import TrustedSetup
        srs_degree = 1
        while srs_degree <= len(circuit.gates):
            srs_degree *= 2
        # PLONK prover needs n+5 for blinding polynomials
        srs_degree += 6
        srs = TrustedSetup.generate(srs_degree, Fr(42))
        t_srs = time.perf_counter() - t0

        # ── Stage 5: PLONK prove ──
        t0 = time.perf_counter()
        from zkml_system.plonk.plonk_prover import PLONKProver
        prover = PLONKProver(srs)
        proof = prover.prove(circuit)
        t_prove = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total

        return HybridProofBundle(
            fingerprint_a=fp_a,
            fingerprint_b=fp_b,
            tda_proof_a=tda_proof_a,
            tda_proof_b=tda_proof_b,
            plonk_proof=proof,
            circuit_gates=len(circuit.gates),
            srs_degree=srs_degree,
            threshold=threshold,
            threshold_sq_fr=public.threshold_sq,
            commitment_a=public.commitment_a,
            commitment_b=public.commitment_b,
            commitment_random=public.commitment_random,
            tda_time_a=t_tda_a,
            tda_time_b=t_tda_b,
            zk_compile_time=t_compile,
            zk_srs_time=t_srs,
            zk_prove_time=t_prove,
            total_time=t_total,
            actual_distance_sq_fp=int(actual_dist_sq * FIXED_POINT_SCALE**2),
            actual_distance_float=actual_dist_float,
        )

    def verify_similarity(
        self,
        bundle: HybridProofBundle,
    ) -> Tuple[bool, str]:
        """Verify a hybrid TDA+ZK similarity proof.

        Performs three checks:
            1. TDA proof A is valid (SHA256 commitment)
            2. TDA proof B is valid (SHA256 commitment)
            3. PLONK proof is valid (ZK distance assertion)

        Returns:
            (valid, reason) tuple
        """
        # Check 1: TDA proof A
        tda_valid_a, tda_reason_a = self.tda_system.verify(bundle.tda_proof_a)
        if not tda_valid_a:
            return False, f"TDA proof A invalid: {tda_reason_a}"

        # Check 2: TDA proof B
        tda_valid_b, tda_reason_b = self.tda_system.verify(bundle.tda_proof_b)
        if not tda_valid_b:
            return False, f"TDA proof B invalid: {tda_reason_b}"

        # Check 3: Reconstruct circuit and verify PLONK proof
        from zkml_system.plonk.fingerprint_circuit import (
            FingerprintCircuitCompiler,
            FingerprintWitness,
            FingerprintPublicInputs,
            fingerprints_to_features,
        )

        feats_a = self._normalize_features(
            bundle.fingerprint_a.features, self.n_features
        )
        feats_b = self._normalize_features(
            bundle.fingerprint_b.features, self.n_features
        )

        features_a = fingerprints_to_features(feats_a)
        features_b = fingerprints_to_features(feats_b)

        witness = FingerprintWitness(
            features_a=features_a,
            features_b=features_b,
        )
        public = FingerprintPublicInputs(
            threshold_sq=bundle.threshold_sq_fr,
            commitment_a=bundle.commitment_a,
            commitment_b=bundle.commitment_b,
            commitment_random=bundle.commitment_random,
        )

        compiler = FingerprintCircuitCompiler(n_features=self.n_features)
        circuit = compiler.compile(witness, public)

        from zkml_system.plonk.plonk_kzg import TrustedSetup
        srs = TrustedSetup.generate(bundle.srs_degree, Fr(42))

        from zkml_system.plonk.plonk_prover import PLONKVerifier
        verifier = PLONKVerifier(srs)
        plonk_valid = verifier.verify(bundle.plonk_proof, circuit)

        if not plonk_valid:
            return False, "PLONK proof invalid: ZK distance assertion failed"

        return True, "Valid: TDA proofs OK, ZK distance proof OK"

    def _normalize_features(
        self,
        features: List[Tuple[int, float, float]],
        n: int,
    ) -> List[Tuple[int, float, float]]:
        """Ensure exactly n features (truncate or pad with zeros)."""
        if len(features) >= n:
            return features[:n]
        padded = list(features)
        while len(padded) < n:
            padded.append((0, 0.0, 0.0))
        return padded


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(42)

    print("=" * 60)
    print("  Hybrid TDA+ZK Bridge — Self-Test")
    print(f"  Backend: {HybridBridge().backend}")
    print("=" * 60)

    # Create two "similar" models (same arch, slightly different weights)
    base_w1 = np.random.randn(8, 4)
    base_w2 = np.random.randn(4, 2)

    weights_a = [base_w1, base_w2]
    weights_b = [base_w1 + 0.01 * np.random.randn(8, 4),
                 base_w2 + 0.01 * np.random.randn(4, 2)]

    print("\n1. Proving similarity (threshold=5.0)...")
    bridge = HybridBridge(n_features=10, threshold=5.0)
    bundle = bridge.prove_similarity(weights_a, weights_b)
    print(bundle.summary())

    print("\n2. Verifying...")
    valid, reason = bridge.verify_similarity(bundle)
    print(f"   Result: valid={valid}, reason={reason}")

    print(f"\n{'=' * 60}")
    print(f"  SELF-TEST: {'PASS' if valid else 'FAIL'}")
    print(f"{'=' * 60}")
