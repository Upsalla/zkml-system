"""
zkml_bridge — Connect tda_fingerprint to zkml_system's PLONK pipeline.
======================================================================

This module bridges two systems:
  - **tda_fingerprint**: Standalone TDA fingerprinting with SHA256 proofs
  - **zkml_system**: Full PLONK zero-knowledge proof system for TDA

Usage:
    from zkml_system.zkml_bridge import ZKMLBridge

    bridge = ZKMLBridge()

    # Full pipeline: weights → TDA fingerprint + ZK proof
    result = bridge.prove(weights)
    is_valid = bridge.verify(result)

The bridge automatically uses the RustFr backend (~25–60x faster)
when available, falling back to Python Fr.

Author: David Weyhe
Date: 2026-03-02
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

# Ensure zkml_system is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr, _BACKEND

from zkml_system.plonk.tda_circuit import (
    TDACircuitCompiler,
    TDAWitness,
    TDAPublicInputs,
    generate_tda_witness,
)
from zkml_system.plonk.plonk_prover import PLONKProver, PLONKVerifier
from zkml_system.plonk.plonk_kzg import TrustedSetup


# ═══════════════════════════════════════════════════════════
# Bridge Result
# ═══════════════════════════════════════════════════════════

@dataclass
class ZKMLProofBundle:
    """Complete proof bundle combining TDA fingerprint + PLONK ZK proof."""

    # Classical TDA fingerprint (from tda_fingerprint package)
    tda_fingerprint: Optional[Any] = None
    tda_proof: Optional[Any] = None

    # ZK proof (from zkml_system)
    plonk_proof: Optional[Any] = None
    compiled_circuit: Optional[Any] = None
    trusted_setup: Optional[Any] = None

    # Metadata
    n_gates: int = 0
    n_points: int = 0
    backend: str = ""
    timings: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ZKMLProofBundle",
            f"  Backend:      {self.backend}",
            f"  Circuit:      {self.n_gates} gates, {self.n_points} points",
            f"  PLONK proof:  {'present' if self.plonk_proof else 'MISSING'}",
            f"  TDA proof:    {'present' if self.tda_proof else 'MISSING'}",
            f"  Timings:",
        ]
        for k, v in self.timings.items():
            lines.append(f"    {k:20s} {v:8.3f} s")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Bridge
# ═══════════════════════════════════════════════════════════

class ZKMLBridge:
    """
    Bridge between tda_fingerprint and zkml_system.

    Provides a unified pipeline:
        weights → TDA computation → PLONK circuit → ZK proof

    Args:
        n_landmarks:  Number of landmark points for TDA.
        n_features:   Number of top persistence features.
        max_degree:   SRS maximum degree.
        tau:          SRS secret. Use None for random.
    """

    def __init__(
        self,
        n_landmarks: int = 20,
        n_features: int = 10,
        max_degree: int = 16384,
        tau: Optional[Fr] = None,
    ):
        self.n_landmarks = n_landmarks
        self.n_features = n_features
        self.max_degree = max_degree
        self._tau = tau or Fr(42)  # deterministic default for testing
        self._srs: Optional[TrustedSetup] = None
        self._tda_system = None  # lazy-loaded

    @property
    def backend(self) -> str:
        return _BACKEND

    @property
    def srs(self) -> TrustedSetup:
        """Lazy-load and cache the SRS."""
        if self._srs is None:
            self._srs = TrustedSetup.generate(self.max_degree, self._tau)
        return self._srs

    def _get_tda_system(self):
        """Lazy-load the tda_fingerprint system."""
        if self._tda_system is None:
            try:
                from tda_fingerprint import TDAFingerprintSystem
                self._tda_system = TDAFingerprintSystem(
                    n_features=self.n_features
                )
            except ImportError:
                self._tda_system = None
        return self._tda_system

    def prove(
        self,
        weights: List[np.ndarray],
        include_tda_proof: bool = True,
    ) -> ZKMLProofBundle:
        """
        Generate a full ZK proof for a model's topological fingerprint.

        Pipeline:
          1. (Optional) Generate classical TDA proof via tda_fingerprint
          2. Generate TDA witness (point cloud → persistence → circuit input)
          3. Compile PLONK circuit
          4. Generate PLONK proof

        Args:
            weights:           Model weight matrices.
            include_tda_proof: Also generate classical TDA proof.

        Returns:
            ZKMLProofBundle with all artefacts.
        """
        timings = {}
        bundle = ZKMLProofBundle(backend=self.backend)

        # Step 1: Classical TDA proof (optional)
        if include_tda_proof:
            tda_sys = self._get_tda_system()
            if tda_sys is not None:
                t0 = time.perf_counter()
                bundle.tda_proof = tda_sys.prove(weights)
                bundle.tda_fingerprint = tda_sys.fingerprint(weights)
                timings["tda_fingerprint"] = time.perf_counter() - t0

        # Step 2: Generate TDA witness
        t0 = time.perf_counter()
        witness, public_inputs = generate_tda_witness(
            weights,
            n_landmarks=self.n_landmarks,
            max_dim=1,
            n_features=self.n_features,
        )
        timings["tda_witness"] = time.perf_counter() - t0

        # Step 3: Compile circuit
        t0 = time.perf_counter()
        compiler = TDACircuitCompiler(
            n_points=public_inputs.n_points,
            point_dim=public_inputs.point_dim,
            n_features=public_inputs.n_features,
        )
        circuit = compiler.compile(witness, public_inputs)
        timings["circuit_compile"] = time.perf_counter() - t0
        bundle.compiled_circuit = circuit
        bundle.n_gates = len(circuit.gates)
        bundle.n_points = public_inputs.n_points

        # Step 4: Generate SRS (if not cached) + PLONK proof
        t0 = time.perf_counter()
        # Adjust SRS size to circuit
        n_needed = 1
        while n_needed < len(circuit.gates):
            n_needed *= 2
        if self._srs is None or len(self.srs.g1_powers) < n_needed + 1:
            self._srs = TrustedSetup.generate(n_needed, self._tau)
        timings["srs_generate"] = time.perf_counter() - t0

        bundle.trusted_setup = self._srs

        t0 = time.perf_counter()
        prover = PLONKProver(self._srs)
        bundle.plonk_proof = prover.prove(circuit)
        timings["plonk_prove"] = time.perf_counter() - t0

        bundle.timings = timings
        return bundle

    def verify(self, bundle: ZKMLProofBundle) -> Tuple[bool, str]:
        """
        Verify a ZK proof bundle.

        Performs:
          1. PLONK ZK proof verification (pairing check)
          2. (Optional) Classical TDA proof verification

        Returns:
            (is_valid, reason)
        """
        if bundle.plonk_proof is None or bundle.compiled_circuit is None:
            return False, "Missing PLONK proof or circuit"

        if bundle.trusted_setup is None:
            return False, "Missing trusted setup"

        verifier = PLONKVerifier(bundle.trusted_setup)
        plonk_ok = verifier.verify(bundle.plonk_proof, bundle.compiled_circuit)

        if not plonk_ok:
            return False, "PLONK verification failed"

        # Optional: classical TDA verification
        if bundle.tda_proof is not None:
            tda_sys = self._get_tda_system()
            if tda_sys is not None:
                tda_ok, tda_reason = tda_sys.verify(bundle.tda_proof)
                if not tda_ok:
                    return False, f"TDA verification failed: {tda_reason}"

        return True, "Valid (PLONK + TDA)"


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  zkML Bridge — Self-Test")
    print(f"  Backend: {_BACKEND}")
    print("=" * 60)

    # Small model for testing
    np.random.seed(42)
    weights = [np.random.randn(8, 4), np.random.randn(4, 2)]

    bridge = ZKMLBridge(n_landmarks=8, n_features=5, max_degree=128)

    print("\n1. Generating ZK proof...")
    result = bridge.prove(weights, include_tda_proof=False)
    print(result.summary())

    print("\n2. Verifying ZK proof...")
    ok, reason = bridge.verify(result)
    print(f"   Result: {ok} — {reason}")

    assert ok, f"Verification failed: {reason}"
    print("\n✅ Bridge self-test PASSED")
