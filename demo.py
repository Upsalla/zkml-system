#!/usr/bin/env python3
"""
Hybrid TDA+ZK Demo — Zero-Knowledge Model Similarity Proof.

Generates two neural networks with slightly different weights,
proves they are structurally similar via TDA fingerprinting + PLONK,
and verifies the proof — all without revealing the model weights.

Usage:
    OPENBLAS_NUM_THREADS=1 python demo.py
"""
import sys
import os
import time

# Ensure imports work regardless of install method
_here = os.path.dirname(os.path.abspath(__file__))
# Add repo root so 'from zkml_system...' works when not pip-installed
sys.path.insert(0, os.path.dirname(_here))
# Also add this directory for direct 'from plonk...' style imports
sys.path.insert(0, _here)
# Add sibling tda_fingerprint repo if it exists
_tda = os.path.join(os.path.dirname(_here), "tda_fingerprint")
if not os.path.isdir(_tda):
    _tda = os.path.join(os.path.dirname(os.path.dirname(_here)), "tda_fingerprint")
if os.path.isdir(_tda):
    sys.path.insert(0, _tda)
sys.stdout.reconfigure(line_buffering=True)

import numpy as np


def main():
    print("=" * 60)
    print("  Hybrid TDA+ZK Demo")
    print("  Zero-Knowledge Model Similarity Proof")
    print("=" * 60)

    # Detect backend
    from zkml_system.hybrid_bridge import HybridBridge
    bridge = HybridBridge(n_features=10, threshold=5.0)
    print(f"  Backend: {bridge.backend}")
    print()

    # ── Step 1: Generate two similar models ──
    print("1. Generating two neural networks...")
    np.random.seed(42)
    w1 = np.random.randn(16, 8)
    w2 = np.random.randn(8, 4)

    weights_a = [w1, w2]
    weights_b = [
        w1 + 0.01 * np.random.randn(16, 8),  # slight perturbation
        w2 + 0.01 * np.random.randn(8, 4),
    ]
    print(f"   Model A: 2 layers, {sum(w.size for w in weights_a)} params")
    print(f"   Model B: same architecture, ~1% weight perturbation")
    print()

    # ── Step 2: Prove similarity ──
    print("2. Proving similarity (threshold=5.0)...")
    t0 = time.perf_counter()
    bundle = bridge.prove_similarity(weights_a, weights_b)
    total = time.perf_counter() - t0

    print(f"   Circuit gates:  {bundle.circuit_gates}")
    print(f"   Actual distance: {bundle.actual_distance_float:.4f}")
    print(f"   Within threshold: {'YES' if bundle.actual_distance_float <= bundle.threshold else 'NO'}")
    print()
    print(f"   Timing breakdown:")
    print(f"     TDA fingerprinting: {bundle.tda_time_a + bundle.tda_time_b:.3f}s")
    print(f"     Circuit compile:    {bundle.zk_compile_time:.3f}s")
    print(f"     SRS generation:     {bundle.zk_srs_time:.3f}s")
    print(f"     PLONK prove:        {bundle.zk_prove_time:.3f}s")
    print(f"     Total:              {total:.1f}s")
    print()

    # ── Step 3: Verify ──
    print("3. Verifying proof...")
    t0 = time.perf_counter()
    valid, reason = bridge.verify_similarity(bundle)
    verify_time = time.perf_counter() - t0

    print(f"   Result: {'VALID' if valid else 'INVALID'}")
    print(f"   Reason: {reason}")
    print(f"   Verify time: {verify_time:.1f}s")
    print()

    # ── Step 4: Tamper test ──
    print("4. Tamper detection test...")
    from zkml_system.plonk.plonk_prover import PLONKProof
    from zkml_system.crypto.bn254.fr_adapter import Fr
    from zkml_system.crypto.bn254.curve import G1Point

    def clone_fr(f):
        return Fr(f.value)
    def clone_g1(p):
        if p.is_identity():
            return G1Point.identity()
        x, y = p.to_affine()
        return G1Point.from_affine(x, y)

    orig = bundle.plonk_proof
    tampered = PLONKProof(
        com_a=clone_g1(orig.com_a), com_b=clone_g1(orig.com_b),
        com_c=clone_g1(orig.com_c), com_z=clone_g1(orig.com_z),
        com_t_lo=clone_g1(orig.com_t_lo), com_t_mid=clone_g1(orig.com_t_mid),
        com_t_hi=clone_g1(orig.com_t_hi),
        a_bar=clone_fr(orig.a_bar) + Fr(1),  # ← tamper
        b_bar=clone_fr(orig.b_bar), c_bar=clone_fr(orig.c_bar),
        s_sigma1_bar=clone_fr(orig.s_sigma1_bar),
        s_sigma2_bar=clone_fr(orig.s_sigma2_bar),
        z_omega_bar=clone_fr(orig.z_omega_bar),
        r_zeta=clone_fr(orig.r_zeta),
        w_zeta=clone_g1(orig.w_zeta),
        w_zeta_omega=clone_g1(orig.w_zeta_omega),
        n=orig.n,
    )
    bundle.plonk_proof = tampered
    valid_t, reason_t = bridge.verify_similarity(bundle)
    print(f"   Tampered proof accepted: {valid_t}")
    print(f"   Expected: False (tamper detected)")
    print()

    # ── Summary ──
    ok = valid and not valid_t
    print("=" * 60)
    print(f"  DEMO: {'PASS' if ok else 'FAIL'}")
    if ok:
        print(f"  ✓ Similarity proof: valid ({total:.1f}s)")
        print(f"  ✓ Tamper detection: working")
        print(f"  ✓ Circuit size: {bundle.circuit_gates} gates")
    print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
