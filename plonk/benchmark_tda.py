"""
ZK-TDA Comprehensive Benchmark Suite.

Tests circuit scaling across:
  1. Landmark count (5 → 25)
  2. Model size (small → large)
  3. Point dimension
  4. Persistence pair count

Outputs structured data for the research report.

Author: David Weyhe
Date: 2026-03-01
"""

import numpy as np
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.plonk.tda_prover import ZKTDAProver


def create_model(layer_sizes, seed=42):
    """Create a random neural network with given layer sizes."""
    np.random.seed(seed)
    weights = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
        weights.append(W)
    return weights


def run_benchmark_suite():
    """Run the full benchmark suite."""

    print("=" * 70)
    print("ZK-TDA Comprehensive Benchmark")
    print("=" * 70)
    print()

    all_results = {}

    # ================================================================
    # Benchmark 1: Landmark Scaling
    # ================================================================
    print("━" * 70)
    print("BENCHMARK 1: Gate Count vs Landmark Count")
    print("  Model: 3-layer (50→30→20→10), 2,500 params")
    print("  Features: 10, Pairs verified: 10")
    print("━" * 70)

    model = create_model([50, 30, 20, 10], seed=42)
    total_params = sum(w.size for w in model)
    total_neurons = sum(w.shape[0] for w in model)
    print(f"  Total params: {total_params}, Total neurons: {total_neurons}")
    print()

    landmark_results = []
    landmarks_to_test = [5, 8, 10, 12, 15, 18, 20]

    print(f"{'N':>4} {'Gates':>8} {'Dist':>7} {'Bnd':>7} "
          f"{'Wires':>7} {'Pairs':>6} {'Simpl':>6} {'ms':>7} {'Valid':>6}")
    print("-" * 65)

    for n in landmarks_to_test:
        if n > total_neurons:
            continue
        prover = ZKTDAProver(n_landmarks=n, n_features=10, verify_top_k_pairs=10)
        proof = prover.prove(model)
        valid, _ = prover.verify(proof)

        row = {
            "landmarks": n,
            "total_gates": proof.combined_gates,
            "distance_gates": proof.distance_circuit_gates,
            "boundary_gates": proof.boundary_circuit_gates,
            "wires": proof.total_wires,
            "pairs": proof.n_persistence_pairs,
            "simplices": proof.n_simplices,
            "time_ms": round(proof.total_proof_time_ms, 1),
            "valid": valid,
        }
        landmark_results.append(row)

        print(f"{n:>4} {proof.combined_gates:>8,} {proof.distance_circuit_gates:>7,} "
              f"{proof.boundary_circuit_gates:>7,} {proof.total_wires:>7,} "
              f"{proof.n_persistence_pairs:>6} {proof.n_simplices:>6} "
              f"{proof.total_proof_time_ms:>7.1f} {'✅' if valid else '❌':>6}")

    all_results["landmark_scaling"] = landmark_results

    # ================================================================
    # Benchmark 2: Model Size Scaling
    # ================================================================
    print()
    print("━" * 70)
    print("BENCHMARK 2: Gate Count vs Model Size")
    print("  Landmarks: 15, Features: 8, Pairs: 10")
    print("━" * 70)
    print()

    model_configs = [
        ("Tiny (3→2)",      [3, 2],           1),
        ("Small (10→5→3)",  [10, 5, 3],       2),
        ("Medium (20→10→5)",[20, 10, 5],      3),
        ("Large (50→30→10)",[50, 30, 10],     4),
        ("XL (100→50→20)",  [100, 50, 20],    5),
    ]

    model_results = []
    print(f"{'Model':>22} {'Params':>8} {'Gates':>8} {'Wires':>7} "
          f"{'Pairs':>6} {'ms':>7} {'Valid':>6}")
    print("-" * 70)

    for name, sizes, seed in model_configs:
        model = create_model(sizes, seed=seed)
        params = sum(w.size for w in model)
        neurons = sum(w.shape[0] for w in model)

        n_land = min(15, neurons)
        prover = ZKTDAProver(n_landmarks=n_land, n_features=8, verify_top_k_pairs=10)
        proof = prover.prove(model)
        valid, _ = prover.verify(proof)

        row = {
            "name": name,
            "layer_sizes": sizes,
            "params": params,
            "neurons": neurons,
            "landmarks_used": n_land,
            "total_gates": proof.combined_gates,
            "wires": proof.total_wires,
            "pairs": proof.n_persistence_pairs,
            "time_ms": round(proof.total_proof_time_ms, 1),
            "valid": valid,
        }
        model_results.append(row)

        print(f"{name:>22} {params:>8} {proof.combined_gates:>8,} "
              f"{proof.total_wires:>7,} {proof.n_persistence_pairs:>6} "
              f"{proof.total_proof_time_ms:>7.1f} {'✅' if valid else '❌':>6}")

    all_results["model_scaling"] = model_results

    # ================================================================
    # Benchmark 3: Persistence Pair Count
    # ================================================================
    print()
    print("━" * 70)
    print("BENCHMARK 3: Gate Count vs Pairs Verified")
    print("  Model: 3-layer (50→30→20→10), Landmarks: 15")
    print("━" * 70)
    print()

    model = create_model([50, 30, 20, 10], seed=42)
    pair_results = []
    pairs_to_test = [1, 3, 5, 8, 10, 15, 20]

    print(f"{'Pairs':>6} {'Gates':>8} {'Bnd':>7} {'Wires':>7} "
          f"{'ms':>7} {'Valid':>6}")
    print("-" * 48)

    for n_pairs in pairs_to_test:
        prover = ZKTDAProver(
            n_landmarks=15, n_features=8, verify_top_k_pairs=n_pairs
        )
        proof = prover.prove(model)
        valid, _ = prover.verify(proof)

        row = {
            "pairs_verified": n_pairs,
            "total_gates": proof.combined_gates,
            "boundary_gates": proof.boundary_circuit_gates,
            "wires": proof.total_wires,
            "time_ms": round(proof.total_proof_time_ms, 1),
            "valid": valid,
        }
        pair_results.append(row)

        print(f"{n_pairs:>6} {proof.combined_gates:>8,} "
              f"{proof.boundary_circuit_gates:>7,} {proof.total_wires:>7,} "
              f"{proof.total_proof_time_ms:>7.1f} {'✅' if valid else '❌':>6}")

    all_results["pair_scaling"] = pair_results

    # ================================================================
    # Benchmark 4: Fingerprint Discrimination
    # ================================================================
    print()
    print("━" * 70)
    print("BENCHMARK 4: Fingerprint Discrimination")
    print("  Do different models produce different fingerprints?")
    print("━" * 70)
    print()

    discrimination_results = []
    models = {
        "MLP-A": create_model([20, 10, 5], seed=1),
        "MLP-B": create_model([20, 10, 5], seed=2),
        "MLP-C": create_model([20, 10, 5], seed=3),
        "Deep":  create_model([20, 15, 10, 5, 3], seed=4),
        "Wide":  create_model([20, 50, 5], seed=5),
    }

    fingerprints = {}
    print(f"{'Model':>8} {'Params':>8} {'Gates':>8} {'Fingerprint':>28}")
    print("-" * 58)

    for name, model in models.items():
        params = sum(w.size for w in model)
        prover = ZKTDAProver(n_landmarks=15, n_features=8)
        proof = prover.prove(model)
        fp = proof.fingerprint_hash[:24]
        fingerprints[name] = proof.fingerprint_hash

        print(f"{name:>8} {params:>8} {proof.combined_gates:>8,} {fp}...")

        discrimination_results.append({
            "name": name,
            "params": params,
            "fingerprint": proof.fingerprint_hash,
        })

    # Check uniqueness
    print()
    unique = len(set(fingerprints.values()))
    total = len(fingerprints)
    print(f"  Unique fingerprints: {unique}/{total} "
          f"{'✅ all distinct' if unique == total else '⚠️ collisions!'}")

    all_results["discrimination"] = discrimination_results

    # ================================================================
    # Benchmark 5: Determinism
    # ================================================================
    print()
    print("━" * 70)
    print("BENCHMARK 5: Determinism")
    print("  Same model → same fingerprint on repeated runs?")
    print("━" * 70)
    print()

    model = create_model([20, 10, 5], seed=42)
    fps = []
    for run in range(3):
        prover = ZKTDAProver(n_landmarks=10, n_features=5)
        proof = prover.prove(model)
        fps.append(proof.fingerprint_hash)
        print(f"  Run {run+1}: {proof.fingerprint_hash[:24]}...")

    deterministic = all(fp == fps[0] for fp in fps)
    print(f"  Deterministic: {'✅ YES' if deterministic else '❌ NO'}")

    all_results["determinism"] = {
        "fingerprints": fps,
        "is_deterministic": deterministic,
    }

    # ================================================================
    # Constraint Analysis
    # ================================================================
    print()
    print("━" * 70)
    print("CONSTRAINT ANALYSIS")
    print("━" * 70)
    print()

    # Use the 15-landmark proof for detailed breakdown
    model = create_model([50, 30, 20, 10], seed=42)
    prover = ZKTDAProver(n_landmarks=15, n_features=8, verify_top_k_pairs=10)
    proof = prover.prove(model)

    print("Gate Breakdown:")
    for gtype, count in sorted(proof.gate_breakdown.items()):
        pct = count / proof.combined_gates * 100
        bar = "█" * int(pct / 2)
        print(f"  {gtype:>8}: {count:>6} ({pct:>5.1f}%) {bar}")

    print()
    print("Sub-Circuit Breakdown:")
    dist_pct = proof.distance_circuit_gates / proof.combined_gates * 100
    bnd_pct = proof.boundary_circuit_gates / proof.combined_gates * 100
    print(f"  Distance + Filtration: {proof.distance_circuit_gates:>6} ({dist_pct:.1f}%)")
    print(f"  Boundary Reduction:    {proof.boundary_circuit_gates:>6} ({bnd_pct:.1f}%)")
    print(f"  Combined:              {proof.combined_gates:>6}")
    print()

    # Theoretical estimates
    N = 15
    D = len(create_model([50, 30, 20, 10], seed=42)[0][0])  # point dimension
    E = N * (N - 1) // 2
    print("Theoretical Scaling:")
    print(f"  N={N} landmarks, D=5 dims, E={E} edges")
    print(f"  Distance matrix: O(N²·D) = O({N}²·5) ≈ {E * 3 * 5} constraints")
    print(f"  Filtration order: O(E) = O({E}) ≈ {E} constraints")
    print(f"  Hash (commitment): O(N·D) ≈ {N * 5 * 3} constraints")
    print(f"  Hash (fingerprint): O(features) ≈ {8 * 3 * 3} constraints")
    print(f"  Boundary mod-2: O(pairs · rows) — data-dependent")
    print()
    print(f"  Actual total: {proof.combined_gates:,} gates")
    print(f"  Budget (20k): {proof.combined_gates / 20000 * 100:.1f}% utilized")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "benchmark_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark_suite()
