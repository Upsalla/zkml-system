#!/usr/bin/env python3
"""
E2E Benchmark: RustFr vs Python Fr Performance
================================================
Part 1: Raw field arithmetic (isolated, no cross-contamination)
Part 2: Pipeline operations using the adapter's selected backend
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def bench(name: str, fn, iterations: int = 1) -> dict:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    per_iter = elapsed / iterations
    return {"name": name, "total_s": round(elapsed, 4), "per_iter_ms": round(per_iter * 1000, 3), "iterations": iterations}


# ═══════════════════════════════════════════════════════════
# PART 1: Raw Field Arithmetic
# ═══════════════════════════════════════════════════════════

def bench_rust_field(n=10000):
    from zkml_rust import RustFr as Fr
    a, b = Fr(123456789), Fr(987654321)
    results = []

    def do_add():
        x = a
        for _ in range(n): x = x + b
    results.append(bench(f"Fr.add x{n}", do_add))

    def do_mul():
        x = a
        for _ in range(n): x = x * b
    results.append(bench(f"Fr.mul x{n}", do_mul))

    def do_inv():
        x = a
        for _ in range(n // 10): x = x.inverse()
    results.append(bench(f"Fr.inverse x{n // 10}", do_inv))

    def do_pow():
        for _ in range(n // 10): _ = a ** 65537
    results.append(bench(f"Fr.pow(65537) x{n // 10}", do_pow))

    def do_pow_large():
        _ = Fr(5) ** ((MODULUS - 1) // 4)
    results.append(bench("Fr.pow(r-1)/4", do_pow_large, iterations=10))

    return results


def bench_python_field(n=10000):
    from zkml_system.crypto.bn254.field import Fr
    a, b = Fr(123456789), Fr(987654321)
    results = []

    def do_add():
        x = a
        for _ in range(n): x = x + b
    results.append(bench(f"Fr.add x{n}", do_add))

    def do_mul():
        x = a
        for _ in range(n): x = x * b
    results.append(bench(f"Fr.mul x{n}", do_mul))

    def do_inv():
        x = a
        for _ in range(n // 10): x = x.inverse()
    results.append(bench(f"Fr.inverse x{n // 10}", do_inv))

    def do_pow():
        for _ in range(n // 10): _ = a ** 65537
    results.append(bench(f"Fr.pow(65537) x{n // 10}", do_pow))

    def do_pow_large():
        _ = Fr(5) ** ((MODULUS - 1) // 4)
    results.append(bench("Fr.pow(r-1)/4", do_pow_large, iterations=10))

    return results


# ═══════════════════════════════════════════════════════════
# PART 2: Pipeline (adapter-selected backend)
# ═══════════════════════════════════════════════════════════

def bench_pipeline():
    """Benchmark pipeline ops — uses whichever backend the adapter selects."""
    from zkml_system.crypto.bn254.fr_adapter import Fr, _BACKEND
    from zkml_system.plonk.polynomial import Polynomial, FFT
    from zkml_system.plonk.kzg import SRS, KZG
    results = []

    n = 64
    coeffs = [Fr(i + 1) for i in range(n)]
    p = Polynomial(coeffs)

    def do_eval():
        _ = p.evaluate(Fr(42))
    results.append(bench(f"Poly.evaluate (deg={n})", do_eval, iterations=100))

    q = Polynomial([Fr(i + 10) for i in range(n // 2)])
    def do_mul():
        _ = p * q
    results.append(bench(f"Poly.mul ({n}x{n // 2})", do_mul, iterations=10))

    fft = FFT(n)
    def do_fft():
        _ = fft.fft(coeffs)
    results.append(bench(f"FFT forward (n={n})", do_fft, iterations=10))

    evals = fft.fft(coeffs)
    def do_ifft():
        _ = fft.ifft(evals)
    results.append(bench(f"FFT inverse (n={n})", do_ifft, iterations=10))

    tau = Fr(12345)
    def do_srs():
        _ = SRS.generate(32, tau)
    results.append(bench("SRS.generate (n=32)", do_srs, iterations=1))

    srs = SRS.generate(32, tau)
    kzg = KZG(srs)
    p32 = Polynomial([Fr(i + 1) for i in range(32)])
    def do_commit():
        _ = kzg.commit(p32)
    results.append(bench("KZG.commit (deg=32)", do_commit, iterations=1))

    return results, _BACKEND


if __name__ == "__main__":
    print("=" * 60)
    print("  zkml_system: RustFr vs Python Fr Benchmark")
    print("=" * 60)

    # Part 1: Raw field comparison
    print("\n" + "=" * 60)
    print("  PART 1: Raw Field Arithmetic")
    print("=" * 60)

    rust_results = bench_rust_field()
    print(f"\n  {'Operation':30s}  {'Rust (ms)':>10s}")
    print(f"  {'-' * 30}  {'-' * 10}")
    for r in rust_results:
        print(f"  {r['name']:30s}  {r['per_iter_ms']:10.3f}")

    py_results = bench_python_field()
    print(f"\n  {'Operation':30s}  {'Python (ms)':>12s}")
    print(f"  {'-' * 30}  {'-' * 12}")
    for r in py_results:
        print(f"  {r['name']:30s}  {r['per_iter_ms']:12.3f}")

    # Comparison
    print(f"\n  {'Operation':30s}  {'Rust':>8s}  {'Python':>10s}  {'Speedup':>8s}")
    print(f"  {'-' * 30}  {'-' * 8}  {'-' * 10}  {'-' * 8}")
    rust_map = {r['name']: r for r in rust_results}
    py_map = {r['name']: r for r in py_results}
    for name in rust_map:
        r_ms = rust_map[name]['per_iter_ms']
        p_ms = py_map[name]['per_iter_ms']
        speedup = p_ms / r_ms if r_ms > 0 else float('inf')
        print(f"  {name:30s}  {r_ms:7.3f}  {p_ms:10.3f}  {speedup:7.1f}x")

    # Part 2: Pipeline
    print("\n" + "=" * 60)
    print("  PART 2: Pipeline Operations (adapter backend)")
    print("=" * 60)
    pipeline_results, backend = bench_pipeline()
    print(f"\n  Active backend: {backend}")
    print(f"  {'Operation':30s}  {'Time (ms)':>10s}")
    print(f"  {'-' * 30}  {'-' * 10}")
    for r in pipeline_results:
        print(f"  {r['name']:30s}  {r['per_iter_ms']:10.3f}")

    # Save
    output = {
        "rust_field": rust_results,
        "python_field": py_results,
        "pipeline": pipeline_results,
        "pipeline_backend": backend,
    }
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
