"""
Comprehensive Benchmark for TDA Model Fingerprinting.

Tests:
1. Uniqueness: Different models should have different fingerprints
2. Stability: Similar models should have similar fingerprints
3. Scalability: Fingerprint size should be constant regardless of model size
4. Performance: Measure computation time for various model sizes
"""

import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tda.fingerprint import TDAFingerprintSystem, ModelFingerprint


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    passed: bool
    details: Dict[str, Any]


class TDABenchmark:
    """Comprehensive benchmark suite for TDA fingerprinting."""
    
    def __init__(self):
        self.system = TDAFingerprintSystem(n_features=20, n_samples=10)
        self.results: List[BenchmarkResult] = []
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""
        self.results = []
        
        print("=" * 70)
        print("TDA Fingerprinting Benchmark Suite")
        print("=" * 70)
        
        self._test_uniqueness()
        self._test_stability()
        self._test_scalability()
        self._test_performance()
        self._test_collision_resistance()
        
        return self.results
    
    def _test_uniqueness(self):
        """Test that different models produce different fingerprints."""
        print("\n1. UNIQUENESS TEST")
        print("-" * 40)
        
        np.random.seed(42)
        n_models = 20
        fingerprints = []
        
        for i in range(n_models):
            weights = [
                np.random.randn(10, 5),
                np.random.randn(8, 10),
                np.random.randn(3, 8)
            ]
            fp = self.system.fingerprint(weights)
            fingerprints.append(fp)
        
        # Check all pairs
        collisions = 0
        total_pairs = n_models * (n_models - 1) // 2
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if fingerprints[i] == fingerprints[j]:
                    collisions += 1
        
        collision_rate = collisions / total_pairs
        passed = collision_rate == 0
        
        print(f"   Models tested: {n_models}")
        print(f"   Pairs compared: {total_pairs}")
        print(f"   Collisions: {collisions}")
        print(f"   Collision rate: {collision_rate:.4%}")
        print(f"   PASSED: {passed}")
        
        self.results.append(BenchmarkResult(
            test_name="uniqueness",
            passed=passed,
            details={
                "n_models": n_models,
                "collisions": collisions,
                "collision_rate": collision_rate
            }
        ))
    
    def _test_stability(self):
        """Test that small perturbations produce similar fingerprints."""
        print("\n2. STABILITY TEST")
        print("-" * 40)
        
        np.random.seed(42)
        
        # Create base model
        base_weights = [
            np.random.randn(10, 5),
            np.random.randn(8, 10),
            np.random.randn(3, 8)
        ]
        base_fp = self.system.fingerprint(base_weights)
        
        # Test various perturbation levels
        perturbation_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
        distances = []
        
        for noise in perturbation_levels:
            perturbed = [w + np.random.randn(*w.shape) * noise for w in base_weights]
            perturbed_fp = self.system.fingerprint(perturbed)
            dist = base_fp.distance(perturbed_fp)
            distances.append(dist)
            print(f"   Noise {noise:.1%}: distance = {dist:.4f}, same = {base_fp == perturbed_fp}")
        
        # Stability criterion: distance should scale roughly with noise
        # Small noise should give small distance
        passed = distances[0] < 0.1 and distances[-1] > distances[0]
        
        print(f"   PASSED: {passed}")
        
        self.results.append(BenchmarkResult(
            test_name="stability",
            passed=passed,
            details={
                "perturbation_levels": perturbation_levels,
                "distances": distances
            }
        ))
    
    def _test_scalability(self):
        """Test that fingerprint size is constant regardless of model size."""
        print("\n3. SCALABILITY TEST")
        print("-" * 40)
        
        np.random.seed(42)
        
        model_sizes = [
            ([10, 5], [8, 10], [3, 8]),           # Small: ~150 params
            ([64, 32], [32, 64], [16, 32]),       # Medium: ~5K params
            ([128, 64], [64, 128], [32, 64]),     # Large: ~20K params
            ([256, 128], [128, 256], [64, 128])   # XLarge: ~80K params
        ]
        
        fingerprint_sizes = []
        param_counts = []
        
        for sizes in model_sizes:
            weights = [np.random.randn(*s) for s in sizes]
            params = sum(w.size for w in weights)
            param_counts.append(params)
            
            fp = self.system.fingerprint(weights)
            fingerprint_sizes.append(fp.size_bytes())
            
            print(f"   {params:,} params → {fp.size_bytes()} bytes fingerprint")
        
        # All fingerprints should be the same size
        all_same_size = len(set(fingerprint_sizes)) == 1
        passed = all_same_size
        
        print(f"   Fingerprint sizes: {fingerprint_sizes}")
        print(f"   All same size: {all_same_size}")
        print(f"   PASSED: {passed}")
        
        self.results.append(BenchmarkResult(
            test_name="scalability",
            passed=passed,
            details={
                "param_counts": param_counts,
                "fingerprint_sizes": fingerprint_sizes,
                "all_same_size": all_same_size
            }
        ))
    
    def _test_performance(self):
        """Test computation time for various model sizes."""
        print("\n4. PERFORMANCE TEST")
        print("-" * 40)
        
        np.random.seed(42)
        
        model_configs = [
            {"name": "tiny", "layers": [(10, 5), (5, 10), (3, 5)]},
            {"name": "small", "layers": [(32, 16), (16, 32), (8, 16)]},
            {"name": "medium", "layers": [(64, 32), (32, 64), (16, 32)]},
            {"name": "large", "layers": [(128, 64), (64, 128), (32, 64)]},
        ]
        
        results = []
        
        for config in model_configs:
            weights = [np.random.randn(*s) for s in config["layers"]]
            params = sum(w.size for w in weights)
            
            # Warm-up
            _ = self.system.fingerprint(weights)
            
            # Measure
            times = []
            for _ in range(3):
                start = time.time()
                fp = self.system.fingerprint(weights)
                times.append((time.time() - start) * 1000)
            
            avg_time = np.mean(times)
            results.append({
                "name": config["name"],
                "params": params,
                "time_ms": avg_time
            })
            
            print(f"   {config['name']:8} ({params:,} params): {avg_time:.2f} ms")
        
        # Performance criterion: should complete in reasonable time
        passed = all(r["time_ms"] < 5000 for r in results)  # < 5 seconds
        
        print(f"   PASSED: {passed}")
        
        self.results.append(BenchmarkResult(
            test_name="performance",
            passed=passed,
            details={"results": results}
        ))
    
    def _test_collision_resistance(self):
        """Test resistance to intentional collision attempts."""
        print("\n5. COLLISION RESISTANCE TEST")
        print("-" * 40)
        
        np.random.seed(42)
        
        # Create target fingerprint
        target_weights = [
            np.random.randn(10, 5),
            np.random.randn(8, 10),
            np.random.randn(3, 8)
        ]
        target_fp = self.system.fingerprint(target_weights)
        
        # Try to find collision with random models
        n_attempts = 100
        collisions = 0
        min_distance = float('inf')
        
        for _ in range(n_attempts):
            random_weights = [
                np.random.randn(10, 5),
                np.random.randn(8, 10),
                np.random.randn(3, 8)
            ]
            random_fp = self.system.fingerprint(random_weights)
            
            if random_fp == target_fp:
                collisions += 1
            
            dist = target_fp.distance(random_fp)
            min_distance = min(min_distance, dist)
        
        passed = collisions == 0
        
        print(f"   Attempts: {n_attempts}")
        print(f"   Collisions found: {collisions}")
        print(f"   Minimum distance: {min_distance:.4f}")
        print(f"   PASSED: {passed}")
        
        self.results.append(BenchmarkResult(
            test_name="collision_resistance",
            passed=passed,
            details={
                "n_attempts": n_attempts,
                "collisions": collisions,
                "min_distance": min_distance
            }
        ))
    
    def generate_report(self) -> str:
        """Generate markdown report."""
        report = []
        report.append("# TDA Fingerprinting Benchmark Report\n")
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        report.append(f"## Summary: {passed}/{total} tests passed\n")
        
        # Results table
        report.append("## Test Results\n")
        report.append("| Test | Status | Details |")
        report.append("|------|--------|---------|")
        
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            details = json.dumps(r.details, default=str)[:50] + "..."
            report.append(f"| {r.test_name} | {status} | {details} |")
        
        # Key findings
        report.append("\n## Key Findings\n")
        
        for r in self.results:
            report.append(f"### {r.test_name.title()}\n")
            if r.test_name == "uniqueness":
                report.append(f"- Tested {r.details['n_models']} random models")
                report.append(f"- Found {r.details['collisions']} collisions")
                report.append(f"- Collision rate: {r.details['collision_rate']:.4%}")
            elif r.test_name == "scalability":
                report.append(f"- Fingerprint size is **constant** ({r.details['fingerprint_sizes'][0]} bytes)")
                report.append(f"- Tested models from {min(r.details['param_counts']):,} to {max(r.details['param_counts']):,} parameters")
            elif r.test_name == "performance":
                for res in r.details['results']:
                    report.append(f"- {res['name']}: {res['time_ms']:.2f} ms")
            report.append("")
        
        return "\n".join(report)


# Main
if __name__ == "__main__":
    benchmark = TDABenchmark()
    results = benchmark.run_all()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.passed)
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    # Save report
    report = benchmark.generate_report()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(output_dir, "BENCHMARK_REPORT.md"), "w") as f:
        f.write(report)
    
    print(f"\nReport saved to {output_dir}/BENCHMARK_REPORT.md")
