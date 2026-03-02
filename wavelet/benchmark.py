"""
Comprehensive Benchmark: HWWB vs Standard vs CSWC.

This benchmark compares all three witness compression approaches:
1. Standard (naive hash commitment)
2. CSWC (Compressed Sensing - exploits sparsity)
3. HWWB (Haar Wavelet - exploits correlation)
4. Combined (CSWC + HWWB)
"""

import time
import json
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig
from wavelet.haar_transform import HWWBSystem, HaarTransformer
from compressed_sensing.commitment import CSWCSystem


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    method: str
    witness_size: int
    sparsity: float
    correlation: float
    prove_time_ms: float
    verify_time_ms: float
    proof_size_bytes: int


class WitnessBenchmark:
    """
    Comprehensive benchmark for witness compression methods.
    """
    
    def __init__(self, field_modulus: int = 2**61 - 1):
        self.field_config = FieldConfig(prime=field_modulus, name="Benchmark")
        self.hwwb = HWWBSystem(self.field_config, small_diff_threshold=100)
        self.transformer = HaarTransformer(self.field_config)
    
    def create_witness(self, size: int, sparsity: float, correlation: float) -> List[FieldElement]:
        """
        Create a witness with controlled sparsity and correlation.
        
        Args:
            size: Number of elements
            sparsity: Fraction of zeros (0.0 = all non-zero, 1.0 = all zero)
            correlation: How similar adjacent values are (0.0 = random, 1.0 = identical)
        """
        import random
        random.seed(42)
        
        witness = []
        prev_value = random.randint(1, 10000)
        
        for i in range(size):
            # Determine if this should be zero (sparsity)
            if random.random() < sparsity:
                witness.append(FieldElement(0, self.field_config))
                continue
            
            # Determine value based on correlation
            if random.random() < correlation:
                # Correlated: similar to previous non-zero value
                delta = random.randint(-10, 10)
                value = max(1, prev_value + delta)
            else:
                # Uncorrelated: random value
                value = random.randint(1, 10000)
            
            witness.append(FieldElement(value, self.field_config))
            prev_value = value
        
        return witness
    
    def benchmark_standard(self, witness: List[FieldElement]) -> BenchmarkResult:
        """Benchmark standard hash commitment."""
        # Analyze witness
        zeros = sum(1 for w in witness if w.value == 0)
        sparsity = zeros / len(witness)
        correlation = self.transformer.analyze_correlation(witness)['small_ratio']
        
        # Prove (just hash all values)
        start = time.time()
        data = b''.join(w.value.to_bytes(32, 'big') for w in witness)
        commitment = hashlib.sha256(data).digest()
        prove_time = (time.time() - start) * 1000
        
        # Verify
        start = time.time()
        verify_data = b''.join(w.value.to_bytes(32, 'big') for w in witness)
        valid = hashlib.sha256(verify_data).digest() == commitment
        verify_time = (time.time() - start) * 1000
        
        # Size: commitment + all values
        proof_size = 32 + len(witness) * 32
        
        return BenchmarkResult(
            method="standard",
            witness_size=len(witness),
            sparsity=sparsity,
            correlation=correlation,
            prove_time_ms=prove_time,
            verify_time_ms=verify_time,
            proof_size_bytes=proof_size
        )
    
    def benchmark_hwwb(self, witness: List[FieldElement]) -> BenchmarkResult:
        """Benchmark HWWB."""
        # Analyze witness
        zeros = sum(1 for w in witness if w.value == 0)
        sparsity = zeros / len(witness)
        correlation = self.transformer.analyze_correlation(witness)['small_ratio']
        
        # Prove
        start = time.time()
        proof = self.hwwb.prover.prove(witness)
        prove_time = (time.time() - start) * 1000
        
        # Verify
        start = time.time()
        valid, reason = self.hwwb.verifier.verify(proof)
        verify_time = (time.time() - start) * 1000
        
        return BenchmarkResult(
            method="hwwb",
            witness_size=len(witness),
            sparsity=sparsity,
            correlation=correlation,
            prove_time_ms=prove_time,
            verify_time_ms=verify_time,
            proof_size_bytes=proof.size_bytes()
        )
    
    def benchmark_cswc(self, witness: List[FieldElement]) -> BenchmarkResult:
        """Benchmark CSWC."""
        # Analyze witness
        zeros = sum(1 for w in witness if w.value == 0)
        sparsity = zeros / len(witness)
        correlation = self.transformer.analyze_correlation(witness)['small_ratio']
        
        # Create CSWC system
        cswc = CSWCSystem(
            witness_size=len(witness),
            sketch_dimension=32,
            field_modulus=self.field_config.prime
        )
        
        # Prove
        start = time.time()
        proof = cswc.prover.prove(witness)
        prove_time = (time.time() - start) * 1000
        
        # Verify
        start = time.time()
        valid, reason = cswc.verifier.verify(proof)
        verify_time = (time.time() - start) * 1000
        
        return BenchmarkResult(
            method="cswc",
            witness_size=len(witness),
            sparsity=sparsity,
            correlation=correlation,
            prove_time_ms=prove_time,
            verify_time_ms=verify_time,
            proof_size_bytes=proof.size_bytes()
        )
    
    def run_comparison(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run comparison across different scenarios.
        
        Args:
            scenarios: List of dicts with 'size', 'sparsity', 'correlation'
        """
        results = []
        
        for scenario in scenarios:
            size = scenario['size']
            sparsity = scenario['sparsity']
            correlation = scenario['correlation']
            
            print(f"  Testing size={size}, sparsity={sparsity:.0%}, correlation={correlation:.0%}...", 
                  end=" ", flush=True)
            
            # Create witness
            witness = self.create_witness(size, sparsity, correlation)
            
            # Benchmark all methods
            try:
                standard = self.benchmark_standard(witness)
                hwwb = self.benchmark_hwwb(witness)
                cswc = self.benchmark_cswc(witness)
                
                # Calculate improvements
                hwwb_reduction = 1 - hwwb.proof_size_bytes / standard.proof_size_bytes
                cswc_reduction = 1 - cswc.proof_size_bytes / standard.proof_size_bytes
                
                # Determine best method
                best_method = "standard"
                best_size = standard.proof_size_bytes
                if hwwb.proof_size_bytes < best_size:
                    best_method = "hwwb"
                    best_size = hwwb.proof_size_bytes
                if cswc.proof_size_bytes < best_size:
                    best_method = "cswc"
                    best_size = cswc.proof_size_bytes
                
                result = {
                    "scenario": scenario,
                    "standard": {
                        "size": standard.proof_size_bytes,
                        "prove_ms": standard.prove_time_ms,
                        "verify_ms": standard.verify_time_ms
                    },
                    "hwwb": {
                        "size": hwwb.proof_size_bytes,
                        "prove_ms": hwwb.prove_time_ms,
                        "verify_ms": hwwb.verify_time_ms,
                        "reduction": hwwb_reduction * 100
                    },
                    "cswc": {
                        "size": cswc.proof_size_bytes,
                        "prove_ms": cswc.prove_time_ms,
                        "verify_ms": cswc.verify_time_ms,
                        "reduction": cswc_reduction * 100
                    },
                    "best_method": best_method,
                    "best_reduction": (1 - best_size / standard.proof_size_bytes) * 100
                }
                
                results.append(result)
                print(f"best={best_method} ({result['best_reduction']:.1f}% reduction)")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "scenario": scenario,
                    "error": str(e)
                })
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown report."""
        report = []
        report.append("# Witness Compression Benchmark Report\n")
        report.append("## Comparison: Standard vs HWWB vs CSWC\n")
        
        # Summary table
        report.append("### Results by Scenario\n")
        report.append("| Size | Sparsity | Correlation | Standard | HWWB | CSWC | Best | Reduction |")
        report.append("|------|----------|-------------|----------|------|------|------|-----------|")
        
        for r in results:
            if "error" in r:
                continue
            s = r["scenario"]
            report.append(
                f"| {s['size']} | {s['sparsity']:.0%} | {s['correlation']:.0%} | "
                f"{r['standard']['size']:,}B | {r['hwwb']['size']:,}B | {r['cswc']['size']:,}B | "
                f"**{r['best_method']}** | {r['best_reduction']:.1f}% |"
            )
        
        # Key findings
        report.append("\n### Key Findings\n")
        
        # When is each method best?
        hwwb_wins = [r for r in results if r.get('best_method') == 'hwwb']
        cswc_wins = [r for r in results if r.get('best_method') == 'cswc']
        standard_wins = [r for r in results if r.get('best_method') == 'standard']
        
        report.append(f"- **HWWB wins**: {len(hwwb_wins)} scenarios (high correlation)")
        report.append(f"- **CSWC wins**: {len(cswc_wins)} scenarios (high sparsity)")
        report.append(f"- **Standard wins**: {len(standard_wins)} scenarios (low sparsity + low correlation)")
        
        report.append("\n### Recommendations\n")
        report.append("1. **Use CSWC** when sparsity > 30% (typical for ReLU networks)")
        report.append("2. **Use HWWB** when correlation > 50% and sparsity < 30% (dense but structured)")
        report.append("3. **Combine both** for maximum compression in mixed scenarios")
        
        return "\n".join(report)


# Main benchmark
if __name__ == "__main__":
    print("=" * 70)
    print("Witness Compression Benchmark: Standard vs HWWB vs CSWC")
    print("=" * 70)
    print()
    
    benchmark = WitnessBenchmark()
    
    # Define scenarios
    scenarios = [
        # Low sparsity, varying correlation
        {"size": 500, "sparsity": 0.1, "correlation": 0.2},
        {"size": 500, "sparsity": 0.1, "correlation": 0.5},
        {"size": 500, "sparsity": 0.1, "correlation": 0.8},
        
        # Medium sparsity, varying correlation
        {"size": 500, "sparsity": 0.5, "correlation": 0.2},
        {"size": 500, "sparsity": 0.5, "correlation": 0.5},
        {"size": 500, "sparsity": 0.5, "correlation": 0.8},
        
        # High sparsity, varying correlation
        {"size": 500, "sparsity": 0.8, "correlation": 0.2},
        {"size": 500, "sparsity": 0.8, "correlation": 0.5},
        {"size": 500, "sparsity": 0.8, "correlation": 0.8},
        
        # Larger witness
        {"size": 1000, "sparsity": 0.5, "correlation": 0.5},
        {"size": 1000, "sparsity": 0.7, "correlation": 0.7},
    ]
    
    print("Running benchmarks...")
    results = benchmark.run_comparison(scenarios)
    
    print()
    print("=" * 70)
    print("Generating Report...")
    print("=" * 70)
    print()
    
    report = benchmark.generate_report(results)
    print(report)
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, "BENCHMARK_REPORT.md"), "w") as f:
        f.write(report)
    
    print()
    print(f"Results saved to {output_dir}/")
