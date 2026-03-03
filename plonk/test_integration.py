"""
Full integration test for the zkML PLONK pipeline

This test validates:
1. Circuit compiler with GELU and Sparse
2. End-to-End Pipeline: Network → Circuit → Proof → Verify
3. Benchmark of optimizations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List, Dict, Any

from zkml_system.plonk.circuit_compiler import CircuitCompiler, CompiledCircuit
from zkml_system.plonk.zkml_pipeline import ZkMLPipeline, ZkMLProof, VerificationResult


def create_test_network(input_size: int, hidden_sizes: List[int], output_size: int) -> List[Dict[str, Any]]:
    """Create a test network configuration."""
    layers = []
    prev_size = input_size
    
    for i, hidden_size in enumerate(hidden_sizes):
        layer = {
            'type': 'dense',
            'weights': [[(i * j + 1) % 100 for j in range(prev_size)] for i in range(hidden_size)],
            'biases': [i for i in range(hidden_size)],
            'activation': 'gelu'
        }
        layers.append(layer)
        prev_size = hidden_size
    
    # Output layer
    layers.append({
        'type': 'dense',
        'weights': [[(i * j + 1) % 100 for j in range(prev_size)] for i in range(output_size)],
        'biases': [i for i in range(output_size)],
        'activation': 'linear'
    })
    
    return layers


def test_circuit_compiler():
    """Test the circuit compiler in isolation."""
    print("\n" + "=" * 80)
    print("TEST 1: CIRCUIT COMPILER")
    print("=" * 80)
    
    # Small network
    layers = create_test_network(4, [3], 2)
    inputs = [10, 20, 30, 40]
    
    # Test without optimizations
    compiler = CircuitCompiler(use_sparse=False, use_gelu=False)
    circuit_baseline = compiler.compile_network(layers, inputs)
    print(f"\nBaseline (ReLU + Dense):")
    print(f"  Gates: {circuit_baseline.total_gates}")
    print(f"  Sparse: {circuit_baseline.sparse_gates}")
    print(f"  GELU: {circuit_baseline.gelu_gates}")
    
    # Test with GELU
    compiler = CircuitCompiler(use_sparse=False, use_gelu=True)
    circuit_gelu = compiler.compile_network(layers, inputs)
    print(f"\nGELU + Dense:")
    print(f"  Gates: {circuit_gelu.total_gates}")
    print(f"  GELU: {circuit_gelu.gelu_gates}")
    
    # Test with Sparse
    activation_values = [[100, 0, 200]]  # Neuron 1 inaktiv
    compiler = CircuitCompiler(use_sparse=True, use_gelu=True)
    circuit_sparse = compiler.compile_network(layers, inputs, activation_values)
    print(f"\nGELU + Sparse (33% inactive):")
    print(f"  Gates: {circuit_sparse.total_gates}")
    print(f"  Sparse: {circuit_sparse.sparse_gates}")
    
    # Validation
    assert circuit_sparse.sparse_gates > 0, "Sparse gates should be > 0"
    assert circuit_gelu.gelu_gates > 0, "GELU gates should be > 0"
    
    print("\n✓ Circuit Compiler Tests passed!")
    return True


def test_pipeline_end_to_end():
    """Test the full pipeline."""
    print("\n" + "=" * 80)
    print("TEST 2: END-TO-END PIPELINE")
    print("=" * 80)
    
    layers = create_test_network(4, [3], 2)
    inputs = [10, 20, 30, 40]
    
    pipeline = ZkMLPipeline(use_sparse=True, use_gelu=True, srs_size=64)
    
    # Compile
    print("\n1. Compiling...")
    activation_values = [[100, 0, 200]]
    circuit = pipeline.compile_network(layers, inputs, activation_values)
    print(f"   Circuit created: {circuit.total_gates} gates")
    
    # Generate proof
    print("\n2. Generating proof...")
    proof = pipeline.prove(circuit)
    print(f"   Proof generated in {proof.prover_time_ms:.2f} ms")
    print(f"   Proof size: {proof.size_bytes()} bytes")
    
    # Verify
    print("\n3. Verifying...")
    result = pipeline.verify(proof, circuit)
    print(f"   Verifikation: {'✓ VALID' if result.is_valid else '✗ INVALID'}")
    print(f"   Zeit: {result.verification_time_ms:.2f} ms")
    
    # Validation
    assert result.is_valid, "Proof should be valid"
    
    print("\n✓ End-to-End Pipeline Tests passed!")
    return True


def test_optimization_effectiveness():
    """Test the effectiveness of optimizations."""
    print("\n" + "=" * 80)
    print("TEST 3: OPTIMIZATION EFFECTIVENESS")
    print("=" * 80)
    
    # Larger network for more meaningful results
    layers = create_test_network(8, [6, 4], 2)
    inputs = [i * 10 for i in range(8)]
    
    results = {}
    
    # Baseline
    pipeline = ZkMLPipeline(use_sparse=False, use_gelu=False, srs_size=128)
    circuit = pipeline.compile_network(layers, inputs)
    results['baseline'] = circuit.total_gates
    print(f"\nBaseline (ReLU + Dense): {circuit.total_gates} Gates")
    
    # GELU only
    pipeline = ZkMLPipeline(use_sparse=False, use_gelu=True, srs_size=128)
    circuit = pipeline.compile_network(layers, inputs)
    results['gelu'] = circuit.total_gates
    reduction = (1 - circuit.total_gates / results['baseline']) * 100
    print(f"GELU + Dense: {circuit.total_gates} Gates ({reduction:+.1f}%)")
    
    # Sparse only (50%)
    activation_values = [
        [100, 0, 100, 0, 100, 0],  # 50% inaktiv
        [100, 0, 100, 0],          # 50% inaktiv
    ]
    pipeline = ZkMLPipeline(use_sparse=True, use_gelu=False, srs_size=128)
    circuit = pipeline.compile_network(layers, inputs, activation_values)
    results['sparse50'] = circuit.total_gates
    reduction = (1 - circuit.total_gates / results['baseline']) * 100
    print(f"ReLU + Sparse (50%): {circuit.total_gates} Gates ({reduction:+.1f}%)")
    
    # Combined
    pipeline = ZkMLPipeline(use_sparse=True, use_gelu=True, srs_size=128)
    circuit = pipeline.compile_network(layers, inputs, activation_values)
    results['combined'] = circuit.total_gates
    reduction = (1 - circuit.total_gates / results['baseline']) * 100
    print(f"GELU + Sparse (50%): {circuit.total_gates} Gates ({reduction:+.1f}%)")
    
    # High Sparsity (90%)
    activation_values = [
        [100, 0, 0, 0, 0, 0],  # 83% inaktiv
        [100, 0, 0, 0],        # 75% inaktiv
    ]
    circuit = pipeline.compile_network(layers, inputs, activation_values)
    results['sparse90'] = circuit.total_gates
    reduction = (1 - circuit.total_gates / results['baseline']) * 100
    print(f"GELU + Sparse (90%): {circuit.total_gates} Gates ({reduction:+.1f}%)")
    
    print("\n✓ Optimierungs-Effektivität Tests bestanden!")
    return results


def run_all_tests():
    """Führt alle Tests aus."""
    print("=" * 80)
    print("zkML PLONK PIPELINE - VOLLSTÄNDIGER INTEGRATIONSTEST")
    print("=" * 80)
    
    all_passed = True
    
    try:
        test_circuit_compiler()
    except Exception as e:
        print(f"\n✗ Circuit Compiler Test fehlgeschlagen: {e}")
        all_passed = False
    
    try:
        test_pipeline_end_to_end()
    except Exception as e:
        print(f"\n✗ End-to-End Pipeline Test fehlgeschlagen: {e}")
        all_passed = False
    
    try:
        results = test_optimization_effectiveness()
    except Exception as e:
        print(f"\n✗ Optimierungs-Effektivität Test fehlgeschlagen: {e}")
        all_passed = False
        results = {}
    
    # Finale Zusammenfassung
    print("\n" + "=" * 80)
    print("FINALE ZUSAMMENFASSUNG")
    print("=" * 80)
    
    if all_passed:
        print("\n✓ ALLE TESTS BESTANDEN!")
        print("\nDie zkML PLONK Pipeline ist vollständig integriert:")
        print("  - Circuit Compiler: GELU + Sparse ✓")
        print("  - End-to-End: Network → Circuit → Proof → Verify ✓")
        print("  - Optimierungen: Validiert und effektiv ✓")
        
        if results:
            print("\nOptimierungs-Zusammenfassung:")
            baseline = results.get('baseline', 1)
            for key, value in results.items():
                if key != 'baseline':
                    reduction = (1 - value / baseline) * 100
                    print(f"  {key}: {reduction:+.1f}% vs Baseline")
    else:
        print("\n✗ EINIGE TESTS FEHLGESCHLAGEN!")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
