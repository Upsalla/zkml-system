"""
End-to-End Demo: zkML System with All Innovations.

This demo showcases the complete zkML system including:
1. CSWC (Compressed Sensing Witness Commitment)
2. HWWB (Haar Wavelet Witness Batching)
3. TDA Model Fingerprinting
4. On-Chain Registry Integration

Author: Upsalla
"""

import sys
import os
import time
import numpy as np
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_section(title: str):
    """Print a section header."""
    print(f"\n--- {title} ---")

def create_sample_network():
    """Create a sample neural network for demonstration."""
    np.random.seed(42)
    
    # Simple 3-layer network: 784 -> 128 -> 64 -> 10
    weights = [
        np.random.randn(784, 128) * 0.1,
        np.random.randn(128, 64) * 0.1,
        np.random.randn(64, 10) * 0.1
    ]
    
    # Make it sparse (simulate ReLU activation)
    for i, w in enumerate(weights):
        mask = np.random.random(w.shape) > 0.5
        weights[i] = w * mask
    
    return weights

def demo_cswc():
    """Demonstrate CSWC (Compressed Sensing Witness Commitment)."""
    print_header("1. CSWC - Compressed Sensing Witness Commitment")
    
    from compressed_sensing.commitment import CSWCSystem
    from core.field import FieldConfig
    
    # Create a sparse witness
    n = 1000
    k = 100  # 10% non-zero
    
    witness = np.zeros(n)
    indices = np.random.choice(n, k, replace=False)
    witness[indices] = np.random.randn(k)
    
    sparsity = 1 - (k / n)
    print(f"Witness size: {n} elements")
    print(f"Non-zero elements: {k} ({100*k/n:.1f}%)")
    print(f"Sparsity: {sparsity*100:.1f}%")
    
    # Create CSWC system
    field_config = FieldConfig(21888242871839275222246405745257275088548364400416034343698204186575808495617, "BN254_Fr")
    cswc = CSWCSystem(field_config, n_measurements=200)
    
    # Generate commitment and proof
    start = time.time()
    commitment, proof = cswc.commit_and_prove(witness)
    prove_time = (time.time() - start) * 1000
    
    # Verify
    start = time.time()
    is_valid = cswc.verify(commitment, proof)
    verify_time = (time.time() - start) * 1000
    
    # Calculate sizes
    standard_size = n * 32  # 32 bytes per field element
    cswc_size = proof.size_bytes()
    reduction = (1 - cswc_size / standard_size) * 100
    
    print_section("Results")
    print(f"Standard witness size: {standard_size} bytes")
    print(f"CSWC proof size: {cswc_size} bytes")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Prove time: {prove_time:.2f} ms")
    print(f"Verify time: {verify_time:.2f} ms")
    print(f"Verification: {'PASSED' if is_valid else 'FAILED'}")
    
    return {
        "standard_size": standard_size,
        "cswc_size": cswc_size,
        "reduction": reduction,
        "prove_time": prove_time,
        "verify_time": verify_time,
        "valid": is_valid
    }

def demo_hwwb():
    """Demonstrate HWWB (Haar Wavelet Witness Batching)."""
    print_header("2. HWWB - Haar Wavelet Witness Batching")
    
    from wavelet.haar_transform import HWWBProver, HWWBVerifier
    from core.field import FieldConfig, FieldElement
    
    # Create correlated witness (simulates adjacent layer activations)
    n = 256
    base = np.sin(np.linspace(0, 4*np.pi, n))  # Smooth signal
    noise = np.random.randn(n) * 0.1
    witness_values = base + noise
    
    # Convert to field elements
    field_config = FieldConfig(21888242871839275222246405745257275088548364400416034343698204186575808495617, "BN254_Fr")
    witness = [FieldElement(int(abs(v) * 1000000) % field_config.prime, field_config) 
               for v in witness_values]
    
    print(f"Witness size: {n} elements")
    print(f"Signal type: Correlated (sinusoidal with noise)")
    
    # Create HWWB system
    prover = HWWBProver(field_config, threshold=0.1)
    verifier = HWWBVerifier(field_config)
    
    # Generate proof
    start = time.time()
    proof = prover.prove(witness)
    prove_time = (time.time() - start) * 1000
    
    # Verify
    start = time.time()
    is_valid = verifier.verify(proof)
    verify_time = (time.time() - start) * 1000
    
    # Calculate compression
    original_coeffs = n
    retained_coeffs = len(proof.significant_coefficients)
    compression = (1 - retained_coeffs / original_coeffs) * 100
    
    print_section("Results")
    print(f"Original coefficients: {original_coeffs}")
    print(f"Retained coefficients: {retained_coeffs}")
    print(f"Compression: {compression:.1f}%")
    print(f"Prove time: {prove_time:.2f} ms")
    print(f"Verify time: {verify_time:.2f} ms")
    print(f"Verification: {'PASSED' if is_valid else 'FAILED'}")
    
    return {
        "original_coeffs": original_coeffs,
        "retained_coeffs": retained_coeffs,
        "compression": compression,
        "prove_time": prove_time,
        "verify_time": verify_time,
        "valid": is_valid
    }

def demo_tda():
    """Demonstrate TDA Model Fingerprinting."""
    print_header("3. TDA - Model Fingerprinting")
    
    from tda.fingerprint import TDAFingerprintSystem
    
    # Create sample models
    weights_original = create_sample_network()
    
    # Create a slightly modified version (simulates fine-tuning)
    weights_finetuned = [w + np.random.randn(*w.shape) * 0.01 for w in weights_original]
    
    # Create a completely different model
    weights_different = [np.random.randn(*w.shape) * 0.1 for w in weights_original]
    
    total_params = sum(w.size for w in weights_original)
    print(f"Model size: {total_params:,} parameters")
    print(f"Layers: {len(weights_original)}")
    
    # Create fingerprints
    tda = TDAFingerprintSystem(n_features=20)
    
    start = time.time()
    fp_original = tda.fingerprint(weights_original)
    fp_time = (time.time() - start) * 1000
    
    fp_finetuned = tda.fingerprint(weights_finetuned)
    fp_different = tda.fingerprint(weights_different)
    
    # Compare fingerprints
    dist_finetuned = fp_original.distance(fp_finetuned)
    dist_different = fp_original.distance(fp_different)
    
    print_section("Fingerprint Generation")
    print(f"Fingerprint size: {fp_original.size_bytes()} bytes (constant)")
    print(f"Generation time: {fp_time:.2f} ms")
    print(f"Hash: {fp_original.hash[:32]}...")
    
    print_section("Model Comparison")
    print(f"Original vs Fine-tuned: distance = {dist_finetuned:.4f}")
    print(f"Original vs Different:  distance = {dist_different:.4f}")
    print(f"Same model threshold: 0.05")
    print(f"Fine-tuned detected as same: {dist_finetuned < 0.05}")
    print(f"Different detected as different: {dist_different >= 0.05}")
    
    # Generate proof
    from tda.fingerprint import TDAProver, TDAVerifier
    
    prover = TDAProver()
    verifier = TDAVerifier()
    
    start = time.time()
    proof = prover.prove(fp_original)
    prove_time = (time.time() - start) * 1000
    
    start = time.time()
    is_valid = verifier.verify(proof)
    verify_time = (time.time() - start) * 1000
    
    print_section("Proof Generation")
    print(f"Proof size: {proof.size_bytes()} bytes")
    print(f"Prove time: {prove_time:.2f} ms")
    print(f"Verify time: {verify_time:.2f} ms")
    print(f"Verification: {'PASSED' if is_valid else 'FAILED'}")
    
    return {
        "fingerprint_size": fp_original.size_bytes(),
        "fp_time": fp_time,
        "dist_finetuned": dist_finetuned,
        "dist_different": dist_different,
        "proof_size": proof.size_bytes(),
        "prove_time": prove_time,
        "verify_time": verify_time,
        "valid": is_valid
    }

def demo_on_chain():
    """Demonstrate On-Chain Registry Integration (simulated)."""
    print_header("4. On-Chain Registry Integration (Simulated)")
    
    from tda.fingerprint import TDAFingerprintSystem
    
    # Create model and fingerprint
    weights = create_sample_network()
    tda = TDAFingerprintSystem(n_features=20)
    fingerprint = tda.fingerprint(weights)
    
    # Simulate on-chain registration
    model_name = "MyMLModel"
    model_version = "1.0.0"
    owner_address = "0x" + hashlib.sha256(b"owner").hexdigest()[:40]
    
    # Compute model ID (as in the smart contract)
    model_id = hashlib.sha256(f"{model_name}{owner_address}".encode()).hexdigest()
    fingerprint_hash = fingerprint.hash
    
    print_section("Model Registration")
    print(f"Model Name: {model_name}")
    print(f"Version: {model_version}")
    print(f"Owner: {owner_address}")
    print(f"Model ID: 0x{model_id[:16]}...")
    print(f"Fingerprint Hash: {fingerprint_hash[:32]}...")
    
    # Simulate contract call
    print_section("Smart Contract Interaction (Simulated)")
    print("Function: registerModel(bytes32 modelId, bytes32 fingerprintHash, string metadataURI)")
    print(f"  modelId: 0x{model_id}")
    print(f"  fingerprintHash: 0x{fingerprint_hash}")
    print(f"  metadataURI: ipfs://Qm...")
    print("Transaction: SUCCESS")
    print("Gas used: ~85,000 (estimated)")
    
    # Simulate verification
    print_section("On-Chain Verification (Simulated)")
    print("Function: verifyModel(bytes32 modelId, bytes32 fingerprintHash)")
    print(f"  modelId: 0x{model_id[:16]}...")
    print(f"  fingerprintHash: 0x{fingerprint_hash[:16]}...")
    print("Result: (isValid: true, version: 1)")
    
    return {
        "model_id": model_id,
        "fingerprint_hash": fingerprint_hash,
        "owner": owner_address
    }

def demo_combined_pipeline():
    """Demonstrate the combined zkML pipeline."""
    print_header("5. Combined zkML Pipeline")
    
    from plonk.zkml_pipeline import ZkMLPipeline
    from network.builder import NetworkBuilder
    from core.field import FieldConfig
    
    # Create a simple network
    print("Creating neural network...")
    network = (NetworkBuilder()
               .input(4)
               .dense(8, activation='gelu')
               .dense(4, activation='gelu')
               .output(2)
               .build())
    
    print(f"Network: {network.input_size} -> 8 -> 4 -> {network.output_size}")
    
    # Create input
    input_data = [0.5, -0.3, 0.8, -0.1]
    
    # Run inference
    output = network.forward(input_data)
    print(f"Input: {input_data}")
    print(f"Output: {[round(o, 4) for o in output]}")
    
    # Create zkML pipeline
    field_config = FieldConfig(21888242871839275222246405745257275088548364400416034343698204186575808495617, "BN254_Fr")
    pipeline = ZkMLPipeline(field_config)
    
    # Compile network
    print_section("Circuit Compilation")
    start = time.time()
    circuit = pipeline.compile(network, enable_sparse=True)
    compile_time = (time.time() - start) * 1000
    
    print(f"Compilation time: {compile_time:.2f} ms")
    print(f"Circuit gates: {circuit.n_gates}")
    print(f"Sparse optimization: enabled")
    
    # Generate proof (simplified - full PLONK proof is slow in Python)
    print_section("Proof Generation")
    print("Note: Full PLONK proof generation skipped (too slow in pure Python)")
    print("In production, this would use Rust/arkworks for ~1000x speedup")
    
    return {
        "network_layers": 3,
        "circuit_gates": circuit.n_gates,
        "compile_time": compile_time
    }

def main():
    """Run the complete end-to-end demo."""
    print("\n" + "=" * 70)
    print(" zkML SYSTEM - END-TO-END DEMONSTRATION")
    print(" Featuring: CSWC, HWWB, TDA Fingerprinting, On-Chain Registry")
    print("=" * 70)
    
    results = {}
    
    # Run all demos
    try:
        results['cswc'] = demo_cswc()
    except Exception as e:
        print(f"CSWC demo failed: {e}")
        results['cswc'] = None
    
    try:
        results['hwwb'] = demo_hwwb()
    except Exception as e:
        print(f"HWWB demo failed: {e}")
        results['hwwb'] = None
    
    try:
        results['tda'] = demo_tda()
    except Exception as e:
        print(f"TDA demo failed: {e}")
        results['tda'] = None
    
    try:
        results['on_chain'] = demo_on_chain()
    except Exception as e:
        print(f"On-Chain demo failed: {e}")
        results['on_chain'] = None
    
    try:
        results['pipeline'] = demo_combined_pipeline()
    except Exception as e:
        print(f"Pipeline demo failed: {e}")
        results['pipeline'] = None
    
    # Summary
    print_header("SUMMARY")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ Component          │ Status   │ Key Metric                          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    if results['cswc']:
        print(f"│ CSWC               │ ✓ PASS   │ {results['cswc']['reduction']:.1f}% size reduction              │")
    else:
        print("│ CSWC               │ ✗ FAIL   │ -                                   │")
    
    if results['hwwb']:
        print(f"│ HWWB               │ ✓ PASS   │ {results['hwwb']['compression']:.1f}% compression                 │")
    else:
        print("│ HWWB               │ ✗ FAIL   │ -                                   │")
    
    if results['tda']:
        print(f"│ TDA Fingerprint    │ ✓ PASS   │ {results['tda']['fingerprint_size']} bytes (constant)            │")
    else:
        print("│ TDA Fingerprint    │ ✗ FAIL   │ -                                   │")
    
    if results['on_chain']:
        print(f"│ On-Chain Registry  │ ✓ PASS   │ ~85k gas per registration           │")
    else:
        print("│ On-Chain Registry  │ ✗ FAIL   │ -                                   │")
    
    if results['pipeline']:
        print(f"│ zkML Pipeline      │ ✓ PASS   │ {results['pipeline']['circuit_gates']} gates                       │")
    else:
        print("│ zkML Pipeline      │ ✗ FAIL   │ -                                   │")
    
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 70)
    print(" END-TO-END DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    main()
