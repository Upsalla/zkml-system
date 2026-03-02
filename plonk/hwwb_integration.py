"""
HWWB Integration into zkML Pipeline.

This module integrates Haar-Wavelet Witness Batching as an optional
optimization component in the zkML proof system.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavelet.haar_transform import HWWBSystem, HWWBProof, HWWBProver
from core.field import FieldConfig


@dataclass
class OptimizationConfig:
    """Configuration for zkML optimizations."""
    use_sparse: bool = True           # Use CSWC for sparse witnesses
    use_wavelet: bool = False         # Use HWWB for correlated witnesses
    sparse_threshold: float = 0.3     # Minimum sparsity to use CSWC
    correlation_threshold: float = 0.5 # Minimum correlation to use HWWB
    auto_select: bool = True          # Automatically select best optimization


class OptimizationSelector:
    """
    Automatically selects the best optimization strategy based on witness characteristics.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
    
    def analyze_witness(self, witness: List[float]) -> Dict[str, float]:
        """
        Analyze witness characteristics.
        
        Returns:
            Dictionary with sparsity, correlation, and recommended optimization.
        """
        arr = np.array(witness)
        
        # Calculate sparsity (fraction of zeros or near-zeros)
        threshold = np.max(np.abs(arr)) * 0.01 if len(arr) > 0 else 0
        sparsity = np.mean(np.abs(arr) < threshold)
        
        # Calculate correlation (using autocorrelation)
        if len(arr) > 1:
            mean = np.mean(arr)
            var = np.var(arr)
            if var > 0:
                autocorr = np.correlate(arr - mean, arr - mean, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
                correlation = np.mean(np.abs(autocorr[1:min(10, len(autocorr))]))
            else:
                correlation = 0
        else:
            correlation = 0
        
        # Recommend optimization
        if self.config.auto_select:
            if sparsity >= self.config.sparse_threshold:
                recommendation = "sparse"
            elif correlation >= self.config.correlation_threshold:
                recommendation = "wavelet"
            else:
                recommendation = "none"
        else:
            if self.config.use_sparse and sparsity >= self.config.sparse_threshold:
                recommendation = "sparse"
            elif self.config.use_wavelet and correlation >= self.config.correlation_threshold:
                recommendation = "wavelet"
            else:
                recommendation = "none"
        
        return {
            "sparsity": sparsity,
            "correlation": correlation,
            "recommendation": recommendation
        }
    
    def select(self, witness: List[float]) -> str:
        """Select optimization strategy for a witness."""
        analysis = self.analyze_witness(witness)
        return analysis["recommendation"]


class HWWBOptimizedProver:
    """
    Prover that can optionally use HWWB for correlated witnesses.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.selector = OptimizationSelector(self.config)
        # Use a small prime for testing
        self.field_config = FieldConfig(prime=101, name="test")
        self.prover_hwwb = HWWBProver(self.field_config)
    
    def prove_with_optimization(self, witness: List[float], 
                                 force_optimization: str = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate proof with automatic or forced optimization.
        
        Args:
            witness: The witness values
            force_optimization: Force specific optimization ("sparse", "wavelet", "none")
            
        Returns:
            Tuple of (proof, metadata)
        """
        # Analyze and select optimization
        analysis = self.selector.analyze_witness(witness)
        optimization = force_optimization or analysis["recommendation"]
        
        metadata = {
            "optimization": optimization,
            "sparsity": analysis["sparsity"],
            "correlation": analysis["correlation"],
            "witness_size": len(witness)
        }
        
        if optimization == "wavelet":
            # Use HWWB
            proof = self._prove_with_hwwb(witness)
            metadata["hwwb_large_diffs"] = len(proof.large_diffs)
            metadata["hwwb_small_diffs"] = len(proof.small_diff_indices)
            metadata["hwwb_compression"] = 1 - (proof.size_bytes() / (len(witness) * 8))
        elif optimization == "sparse":
            # Use standard sparse proof (placeholder - would integrate with CSWC)
            proof = self._prove_standard(witness)
            metadata["note"] = "Using CSWC (sparse optimization)"
        else:
            # No optimization
            proof = self._prove_standard(witness)
            metadata["note"] = "No optimization applied"
        
        return proof, metadata
    
    def _prove_with_hwwb(self, witness: List[float]) -> HWWBProof:
        """Generate HWWB proof."""
        from core.field import FieldElement
        # Convert floats to FieldElements (quantize to integers)
        field_witness = [
            FieldElement(int(abs(w) * 1000) % self.field_config.prime, self.field_config)
            for w in witness
        ]
        return self.prover_hwwb.prove(field_witness)
    
    def _prove_standard(self, witness: List[float]) -> Dict[str, Any]:
        """Generate standard proof (placeholder)."""
        return {
            "type": "standard",
            "witness_commitment": hash(tuple(witness)),
            "size_bytes": len(witness) * 8
        }


class IntegratedZkMLPipeline:
    """
    Integrated zkML pipeline with optional HWWB optimization.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.prover = HWWBOptimizedProver(self.config)
    
    def compile_and_prove(self, network_weights: List[np.ndarray], 
                          input_data: np.ndarray) -> Dict[str, Any]:
        """
        Compile network and generate proof with optimizations.
        
        Args:
            network_weights: List of weight matrices
            input_data: Input to the network
            
        Returns:
            Dictionary with proof and metadata
        """
        # Simulate forward pass to get witness
        witness = self._compute_witness(network_weights, input_data)
        
        # Generate optimized proof
        proof, metadata = self.prover.prove_with_optimization(witness)
        
        return {
            "proof": proof,
            "metadata": metadata,
            "network_stats": {
                "n_layers": len(network_weights),
                "total_params": sum(w.size for w in network_weights)
            }
        }
    
    def _compute_witness(self, weights: List[np.ndarray], 
                         input_data: np.ndarray) -> List[float]:
        """Compute witness from network execution."""
        # Simple forward pass
        x = input_data.flatten()
        witness = list(x)
        
        for W in weights:
            if len(W.shape) == 1:
                W = W.reshape(-1, 1)
            
            # Pad input if needed
            if len(x) < W.shape[1]:
                x = np.pad(x, (0, W.shape[1] - len(x)))
            elif len(x) > W.shape[1]:
                x = x[:W.shape[1]]
            
            x = W @ x
            x = np.maximum(x, 0)  # ReLU
            witness.extend(x.tolist())
        
        return witness


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("HWWB Integration Tests")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test 1: Optimization selector
    print("\n1. Testing optimization selector")
    selector = OptimizationSelector()
    
    # Sparse witness
    sparse_witness = [0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0]
    analysis = selector.analyze_witness(sparse_witness)
    print(f"   Sparse witness: sparsity={analysis['sparsity']:.2f}, "
          f"correlation={analysis['correlation']:.2f}, "
          f"recommendation={analysis['recommendation']}")
    
    # Correlated witness
    correlated_witness = [1, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9]
    analysis = selector.analyze_witness(correlated_witness)
    print(f"   Correlated witness: sparsity={analysis['sparsity']:.2f}, "
          f"correlation={analysis['correlation']:.2f}, "
          f"recommendation={analysis['recommendation']}")
    
    # Random witness
    random_witness = list(np.random.randn(16))
    analysis = selector.analyze_witness(random_witness)
    print(f"   Random witness: sparsity={analysis['sparsity']:.2f}, "
          f"correlation={analysis['correlation']:.2f}, "
          f"recommendation={analysis['recommendation']}")
    
    # Test 2: HWWB optimized prover
    print("\n2. Testing HWWB optimized prover")
    prover = HWWBOptimizedProver()
    
    proof, metadata = prover.prove_with_optimization(correlated_witness)
    print(f"   Optimization used: {metadata['optimization']}")
    print(f"   Sparsity: {metadata['sparsity']:.2f}")
    print(f"   Correlation: {metadata['correlation']:.2f}")
    if 'hwwb_compression' in metadata:
        print(f"   HWWB compression: {metadata['hwwb_compression']:.1%}")
    
    # Test 3: Integrated pipeline
    print("\n3. Testing integrated pipeline")
    pipeline = IntegratedZkMLPipeline()
    
    weights = [
        np.random.randn(8, 4),
        np.random.randn(4, 8),
        np.random.randn(2, 4)
    ]
    input_data = np.random.randn(4)
    
    result = pipeline.compile_and_prove(weights, input_data)
    print(f"   Network: {result['network_stats']['n_layers']} layers, "
          f"{result['network_stats']['total_params']} params")
    print(f"   Optimization: {result['metadata']['optimization']}")
    print(f"   Witness size: {result['metadata']['witness_size']}")
    
    # Test 4: Force HWWB
    print("\n4. Testing forced HWWB optimization")
    proof, metadata = prover.prove_with_optimization(random_witness, force_optimization="wavelet")
    print(f"   Forced optimization: {metadata['optimization']}")
    if 'hwwb_compression' in metadata:
        print(f"   HWWB compression: {metadata['hwwb_compression']:.1%}")
    
    print("\n" + "=" * 60)
    print("HWWB Integration Complete!")
    print("=" * 60)
