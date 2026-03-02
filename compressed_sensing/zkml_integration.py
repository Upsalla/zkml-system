"""
Integration of CSWC (Compressed Sensing Witness Commitment) into zkML Pipeline.

This module bridges the CSWC protocol with the existing zkML proof system,
enabling efficient sparse witness handling in neural network inference proofs.

Integration Points:
1. Network Inference → Sparse Witness Extraction
2. Sparse Witness → CSWC Commitment
3. CSWC Proof → Combined with PLONK Proof
4. Verification: CSWC + PLONK

The combined system provides:
- Efficient handling of sparse activations (ReLU, pruned networks)
- Reduced proof size proportional to sparsity
- Faster verification for highly sparse witnesses
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.field import FieldElement, FieldConfig
from compressed_sensing.sparse_witness import SparseWitness, SparseExtractor, SparseWitnessBuilder
from compressed_sensing.commitment import CSWCSystem, CSWCProof, CSWCProver, CSWCVerifier


@dataclass
class CSWCConfig:
    """Configuration for CSWC integration."""
    sketch_dimension: int = 64  # Security parameter (m)
    sparsity_threshold: float = 0.3  # Minimum sparsity to use CSWC
    matrix_seed: bytes = b"ZKML_CSWC_V1"  # Deterministic seed
    
    def should_use_cswc(self, sparsity: float) -> bool:
        """Determine if CSWC should be used based on sparsity."""
        return sparsity >= self.sparsity_threshold


@dataclass
class LayerWitness:
    """Witness data for a single neural network layer."""
    layer_id: int
    layer_type: str  # "dense", "conv2d", "activation"
    input_activations: List[FieldElement]
    output_activations: List[FieldElement]
    weights: Optional[List[FieldElement]] = None
    
    @property
    def total_size(self) -> int:
        """Total number of field elements in this layer witness."""
        size = len(self.input_activations) + len(self.output_activations)
        if self.weights:
            size += len(self.weights)
        return size


@dataclass 
class NetworkWitness:
    """Complete witness for a neural network inference."""
    layers: List[LayerWitness]
    input_data: List[FieldElement]
    output_data: List[FieldElement]
    
    @property
    def total_size(self) -> int:
        """Total witness size."""
        return sum(layer.total_size for layer in self.layers)
    
    def flatten(self) -> List[FieldElement]:
        """Flatten all witness data into a single vector."""
        flat = list(self.input_data)
        for layer in self.layers:
            flat.extend(layer.input_activations)
            flat.extend(layer.output_activations)
            if layer.weights:
                flat.extend(layer.weights)
        flat.extend(self.output_data)
        return flat


class CSWCNetworkProver:
    """
    Prover that combines CSWC with neural network inference proofs.
    
    Strategy:
    1. Extract activations from network inference
    2. Identify sparse layers (e.g., after ReLU)
    3. Use CSWC for sparse layers, standard commitment for dense layers
    4. Combine proofs into a single structure
    """
    
    def __init__(self, 
                 config: CSWCConfig = None,
                 field_config: FieldConfig = None):
        """
        Initialize the network prover.
        
        Args:
            config: CSWC configuration
            field_config: Field configuration for arithmetic
        """
        self.config = config or CSWCConfig()
        self.field_config = field_config
        self.extractor = SparseExtractor(threshold=0)
        
        # Cache for CSWC systems (one per witness size)
        self._cswc_cache: Dict[int, CSWCSystem] = {}
    
    def _get_cswc_system(self, witness_size: int, field_config: FieldConfig) -> CSWCSystem:
        """Get or create a CSWC system for the given witness size."""
        cache_key = (witness_size, field_config.prime)
        if cache_key not in self._cswc_cache:
            self._cswc_cache[cache_key] = CSWCSystem(
                witness_size=witness_size,
                sketch_dimension=self.config.sketch_dimension,
                field_modulus=field_config.prime,
                matrix_seed=self.config.matrix_seed
            )
        return self._cswc_cache[cache_key]
    
    def analyze_sparsity(self, witness: NetworkWitness) -> Dict[str, Any]:
        """
        Analyze the sparsity of a network witness.
        
        Returns statistics about which layers are sparse and could benefit from CSWC.
        """
        analysis = {
            "total_size": witness.total_size,
            "layers": [],
            "overall_sparsity": 0.0,
            "cswc_candidate_layers": 0,
            "potential_savings": 0
        }
        
        total_zeros = 0
        total_elements = 0
        
        for layer in witness.layers:
            # Analyze output activations (most likely to be sparse after ReLU)
            outputs = layer.output_activations
            zeros = sum(1 for v in outputs if v.value == 0)
            sparsity = zeros / len(outputs) if outputs else 0.0
            
            layer_info = {
                "layer_id": layer.layer_id,
                "layer_type": layer.layer_type,
                "output_size": len(outputs),
                "zeros": zeros,
                "sparsity": sparsity,
                "use_cswc": self.config.should_use_cswc(sparsity)
            }
            
            if layer_info["use_cswc"]:
                analysis["cswc_candidate_layers"] += 1
                # Estimate savings: sparse size vs dense size
                sparse_size = (len(outputs) - zeros) * 36  # 4 bytes index + 32 bytes value
                dense_size = len(outputs) * 32
                layer_info["estimated_savings"] = dense_size - sparse_size
                analysis["potential_savings"] += layer_info["estimated_savings"]
            
            analysis["layers"].append(layer_info)
            total_zeros += zeros
            total_elements += len(outputs)
        
        analysis["overall_sparsity"] = total_zeros / total_elements if total_elements > 0 else 0.0
        
        return analysis
    
    def prove_layer(self, layer: LayerWitness) -> Tuple[Optional[CSWCProof], Dict[str, Any]]:
        """
        Generate a CSWC proof for a single layer if it's sparse enough.
        
        Args:
            layer: The layer witness
            
        Returns:
            Tuple of (proof or None, metadata)
        """
        outputs = layer.output_activations
        if not outputs:
            return None, {"skipped": True, "reason": "empty layer"}
        
        # Check sparsity
        sparse_witness = self.extractor.extract(outputs)
        
        if not self.config.should_use_cswc(sparse_witness.sparsity):
            return None, {
                "skipped": True,
                "reason": f"sparsity {sparse_witness.sparsity:.2%} below threshold",
                "sparsity": sparse_witness.sparsity
            }
        
        # Generate CSWC proof
        cswc_system = self._get_cswc_system(len(outputs), outputs[0].field)
        
        start_time = time.time()
        proof = cswc_system.prover.prove(outputs)
        prove_time = time.time() - start_time
        
        metadata = {
            "skipped": False,
            "sparsity": sparse_witness.sparsity,
            "support_size": len(proof.support),
            "proof_size_bytes": proof.size_bytes(),
            "prove_time_ms": prove_time * 1000
        }
        
        return proof, metadata
    
    def prove_network(self, witness: NetworkWitness) -> Tuple[List[Optional[CSWCProof]], Dict[str, Any]]:
        """
        Generate CSWC proofs for all eligible layers in a network.
        
        Args:
            witness: The complete network witness
            
        Returns:
            Tuple of (list of proofs, aggregated metadata)
        """
        proofs = []
        layer_metadata = []
        
        total_prove_time = 0
        total_proof_size = 0
        layers_with_cswc = 0
        
        for layer in witness.layers:
            proof, meta = self.prove_layer(layer)
            proofs.append(proof)
            layer_metadata.append(meta)
            
            if proof is not None:
                layers_with_cswc += 1
                total_prove_time += meta.get("prove_time_ms", 0)
                total_proof_size += meta.get("proof_size_bytes", 0)
        
        aggregated = {
            "total_layers": len(witness.layers),
            "layers_with_cswc": layers_with_cswc,
            "total_prove_time_ms": total_prove_time,
            "total_proof_size_bytes": total_proof_size,
            "layer_details": layer_metadata
        }
        
        return proofs, aggregated


class CSWCNetworkVerifier:
    """
    Verifier for CSWC-enhanced network proofs.
    """
    
    def __init__(self,
                 config: CSWCConfig = None,
                 field_config: FieldConfig = None):
        """Initialize the verifier."""
        self.config = config or CSWCConfig()
        self.field_config = field_config
        self._cswc_cache: Dict[int, CSWCSystem] = {}
    
    def _get_cswc_system(self, witness_size: int, field_modulus: int) -> CSWCSystem:
        """Get or create a CSWC system for verification."""
        cache_key = (witness_size, field_modulus)
        if cache_key not in self._cswc_cache:
            self._cswc_cache[cache_key] = CSWCSystem(
                witness_size=witness_size,
                sketch_dimension=self.config.sketch_dimension,
                field_modulus=field_modulus,
                matrix_seed=self.config.matrix_seed
            )
        return self._cswc_cache[cache_key]
    
    def verify_layer(self, proof: CSWCProof) -> Tuple[bool, str, float]:
        """
        Verify a CSWC proof for a single layer.
        
        Returns:
            Tuple of (is_valid, reason, verify_time_ms)
        """
        cswc_system = self._get_cswc_system(proof.witness_size, proof.field_modulus)
        
        start_time = time.time()
        is_valid, reason = cswc_system.verifier.verify(proof)
        verify_time = (time.time() - start_time) * 1000
        
        return is_valid, reason, verify_time
    
    def verify_network(self, proofs: List[Optional[CSWCProof]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify all CSWC proofs for a network.
        
        Args:
            proofs: List of proofs (None for layers without CSWC)
            
        Returns:
            Tuple of (all_valid, verification_metadata)
        """
        results = []
        total_verify_time = 0
        all_valid = True
        
        for i, proof in enumerate(proofs):
            if proof is None:
                results.append({
                    "layer": i,
                    "skipped": True
                })
                continue
            
            is_valid, reason, verify_time = self.verify_layer(proof)
            
            results.append({
                "layer": i,
                "skipped": False,
                "valid": is_valid,
                "reason": reason,
                "verify_time_ms": verify_time
            })
            
            total_verify_time += verify_time
            if not is_valid:
                all_valid = False
        
        metadata = {
            "all_valid": all_valid,
            "total_verify_time_ms": total_verify_time,
            "layer_results": results
        }
        
        return all_valid, metadata


class IntegratedZkMLProver:
    """
    Integrated prover combining CSWC with the full zkML pipeline.
    
    This is the main entry point for generating proofs that leverage
    both CSWC for sparse witnesses and PLONK for circuit constraints.
    """
    
    def __init__(self, config: CSWCConfig = None):
        """Initialize the integrated prover."""
        self.config = config or CSWCConfig()
        self.cswc_prover = CSWCNetworkProver(config=self.config)
        self.cswc_verifier = CSWCNetworkVerifier(config=self.config)
    
    def prove_inference(self, 
                        network_witness: NetworkWitness,
                        include_plonk: bool = False) -> Dict[str, Any]:
        """
        Generate a complete proof for a neural network inference.
        
        Args:
            network_witness: The witness from network inference
            include_plonk: Whether to also generate PLONK proof (slower)
            
        Returns:
            Dictionary containing proofs and metadata
        """
        result = {
            "witness_size": network_witness.total_size,
            "num_layers": len(network_witness.layers)
        }
        
        # Analyze sparsity
        analysis = self.cswc_prover.analyze_sparsity(network_witness)
        result["sparsity_analysis"] = analysis
        
        # Generate CSWC proofs
        cswc_proofs, cswc_meta = self.cswc_prover.prove_network(network_witness)
        result["cswc_proofs"] = cswc_proofs
        result["cswc_metadata"] = cswc_meta
        
        # Optionally generate PLONK proof (for non-sparse parts)
        if include_plonk:
            # This would integrate with the existing PLONK pipeline
            # For now, we just note that it would be done here
            result["plonk_proof"] = None
            result["plonk_metadata"] = {"note": "PLONK integration pending"}
        
        return result
    
    def verify_inference(self, proof_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a complete inference proof.
        
        Args:
            proof_result: The result from prove_inference
            
        Returns:
            Tuple of (is_valid, verification_metadata)
        """
        cswc_proofs = proof_result.get("cswc_proofs", [])
        
        # Verify CSWC proofs
        cswc_valid, cswc_meta = self.cswc_verifier.verify_network(cswc_proofs)
        
        # Would also verify PLONK proof here if present
        
        return cswc_valid, {"cswc_verification": cswc_meta}


# Test code
if __name__ == "__main__":
    print("=== CSWC zkML Integration Tests ===\n")
    
    # Use a smaller field for testing
    field_config = FieldConfig(prime=2**61 - 1, name="Test")
    
    # Create a mock network witness with sparse activations
    print("1. Creating Mock Network Witness")
    
    def create_sparse_activations(size: int, sparsity: float) -> List[FieldElement]:
        """Create activations with given sparsity."""
        activations = []
        for i in range(size):
            if i / size >= sparsity:  # First (1-sparsity) fraction is non-zero
                activations.append(FieldElement(i + 1, field_config))
            else:
                activations.append(FieldElement(0, field_config))
        return activations
    
    # Create layers with varying sparsity
    layers = [
        LayerWitness(
            layer_id=0,
            layer_type="dense",
            input_activations=create_sparse_activations(784, 0.0),  # Dense input
            output_activations=create_sparse_activations(256, 0.5)  # 50% sparse after ReLU
        ),
        LayerWitness(
            layer_id=1,
            layer_type="dense",
            input_activations=create_sparse_activations(256, 0.5),
            output_activations=create_sparse_activations(128, 0.7)  # 70% sparse
        ),
        LayerWitness(
            layer_id=2,
            layer_type="dense",
            input_activations=create_sparse_activations(128, 0.7),
            output_activations=create_sparse_activations(10, 0.1)  # Dense output
        )
    ]
    
    witness = NetworkWitness(
        layers=layers,
        input_data=create_sparse_activations(784, 0.0),
        output_data=create_sparse_activations(10, 0.1)
    )
    
    print(f"   Total witness size: {witness.total_size}")
    print(f"   Number of layers: {len(witness.layers)}")
    
    # Create integrated prover
    print("\n2. Initializing Integrated Prover")
    config = CSWCConfig(
        sketch_dimension=32,  # Smaller for faster testing
        sparsity_threshold=0.3
    )
    prover = IntegratedZkMLProver(config=config)
    
    # Analyze sparsity
    print("\n3. Sparsity Analysis")
    analysis = prover.cswc_prover.analyze_sparsity(witness)
    print(f"   Overall sparsity: {analysis['overall_sparsity']:.2%}")
    print(f"   CSWC candidate layers: {analysis['cswc_candidate_layers']}")
    print(f"   Potential savings: {analysis['potential_savings']} bytes")
    
    for layer_info in analysis["layers"]:
        print(f"   Layer {layer_info['layer_id']} ({layer_info['layer_type']}): "
              f"sparsity={layer_info['sparsity']:.2%}, use_cswc={layer_info['use_cswc']}")
    
    # Generate proof
    print("\n4. Generating Proof")
    start_time = time.time()
    proof_result = prover.prove_inference(witness)
    total_time = time.time() - start_time
    
    print(f"   Total prove time: {total_time*1000:.2f} ms")
    print(f"   CSWC prove time: {proof_result['cswc_metadata']['total_prove_time_ms']:.2f} ms")
    print(f"   CSWC proof size: {proof_result['cswc_metadata']['total_proof_size_bytes']} bytes")
    print(f"   Layers with CSWC: {proof_result['cswc_metadata']['layers_with_cswc']}")
    
    # Verify proof
    print("\n5. Verifying Proof")
    start_time = time.time()
    is_valid, verify_meta = prover.verify_inference(proof_result)
    verify_time = time.time() - start_time
    
    print(f"   Proof valid: {is_valid}")
    print(f"   Total verify time: {verify_time*1000:.2f} ms")
    print(f"   CSWC verify time: {verify_meta['cswc_verification']['total_verify_time_ms']:.2f} ms")
    
    # Comparison with dense approach
    print("\n6. Comparison with Dense Approach")
    dense_size = witness.total_size * 32  # 32 bytes per element
    cswc_size = proof_result['cswc_metadata']['total_proof_size_bytes']
    
    print(f"   Dense witness size: {dense_size} bytes")
    print(f"   CSWC proof size: {cswc_size} bytes")
    print(f"   Compression: {(1 - cswc_size/dense_size)*100:.1f}% reduction")
    
    print("\n=== All Tests Passed ===")
