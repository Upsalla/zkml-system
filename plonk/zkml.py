"""
zkML Unified API

This module provides the main entry point for the zkML system,
offering a clean pipeline from neural network to zero-knowledge proof:

    Network → Circuit → Witness → Proof → Verification

Example usage:
    from zkml_system.plonk.zkml import ZkML, NetworkConfig
    
    # Configure network
    config = NetworkConfig(
        layers=[
            ('dense', {'input_size': 784, 'output_size': 128}),
            ('relu', {}),
            ('dense', {'input_size': 128, 'output_size': 10}),
            ('argmax', {})
        ]
    )
    
    # Create zkML instance
    zkml = ZkML(config)
    
    # Generate proof
    input_data = [...]  # 784 values
    proof = zkml.prove(input_data)
    
    # Verify
    is_valid = zkml.verify(proof)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import time
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.warn(
    "plonk.zkml provides a simplified demo API. For production-grade PLONK "
    "proofs use PLONKProver/PLONKVerifier from plonk.plonk_prover.",
    DeprecationWarning,
    stacklevel=2,
)

from plonk.core import (
    Field, Polynomial, FFT, SRS, KZG, KZGCommitment, KZGProof,
    Circuit, Witness, Gate, GateType, Wire, Fr, G1Point
)
from plonk.optimizations import (
    PLONKOptimizer, OptimizationConfig, OptimizationStats,
    SparseWitness, TropicalOptimizer
)


# =============================================================================
# Layer Types
# =============================================================================

class LayerType(Enum):
    """Supported neural network layer types."""
    DENSE = "dense"
    CONV2D = "conv2d"
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    ARGMAX = "argmax"
    SOFTMAX = "softmax"
    FLATTEN = "flatten"
    BATCH_NORM = "batch_norm"


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    layer_type: LayerType
    params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def input_size(self) -> int:
        return self.params.get('input_size', 0)
    
    @property
    def output_size(self) -> int:
        return self.params.get('output_size', self.input_size)


# =============================================================================
# Network Configuration
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuration for a neural network."""
    layers: List[Tuple[str, Dict[str, Any]]]
    name: str = "zkml_network"
    
    def get_layer_configs(self) -> List[LayerConfig]:
        """Convert to LayerConfig objects."""
        configs = []
        for layer_type, params in self.layers:
            configs.append(LayerConfig(
                layer_type=LayerType(layer_type),
                params=params
            ))
        return configs
    
    @property
    def input_size(self) -> int:
        """Get network input size."""
        if self.layers:
            return self.layers[0][1].get('input_size', 0)
        return 0
    
    @property
    def output_size(self) -> int:
        """Get network output size."""
        if self.layers:
            last_params = self.layers[-1][1]
            return last_params.get('output_size', last_params.get('input_size', 0))
        return 0


# =============================================================================
# Circuit Compiler
# =============================================================================

class CircuitCompiler:
    """
    Compiles neural network layers to PLONK circuits.
    
    Applies tropical optimizations for max operations.
    """
    
    def __init__(self, optimizer: Optional[PLONKOptimizer] = None):
        self.optimizer = optimizer or PLONKOptimizer()
        self.tropical = TropicalOptimizer()
    
    def compile_dense(self, circuit: Circuit, 
                      inputs: List[Wire],
                      weights: List[List[Fr]],
                      biases: List[Fr]) -> List[Wire]:
        """Compile a dense layer."""
        outputs = []
        output_size = len(biases)
        
        for i in range(output_size):
            # Compute weighted sum: sum(w[i][j] * x[j]) + b[i]
            acc = circuit.add_constant(biases[i], f"bias_{i}")
            
            for j, inp in enumerate(inputs):
                if j < len(weights[i]):
                    w = circuit.add_constant(weights[i][j], f"w_{i}_{j}")
                    prod = circuit.mul(w, inp, f"prod_{i}_{j}")
                    acc = circuit.add(acc, prod, f"acc_{i}_{j}")
            
            outputs.append(acc)
        
        return outputs
    
    def compile_relu(self, circuit: Circuit, 
                     inputs: List[Wire]) -> List[Wire]:
        """
        Compile ReLU activation.
        
        Uses tropical optimization: max(0, x) with 2 constraints per element.
        """
        outputs = []
        zero = circuit.add_constant(Fr.zero(), "zero")
        
        for i, inp in enumerate(inputs):
            # ReLU: max(0, x)
            # Tropical: 2 constraints instead of bit_width
            out = circuit.new_wire(f"relu_{i}")
            # Add constraint: out = max(0, inp)
            circuit.gates.append(Gate(
                gate_type=GateType.CUSTOM,
                left=zero,
                right=inp,
                output=out,
                selector={'type': 'tropical_max'}
            ))
            outputs.append(out)
        
        return outputs
    
    def compile_max_pool(self, circuit: Circuit,
                         inputs: List[Wire],
                         pool_size: int) -> List[Wire]:
        """
        Compile max pooling layer.
        
        Uses tropical optimization: 90%+ constraint reduction.
        """
        outputs = []
        num_pools = len(inputs) // pool_size
        
        for p in range(num_pools):
            pool_inputs = inputs[p * pool_size:(p + 1) * pool_size]
            
            # Tropical max: O(n) constraints
            current = pool_inputs[0]
            for i, inp in enumerate(pool_inputs[1:], 1):
                out = circuit.new_wire(f"pool_{p}_max_{i}")
                circuit.gates.append(Gate(
                    gate_type=GateType.CUSTOM,
                    left=current,
                    right=inp,
                    output=out,
                    selector={'type': 'tropical_max'}
                ))
                current = out
            
            outputs.append(current)
        
        return outputs
    
    def compile_argmax(self, circuit: Circuit,
                       inputs: List[Wire]) -> Wire:
        """
        Compile argmax operation.
        
        Uses tropical optimization for efficient comparison.
        """
        # Track both max value and index
        max_val = inputs[0]
        max_idx = circuit.add_constant(Fr.zero(), "idx_0")
        
        for i, inp in enumerate(inputs[1:], 1):
            # Compare and update
            new_max = circuit.new_wire(f"argmax_val_{i}")
            new_idx = circuit.new_wire(f"argmax_idx_{i}")
            
            circuit.gates.append(Gate(
                gate_type=GateType.CUSTOM,
                left=max_val,
                right=inp,
                output=new_max,
                selector={'type': 'tropical_argmax', 'index': i}
            ))
            
            max_val = new_max
            max_idx = new_idx
        
        return max_idx
    
    def compile_network(self, config: NetworkConfig,
                        weights: Optional[Dict[str, Any]] = None) -> Circuit:
        """
        Compile a complete network to a circuit.
        
        Args:
            config: Network configuration
            weights: Optional pre-trained weights
        
        Returns:
            Compiled Circuit
        """
        circuit = Circuit()
        weights = weights or {}
        
        # Create input wires
        input_size = config.input_size
        current_wires = [circuit.add_public_input(f"input_{i}") 
                        for i in range(input_size)]
        
        # Compile each layer
        for layer_idx, (layer_type, params) in enumerate(config.layers):
            lt = LayerType(layer_type)
            
            if lt == LayerType.DENSE:
                out_size = params.get('output_size', len(current_wires))
                in_size = len(current_wires)
                
                # Get or generate weights
                layer_weights = weights.get(f"layer_{layer_idx}_weights", 
                    [[Fr.one() for _ in range(in_size)] for _ in range(out_size)])
                layer_biases = weights.get(f"layer_{layer_idx}_biases",
                    [Fr.zero() for _ in range(out_size)])
                
                current_wires = self.compile_dense(
                    circuit, current_wires, layer_weights, layer_biases
                )
            
            elif lt == LayerType.RELU:
                current_wires = self.compile_relu(circuit, current_wires)
            
            elif lt == LayerType.MAX_POOL:
                pool_size = params.get('pool_size', 2)
                current_wires = self.compile_max_pool(
                    circuit, current_wires, pool_size
                )
            
            elif lt == LayerType.ARGMAX:
                result = self.compile_argmax(circuit, current_wires)
                current_wires = [result]
            
            elif lt == LayerType.FLATTEN:
                # Flatten is a no-op in circuit representation
                pass
            
            else:
                raise NotImplementedError(
                    f"Layer type {lt.value!r} not yet supported in circuit compiler. "
                    f"Supported: DENSE, RELU, MAX_POOL, ARGMAX, FLATTEN."
                )
        
        return circuit


# =============================================================================
# Prover
# =============================================================================

@dataclass
class ZkMLProof:
    """A complete zkML proof."""
    # Public inputs
    input_commitment: KZGCommitment
    output_value: Fr
    
    # PLONK proof components
    wire_commitments: List[KZGCommitment]
    quotient_commitment: KZGCommitment
    opening_proof: G1Point
    
    # Evaluation values
    evaluations: Dict[str, Fr]
    
    # Metadata
    network_hash: bytes
    timestamp: float
    optimization_stats: Optional[OptimizationStats] = None
    
    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        data = b""
        data += self.input_commitment.to_bytes()
        data += self.output_value.to_int().to_bytes(32, 'big')
        for wc in self.wire_commitments:
            data += wc.to_bytes()
        data += self.quotient_commitment.to_bytes()
        if not self.opening_proof.is_identity():
            px, py = self.opening_proof.to_affine()
            data += px.to_int().to_bytes(32, 'big')
            data += py.to_int().to_bytes(32, 'big')
        data += self.network_hash
        return data
    
    @property
    def size_bytes(self) -> int:
        """Return proof size in bytes."""
        return len(self.to_bytes())


class ZkMLProver:
    """
    Prover for zkML proofs.
    
    Generates PLONK proofs for neural network inference.
    """
    
    def __init__(self, srs: SRS, optimizer: Optional[PLONKOptimizer] = None):
        self.srs = srs
        self.kzg = KZG(srs)
        self.optimizer = optimizer or PLONKOptimizer()
    
    def prove(self, circuit: Circuit, witness: Witness,
              public_input: List[Fr], network_hash: bytes) -> ZkMLProof:
        """
        Generate a zkML proof.
        
        Raises:
            NotImplementedError: Always. This simplified API does not implement
                the PLONK quotient polynomial. Use PLONKProver from
                plonk_prover.py for production-grade proofs.
        """
        # HONEST BOUNDARY: Quotient polynomial requires the full PLONK
        # protocol (5-round Fiat-Shamir). This simplified API does not
        # implement it. Use PLONKProver from plonk_prover.py for real proofs.
        raise NotImplementedError(
            "ZkMLProver.prove() does not implement the PLONK quotient polynomial. "
            "The full 5-round Fiat-Shamir protocol is available via "
            "plonk.plonk_prover.PLONKProver.prove(). This API cannot generate "
            "cryptographically sound proofs."
        )


# =============================================================================
# Verifier
# =============================================================================

class ZkMLVerifier:
    """
    Verifier for zkML proofs.
    
    Verifies PLONK proofs for neural network inference.
    """
    
    def __init__(self, srs: SRS):
        self.srs = srs
        self.kzg = KZG(srs)
    
    def verify(self, proof: ZkMLProof, 
               expected_network_hash: bytes) -> Tuple[bool, str]:
        """
        Verify a zkML proof.
        
        Args:
            proof: The proof to verify
            expected_network_hash: Expected hash of network configuration
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check network hash
        if proof.network_hash != expected_network_hash:
            return False, "Network hash mismatch"
        
        # STRUCTURAL CHECKS ONLY — no pairing verification.
        # This verifier does NOT check constraint satisfaction or
        # KZG opening proofs. For cryptographic verification, use
        # PLONKVerifier from plonk_prover.py.
        warnings.warn(
            "ZkMLVerifier.verify() performs structural checks only "
            "(curve point validity). It does NOT verify constraint "
            "satisfaction or KZG openings. Use PLONKVerifier for soundness.",
            UserWarning,
            stacklevel=2,
        )
        
        # Check commitments are valid curve points
        for wc in proof.wire_commitments:
            if not wc.point.is_on_curve():
                return False, "Invalid wire commitment"
        
        if not proof.quotient_commitment.point.is_on_curve():
            return False, "Invalid quotient commitment"
        
        return True, "Proof structurally valid (NOT cryptographically verified)"


# =============================================================================
# Main zkML Interface
# =============================================================================

class ZkML:
    """
    Main zkML interface.
    
    Provides a simple API for:
    1. Configuring neural networks
    2. Compiling to circuits
    3. Generating proofs
    4. Verifying proofs
    
    Example:
        zkml = ZkML(network_config)
        proof = zkml.prove(input_data)
        is_valid = zkml.verify(proof)
    """
    
    def __init__(self, config: NetworkConfig,
                 optimization_config: Optional[OptimizationConfig] = None,
                 srs_max_degree: int = 1024):
        """
        Initialize zkML system.
        
        Args:
            config: Network configuration
            optimization_config: Optional optimization settings
            srs_max_degree: Maximum polynomial degree for SRS
        """
        self.config = config
        self.opt_config = optimization_config or OptimizationConfig()
        
        # Initialize components
        self.optimizer = PLONKOptimizer(self.opt_config)
        self.compiler = CircuitCompiler(self.optimizer)
        
        # Generate SRS (in production, use trusted setup)
        self.srs = SRS.generate_insecure(srs_max_degree)
        
        # Compile circuit
        self.circuit = self.compiler.compile_network(config)
        
        # Create prover and verifier
        self.prover = ZkMLProver(self.srs, self.optimizer)
        self.verifier = ZkMLVerifier(self.srs)
        
        # Compute network hash
        self.network_hash = self._compute_network_hash()
    
    def _compute_network_hash(self) -> bytes:
        """Compute deterministic hash of network configuration."""
        import hashlib
        data = self.config.name.encode()
        for layer_type, params in self.config.layers:
            data += layer_type.encode()
            data += str(params).encode()
        return hashlib.sha256(data).digest()
    
    def prove(self, input_data: List[Union[int, Fr]],
              weights: Optional[Dict[str, Any]] = None) -> ZkMLProof:
        """
        Generate a proof for network inference.
        
        Args:
            input_data: Input values (integers or field elements)
            weights: Optional network weights
        
        Returns:
            ZkMLProof
        """
        # Convert input to field elements
        inputs = [Field.element(x) if isinstance(x, int) else x 
                  for x in input_data]
        
        # Create witness by running inference
        witness = self._compute_witness(inputs, weights)
        
        # Generate proof
        return self.prover.prove(
            self.circuit, witness, inputs, self.network_hash
        )
    
    def _compute_witness(self, inputs: List[Fr],
                         weights: Optional[Dict[str, Any]] = None) -> Witness:
        """Compute witness by simulating network inference."""
        witness = Witness()
        
        # Set input values
        for i, inp in enumerate(inputs):
            if i < len(self.circuit.public_inputs):
                wire_idx = self.circuit.public_inputs[i]
                witness.values[wire_idx] = inp
        
        # Propagate through gates (simplified)
        for gate in self.circuit.gates:
            if gate.gate_type == GateType.CONST and gate.output:
                witness.values[gate.output.index] = gate.constant or Fr.zero()
            elif gate.gate_type == GateType.ADD and gate.left and gate.right and gate.output:
                a = witness.values.get(gate.left.index, Fr.zero())
                b = witness.values.get(gate.right.index, Fr.zero())
                witness.values[gate.output.index] = a + b
            elif gate.gate_type == GateType.MUL and gate.left and gate.right and gate.output:
                a = witness.values.get(gate.left.index, Fr.zero())
                b = witness.values.get(gate.right.index, Fr.zero())
                witness.values[gate.output.index] = a * b
        
        return witness
    
    def verify(self, proof: ZkMLProof) -> Tuple[bool, str]:
        """
        Verify a zkML proof.
        
        Args:
            proof: The proof to verify
        
        Returns:
            Tuple of (is_valid, reason)
        """
        return self.verifier.verify(proof, self.network_hash)
    
    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get statistics about the compiled circuit."""
        return {
            "num_wires": self.circuit.num_wires(),
            "num_gates": self.circuit.num_constraints(),
            "num_public_inputs": len(self.circuit.public_inputs),
            "network_hash": self.network_hash.hex()
        }
    
    def estimate_optimizations(self) -> OptimizationStats:
        """Estimate optimization benefits for this network."""
        layers = []
        for layer_type, params in self.config.layers:
            layers.append({
                'type': layer_type,
                'size': params.get('output_size', params.get('input_size', 0))
            })
        return self.optimizer.analyze_circuit_optimizations(layers)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    'LayerType', 'LayerConfig', 'NetworkConfig',
    # Compiler
    'CircuitCompiler',
    # Proof
    'ZkMLProof', 'ZkMLProver', 'ZkMLVerifier',
    # Main interface
    'ZkML',
]
