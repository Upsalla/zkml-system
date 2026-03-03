"""
Neural Network Builder for zkML
===============================

Enables easy construction of neural networks
with automatic constraint calculation and optimization.

Beispiel:
    network = NetworkBuilder(prime=101)
    network.add_input(784)
    network.add_dense(128, activation="gelu")
    network.add_dense(10, activation="none")
    network.add_output()
    
    model = network.build()
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.witness import Witness, WitnessBuilder
from network.layers import (
    DenseLayer, InputLayer, OutputLayer,
    LayerConfig, LayerWeights, LayerOutput
)
from activations.optimized import get_activation


@dataclass
class NetworkStats:
    """Network statistics."""
    num_layers: int
    total_parameters: int
    total_constraints: int
    constraint_breakdown: Dict[str, int]
    
    # Comparison with ReLU
    relu_constraints: int
    constraint_savings: float
    
    # Sparsity (after forward pass)
    avg_sparsity: float = 0.0
    sparse_constraints: int = 0
    
    def __repr__(self) -> str:
        return (
            f"NetworkStats(\n"
            f"  layers: {self.num_layers}\n"
            f"  parameters: {self.total_parameters}\n"
            f"  constraints: {self.total_constraints} (ReLU: {self.relu_constraints}, saved: {self.constraint_savings:.1%})\n"
            f"  avg_sparsity: {self.avg_sparsity:.1%}\n"
            f"  with_sparsity: {self.sparse_constraints} constraints\n"
            f")"
        )


@dataclass
class LayerSpec:
    """Specification for a layer (before build)."""
    layer_type: str  # "input", "dense", "output"
    size: int
    activation: str = "none"
    use_bias: bool = True
    name: str = ""


class Network:
    """
    A built neural network.
    
    Contains all layers and can perform forward passes.
    """
    
    def __init__(
        self, 
        prime: int,
        input_layer: InputLayer,
        hidden_layers: List[DenseLayer],
        output_layer: OutputLayer
    ):
        self.prime = prime
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        
        self._calculate_stats()
    
    def _calculate_stats(self) -> None:
        """Calculate network statistics."""
        total_constraints = 0
        total_params = 0
        relu_constraints = 0
        breakdown = {"matmul": 0, "bias": 0, "activation": 0}
        
        for layer in self.hidden_layers:
            cb = layer.get_constraint_breakdown()
            total_constraints += cb["total"]
            breakdown["matmul"] += cb["matmul"]
            breakdown["bias"] += cb["bias"]
            breakdown["activation"] += cb["activation"]
            
            # Count parameters
            total_params += layer.config.input_size * layer.config.output_size
            if layer.config.use_bias:
                total_params += layer.config.output_size
            
            # ReLU comparison (258 constraints per activation)
            relu_constraints += cb["matmul"] + cb["bias"] + 258 * layer.config.output_size
        
        savings = 1 - total_constraints / relu_constraints if relu_constraints > 0 else 0
        
        self.stats = NetworkStats(
            num_layers=len(self.hidden_layers),
            total_parameters=total_params,
            total_constraints=total_constraints,
            constraint_breakdown=breakdown,
            relu_constraints=relu_constraints,
            constraint_savings=savings
        )
    
    def forward(self, inputs: List[int]) -> Tuple[List[int], Witness, NetworkStats]:
        """
        Perform a forward pass.
        
        Args:
            inputs: The input values
            
        Returns:
            Tuple of (output values, witness, updated stats)
        """
        witness = Witness(self.prime)
        
        # Register inputs
        current_indices = self.input_layer.forward(inputs, witness, public=True)
        
        # Process hidden layers
        sparsities = []
        for i, layer in enumerate(self.hidden_layers):
            output = layer.forward(current_indices, witness, layer_id=i+1)
            current_indices = output.output_indices
            sparsities.append(output.sparsity)
        
        # Mark outputs
        output_indices = self.output_layer.forward(current_indices, witness)
        
        # Extract output values
        outputs = [witness.get(idx) for idx in output_indices]
        
        # Update stats with sparsity
        avg_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0
        
        # Calculate sparse constraints
        sparse_constraints = 0
        for i, layer in enumerate(self.hidden_layers):
            cb = layer.get_constraint_breakdown()
            active_ratio = 1 - sparsities[i]
            
            # Active neurons: full cost
            # Inactive neurons: only 1 constraint (zero proof)
            active_neurons = int(layer.config.output_size * active_ratio)
            inactive_neurons = layer.config.output_size - active_neurons
            
            per_neuron = cb["per_neuron"]
            sparse_constraints += active_neurons * per_neuron + inactive_neurons * 1
        
        updated_stats = NetworkStats(
            num_layers=self.stats.num_layers,
            total_parameters=self.stats.total_parameters,
            total_constraints=self.stats.total_constraints,
            constraint_breakdown=self.stats.constraint_breakdown,
            relu_constraints=self.stats.relu_constraints,
            constraint_savings=self.stats.constraint_savings,
            avg_sparsity=avg_sparsity,
            sparse_constraints=sparse_constraints
        )
        
        return outputs, witness, updated_stats
    
    def __repr__(self) -> str:
        layers_str = "\n  ".join([str(self.input_layer)] + 
                                  [str(l) for l in self.hidden_layers] + 
                                  [str(self.output_layer)])
        return f"Network(\n  {layers_str}\n)"


class NetworkBuilder:
    """
    Builder for neural networks.
    
    Enables step-by-step construction of a network
    with automatic configuration.
    """
    
    def __init__(self, prime: int):
        """
        Initialize the builder.
        
        Args:
            prime: The prime field for all computations
        """
        self.prime = prime
        self.layer_specs: List[LayerSpec] = []
        self.weights: Dict[int, LayerWeights] = {}  # layer_index -> weights
        
        self._input_size: Optional[int] = None
        self._output_size: Optional[int] = None
    
    def add_input(self, size: int, name: str = "input") -> 'NetworkBuilder':
        """Add the input layer."""
        if self._input_size is not None:
            raise ValueError("Input layer already defined")
        
        self._input_size = size
        self.layer_specs.append(LayerSpec(
            layer_type="input",
            size=size,
            name=name
        ))
        return self
    
    def add_dense(
        self, 
        size: int, 
        activation: str = "gelu",
        use_bias: bool = True,
        name: str = "",
        weights: Optional[List[List[int]]] = None,
        biases: Optional[List[int]] = None
    ) -> 'NetworkBuilder':
        """
        Add a dense layer.
        
        Args:
            size: Number of neurons
            activation: Activation function ("gelu", "swish", "quadratic", "relu", "none")
            use_bias: Whether to use bias
            name: Layer name
            weights: Optional weights (otherwise randomly initialized)
            biases: Optional biases (otherwise randomly initialized)
        """
        if self._input_size is None:
            raise ValueError("Input layer must be defined first")
        
        layer_idx = len([s for s in self.layer_specs if s.layer_type == "dense"])
        
        if not name:
            name = f"dense_{layer_idx}"
        
        self.layer_specs.append(LayerSpec(
            layer_type="dense",
            size=size,
            activation=activation,
            use_bias=use_bias,
            name=name
        ))
        
        # Store weights if provided
        if weights is not None:
            if biases is None and use_bias:
                biases = [0] * size
            self.weights[layer_idx] = LayerWeights(weights=weights, biases=biases)
        
        return self
    
    def add_output(self, name: str = "output") -> 'NetworkBuilder':
        """Add the output layer."""
        if self._output_size is not None:
            raise ValueError("Output layer already defined")
        
        # Output size is the size of the last dense layer
        dense_specs = [s for s in self.layer_specs if s.layer_type == "dense"]
        if not dense_specs:
            raise ValueError("At least one dense layer must be defined before output")
        
        self._output_size = dense_specs[-1].size
        
        self.layer_specs.append(LayerSpec(
            layer_type="output",
            size=self._output_size,
            name=name
        ))
        return self
    
    def build(self, random_seed: Optional[int] = None) -> Network:
        """
        Build the network.
        
        Args:
            random_seed: Seed for random weight initialization
            
        Returns:
            The final Network object
        """
        if self._input_size is None:
            raise ValueError("Input layer not defined")
        if self._output_size is None:
            raise ValueError("Output layer not defined")
        
        import random
        if random_seed is not None:
            random.seed(random_seed)
        
        # Input Layer
        input_spec = next(s for s in self.layer_specs if s.layer_type == "input")
        input_layer = InputLayer(input_spec.size, input_spec.name)
        
        # Dense Layers
        hidden_layers = []
        current_input_size = self._input_size
        
        dense_idx = 0
        for spec in self.layer_specs:
            if spec.layer_type != "dense":
                continue
            
            config = LayerConfig(
                input_size=current_input_size,
                output_size=spec.size,
                activation=spec.activation,
                use_bias=spec.use_bias,
                name=spec.name
            )
            
            # Get or generate weights
            if dense_idx in self.weights:
                weights = self.weights[dense_idx]
            else:
                # Random initialization (small values)
                max_val = min(10, self.prime // 10)
                w = [[random.randint(-max_val, max_val) % self.prime 
                      for _ in range(current_input_size)] 
                     for _ in range(spec.size)]
                b = [random.randint(-max_val, max_val) % self.prime 
                     for _ in range(spec.size)] if spec.use_bias else None
                weights = LayerWeights(weights=w, biases=b)
            
            layer = DenseLayer(config, weights, self.prime)
            hidden_layers.append(layer)
            
            current_input_size = spec.size
            dense_idx += 1
        
        # Output Layer
        output_spec = next(s for s in self.layer_specs if s.layer_type == "output")
        output_layer = OutputLayer(output_spec.size, output_spec.name)
        
        return Network(self.prime, input_layer, hidden_layers, output_layer)
    
    def summary(self) -> str:
        """Return a summary of the network."""
        lines = ["Network Summary:", "=" * 50]
        
        current_size = self._input_size
        total_params = 0
        
        for spec in self.layer_specs:
            if spec.layer_type == "input":
                lines.append(f"Input:  {spec.size}")
            elif spec.layer_type == "dense":
                params = current_size * spec.size + (spec.size if spec.use_bias else 0)
                total_params += params
                lines.append(f"Dense:  {current_size} -> {spec.size} ({spec.activation}), params={params}")
                current_size = spec.size
            elif spec.layer_type == "output":
                lines.append(f"Output: {spec.size}")
        
        lines.append("=" * 50)
        lines.append(f"Total Parameters: {total_params}")
        
        return "\n".join(lines)


# Tests
if __name__ == "__main__":
    print("=== Network Builder Tests ===\n")
    
    prime = 101
    
    # Build a simple network
    builder = NetworkBuilder(prime)
    builder.add_input(4)
    builder.add_dense(8, activation="gelu")
    builder.add_dense(4, activation="gelu")
    builder.add_dense(2, activation="none")
    builder.add_output()
    
    print(builder.summary())
    
    # Build the network
    network = builder.build(random_seed=42)
    print(f"\n{network}")
    print(f"\nStats: {network.stats}")
    
    # Forward Pass
    inputs = [5, 3, 2, 1]
    outputs, witness, updated_stats = network.forward(inputs)
    
    print(f"\nInputs: {inputs}")
    print(f"Outputs: {outputs}")
    print(f"Witness size: {witness.size()}")
    print(f"\nUpdated Stats: {updated_stats}")
    
    # Comparison: With vs. without optimizations
    print("\n" + "=" * 50)
    print("COMPARISON: Optimized vs. ReLU")
    print("=" * 50)
    print(f"With GELU:     {updated_stats.total_constraints} constraints")
    print(f"With ReLU:     {updated_stats.relu_constraints} constraints")
    print(f"Savings:       {updated_stats.constraint_savings:.1%}")
    print(f"With Sparsity: {updated_stats.sparse_constraints} constraints")
    
    if updated_stats.sparse_constraints > 0:
        total_savings = 1 - updated_stats.sparse_constraints / updated_stats.relu_constraints
        print(f"Total savings: {total_savings:.1%}")
