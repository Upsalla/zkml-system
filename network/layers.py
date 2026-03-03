"""
Neural Network Layers for zkML
==============================

Implements the basic layer types for neural networks,
optimized for zkML with minimal constraints.

Supported layers:
- DenseLayer: Fully connected layer
- (Extensible: Conv2D, BatchNorm, etc.)
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.witness import Witness, WitnessBuilder
from activations.base import ActivationFunction, FIXED_POINT_SCALE
from activations.optimized import get_activation


@dataclass
class LayerConfig:
    """Configuration for a layer."""
    input_size: int
    output_size: int
    activation: str = "gelu"  # "gelu", "swish", "quadratic", "relu", "none"
    use_bias: bool = True
    name: str = ""


@dataclass
class LayerWeights:
    """Weights for a layer."""
    weights: List[List[int]]  # [output_size][input_size]
    biases: Optional[List[int]] = None  # [output_size]
    
    def validate(self, config: LayerConfig) -> bool:
        """Validate that weights match the configuration."""
        if len(self.weights) != config.output_size:
            return False
        if any(len(row) != config.input_size for row in self.weights):
            return False
        if config.use_bias and (self.biases is None or len(self.biases) != config.output_size):
            return False
        return True


@dataclass
class LayerOutput:
    """Output of a layer forward pass."""
    output_indices: List[int]  # Indices of outputs in witness
    activation_indices: List[int]  # Indices of activations (before activation function)
    num_constraints: int
    sparsity: float  # Fraction of inactive neurons


class DenseLayer:
    """
    Fully connected (dense) layer.
    
    Computation: output = activation(W @ input + b)
    
    Constraint analysis:
    - Matrix multiplication: input_size constraints per output neuron
    - Bias addition: 1 constraint per output neuron (if use_bias)
    - Activation: Depends on the activation function
    
    Total per neuron: input_size + 1 + activation_constraints
    """
    
    def __init__(self, config: LayerConfig, weights: LayerWeights, prime: int):
        """
        Initialize the dense layer.
        
        Args:
            config: Layer configuration
            weights: Weights and biases
            prime: The prime field
        """
        if not weights.validate(config):
            raise ValueError("Weights do not match configuration")
        
        self.config = config
        self.weights = weights
        self.prime = prime
        
        # Activation function
        if config.activation.lower() != "none":
            self.activation = get_activation(config.activation)
        else:
            self.activation = None
        
        # Constraint calculation
        self._calculate_constraint_cost()
    
    def _calculate_constraint_cost(self) -> None:
        """Calculate the constraint cost for this layer."""
        # Matrixmultiplikation: input_size Multiplikationen pro Output
        matmul_constraints = self.config.input_size * self.config.output_size
        
        # Bias: 1 Addition pro Output (als Constraint: (sum + bias) * 1 = result)
        bias_constraints = self.config.output_size if self.config.use_bias else 0
        
        # Activation
        if self.activation:
            activation_constraints = self.activation.constraint_count * self.config.output_size
        else:
            activation_constraints = 0
        
        self.matmul_constraints = matmul_constraints
        self.bias_constraints = bias_constraints
        self.activation_constraints = activation_constraints
        self.total_constraints = matmul_constraints + bias_constraints + activation_constraints
    
    def forward(
        self, 
        input_indices: List[int], 
        witness: Witness,
        layer_id: int = 0
    ) -> LayerOutput:
        """
        Perform forward pass and generate witness entries.
        
        Args:
            input_indices: Indices of inputs in the witness
            witness: The witness
            layer_id: ID of this layer (for tracking)
            
        Returns:
            LayerOutput with output indices and statistics
        """
        if len(input_indices) != self.config.input_size:
            raise ValueError(f"Expected {self.config.input_size} inputs, got {len(input_indices)}")
        
        # Get input values
        inputs = [witness.get(idx) for idx in input_indices]
        
        output_indices = []
        activation_indices = []
        inactive_count = 0
        
        for neuron_idx in range(self.config.output_size):
            # Compute weighted sum
            weighted_sum = 0
            for i, (inp, w) in enumerate(zip(inputs, self.weights.weights[neuron_idx])):
                weighted_sum = (weighted_sum + inp * w) % self.prime
            
            # Add bias
            if self.config.use_bias:
                weighted_sum = (weighted_sum + self.weights.biases[neuron_idx]) % self.prime
            
            # Store intermediate value (before activation) in witness
            pre_act_idx = witness.allocate(
                weighted_sum,
                name=f"{self.config.name}_pre_act_{neuron_idx}",
                layer=layer_id,
                neuron=neuron_idx,
                var_type="intermediate"
            )
            activation_indices.append(pre_act_idx)
            
            # Apply activation
            if self.activation:
                output_value = self.activation.compute(weighted_sum, self.prime)
                
                # Check for inactivity (for sparsity tracking)
                if output_value == 0:
                    inactive_count += 1
            else:
                output_value = weighted_sum
            
            # Store output in witness
            out_idx = witness.allocate(
                output_value,
                name=f"{self.config.name}_out_{neuron_idx}",
                layer=layer_id,
                neuron=neuron_idx,
                var_type="activation" if self.activation else "output"
            )
            output_indices.append(out_idx)
        
        sparsity = inactive_count / self.config.output_size if self.config.output_size > 0 else 0
        
        return LayerOutput(
            output_indices=output_indices,
            activation_indices=activation_indices,
            num_constraints=self.total_constraints,
            sparsity=sparsity
        )
    
    def get_constraint_breakdown(self) -> dict:
        """Gibt eine Aufschlüsselung der Constraints zurück."""
        return {
            "matmul": self.matmul_constraints,
            "bias": self.bias_constraints,
            "activation": self.activation_constraints,
            "total": self.total_constraints,
            "per_neuron": self.total_constraints // self.config.output_size if self.config.output_size > 0 else 0
        }
    
    def __repr__(self) -> str:
        act_name = self.activation.name if self.activation else "None"
        return (
            f"DenseLayer({self.config.input_size} -> {self.config.output_size}, "
            f"activation={act_name}, constraints={self.total_constraints})"
        )


class InputLayer:
    """
    Eingabe-Layer.
    
    Kein echter Layer, sondern ein Helfer zum Registrieren von Eingaben im Witness.
    """
    
    def __init__(self, size: int, name: str = "input"):
        self.size = size
        self.name = name
    
    def forward(self, values: List[int], witness: Witness, public: bool = True) -> List[int]:
        """
        Registriert Eingaben im Witness.
        
        Args:
            values: Die Eingabewerte
            witness: Der Witness
            public: Ob die Eingaben öffentlich sind
            
        Returns:
            Liste der Witness-Indices
        """
        if len(values) != self.size:
            raise ValueError(f"Erwartete {self.size} Eingaben, bekam {len(values)}")
        
        indices = []
        for i, val in enumerate(values):
            idx = witness.allocate(
                val,
                name=f"{self.name}_{i}",
                is_public=public,
                layer=0,
                var_type="input"
            )
            indices.append(idx)
        
        return indices
    
    def __repr__(self) -> str:
        return f"InputLayer(size={self.size})"


class OutputLayer:
    """
    Ausgabe-Layer.
    
    Markiert bestimmte Witness-Einträge als öffentliche Ausgaben.
    """
    
    def __init__(self, size: int, name: str = "output"):
        self.size = size
        self.name = name
    
    def forward(self, input_indices: List[int], witness: Witness) -> List[int]:
        """
        Markiert Eingaben als öffentliche Ausgaben.
        
        In der aktuellen Implementierung kopieren wir die Werte
        in neue "Output"-Einträge.
        """
        if len(input_indices) != self.size:
            raise ValueError(f"Erwartete {self.size} Eingaben, bekam {len(input_indices)}")
        
        output_indices = []
        for i, in_idx in enumerate(input_indices):
            val = witness.get(in_idx)
            out_idx = witness.allocate(
                val,
                name=f"{self.name}_{i}",
                is_public=True,
                var_type="output"
            )
            output_indices.append(out_idx)
        
        return output_indices
    
    def __repr__(self) -> str:
        return f"OutputLayer(size={self.size})"


# Tests
if __name__ == "__main__":
    print("=== Neural Network Layer Tests ===\n")
    
    prime = 101
    
    # Erstelle einen einfachen Dense Layer
    config = LayerConfig(
        input_size=3,
        output_size=2,
        activation="gelu",
        use_bias=True,
        name="dense1"
    )
    
    weights = LayerWeights(
        weights=[[1, 2, 3], [4, 5, 6]],  # 2 Neuronen, 3 Eingaben
        biases=[1, 2]
    )
    
    layer = DenseLayer(config, weights, prime)
    print(f"Layer: {layer}")
    print(f"Constraint Breakdown: {layer.get_constraint_breakdown()}")
    
    # Forward Pass
    witness = Witness(prime)
    
    # Eingaben registrieren
    input_layer = InputLayer(3, "input")
    input_indices = input_layer.forward([5, 3, 2], witness, public=True)
    print(f"\nInput indices: {input_indices}")
    
    # Dense Layer Forward
    output = layer.forward(input_indices, witness, layer_id=1)
    print(f"Output: {output}")
    print(f"Output values: {[witness.get(idx) for idx in output.output_indices]}")
    
    print(f"\nWitness size: {witness.size()}")
    print(f"Sparsity: {output.sparsity:.1%}")
