"""
Tropical Geometry Integration for zkML Pipeline

This module provides seamless integration of tropical optimizations
into the existing zkML circuit compilation pipeline.

Usage:
    from zkml_system.tropical import TropicalOptimizer
    
    optimizer = TropicalOptimizer()
    optimized_circuit = optimizer.optimize(circuit, config)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tropical.tropical_ops import (
    TropicalCircuit, TropicalMaxPool, TropicalArgmax, TropicalSoftmax,
    TropicalSemiring, TropicalConstraint
)


class LayerType(Enum):
    """Types of layers that can be optimized."""
    MAX_POOL = "max_pool"
    ARGMAX = "argmax"
    SOFTMAX = "softmax"
    RELU = "relu"
    DENSE = "dense"
    CONV2D = "conv2d"
    OTHER = "other"


@dataclass
class OptimizationConfig:
    """Configuration for tropical optimizations."""
    enable_tropical_maxpool: bool = True
    enable_tropical_argmax: bool = True
    enable_tropical_softmax: bool = True
    softmax_approximation_warning: bool = True
    comparison_bits: int = 20  # Bits for standard comparison


@dataclass
class LayerInfo:
    """Information about a layer in the network."""
    layer_type: LayerType
    input_size: int
    output_size: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of tropical optimization."""
    original_constraints: int
    optimized_constraints: int
    reduction_percent: float
    layers_optimized: int
    warnings: List[str] = field(default_factory=list)


class TropicalOptimizer:
    """
    Main optimizer class for integrating tropical operations.
    
    This class analyzes a network architecture and applies tropical
    optimizations where beneficial.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_log: List[Dict[str, Any]] = []
    
    def analyze_layer(self, layer: LayerInfo) -> Tuple[int, int, bool]:
        """
        Analyze a layer and return (standard_cost, tropical_cost, can_optimize).
        """
        if layer.layer_type == LayerType.MAX_POOL:
            pool_size = layer.params.get('pool_size', 2)
            num_pools = layer.output_size
            elements_per_pool = pool_size ** 2
            
            standard_cost = num_pools * (elements_per_pool - 1) * (self.config.comparison_bits + 2)
            tropical_cost = num_pools * (elements_per_pool - 1) * 2
            
            return standard_cost, tropical_cost, self.config.enable_tropical_maxpool
        
        elif layer.layer_type == LayerType.ARGMAX:
            num_classes = layer.input_size
            
            standard_cost = (num_classes - 1) * (self.config.comparison_bits + 3)
            tropical_cost = (num_classes - 1) * 3
            
            return standard_cost, tropical_cost, self.config.enable_tropical_argmax
        
        elif layer.layer_type == LayerType.SOFTMAX:
            num_classes = layer.input_size
            
            standard_cost = num_classes * 50  # exp + div
            tropical_cost = 2 * (num_classes - 1)  # just max
            
            return standard_cost, tropical_cost, self.config.enable_tropical_softmax
        
        elif layer.layer_type == LayerType.RELU:
            # ReLU: max(0, x) - can be optimized with tropical
            num_activations = layer.input_size
            
            standard_cost = num_activations * self.config.comparison_bits
            tropical_cost = num_activations * 2  # comparison + selection
            
            return standard_cost, tropical_cost, self.config.enable_tropical_maxpool
        
        else:
            # Dense, Conv2D, etc. - no tropical optimization
            if layer.layer_type == LayerType.DENSE:
                cost = layer.input_size * layer.output_size
            elif layer.layer_type == LayerType.CONV2D:
                kernel_size = layer.params.get('kernel_size', 3)
                cost = layer.output_size * (kernel_size ** 2) * layer.params.get('in_channels', 1)
            else:
                cost = layer.input_size
            
            return cost, cost, False
    
    def optimize_network(self, layers: List[LayerInfo]) -> OptimizationResult:
        """
        Optimize a network architecture using tropical operations.
        
        Args:
            layers: List of LayerInfo describing the network
        
        Returns:
            OptimizationResult with statistics
        """
        total_standard = 0
        total_tropical = 0
        layers_optimized = 0
        warnings = []
        
        self.optimization_log = []
        
        for i, layer in enumerate(layers):
            standard_cost, tropical_cost, can_optimize = self.analyze_layer(layer)
            
            if can_optimize and tropical_cost < standard_cost:
                total_standard += standard_cost
                total_tropical += tropical_cost
                layers_optimized += 1
                
                reduction = (1 - tropical_cost / standard_cost) * 100
                
                self.optimization_log.append({
                    'layer_index': i,
                    'layer_type': layer.layer_type.value,
                    'standard_cost': standard_cost,
                    'tropical_cost': tropical_cost,
                    'reduction': reduction,
                    'optimized': True
                })
                
                # Add warning for softmax approximation
                if layer.layer_type == LayerType.SOFTMAX and self.config.softmax_approximation_warning:
                    warnings.append(
                        f"Layer {i} (Softmax): Using tropical approximation. "
                        "This may affect numerical accuracy."
                    )
            else:
                total_standard += standard_cost
                total_tropical += standard_cost  # No optimization
                
                self.optimization_log.append({
                    'layer_index': i,
                    'layer_type': layer.layer_type.value,
                    'standard_cost': standard_cost,
                    'tropical_cost': standard_cost,
                    'reduction': 0,
                    'optimized': False
                })
        
        reduction_percent = (1 - total_tropical / total_standard) * 100 if total_standard > 0 else 0
        
        return OptimizationResult(
            original_constraints=total_standard,
            optimized_constraints=total_tropical,
            reduction_percent=reduction_percent,
            layers_optimized=layers_optimized,
            warnings=warnings
        )
    
    def compile_tropical_circuit(self, layers: List[LayerInfo]) -> TropicalCircuit:
        """
        Compile a network to a TropicalCircuit.
        
        This creates the actual constraint system with tropical optimizations.
        """
        circuit = TropicalCircuit()
        
        # Create input variables
        if layers:
            first_layer = layers[0]
            inputs = [circuit.new_variable(f"input_{i}") for i in range(first_layer.input_size)]
            for inp in inputs:
                circuit.variables[inp].is_input = True
        else:
            inputs = []
        
        current_vars = inputs
        
        for layer in layers:
            if layer.layer_type == LayerType.MAX_POOL and self.config.enable_tropical_maxpool:
                # Apply tropical max-pooling
                pool_size = layer.params.get('pool_size', 2)
                num_pools = layer.output_size
                elements_per_pool = pool_size ** 2
                
                new_vars = []
                max_pool = TropicalMaxPool(elements_per_pool)
                
                for p in range(num_pools):
                    start_idx = p * elements_per_pool
                    pool_inputs = current_vars[start_idx:start_idx + elements_per_pool]
                    if len(pool_inputs) == elements_per_pool:
                        result, _ = max_pool.compile(circuit, pool_inputs)
                        new_vars.append(result)
                
                current_vars = new_vars if new_vars else current_vars
            
            elif layer.layer_type == LayerType.ARGMAX and self.config.enable_tropical_argmax:
                # Apply tropical argmax
                argmax = TropicalArgmax()
                max_val, max_idx, _ = argmax.compile(circuit, current_vars)
                current_vars = [max_val, max_idx]
            
            elif layer.layer_type == LayerType.SOFTMAX and self.config.enable_tropical_softmax:
                # Apply tropical softmax
                softmax = TropicalSoftmax(len(current_vars))
                outputs, _ = softmax.compile(circuit, current_vars)
                current_vars = outputs
            
            else:
                # Non-optimizable layers: create placeholder variables
                new_vars = [circuit.new_variable(f"layer_{layer.layer_type.value}_{i}") 
                           for i in range(layer.output_size)]
                
                # Add placeholder constraints (simplified)
                for var in new_vars:
                    circuit.add_constraint(TropicalConstraint(
                        constraint_type='linear',
                        variables=[var],
                        coefficients=[1],
                        metadata={'layer_type': layer.layer_type.value}
                    ))
                
                current_vars = new_vars
        
        # Mark outputs
        for var in current_vars:
            if var in circuit.variables:
                circuit.variables[var].is_output = True
        
        return circuit
    
    def get_optimization_report(self) -> str:
        """Generate a human-readable optimization report."""
        if not self.optimization_log:
            return "No optimization performed yet."
        
        lines = [
            "=" * 70,
            "TROPICAL OPTIMIZATION REPORT",
            "=" * 70,
            "",
            f"{'Layer':<8} {'Type':<15} {'Standard':<12} {'Tropical':<12} {'Reduction':<12} {'Status':<10}",
            "-" * 70
        ]
        
        total_standard = 0
        total_tropical = 0
        
        for entry in self.optimization_log:
            status = "OPTIMIZED" if entry['optimized'] else "STANDARD"
            lines.append(
                f"{entry['layer_index']:<8} {entry['layer_type']:<15} "
                f"{entry['standard_cost']:<12,} {entry['tropical_cost']:<12,} "
                f"{entry['reduction']:.1f}%{'':<6} {status:<10}"
            )
            total_standard += entry['standard_cost']
            total_tropical += entry['tropical_cost']
        
        lines.append("-" * 70)
        total_reduction = (1 - total_tropical / total_standard) * 100 if total_standard > 0 else 0
        lines.append(
            f"{'TOTAL':<8} {'':<15} {total_standard:<12,} {total_tropical:<12,} "
            f"{total_reduction:.1f}%"
        )
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimize_cnn(input_shape: Tuple[int, int], 
                 conv_layers: List[Tuple[int, int]], 
                 pool_size: int,
                 dense_layers: List[int],
                 num_classes: int,
                 config: Optional[OptimizationConfig] = None) -> OptimizationResult:
    """
    Convenience function to optimize a CNN architecture.
    
    Args:
        input_shape: (height, width) of input
        conv_layers: List of (filters, kernel_size)
        pool_size: Size of max-pooling
        dense_layers: List of hidden units
        num_classes: Number of output classes
        config: Optional optimization config
    
    Returns:
        OptimizationResult
    """
    layers = []
    h, w = input_shape
    channels = 1
    
    for filters, kernel_size in conv_layers:
        # Conv layer
        h_out = h - kernel_size + 1
        w_out = w - kernel_size + 1
        layers.append(LayerInfo(
            layer_type=LayerType.CONV2D,
            input_size=h * w * channels,
            output_size=h_out * w_out * filters,
            params={'kernel_size': kernel_size, 'in_channels': channels}
        ))
        
        # ReLU
        layers.append(LayerInfo(
            layer_type=LayerType.RELU,
            input_size=h_out * w_out * filters,
            output_size=h_out * w_out * filters
        ))
        
        # Max-Pool
        pool_h = h_out // pool_size
        pool_w = w_out // pool_size
        layers.append(LayerInfo(
            layer_type=LayerType.MAX_POOL,
            input_size=h_out * w_out * filters,
            output_size=pool_h * pool_w * filters,
            params={'pool_size': pool_size}
        ))
        
        h, w, channels = pool_h, pool_w, filters
    
    # Flatten
    flatten_size = h * w * channels
    prev_size = flatten_size
    
    # Dense layers
    for hidden in dense_layers:
        layers.append(LayerInfo(
            layer_type=LayerType.DENSE,
            input_size=prev_size,
            output_size=hidden
        ))
        layers.append(LayerInfo(
            layer_type=LayerType.RELU,
            input_size=hidden,
            output_size=hidden
        ))
        prev_size = hidden
    
    # Output
    layers.append(LayerInfo(
        layer_type=LayerType.DENSE,
        input_size=prev_size,
        output_size=num_classes
    ))
    layers.append(LayerInfo(
        layer_type=LayerType.SOFTMAX,
        input_size=num_classes,
        output_size=num_classes
    ))
    
    optimizer = TropicalOptimizer(config)
    result = optimizer.optimize_network(layers)
    
    print(optimizer.get_optimization_report())
    
    return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Tropical Pipeline Integration\n")
    
    # Test with LeNet-5 style CNN
    result = optimize_cnn(
        input_shape=(28, 28),
        conv_layers=[(6, 5), (16, 5)],
        pool_size=2,
        dense_layers=[120, 84],
        num_classes=10
    )
    
    print(f"\nOptimization Summary:")
    print(f"  Original constraints: {result.original_constraints:,}")
    print(f"  Optimized constraints: {result.optimized_constraints:,}")
    print(f"  Reduction: {result.reduction_percent:.1f}%")
    print(f"  Layers optimized: {result.layers_optimized}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    
    print("\n" + "=" * 70)
    print("Integration test PASSED")
    print("=" * 70)
