"""
Tropical Geometry Module for zkML

This module provides tropical arithmetic optimizations for
zero-knowledge circuits, particularly effective for:
- Max-Pooling operations (90% constraint reduction)
- Argmax operations (87% constraint reduction)
- Softmax approximation (96% constraint reduction)

Usage:
    from zkml_system.tropical import TropicalOptimizer, OptimizationConfig
    
    optimizer = TropicalOptimizer()
    result = optimizer.optimize_network(layers)
"""

from tropical.tropical_ops import (
    TropicalSemiring,
    TropicalElement,
    TropicalCircuit,
    TropicalMaxPool,
    TropicalArgmax,
    TropicalSoftmax,
)

from tropical.pipeline_integration import (
    TropicalOptimizer,
    OptimizationConfig,
    LayerType,
    LayerInfo,
    OptimizationResult,
    optimize_cnn,
)

__all__ = [
    # Core tropical operations
    "TropicalSemiring",
    "TropicalElement", 
    "TropicalCircuit",
    "TropicalMaxPool",
    "TropicalArgmax",
    "TropicalSoftmax",
    # Pipeline integration
    "TropicalOptimizer",
    "OptimizationConfig",
    "LayerType",
    "LayerInfo",
    "OptimizationResult",
    "optimize_cnn",
]
