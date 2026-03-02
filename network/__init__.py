"""
Network Module für zkML
=======================

Enthält Layer-Definitionen und den Netzwerk-Builder.
"""

from .layers import (
    LayerConfig, LayerWeights, LayerOutput,
    DenseLayer, InputLayer, OutputLayer
)

from .builder import (
    NetworkStats, LayerSpec, Network, NetworkBuilder
)

__all__ = [
    'LayerConfig', 'LayerWeights', 'LayerOutput',
    'DenseLayer', 'InputLayer', 'OutputLayer',
    'NetworkStats', 'LayerSpec', 'Network', 'NetworkBuilder'
]
