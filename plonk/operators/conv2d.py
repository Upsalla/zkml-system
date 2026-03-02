"""
Conv2D Operator for zkML

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

This module implements the 2D convolution operator for CNNs in zkML.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkml_system.crypto.bn254.fr_adapter import Fr
from plonk.core import Wire, Gate, GateType, Circuit, Witness


@dataclass
class Conv2DConfig:
    """Configuration for a Conv2D layer."""
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    
    def output_size(self, input_height: int, input_width: int) -> Tuple[int, int]:
        """Calculate output dimensions."""
        h_out = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out
    
    def constraint_count(self, input_height: int, input_width: int, batch: int = 1) -> Dict[str, int]:
        """Estimate constraint count."""
        h_out, w_out = self.output_size(input_height, input_width)
        outputs = batch * self.out_channels * h_out * w_out
        mults_per_output = self.in_channels * self.kernel_size * self.kernel_size
        adds_per_output = mults_per_output - 1
        
        return {
            'outputs': outputs,
            'multiplications': outputs * mults_per_output,
            'additions': outputs * adds_per_output,
            'total_gates': outputs * (mults_per_output + adds_per_output)
        }


class Conv2DOperator:
    """
    Conv2D operator that generates witnesses and constraints.
    
    Implements im2col-style convolution for efficient circuit generation.
    """
    
    def __init__(self, config: Conv2DConfig):
        self.config = config
    
    def forward(
        self,
        input_tensor: List[List[List[Fr]]],  # [C_in, H, W]
        kernel: List[List[List[List[Fr]]]]   # [C_out, C_in, K, K]
    ) -> Tuple[List[List[List[Fr]]], Dict]:
        """
        Compute Conv2D forward pass.
        
        Args:
            input_tensor: Input [C_in, H, W]
            kernel: Kernel [C_out, C_in, K, K]
        
        Returns:
            (output_tensor, computation_trace)
        """
        cfg = self.config
        c_in = len(input_tensor)
        h_in = len(input_tensor[0])
        w_in = len(input_tensor[0][0])
        
        h_out, w_out = cfg
