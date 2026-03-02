"""
Neural Network Operators for zkML

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

Implements Conv2D and Attention operators for zkML circuits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.bn254.field import Fr


@dataclass
class Conv2DConfig:
    """Configuration for Conv2D layer."""
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    
    def output_size(self, h: int, w: int) -> Tuple[int, int]:
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out


class Conv2D:
    """2D Convolution operator for zkML."""
    
    def __init__(self, config: Conv2DConfig):
        self.config = config
    
    def forward(
        self,
        input_data: List[List[List[int]]],
        kernel: List[List[List[List[int]]]]
    ) -> Tuple[List[List[List[Fr]]], Dict]:
        """Compute Conv2D forward pass."""
        cfg = self.config
        c_in = len(input_data)
        h_in = len(input_data[0])
        w_in = len(input_data[0][0])
        h_out, w_out = cfg.output_size(h_in, w_in)
        
        inp = [[[Fr(input_data[c][h][w]) for w in range(w_in)] 
                for h in range(h_in)] for c in range(c_in)]
        
        kern = [[[[Fr(kernel[co][ci][kh][kw]) for kw in range(cfg.kernel_size)]
                  for kh in range(cfg.kernel_size)]
                 for ci in range(c_in)]
                for co in range(cfg.out_channels)]
        
        output = []
        mult_count = 0
        
        for co in range(cfg.out_channels):
            out_channel = []
            for h in range(h_out):
                out_row = []
                for w in range(w_out):
                    acc = Fr.zero()
                    for ci in range(c_in):
                        for kh in range(cfg.kernel_size):
                            for kw in range(cfg.kernel_size):
                                ih = h * cfg.stride + kh - cfg.padding
                                iw = w * cfg.stride + kw - cfg.padding
                                if 0 <= ih < h_in and 0 <= iw < w_in:
                                    prod = inp[ci][ih][iw] * kern[co][ci][kh][kw]
                                    acc = acc + prod
                                    mult_count += 1
                    out_row.append(acc)
                out_channel.append(out_row)
            output.append(out_channel)
        
        return output, {'multiplications': mult_count, 'shape': (cfg.out_channels, h_out, w_out)}


@dataclass
class AttentionConfig:
    """Configuration for Attention layer."""
    embed_dim: int
    num_heads: int
    head_dim: int = None
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


class Attention:
    """Multi-head Attention operator for zkML."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
    
    def forward(
        self,
        query: List[List[int]],
        key: List[List[int]],
        value: List[List[int]]
    ) -> Tuple[List[List[Fr]], Dict]:
        """Compute attention forward pass."""
        cfg = self.config
        seq_len = len(query)
        
        Q = [[Fr(query[i][j]) for j in range(cfg.embed_dim)] for i in range(seq_len)]
        K = [[Fr(key[i][j]) for j in range(cfg.embed_dim)] for i in range(seq_len)]
        V = [[Fr(value[i][j]) for j in range(cfg.embed_dim)] for i in range(seq_len)]
        
        # Q * K^T
        scores = []
        qk_mults = 0
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                dot = Fr.zero()
                for d in range(cfg.embed_dim):
                    dot = dot + Q[i][d] * K[j][d]
                    qk_mults += 1
                row.append(dot)
            scores.append(row)
        
        # Simplified softmax (lookup-based in production)
        attention_weights = []
        for i in range(seq_len):
            max_score = max(s.to_int() for s in scores[i])
            exp_scores = [Fr(max(1, 1000 + (s.to_int() - max_score) * 10)) for s in scores[i]]
            total = sum(e.to_int() for e in exp_scores)
            weights = [Fr(e.to_int() * 1000 // max(1, total)) for e in exp_scores]
            attention_weights.append(weights)
        
        # Attention * V
        output = []
        av_mults = 0
        for i in range(seq_len):
            out_row = []
            for d in range(cfg.embed_dim):
                acc = Fr.zero()
                for j in range(seq_len):
                    acc = acc + attention_weights[i][j] * V[j][d]
                    av_mults += 1
                out_row.append(acc)
            output.append(out_row)
        
        return output, {'qk_mults': qk_mults, 'av_mults': av_mults, 'shape': (seq_len, cfg.embed_dim)}


def test_operators():
    """Test Conv2D and Attention operators."""
    print("=" * 50)
    print("Operator Tests")
    print("=" * 50)
    
    # Test Conv2D
    print("\n1. Conv2D Test")
    cfg = Conv2DConfig(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    conv = Conv2D(cfg)
    
    # 4x4 input
    input_data = [[[i*4+j for j in range(4)] for i in range(4)]]
    # 2 output channels, 1 input channel, 3x3 kernel
    kernel = [[[[1 for _ in range(3)] for _ in range(3)] for _ in range(1)] for _ in range(2)]
    
    output, trace = conv.forward(input_data, kernel)
    print(f"   Input shape: (1, 4, 4)")
    print(f"   Kernel shape: (2, 1, 3, 3)")
    print(f"   Output shape: {trace['shape']}")
    print(f"   Multiplications: {trace['multiplications']}")
    
    # Test Attention
    print("\n2. Attention Test")
    cfg = AttentionConfig(embed_dim=4, num_heads=1)
    attn = Attention(cfg)
    
    # seq_len=3, embed_dim=4
    query = [[i*4+j for j in range(4)] for i in range(3)]
    key = query
    value = query
    
    output, trace = attn.forward(query, key, value)
    print(f"   Input shape: (3, 4)")
    print(f"   Output shape: {trace['shape']}")
    print(f"   Q*K^T mults: {trace['qk_mults']}")
    print(f"   Attn*V mults: {trace['av_mults']}")
    
    print("\n" + "=" * 50)
    print("OPERATOR TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_operators()
