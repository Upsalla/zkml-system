"""
Pooling Layers for zkML

This module implements pooling layers optimized for zero-knowledge proofs.
Average pooling is preferred over max pooling as it requires no comparisons,
which are expensive in R1CS.

Key design decisions:
1. AvgPool2D: Zero additional constraints (just linear combinations)
2. MaxPool2D: Requires comparison constraints (expensive, use sparingly)
3. GlobalAvgPool: Efficient for classification heads
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class PoolConfig:
    """Configuration for a pooling layer."""
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = None
    padding: Union[int, Tuple[int, int]] = 0

    def __post_init__(self):
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if self.stride is None:
            self.stride = self.kernel_size
        elif isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)


class AvgPool2D:
    """
    2D Average Pooling Layer.

    Average pooling computes the mean of elements in each pooling window.
    This is highly efficient for zkML as it only requires linear combinations,
    which are essentially free in R1CS (no multiplication constraints).

    Constraint cost: 0 multiplication constraints (additions only).
    """

    def __init__(self, config: PoolConfig):
        """
        Initialize an AvgPool2D layer.

        Args:
            config: Pooling configuration.
        """
        self.config = config
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.padding

        # Precompute the divisor (1 / pool_size)
        self.pool_size = self.kernel_size[0] * self.kernel_size[1]
        self.scale = 1.0 / self.pool_size

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Compute the output shape given an input shape.

        Args:
            input_shape: (channels, height, width)

        Returns:
            Output shape (channels, out_height, out_width)
        """
        c, h_in, w_in = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        h_out = (h_in + 2 * ph - kh) // sh + 1
        w_out = (w_in + 2 * pw - kw) // sw + 1

        return (c, h_out, w_out)

    def forward(self, input_tensor: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Perform average pooling.

        Args:
            input_tensor: Input [channels, height, width].

        Returns:
            Output [channels, out_height, out_width].
        """
        c, h_in, w_in = len(input_tensor), len(input_tensor[0]), len(input_tensor[0][0])
        _, h_out, w_out = self.compute_output_shape((c, h_in, w_in))

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        output = [[[0.0 for _ in range(w_out)] for _ in range(h_out)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    total = 0.0
                    count = 0

                    for kh_idx in range(kh):
                        for kw_idx in range(kw):
                            ih = oh * sh - ph + kh_idx
                            iw = ow * sw - pw + kw_idx

                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                total += input_tensor[ch][ih][iw]
                                count += 1

                    output[ch][oh][ow] = total / count if count > 0 else 0.0

        return output

    def count_constraints(self, input_shape: Tuple[int, int, int]) -> int:
        """
        Count R1CS constraints.

        Average pooling requires no multiplication constraints.
        The division by pool_size is a constant multiplication.
        """
        return 0


class MaxPool2D:
    """
    2D Max Pooling Layer.

    Max pooling selects the maximum element in each pooling window.
    This is expensive in zkML as it requires comparison constraints.

    For each max operation over k elements, we need:
    - k-1 comparison constraints
    - Each comparison requires ~3 multiplication constraints

    Constraint cost: ~3 * (pool_size - 1) * num_outputs multiplication constraints.

    WARNING: Use AvgPool2D instead when possible for zkML efficiency.
    """

    def __init__(self, config: PoolConfig):
        """
        Initialize a MaxPool2D layer.

        Args:
            config: Pooling configuration.
        """
        self.config = config
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.padding

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Compute output shape."""
        c, h_in, w_in = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        h_out = (h_in + 2 * ph - kh) // sh + 1
        w_out = (w_in + 2 * pw - kw) // sw + 1

        return (c, h_out, w_out)

    def forward(self, input_tensor: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Perform max pooling.

        Args:
            input_tensor: Input [channels, height, width].

        Returns:
            Output [channels, out_height, out_width].
        """
        c, h_in, w_in = len(input_tensor), len(input_tensor[0]), len(input_tensor[0][0])
        _, h_out, w_out = self.compute_output_shape((c, h_in, w_in))

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        output = [[[float('-inf') for _ in range(w_out)] for _ in range(h_out)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    max_val = float('-inf')

                    for kh_idx in range(kh):
                        for kw_idx in range(kw):
                            ih = oh * sh - ph + kh_idx
                            iw = ow * sw - pw + kw_idx

                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                max_val = max(max_val, input_tensor[ch][ih][iw])

                    output[ch][oh][ow] = max_val

        return output

    def count_constraints(self, input_shape: Tuple[int, int, int]) -> int:
        """
        Count R1CS constraints for max pooling.

        Each max over k elements requires k-1 comparisons.
        Each comparison requires ~3 multiplication constraints.
        """
        c, h_out, w_out = self.compute_output_shape(input_shape)
        pool_size = self.kernel_size[0] * self.kernel_size[1]

        comparisons_per_output = pool_size - 1
        constraints_per_comparison = 3  # Approximate

        return c * h_out * w_out * comparisons_per_output * constraints_per_comparison


class GlobalAvgPool2D:
    """
    Global Average Pooling Layer.

    Reduces each channel to a single value by averaging all spatial positions.
    This is commonly used before the final classification layer.

    Constraint cost: 0 multiplication constraints.
    """

    def forward(self, input_tensor: List[List[List[float]]]) -> List[float]:
        """
        Perform global average pooling.

        Args:
            input_tensor: Input [channels, height, width].

        Returns:
            Output [channels] (1D).
        """
        c = len(input_tensor)
        h = len(input_tensor[0])
        w = len(input_tensor[0][0])

        output = []
        for ch in range(c):
            total = sum(input_tensor[ch][i][j] for i in range(h) for j in range(w))
            output.append(total / (h * w))

        return output

    def count_constraints(self, input_shape: Tuple[int, int, int]) -> int:
        """Global average pooling requires no multiplication constraints."""
        return 0


class AdaptiveAvgPool2D:
    """
    Adaptive Average Pooling Layer.

    Produces output of specified size regardless of input size.
    Useful for handling variable input sizes.

    Constraint cost: 0 multiplication constraints.
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        """
        Initialize an AdaptiveAvgPool2D layer.

        Args:
            output_size: Target output size (height, width) or single int for square.
        """
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, input_tensor: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Perform adaptive average pooling.

        Args:
            input_tensor: Input [channels, height, width].

        Returns:
            Output [channels, out_height, out_width].
        """
        c = len(input_tensor)
        h_in = len(input_tensor[0])
        w_in = len(input_tensor[0][0])
        h_out, w_out = self.output_size

        output = [[[0.0 for _ in range(w_out)] for _ in range(h_out)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    # Compute input region for this output
                    h_start = (oh * h_in) // h_out
                    h_end = ((oh + 1) * h_in) // h_out
                    w_start = (ow * w_in) // w_out
                    w_end = ((ow + 1) * w_in) // w_out

                    total = 0.0
                    count = 0
                    for ih in range(h_start, h_end):
                        for iw in range(w_start, w_end):
                            total += input_tensor[ch][ih][iw]
                            count += 1

                    output[ch][oh][ow] = total / count if count > 0 else 0.0

        return output

    def count_constraints(self, input_shape: Tuple[int, int, int]) -> int:
        """Adaptive average pooling requires no multiplication constraints."""
        return 0
