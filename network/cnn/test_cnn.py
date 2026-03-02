"""
Unit tests for CNN layers (Conv2D and Pooling).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.network.cnn.conv2d import Conv2DConfig, Conv2DLayer, Conv2DWinograd
from zkml_system.network.cnn.pooling import PoolConfig, AvgPool2D, MaxPool2D, GlobalAvgPool2D


def test_conv2d_basic():
    """Test basic Conv2D forward pass."""
    print("Testing Conv2D basic forward pass...")

    # Simple 1x1 convolution (essentially a linear transformation per pixel)
    config = Conv2DConfig(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0
    )

    # Weight = 2.0, bias = 1.0
    weights = [[[[2.0]]]]
    bias = [1.0]

    layer = Conv2DLayer(config, weights, bias)

    # Input: 1 channel, 3x3
    input_tensor = [[[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]]]

    output = layer.forward(input_tensor)

    # Expected: 2*input + 1
    expected = [[[3.0, 5.0, 7.0],
                 [9.0, 11.0, 13.0],
                 [15.0, 17.0, 19.0]]]

    assert len(output) == 1, "Output channels mismatch"
    assert len(output[0]) == 3, "Output height mismatch"
    assert len(output[0][0]) == 3, "Output width mismatch"

    for i in range(3):
        for j in range(3):
            assert abs(output[0][i][j] - expected[0][i][j]) < 1e-6, \
                f"Output mismatch at ({i}, {j}): {output[0][i][j]} != {expected[0][i][j]}"

    print("  Conv2D basic test passed!")


def test_conv2d_3x3():
    """Test 3x3 Conv2D."""
    print("Testing Conv2D 3x3 kernel...")

    config = Conv2DConfig(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=0
    )

    # Simple averaging kernel
    weights = [[[[1/9, 1/9, 1/9],
                 [1/9, 1/9, 1/9],
                 [1/9, 1/9, 1/9]]]]
    bias = [0.0]

    layer = Conv2DLayer(config, weights, bias)

    # Input: 1 channel, 5x5
    input_tensor = [[[1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0]]]

    output = layer.forward(input_tensor)

    # Output should be 3x3, all values = 1.0 (average of all 1s)
    assert len(output[0]) == 3, f"Output height mismatch: {len(output[0])}"
    assert len(output[0][0]) == 3, f"Output width mismatch: {len(output[0][0])}"

    for i in range(3):
        for j in range(3):
            assert abs(output[0][i][j] - 1.0) < 1e-6, \
                f"Output mismatch at ({i}, {j}): {output[0][i][j]} != 1.0"

    print("  Conv2D 3x3 test passed!")


def test_conv2d_output_shape():
    """Test Conv2D output shape calculation."""
    print("Testing Conv2D output shape calculation...")

    config = Conv2DConfig(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1
    )

    # Dummy weights
    weights = [[[[0.0] * 3 for _ in range(3)] for _ in range(3)] for _ in range(16)]

    layer = Conv2DLayer(config, weights)

    # Input shape: (3, 28, 28)
    output_shape = layer.compute_output_shape((3, 28, 28))

    # Expected: (16, 14, 14) with stride=2, padding=1, kernel=3
    # h_out = (28 + 2*1 - 3) / 2 + 1 = 14
    assert output_shape == (16, 14, 14), f"Shape mismatch: {output_shape}"

    print("  Conv2D output shape test passed!")


def test_conv2d_constraint_count():
    """Test Conv2D constraint counting."""
    print("Testing Conv2D constraint count...")

    config = Conv2DConfig(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    )

    weights = [[[[0.0] * 3 for _ in range(3)] for _ in range(3)] for _ in range(16)]
    layer = Conv2DLayer(config, weights)

    # Input shape: (3, 8, 8)
    constraints = layer.count_constraints((3, 8, 8))

    # Expected: 16 * 8 * 8 * (3 * 3 * 3) = 16 * 64 * 27 = 27648
    expected = 16 * 8 * 8 * 27
    assert constraints == expected, f"Constraint count mismatch: {constraints} != {expected}"

    print("  Conv2D constraint count test passed!")


def test_winograd_conv2d():
    """Test Winograd Conv2D."""
    print("Testing Winograd Conv2D...")

    # Create standard and Winograd layers with same weights
    weights = [[[[1.0, 0.0, -1.0],
                 [2.0, 0.0, -2.0],
                 [1.0, 0.0, -1.0]]]]  # Sobel-like kernel

    bias = [0.0]

    config = Conv2DConfig(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=0
    )

    standard_layer = Conv2DLayer(config, weights, bias)
    winograd_layer = Conv2DWinograd(1, 1, weights, bias)

    # Input: 1 channel, 6x6
    input_tensor = [[[float(i * 6 + j) for j in range(6)] for i in range(6)]]

    standard_output = standard_layer.forward(input_tensor)
    winograd_output = winograd_layer.forward(input_tensor)

    # Compare outputs (should be approximately equal)
    for i in range(len(standard_output[0])):
        for j in range(len(standard_output[0][0])):
            diff = abs(standard_output[0][i][j] - winograd_output[0][i][j])
            assert diff < 1e-4, \
                f"Winograd mismatch at ({i}, {j}): {standard_output[0][i][j]} vs {winograd_output[0][i][j]}"

    print("  Winograd Conv2D test passed!")


def test_winograd_constraint_reduction():
    """Test that Winograd reduces constraint count."""
    print("Testing Winograd constraint reduction...")

    weights = [[[[0.0] * 3 for _ in range(3)] for _ in range(3)] for _ in range(16)]

    config = Conv2DConfig(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=0
    )

    standard_layer = Conv2DLayer(config, weights)
    winograd_layer = Conv2DWinograd(3, 16, weights)

    # Input shape: (3, 8, 8) -> output (16, 6, 6)
    standard_constraints = standard_layer.count_constraints((3, 8, 8))
    winograd_constraints = winograd_layer.count_constraints((3, 8, 8))

    reduction = 1 - (winograd_constraints / standard_constraints)
    print(f"  Standard constraints: {standard_constraints}")
    print(f"  Winograd constraints: {winograd_constraints}")
    print(f"  Reduction: {reduction * 100:.1f}%")

    assert winograd_constraints < standard_constraints, "Winograd should reduce constraints"

    print("  Winograd constraint reduction test passed!")


def test_avgpool2d():
    """Test AvgPool2D."""
    print("Testing AvgPool2D...")

    config = PoolConfig(kernel_size=2, stride=2)
    layer = AvgPool2D(config)

    # Input: 1 channel, 4x4
    input_tensor = [[[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0],
                     [9.0, 10.0, 11.0, 12.0],
                     [13.0, 14.0, 15.0, 16.0]]]

    output = layer.forward(input_tensor)

    # Expected: 2x2 with averages of 2x2 blocks
    # [1+2+5+6]/4=3.5, [3+4+7+8]/4=5.5
    # [9+10+13+14]/4=11.5, [11+12+15+16]/4=13.5
    expected = [[[3.5, 5.5],
                 [11.5, 13.5]]]

    assert len(output[0]) == 2, "Output height mismatch"
    assert len(output[0][0]) == 2, "Output width mismatch"

    for i in range(2):
        for j in range(2):
            assert abs(output[0][i][j] - expected[0][i][j]) < 1e-6, \
                f"AvgPool mismatch at ({i}, {j})"

    # Verify zero constraints
    assert layer.count_constraints((1, 4, 4)) == 0, "AvgPool should have 0 constraints"

    print("  AvgPool2D test passed!")


def test_maxpool2d():
    """Test MaxPool2D."""
    print("Testing MaxPool2D...")

    config = PoolConfig(kernel_size=2, stride=2)
    layer = MaxPool2D(config)

    # Input: 1 channel, 4x4
    input_tensor = [[[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0],
                     [9.0, 10.0, 11.0, 12.0],
                     [13.0, 14.0, 15.0, 16.0]]]

    output = layer.forward(input_tensor)

    # Expected: 2x2 with max of 2x2 blocks
    expected = [[[6.0, 8.0],
                 [14.0, 16.0]]]

    assert len(output[0]) == 2, "Output height mismatch"
    assert len(output[0][0]) == 2, "Output width mismatch"

    for i in range(2):
        for j in range(2):
            assert abs(output[0][i][j] - expected[0][i][j]) < 1e-6, \
                f"MaxPool mismatch at ({i}, {j})"

    # Verify non-zero constraints
    constraints = layer.count_constraints((1, 4, 4))
    assert constraints > 0, "MaxPool should have non-zero constraints"
    print(f"  MaxPool constraints: {constraints}")

    print("  MaxPool2D test passed!")


def test_global_avgpool():
    """Test GlobalAvgPool2D."""
    print("Testing GlobalAvgPool2D...")

    layer = GlobalAvgPool2D()

    # Input: 2 channels, 3x3
    input_tensor = [[[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]],
                    [[9.0, 8.0, 7.0],
                     [6.0, 5.0, 4.0],
                     [3.0, 2.0, 1.0]]]

    output = layer.forward(input_tensor)

    # Expected: [mean of channel 0, mean of channel 1]
    # Channel 0: (1+2+...+9)/9 = 45/9 = 5.0
    # Channel 1: same = 5.0
    expected = [5.0, 5.0]

    assert len(output) == 2, "Output length mismatch"
    for i in range(2):
        assert abs(output[i] - expected[i]) < 1e-6, f"GlobalAvgPool mismatch at {i}"

    print("  GlobalAvgPool2D test passed!")


def run_all_tests():
    """Run all CNN tests."""
    print("=" * 60)
    print("CNN Layer Tests")
    print("=" * 60)

    test_conv2d_basic()
    test_conv2d_3x3()
    test_conv2d_output_shape()
    test_conv2d_constraint_count()
    test_winograd_conv2d()
    test_winograd_constraint_reduction()
    test_avgpool2d()
    test_maxpool2d()
    test_global_avgpool()

    print("=" * 60)
    print("ALL CNN TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
