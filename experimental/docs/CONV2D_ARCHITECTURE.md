# Conv2D Layer Architecture for zkML

**Author**: Manus AI
**Date**: January 26, 2026
**Status**: **PARTIALLY IMPLEMENTED** — `network/cnn/conv2d.py` and `network/cnn/pooling.py` exist. Fused BatchNorm and LeNet-5 model not yet implemented.

## 1. Summary

This document describes the technical architecture for extending the zkML system with **Convolutional Neural Network (CNN)** components. The introduction of Conv2D, Pooling, and Batch Normalization layers is critical for enabling standard computer vision tasks such as image classification (e.g., MNIST, CIFAR-10).

## 2. Problem Statement and Goals

Direct translation of CNN operations into R1CS leads to a constraint explosion:

- **Convolution**: A single 3×3 convolution over 64 channels requires `3 × 3 × 64 = 576` multiplication constraints per output pixel.
- **ReLU activation**: Hundreds of constraints per activation.
- **Max-Pooling**: `max(a, b)` requires expensive bit decomposition (~1000 constraints).

**Project Goals**:

1. ✅ **CNN base layers**: Conv2D and Pooling layers → **Implemented** (`network/cnn/`)
2. **Constraint efficiency**: Systematic optimization → **Partially implemented**
3. **Standard model support**: LeNet-5 → **Not yet implemented**
4. ✅ **Modularity**: Independent testing → **Implemented** (`network/cnn/test_cnn.py`)

## 3. Architecture and Constraint Optimization

### 3.1 Conv2D Layer in R1CS [IMPLEMENTED]

A convolution operation is fundamentally a dot product between the kernel and an input patch. For each output pixel:

`y[i,j,k] = Σ_{di,dj,c} x[i+di, j+dj, c] * w[di, dj, c, k] + bias[k]`

Each multiplication `x * w` generates an R1CS constraint. Summation generates no additional constraints as it can be a single `LinearCombination`.

### 3.2 Pooling Layer [IMPLEMENTED]

| Pooling Type | Operation | R1CS Cost | Recommendation |
| :--- | :--- | :--- | :--- |
| **Max-Pooling** | `y = max(x1, x2, x3, x4)` | **Extremely high** (~1000 constraints) | **Avoid** |
| **Average-Pooling** | `y = (x1+x2+x3+x4) / 4` | **Very low** (1 constraint for division) | **Standard** |

**Average-Pooling is the standard for all zkML CNNs.** Implementation: `network/cnn/pooling.py`.

### 3.3 Fused Batch Normalization [NOT IMPLEMENTED]

BatchNorm parameters `γ, μ, σ, β` can be fused into the preceding Conv2D layer's weights during inference, resulting in **zero additional constraints**.

- `w_fused = w * (γ / σ)`
- `b_fused = (b - μ) * (γ / σ) + β`

## 4. Module Structure [PARTIALLY IMPLEMENTED]

```plaintext
zkml_system/
├── network/
│   ├── cnn/
│   │   ├── __init__.py        ✅
│   │   ├── conv2d.py          ✅ Convolutional Layer
│   │   ├── pooling.py         ✅ AvgPool2D Layer
│   │   ├── test_cnn.py        ✅ Tests
│   │   ├── batchnorm.py       ❌ Not yet implemented
│   │   └── flatten.py         ❌ Not yet implemented
│   │
│   ├── builder.py             ⚠️ CNN integration pending
│   │
│   └── models/                ❌ Not yet implemented
│       ├── lenet.py
│       └── resnet_tiny.py
```

## 5. Implementation Milestones

| Phase | Task | Status |
| :--- | :--- | :--- |
| 1 | Conv2D Layer | ✅ Done |
| 2 | Pooling & Flatten | ⚠️ Pooling done, flatten missing |
| 3 | Fused BatchNorm | ❌ Not started |
| 4 | Builder integration | ❌ Not started |
| 5 | LeNet-5 model | ❌ Not started |
| 6 | End-to-end test | ❌ Not started |

## 6. References

[1] Lavin, A., & Gray, S. (2016). *Fast Algorithms for Convolutional Neural Networks*. [arXiv:1509.09308](https://arxiv.org/abs/1509.09308)
[2] Jacob, B., et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
[3] LeCun, Y., et al. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE.
