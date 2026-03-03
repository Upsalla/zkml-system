# Witness Compression Benchmark Report

## Comparison: Standard vs HWWB vs CSWC

### Results by Scenario

| Size | Sparsity | Correlation | Standard | HWWB | CSWC | Best | Reduction |
|------|----------|-------------|----------|------|------|------|-----------|
| 500 | 10% | 20% | 16,032B | 15,928B | 17,056B | **hwwb** | 0.6% |
| 500 | 10% | 50% | 16,032B | 13,880B | 17,128B | **hwwb** | 13.4% |
| 500 | 10% | 80% | 16,032B | 11,736B | 17,200B | **hwwb** | 26.8% |
| 500 | 50% | 20% | 16,032B | 14,744B | 10,108B | **cswc** | 37.0% |
| 500 | 50% | 50% | 16,032B | 14,136B | 10,072B | **cswc** | 37.2% |
| 500 | 50% | 80% | 16,032B | 13,336B | 10,360B | **cswc** | 35.4% |
| 500 | 80% | 20% | 16,032B | 11,608B | 4,384B | **cswc** | 72.7% |
| 500 | 80% | 50% | 16,032B | 11,576B | 4,384B | **cswc** | 72.7% |
| 500 | 80% | 80% | 16,032B | 11,384B | 4,384B | **cswc** | 72.7% |

### Key Findings

- **HWWB wins**: 3 scenarios (high correlation)
- **CSWC wins**: 6 scenarios (high sparsity)
- **Standard wins**: 0 scenarios (low sparsity + low correlation)

### Recommendations

1. **Use CSWC** when sparsity > 30% (typical for ReLU networks)
2. **Use HWWB** when correlation > 50% and sparsity < 30% (dense but structured)
3. **Combine both** for maximum compression in mixed scenarios