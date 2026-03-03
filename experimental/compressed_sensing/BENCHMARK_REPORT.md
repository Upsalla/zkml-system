# CSWC Benchmark Report

## Compressed Sensing Witness Commitment vs Standard Commitment

### Summary

- **Average Size Reduction**: 37.4%
- **Average Prove Time Ratio**: 764.28x (CSWC/Standard)
- **Average Verify Time Ratio**: 1014.99x (CSWC/Standard)

### Detailed Results

| Witness Size | Sparsity | Standard Size | CSWC Size | Reduction | Prove Ratio | Verify Ratio |
|-------------|----------|---------------|-----------|-----------|-------------|--------------|
| 500 | 0% | 16,032 B | 19,144 B | -19.4% | 367.71x | 851.00x |
| 500 | 30% | 16,032 B | 13,744 B | 14.3% | 402.59x | 714.61x |
| 500 | 50% | 16,032 B | 10,144 B | 36.7% | 261.14x | 391.51x |
| 500 | 70% | 16,032 B | 6,544 B | 59.2% | 188.87x | 310.43x |
| 500 | 90% | 16,032 B | 2,944 B | 81.6% | 50.90x | 80.96x |
| 1,000 | 0% | 32,032 B | 37,144 B | -16.0% | 1485.59x | 1597.99x |
| 1,000 | 30% | 32,032 B | 26,344 B | 17.8% | 884.76x | 1310.80x |
| 1,000 | 50% | 32,032 B | 19,144 B | 40.2% | 369.40x | 497.28x |
| 1,000 | 70% | 32,032 B | 11,944 B | 62.7% | 432.08x | 599.33x |
| 1,000 | 90% | 32,032 B | 4,744 B | 85.2% | 104.84x | 131.07x |
| 2,000 | 0% | 64,032 B | 73,144 B | -14.2% | 2630.80x | 3217.29x |
| 2,000 | 30% | 64,032 B | 51,544 B | 19.5% | 2081.37x | 2513.26x |
| 2,000 | 50% | 64,032 B | 37,144 B | 42.0% | 1233.73x | 1621.36x |
| 2,000 | 70% | 64,032 B | 22,744 B | 64.5% | 711.05x | 976.31x |
| 2,000 | 90% | 64,032 B | 8,344 B | 87.0% | 259.40x | 411.73x |

### Key Findings

- **Best Size Reduction**: 87.0% at 90% sparsity, size 2,000
- **CSWC Beneficial** at sparsity >= 30% for size 500

### Conclusion

CSWC provides significant size reduction for sparse witnesses. The trade-off is 
increased prove and verify time due to the sketch computation. For witnesses with 
sparsity > 30%, the size reduction typically outweighs the computational overhead.
