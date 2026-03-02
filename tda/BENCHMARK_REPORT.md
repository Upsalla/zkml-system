# TDA Fingerprinting Benchmark Report

## Summary: 5/5 tests passed

## Test Results

| Test | Status | Details |
|------|--------|---------|
| uniqueness | ✅ PASS | {"n_models": 20, "collisions": 0, "collision_rate"... |
| stability | ✅ PASS | {"perturbation_levels": [0.001, 0.01, 0.05, 0.1, 0... |
| scalability | ✅ PASS | {"param_counts": [154, 4608, 18432, 73728], "finge... |
| performance | ✅ PASS | {"results": [{"name": "tiny", "params": 115, "time... |
| collision_resistance | ✅ PASS | {"n_attempts": 100, "collisions": 0, "min_distance... |

## Key Findings

### Uniqueness

- Tested 20 random models
- Found 0 collisions
- Collision rate: 0.0000%

### Stability


### Scalability

- Fingerprint size is **constant** (212 bytes)
- Tested models from 154 to 73,728 parameters

### Performance

- tiny: 2.08 ms
- small: 23.15 ms
- medium: 75.61 ms
- large: 406.20 ms

### Collision_Resistance

