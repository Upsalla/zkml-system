# zkml-system

Zero-knowledge proofs for verifiable neural network inference and model similarity — built from scratch on BN254/PLONK.

**Status:** Research prototype. Not audited for production use.

## What This Does

zkml-system provides two capabilities:

1. **Verifiable Inference:** Compile neural network layers (Dense, ReLU, GELU) into PLONK arithmetic circuits and prove correct computation without revealing weights.
2. **Hybrid TDA+ZK Model Similarity:** Prove that two neural networks are structurally similar (via TDA fingerprints) without revealing the models. This combines classical Topological Data Analysis with a zero-knowledge distance proof.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Hybrid TDA+ZK Bridge                                   │
│  (Classical TDA fingerprinting + ZK similarity proof)    │
├─────────────────────────────────────────────────────────┤
│  PLONK Prover / Verifier (5-round protocol)             │
│  ├─ KZG Polynomial Commitments                          │
│  ├─ FFT / NTT (Cooley-Tukey)                            │
│  ├─ Fiat-Shamir Transcript                              │
│  └─ Circuit Compiler (Dense, ReLU, GELU, custom gates)  │
├─────────────────────────────────────────────────────────┤
│  BN254 Cryptographic Primitives                          │
│  ├─ Python: Fr, Fp, G1Point, G2Point, Pairing           │
│  └─ Rust backend (PyO3): MSM, field arithmetic (5-60x)  │
└─────────────────────────────────────────────────────────┘
```

## Install

```bash
git clone https://github.com/Upsalla/zkml-system.git
cd zkml-system

pip install -e .

# Optional: Rust backend for 5-60x speedup on field ops
cd rust_backend
pip install maturin
maturin develop --release
cd ..
```

## Quick Demo

```bash
# Run the hybrid TDA+ZK demo (generates 2 models, proves similarity, verifies)
OPENBLAS_NUM_THREADS=1 python demo.py
```

Expected output:
```
Hybrid TDA+ZK Demo
  Backend: rust
  1. Generating two similar models...
  2. Proving similarity (threshold=5.0)...
     Circuit gates: 374
     TDA time:      0.006s
     ZK prove time: 4.9s
     Total:         10.5s
  3. Verifying...
     Result: VALID
```

### Python API

```python
from zkml_system.hybrid_bridge import HybridBridge
import numpy as np

# Two models with slightly different weights
weights_a = [np.random.randn(64, 32)]
weights_b = [weights_a[0] + 0.01 * np.random.randn(64, 32)]

# Prove similarity
bridge = HybridBridge(n_features=20, threshold=5.0)
bundle = bridge.prove_similarity(weights_a, weights_b)

# Verify
valid, reason = bridge.verify_similarity(bundle)
assert valid
```

### Low-Level PLONK API

```python
from zkml_system.plonk.plonk_prover import PLONKProver, PLONKVerifier
from zkml_system.plonk.plonk_kzg import TrustedSetup
from zkml_system.plonk.circuit_compiler import CircuitCompiler
from zkml_system.crypto.bn254.fr_adapter import Fr

# Define circuit: a * b = c
cc = CircuitCompiler()
a = cc.add_wire('a', is_public=True)
b = cc.add_wire('b', is_public=True)
c = cc.add_wire('c')
cc.add_mul_gate(a, b, c)
circuit = cc.compile()

# Assign witness
circuit.wires[a].value = Fr(3)
circuit.wires[b].value = Fr(5)
circuit.wires[c].value = Fr(15)

# Prove and verify
srs = TrustedSetup.generate(max_degree=32)
proof = PLONKProver(srs).prove(circuit)
assert PLONKVerifier(srs).verify(proof, circuit)
```

## Performance

### Hybrid TDA+ZK (10 features)

| Stage | Time | Detail |
|-------|------|--------|
| TDA Fingerprint (×2) | 0.006s | SHA256 commitment chain |
| Circuit Compilation | 0.002s | 374 gates |
| SRS Generation | 5.5s | 518 G1 scalar multiplications |
| PLONK Prove | 4.9s | Rust backend |
| **Total** | **10.5s** | |

### Rust Backend Speedups

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Fr multiplication | 1.0x | 5x | field arithmetic |
| G1 scalar mul | 1.0x | 60x | MSM |
| SRS generation (512) | ~300s | ~5s | combined |

## Tests

```bash
# Full suite (112 tests, ~6 min)
OPENBLAS_NUM_THREADS=1 python -m pytest tests/ -v

# Quick sanity check
python demo.py
```

## Project Structure

```
plonk/                  # PLONK protocol: prover, verifier, KZG, FFT, circuits
  ├─ plonk_prover.py    # PLONKProver + PLONKVerifier (cryptographic)
  ├─ plonk_kzg.py       # KZG polynomial commitment scheme
  ├─ circuit_compiler.py # Gate-level circuit construction
  ├─ tda_gadgets.py     # TDA-specific circuit gadgets (sub, mul, range_check)
  └─ fingerprint_circuit.py  # Fingerprint similarity circuit (374 gates)
crypto/bn254/           # BN254 field arithmetic, curve ops, pairing
  ├─ fr_adapter.py      # Unified Fr interface (Rust or Python backend)
  └─ curve.py           # G1/G2 point operations
rust_backend/           # PyO3/Rust backend for field ops + MSM
hybrid_bridge.py        # End-to-end TDA+ZK pipeline
core/                   # Field abstractions, R1CS, witness
activations/            # Activation function circuit compilation
network/                # Neural network layer compilation
experimental/           # Unverified optimization modules (CSWC, Tropical, Wavelet)
```

## Limitations

This is a research prototype. Known limitations:

| Limitation | Detail |
|------------|--------|
| **Additive commitment** | Uses `Σ vᵢ·rⁱ` instead of Poseidon hash — binding under DLog, not collision-resistant |
| **Trust model gap** | ZK proves distance correctness; fingerprint authenticity relies on out-of-band TDA verification |
| **SRS is not MPC-generated** | Uses deterministic tau — proofs are forgeable by anyone who knows tau |
| **No formal security proof** | Circuit correctness verified by tests, not by formal methods |
| **Python-speed SRS** | SRS generation is the bottleneck (~5s for 518 degree, even with Rust) |

For production use, you would need: Poseidon commitments, MPC ceremony for SRS, formal audit.

## Related

- [tda_fingerprint](https://github.com/dweyhe/tda_fingerprint) — the TDA fingerprinting library used by the hybrid bridge
- [PLONK paper](https://eprint.iacr.org/2019/953) — Gabizon, Williamson, Ciobotaru (2019)

## License

MIT — see [LICENSE](LICENSE).
