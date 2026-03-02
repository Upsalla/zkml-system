# zkML System

Zero-knowledge proofs for verifiable neural network inference over BN254/PLONK.

## What This Does

zkML compiles neural network inference (dense layers, ReLU, GELU, Conv2D, MaxPool, Argmax) into arithmetic circuits and generates cryptographic proofs that the computation was performed correctly — without revealing model weights or intermediate values.

**Status:** Research prototype. Not audited for production use.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Network Layer Compilation                           │
│  (Dense, Conv2D, ReLU, GELU, MaxPool, Argmax)       │
├──────────────────────────────────────────────────────┤
│  PLONK Prover / Verifier (5-round protocol)         │
│  ├─ KZG Polynomial Commitments                      │
│  ├─ FFT / NTT (Cooley-Tukey)                        │
│  └─ Fiat-Shamir Transcript                          │
├──────────────────────────────────────────────────────┤
│  BN254 Cryptographic Primitives                      │
│  ├─ Python: Fr, Fp, G1Point, G2Point, Pairing       │
│  └─ Rust backend (optional): MSM, field arithmetic  │
└──────────────────────────────────────────────────────┘
```

## Install

```bash
# Clone
git clone https://github.com/Upsalla/zkml-system.git
cd zkml-system

# Python dependencies
pip install -e .

# Optional: Rust backend for 5-60x speedup on field ops
cd rust_backend
pip install maturin
maturin develop --release
cd ..
```

## Quick Start

```python
from zkml_system.plonk.plonk_prover import PLONKProver, PLONKVerifier
from zkml_system.plonk.plonk_kzg import TrustedSetup
from zkml_system.plonk.circuit_compiler import CircuitCompiler
from zkml_system.crypto.bn254.fr_adapter import Fr

# 1. Define a circuit: a * b = c
cc = CircuitCompiler()
a = cc.add_wire('a', is_public=True)
b = cc.add_wire('b', is_public=True)
c = cc.add_wire('c', is_public=True)
cc.add_mul_gate(a, b, c)
circuit = cc.compile()

# 2. Assign witness values
circuit.wires[a].value = Fr(3)
circuit.wires[b].value = Fr(5)
circuit.wires[c].value = Fr(15)

# 3. Generate SRS and prove
srs = TrustedSetup.generate(max_degree=64)
prover = PLONKProver(srs)
proof = prover.prove(circuit)

# 4. Verify
verifier = PLONKVerifier(srs)
assert verifier.verify(proof, circuit), "Verification failed"
```

## Optimizations

| Technique | Reduction | Target |
|---|---|---|
| CSWC (Compressed Sensing) | 49-87% | Sparse witness vectors |
| Tropical Geometry | 90-96% | MaxPool / Softmax |
| HWWB (Haar Wavelet) | ~27% | Correlated data |
| Rust MSM | 5-60x speedup | Scalar multiplication, field ops |

## Tests

```bash
# Full suite (100 tests)
python -m pytest plonk/ crypto/ network/cnn/ -v

# Rust backend only
cd rust_backend && cargo test
```

## Project Structure

```
crypto/bn254/      # BN254 field arithmetic, curve ops, pairing
plonk/             # PLONK protocol: prover, verifier, KZG, FFT, circuits
network/cnn/       # Neural network layer compilation to circuits
rust_backend/      # PyO3/Rust backend for performance-critical ops
```

## License

MIT — see [LICENSE](LICENSE).
