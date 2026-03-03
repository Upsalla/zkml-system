# SNARK Architecture Plan: PLONK Integration

**Author**: Manus AI
**Date**: January 26, 2026
**Status**: **PARTIALLY IMPLEMENTED** — Prover (`plonk/plonk_prover.py`), Verifier, KZG, Polynomial ops, and SRS all exist. R1CS-to-PLONK compiler exists as `plonk/circuit_compiler.py`. Proposed module structure differs from actual implementation (flat `plonk/` instead of `snark/` + `compiler/`).

## 1. Summary

This document describes the technical architecture for integrating a **PLONK** (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge) ZK-SNARK system into the existing zkML framework. The goal is to replace the current non-universal, non-fully-zero-knowledge proof mechanism with a production-ready, universal SNARK system. The decision for PLONK is based on its universal and updatable trusted setup as well as its growing popularity and flexibility compared to alternatives like Groth16.

## 2. Problem Statement and Goals

The current proof system has several limitations that prevent production use:

- **No true zero-knowledge property**: The simplified Schnorr-style proof does not reveal the entire witness but does not provide the formal guarantees of a true ZK-SNARK.
- **No on-chain verifiability**: The protocol is not designed for efficient verification in smart contracts.
- **Non-universal setup**: A new setup would be required for every change to the network architecture, which is untenable in practice.

**Project Goals**:

1. **Implementation of a complete PLONK system**: Including prover, verifier, and a universal trusted setup (SRS). → **[IMPLEMENTED]**
2. **Ensuring the zero-knowledge property**: Ensuring that no information about the private witness can be derived from the proof. → **[IMPLEMENTED]** (blinding factors in plonk_prover.py)
3. **Efficient verification**: The verifier must be performant enough for on-chain verification with acceptable gas costs. → **[PARTIALLY IMPLEMENTED]** (off-chain verifier exists, Solidity contracts exist)
4. **Universality**: The system must work with a single trusted setup for different neural networks (within a certain size). → **[IMPLEMENTED]** (SRS.generate_insecure + TrustedSetup)

## 3. Architecture Decision: PLONK

The choice fell on PLONK over the more established Groth16 for several strategic reasons:

| Criterion | Groth16 | PLONK | Rationale for PLONK |
| :--- | :--- | :--- | :--- |
| **Trusted Setup** | Per Circuit | **Universal & Updatable** | A single setup for all circuits, drastically reducing operational complexity. |
| **Proof Size** | **~192 bytes** | ~400-600 bytes | Although proofs are larger, the difference is acceptable for most applications. |
| **Verification Time** | **~2-3 ms** | ~5-10 ms | The slightly longer verification time is offset by the immense flexibility. |
| **Flexibility** | Low | **High** | PLONK supports custom gates and is better suited for complex, repeated structures. |
| **Developer Ecosystem** | Established | **Growing** | PLONK is rapidly gaining popularity and is strongly supported by modern frameworks like Aztec and Zcash. |

## 4. System Architecture [IMPLEMENTED — different structure]

> [!NOTE]
> The implementation uses a flat `plonk/` module structure instead of the proposed
> `snark/` + `compiler/` hierarchy. All components are in `plonk/`:
> - `plonk_prover.py` — PLONK Prover + Verifier
> - `circuit_compiler.py` — Circuit compiler (R1CS-to-PLONK, GELU gates, sparse optimization)
> - `polynomial.py` — Polynomial arithmetic
> - `kzg.py` — KZG commitment scheme
> - `poseidon.py` — Poseidon hash for transcript
> - `trusted_setup.py` — SRS management

```
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│  Neural Network   │──────▶│ Circuit Compiler   │──────▶│ PLONK Prover      │
│ (e.g. LeNet-5)    │       │ (circuit_compiler) │       │ (plonk_prover)    │
└───────────────────┘       └───────────────────┘       └───────────────────┘
        │                                                         │
        │ (Forward Pass)                                          │
        ▼                                                         ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│   Witness Vector  │       │  Universal SRS    │       │    PLONK Proof    │
│   (Private)       │◀──────│ (trusted_setup)   │──────▶│                   │
└───────────────────┘       └───────────────────┘       └───────────────────┘
                                                                │
                                                                ▼
                                                        ┌───────────────────┐
│  Universal SRS    │──────▶│   PLONK Verifier  │──────▶│  Accept / Reject  │
│ (trusted_setup)   │       │   (plonk_prover)  │       │                   │
└───────────────────┘       └───────────────────┘       └───────────────────┘
```

## 5. Core Components in Detail

### 5.1 R1CS-to-PLONK Compiler [IMPLEMENTED]

This is a critical component that converts the existing R1CS representation into the arithmetic circuit format required for PLONK. An R1CS constraint `A(x) * B(x) - C(x) = 0` is translated into one or more PLONK gates `qL·a + qR·b + qO·c + qM·(a·b) + qC = 0`.

**Implementation**: `plonk/circuit_compiler.py` — includes GELU polynomial gates, sparse neuron optimization (CSWC), and network compilation.

### 5.2 KZG Polynomial Commitment [IMPLEMENTED]

The heart of PLONK is the KZG commitment scheme. **Implementation**: `plonk/kzg.py` with `commit()`, `create_proof()`, and `batch_verify()`.

### 5.3 PLONK Prover Protocol [IMPLEMENTED]

The 5-round Fiat-Shamir protocol is implemented in `plonk/plonk_prover.py`:

1. **Round 1 (Wire Commitments)**: Commit to witness polynomials a(x), b(x), c(x) via KZG.
2. **Round 2 (Permutation)**: Generate permutation polynomial z(x) ensuring copy constraints.
3. **Round 3 (Quotient)**: Combine all gate, permutation, and public input constraints into t(x). Compute quotient q(x) = t(x) / Z_H(x).
4. **Round 4 (Linearization)**: Linearize equations to reduce expensive opening operations.
5. **Round 5 (Opening)**: Generate KZG opening proofs at Fiat-Shamir challenge points.

## 6. Implementation Plan and Milestones

| Phase | Task | Duration | Status |
| :--- | :--- | :--- | :--- |
| 1 | **Foundations**: Polynomial arithmetic, FFT | 1 week | **✅ Done** (`plonk/polynomial.py`) |
| 2 | **Commitment**: KZG scheme | 1 week | **✅ Done** (`plonk/kzg.py`) |
| 3 | **Compiler**: R1CS-to-PLONK converter | 2 weeks | **✅ Done** (`plonk/circuit_compiler.py`) |
| 4 | **Prover**: PLONK prover implementation | 2 weeks | **✅ Done** (`plonk/plonk_prover.py`) |
| 5 | **Verifier**: PLONK verifier implementation | 1 week | **✅ Done** (in `plonk/plonk_prover.py`) |
| 6 | **Integration**: Connection to zkML system | 1 week | **⚠️ Partial** (`zkml_bridge.py` exists, `zkml.py` quarantined) |

## 7. Risks

- **Cryptographic complexity**: Mitigated via TDD, Poseidon test vectors, OpenAI co-review audit.
- **Performance**: Rust backend (`zkml_rs`) provides optimized field arithmetic via PyO3.
- **Trusted setup security**: `SRS.generate_insecure()` used for testing. Production use requires a ceremony SRS (see `trusted_setup.py`).

## 8. References

[1] Gabizon, A., Williamson, Z. J., & Ciobotaru, O. (2019). *PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge*. [ePrint 2019/953](https://eprint.iacr.org/2019/953)
[2] Kate, A., Zaverucha, G., & Goldberg, I. (2010). *Constant-Size Commitments to Polynomials and Their Applications*. [ASIACRYPT 2010](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)
[3] Vitalik Buterin (2019). *Understanding PLONK*. [blog.ethereum.org](https://blog.ethereum.org/2019/09/22/plonk-by-hand-part-1)
