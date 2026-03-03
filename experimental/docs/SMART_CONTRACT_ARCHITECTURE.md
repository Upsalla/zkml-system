# Smart Contract Architecture for On-Chain zkML Verification

**Author**: Upsalla
**Date**: January 26, 2026
**Status**: **PARTIALLY IMPLEMENTED** — Contracts exist in `contracts/` (`PlonkVerifier.sol`, `PlonkVerifierOptimized.sol`, `ZkMLVerifier.sol`, `ModelRegistry.sol`, `IZkMLVerifier.sol`). Not deployed or end-to-end tested.

## 1. Summary

This document describes the architecture of smart contracts for on-chain verification of zkML proofs. The goal is to create a decentralized, secure, and gas-efficient system on the Ethereum blockchain that enables trustless verification of machine learning inference integrity.

## 2. Problem Statement and Goals

Off-chain verification, even if cryptographically secure, requires trust in the verifying party. True decentralization and trustlessness requires on-chain verification.

**Project Goals**:

1. ⚠️ **Trustless verification**: Any participant can verify by sending a transaction. → Contracts written, not deployed.
2. ⚠️ **Gas efficiency**: Target < 500,000 gas per verification. → Contract exists, not benchmarked.
3. ✅ **Ethereum compatibility**: Solidity contracts using BN254 precompiles.
4. ✅ **Modularity**: Separate Verifier and Registry contracts.

## 3. On-Chain Architecture [IMPLEMENTED — contracts exist]

```
┌───────────────────────────┐
│      Application Layer    │
│ (e.g., dApp, Script)      │
└─────────────┬─────────────┘
              │ 1. submitProof(modelHash, proof, inputs)
              ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│   ModelRegistry.sol       │──────▶│   PlonkVerifier.sol       │
│ (Manages models & VKs)    │       │ (Core verifier logic)     │
└─────────────┬─────────────┘       └─────────────┬─────────────┘
              │                                   │
              ▼                                   ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│       Event Logs          │       │ Ethereum Precompiles      │
│ (ProofVerified, etc.)     │       │ (0x06, 0x07, 0x08)        │
└───────────────────────────┘       └───────────────────────────┘
```

### 3.1 Components

- **PlonkVerifier.sol** / **PlonkVerifierOptimized.sol**: Core contracts containing the PLONK `verify` function. Accept proof and public inputs, return `true`/`false`.
- **ZkMLVerifier.sol**: Higher-level wrapper contract.
- **ModelRegistry.sol**: Maps model hashes to verification keys (VKs). Allows adding new models without redeploying the verifier.
- **IZkMLVerifier.sol**: Interface definition.

### 3.2 Gas Analysis

| Operation | Gas Cost (approx.) | Count | Total Gas (approx.) |
| :--- | :--- | :--- | :--- |
| `ecPairing` (4 pairs) | `45,000 + 34,000 × 4` | 1 | 181,000 |
| `ecMul` (per public input) | `6,000` | `n` | `6,000 × n` |
| `ecAdd` (per public input) | `150` | `n` | `150 × n` |
| `SLOAD` (for VK) | `2,100` | `~5` | 10,500 |
| Other (Memory, etc.) | - | - | 20,000 |
| **Total (n=2 inputs)** | | | **~225,000** |

## 4. Implementation Milestones

| Phase | Task | Status |
| :--- | :--- | :--- |
| 1 | Pairing library (`PairingLib.sol`) | ⚠️ Likely in the contracts |
| 2 | Registry contract | ✅ `ModelRegistry.sol` exists |
| 3 | Verifier contract | ✅ `PlonkVerifier.sol` + optimized version exist |
| 4 | Deployment & test scripts | ❌ Not yet created |
| 5 | Gas optimization & audit | ❌ Not yet started |

## 5. Risks

- **Gas limit exceeded**: With many public inputs, costs could exceed block gas limit. → Mitigation: Limit public inputs; consider L2 rollups.
- **Precompile interface errors**: Low-level precompile interface is error-prone. → Mitigation: Reference established libraries (OpenZeppelin, other ZK projects).
- **Upgradeability**: Errors in verifier require redeployment. → Mitigation: Use proxy patterns (UUPS) for the verifier.

## 6. References

[1] Ethereum Foundation. *EIP-197: Precompiled contracts for optimal ate pairing check on the elliptic curve alt_bn128*. [eips.ethereum.org](https://eips.ethereum.org/EIPS/eip-197)
[2] Solidity Documentation. *Precompiled Contracts*. [docs.soliditylang.org](https://docs.soliditylang.org/en/latest/contracts.html#precompiled-contracts)
