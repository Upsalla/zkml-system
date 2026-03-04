# Experimental Modules

These modules contain experimental optimization approaches that are **not verified** and **not integrated** into the core system. They are preserved for reference and potential future development.

> **Note:** None of these modules are required for the core zkml-system functionality. All verified features live in the parent directory.

| Module | Description | Contents | Status |
|--------|-------------|----------|--------|
| `audit/` | Security fuzzing, gas analysis, and profiling tools | `fuzzing.py`, `gas_audit.py`, `profiling.py`, audit report | Reference |
| `benchmarks/` | Performance benchmarks for activations, sparse operations, and E2E | 5 benchmark scripts + validation report | Reference |
| `compressed_sensing/` | CSWC (Compressed Sensing Witness Compression) for reducing witness size | Feasibility analysis, benchmarks, design docs | Untested |
| `contracts/` | Solidity smart contract templates for on-chain PLONK verification | `PlonkVerifier.sol`, `ModelRegistry.sol`, `IZkMLVerifier.sol` | Stub |
| `demo/` | End-to-end demos including MNIST inference | `e2e_demo.py`, `mnist_demo.py` | Reference |
| `deployment/` | REST API server (FastAPI) + CLI tool for zkml-as-a-service | Server, CLI, deploy script | Stub |
| `docs/` | Technical documentation, architecture diagrams, presentations | BN254 architecture, Conv2D design, presentation slides | Reference |
| `research/` | Research notes on compressed sensing, holographic proofs | Feasibility analyses and architecture designs | Notes |
| `tropical/` | Tropical geometry approach for MaxPool/Softmax circuit optimization | `tropical_ops.py`, integration benchmark, final report | Untested |
| `wavelet/` | Haar wavelet transform for witness batching / compression | `haar_transform.py`, benchmark results | Untested |
