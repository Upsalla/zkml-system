# Roadmap

Post-v0.1.0 development directions for zkml-system. Ordered by impact and feasibility.

---

## 7. Native Pairing (Remove py_ecc Dependency)

**Goal:** Make the BN254 stack 100% self-contained by replacing the `py_ecc` delegation in `pairing()` with the native Miller loop implementation.

**Current state:**
- `crypto/bn254/pairing.py` contains a complete Miller loop (line functions, Frobenius, final exponentiation) — 574 LOC
- The public `pairing()` function (line 441) delegates to `py_ecc` for correctness
- The native implementation exists but the final exponentiation hard part uses simplified formulas that need validation

**Work required:**
1. **Validate Frobenius coefficients** — The precomputed constants in `_get_frobenius_coeff_*()` need to be cross-checked against the BN254 specification. These are large field elements that must be exact.
2. **Fix `_frobenius_p2()`** — Currently uses `f ** (p * p)` which is a brute-force exponentiation. Replace with optimized Frobenius using precomputed coefficients: `Fp12.frobenius_map(2)`. This is the main performance bottleneck.
3. **Implement cyclotomic squaring** — The `_hard_part()` uses `cyclotomic_exp()` which needs an optimized implementation for elements in the cyclotomic subgroup (GΦ₆ ⊂ Fp12*). Cyclotomic squaring avoids half the Fp2 multiplications.
4. **Validate against test vectors** — Use the py_ecc pairing output as ground truth. Test: `e(G1, G2)`, `e(aP, bQ) == e(P, Q)^ab`, `e(P, -Q) * e(P, Q) == 1`.
5. **Remove py_ecc from `install_requires`** — Once validated, `py_ecc` becomes an optional test dependency.

**Estimated effort:** 2-3 days  
**Impact:** Eliminates the only external cryptographic dependency. Makes the project fully self-contained.

---

## 8. Rust Backend for Poseidon

**Goal:** Move the Poseidon permutation into Rust for 10-50x speedup on hash-intensive circuits.

**Current state:**
- `plonk/poseidon.py` implements the full Poseidon hash in pure Python (503 LOC)
- Each hash invocation runs 65 rounds × MDS multiplication × S-box (x^5)
- In-circuit: 1,722 gates per `hash_two()` call
- The Poseidon execution dominates circuit compilation time for TDA circuits (80% of gate count is Poseidon hashing)

**Work required:**
1. **Implement `poseidon_permutation()` in Rust** — Port the permutation loop (lines 185-246) into `rust_backend/src/`. Use BN254 Fr from the existing Rust field implementation.
2. **Generate and embed round constants** — The Grain-LFSR constant generation can stay in Python (one-time computation), but the 195 constants must be embedded as compile-time constants in Rust.
3. **PyO3 binding** — Expose `fn poseidon_hash_two(a: Fr, b: Fr) -> Fr` and `fn poseidon_hash_many(inputs: Vec<Fr>) -> Fr` via PyO3.
4. **Ensure offline/in-circuit consistency** — The Rust hash must produce identical output to the Python implementation. Use the existing self-test (lines 451-502) as the validation suite.
5. **Benchmark** — Target: < 1ms per `hash_two()` call (currently ~50ms in Python).

**Estimated effort:** 1 week  
**Impact:** Enables practical circuit sizes for real TDA fingerprints (currently limited by Poseidon computation time).

---

## 9. WASM Target (Browser Demos)

**Goal:** Compile the Rust backend to WebAssembly for in-browser proof generation.

**Current state:**
- `rust_backend/` uses PyO3 for Python bindings
- The core field arithmetic (`src/field.rs`) and MSM (`src/msm.rs`) are Rust-native
- No WASM target exists

**Work required:**
1. **Refactor Rust backend** — Separate the core cryptographic library from the PyO3 bindings. Create a `zkml-core` crate (pure Rust, no PyO3) and a `zkml-python` crate (PyO3 wrapper).
2. **Add `wasm-bindgen` target** — Create a `zkml-wasm` crate that wraps `zkml-core` with `#[wasm_bindgen]` exports. Expose: `prove(circuit_json: &str) -> JsValue`, `verify(proof_json: &str, circuit_json: &str) -> bool`.
3. **Circuit serialization** — Define a JSON schema for circuits that can be constructed in JavaScript and deserialized in Rust/WASM. The schema should cover gates, wires, selectors, and public inputs.
4. **Build pipeline** — `wasm-pack build --target web` → npm package. Include a minimal HTML demo page.
5. **Browser demo** — Interactive page: user inputs two weight matrices → TDA fingerprint → ZK proof → verify. All client-side.

**Estimated effort:** 3-5 days  
**Impact:** Massively increases visibility. Paper reviewers, conference demos, and GitHub visitors can try the system without installing anything. "Try it live" is the strongest GitHub marketing signal.

**Dependencies:** Benefits from #8 (Rust Poseidon) being completed first, as pure-Python Poseidon cannot run in WASM.

---

## 10. Formal Verification (Lean/Coq)

**Goal:** Formally verify the soundness of the PLONK implementation.

**Current state:**
- 114 tests cover circuit compilation, gate satisfiability, copy constraints, KZG commitments, and E2E roundtrips
- `plonk/soundness_analysis.md` documents the security model informally
- No formal proofs exist

**What to verify:**
1. **Gate constraint completeness** — For every valid witness, the gate equation `q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C = 0` is always satisfiable (completeness).
2. **Permutation argument soundness** — The grand product accumulator `z(X)` correctly encodes the copy constraints, and any cheating prover is caught with overwhelming probability.
3. **Quotient polynomial degree bound** — Prove that `t(X)` has degree < 3n when the constraint system is satisfied, and degree ≥ 3n otherwise (Schwartz-Zippel).
4. **Fiat-Shamir security** — The transcript produces challenges with sufficient entropy, and the binding property holds.

**Approach options:**
- **Lean 4** — Best tooling, growing ZK verification community (see `lean-crypto` projects). Estimated: 4-8 weeks for the algebraic core.
- **Coq** — More mature, but steeper learning curve. The `Fiat-Crypto` project provides reusable field arithmetic proofs.
- **Partial verification** — Start with just the gate equation and permutation (items 1-2 above). Skip KZG and Fiat-Shamir initially.

**Estimated effort:** 4-8 weeks (partial), months (full)  
**Impact:** Academic credibility. Required for any serious security claim. Would be a differentiator — very few ZK implementations have formal verification.

---

## Future Direction: Agent System Integration (SAF)

> This is a speculative direction for integrating zkml-system with autonomous agent frameworks.

### Feasible Use Cases

| Use Case | Circuit Size | Feasibility |
|----------|-------------|-------------|
| **Agent Model Attestation** — Prove "this agent uses model X" via TDA fingerprint without revealing weights | ~500 gates | ✅ Works now |
| **Verifiable Audit Trail** — Poseidon hash-chain over agent actions, ZK-prove log consistency | ~1,700 gates/entry | ✅ Feasible |
| **Tool Policy Compliance** — Prove "sandbox policy was correctly applied" for a small ruleset | ~100-500 gates | ✅ Feasible |

### Not Yet Feasible

| Use Case | Gate Count | Gap |
|----------|-----------|-----|
| **Verifiable LLM Inference** — Prove full forward-pass correctness | ~100M+ gates | 6 orders of magnitude |
| **Real-Time Verification** — Prove each agent action in-line | Millisecond budget | PLONK proves in seconds |

### Integration Path

If pursued, the recommended approach:

1. **Phase 1:** Agent identity via TDA fingerprinting (uses existing `HybridBridge`)
2. **Phase 2:** Poseidon-based audit log hash chain
3. **Phase 3:** Small policy compliance circuits (tool whitelist, rate limits)
4. **Phase 4:** (Long-term) Recursive proof aggregation for batch verification
