# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.1.x   | :white_check_mark: |
| < 3.1   | :x:                |

## Known Cryptographic Limitations

This is a **research prototype**. The following are known limitations, not bugs:

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Deterministic SRS (`tau`) | Proofs forgeable by anyone who knows tau | Use MPC ceremony for production |
| Additive commitment (not Poseidon) | Binding under DLog, not collision-resistant | Sufficient for research; use Poseidon for production |
| No formal verification | Circuit correctness via tests only | 114 tests cover positive + negative cases |
| `zkml_pipeline.py` verifier | Structural checks only, no pairing verification | Use `PLONKVerifier` from `plonk_prover.py` |

## Reporting a Vulnerability

If you find a security issue:

1. **Do not open a public issue.**
2. Email: `dweyh@users.noreply.github.com`
3. Include: affected component, reproduction steps, impact assessment.
4. Expected response time: 7 days.

For cryptographic soundness concerns (e.g., constraint system bypasses),
please include a concrete witness that satisfies constraints but violates
the intended statement.
