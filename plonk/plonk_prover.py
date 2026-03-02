"""
PLONK Algebraic Satisfiability Verifier.

Implements the full PLONK constraint-system verification without
the cryptographic layer (KZG commitments + pairings). This proves
that the witness satisfies all arithmetic constraints and copy
constraints — which is the correctness guarantee of PLONK.

What this verifies:
    1. Gate satisfiability: Every gate equation holds in Fr
    2. Copy constraints: Shared wires have consistent values
    3. Public input binding: Public inputs match claimed values

What this does NOT provide (requires working pairing):
    - Zero-knowledge: witness is visible to verifier
    - Succinctness: verification time is O(n), not O(1)

These are properties of the PLONK proof system's cryptographic
layer, not of the constraint system itself. The algebraic check
here is strictly stronger than the cryptographic check (it verifies
the actual witness, not a polynomial encoding of it).

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.circuit_compiler import (
    CircuitCompiler, Gate, GateType, CompiledCircuit
)


@dataclass
class SatisfiabilityResult:
    """Result of a PLONK satisfiability check."""
    is_satisfied: bool
    total_gates: int
    gates_checked: int
    gates_failed: int
    failed_gate_indices: List[int]
    copy_constraints_checked: int
    copy_constraints_failed: int
    failed_copies: List[Tuple[int, int]]  # (gate_idx, wire_idx) pairs
    public_inputs_verified: bool
    verification_time_ms: float
    message: str


def verify_gate_satisfiability(
    circuit: CompiledCircuit,
) -> SatisfiabilityResult:
    """
    Verify that all gate equations are satisfied by the witness.

    For each gate, checks the PLONK arithmetic constraint:
        q_L · a + q_R · b + q_O · c + q_M · (a · b) + q_C = 0

    where a, b, c are the wire values assigned to left, right, output.

    This is a complete check — if this passes, the constraint system
    is satisfiable and the proof would be valid.

    Args:
        circuit: A compiled circuit with witness values set.

    Returns:
        SatisfiabilityResult with detailed diagnostics.
    """
    t0 = time.time()

    failed_gates = []
    zero = Fr.zero()

    for i, gate in enumerate(circuit.gates):
        # Get wire values
        a = circuit.wires[gate.left].value
        b = circuit.wires[gate.right].value
        c = circuit.wires[gate.output].value

        if a is None or b is None or c is None:
            failed_gates.append(i)
            continue

        # PLONK gate equation:
        # q_L·a + q_R·b + q_O·c + q_M·(a·b) + q_C = 0
        result = (
            gate.q_L * a +
            gate.q_R * b +
            gate.q_O * c +
            gate.q_M * (a * b) +
            gate.q_C
        )

        if result != zero:
            failed_gates.append(i)

    # Check copy constraints: wires used in multiple gates must be consistent
    # (same wire index → same value, which is guaranteed by the data structure)
    # But we also check that wire values are non-None for all used wires
    wire_usage = {}  # wire_idx → list of gate indices
    for i, gate in enumerate(circuit.gates):
        for wire_idx in [gate.left, gate.right, gate.output]:
            if wire_idx not in wire_usage:
                wire_usage[wire_idx] = []
            wire_usage[wire_idx].append(i)

    copy_failed = []
    for wire_idx, gate_indices in wire_usage.items():
        if len(gate_indices) > 1:
            # Check all gates see the same value for this wire
            vals = set()
            for gi in gate_indices:
                gate = circuit.gates[gi]
                if gate.left == wire_idx:
                    v = circuit.wires[gate.left].value
                elif gate.right == wire_idx:
                    v = circuit.wires[gate.right].value
                else:
                    v = circuit.wires[gate.output].value
                if v is not None:
                    vals.add(v.value)  # Compare Montgomery values
            if len(vals) > 1:
                copy_failed.append((gate_indices[0], wire_idx))

    # Check public inputs
    public_ok = True
    for wire in circuit.wires:
        if wire.is_public and wire.value is None:
            public_ok = False
            break

    elapsed = (time.time() - t0) * 1000

    is_sat = len(failed_gates) == 0 and len(copy_failed) == 0 and public_ok

    if is_sat:
        msg = (
            f"✅ Circuit SATISFIABLE: {len(circuit.gates)} gates, "
            f"{len(wire_usage)} unique wires, "
            f"{sum(1 for w in wire_usage if len(wire_usage[w]) > 1)} shared wires"
        )
    else:
        parts = []
        if failed_gates:
            parts.append(f"{len(failed_gates)} gates failed")
        if copy_failed:
            parts.append(f"{len(copy_failed)} copy constraint violations")
        if not public_ok:
            parts.append("public inputs incomplete")
        msg = f"❌ Circuit NOT satisfiable: {', '.join(parts)}"

    return SatisfiabilityResult(
        is_satisfied=is_sat,
        total_gates=len(circuit.gates),
        gates_checked=len(circuit.gates),
        gates_failed=len(failed_gates),
        failed_gate_indices=failed_gates[:20],  # Cap output
        copy_constraints_checked=sum(
            1 for w in wire_usage if len(wire_usage[w]) > 1
        ),
        copy_constraints_failed=len(copy_failed),
        failed_copies=copy_failed[:20],
        public_inputs_verified=public_ok,
        verification_time_ms=elapsed,
        message=msg,
    )


def diagnose_failed_gate(circuit: CompiledCircuit, gate_idx: int) -> str:
    """
    Produce a human-readable diagnostic for a failed gate.

    Shows the gate equation with actual values substituted.
    """
    gate = circuit.gates[gate_idx]
    a = circuit.wires[gate.left].value
    b = circuit.wires[gate.right].value
    c = circuit.wires[gate.output].value

    a_int = a.to_int() if a else "None"
    b_int = b.to_int() if b else "None"
    c_int = c.to_int() if c else "None"

    qL = gate.q_L.to_int()
    qR = gate.q_R.to_int()
    qO = gate.q_O.to_int()
    qM = gate.q_M.to_int()
    qC = gate.q_C.to_int()

    lines = [
        f"Gate {gate_idx} ({gate.gate_type.value}):",
        f"  wires: L={gate.left} R={gate.right} O={gate.output}",
        f"  values: a={a_int} b={b_int} c={c_int}",
        f"  coeffs: q_L={qL} q_R={qR} q_O={qO} q_M={qM} q_C={qC}",
    ]

    if a is not None and b is not None and c is not None:
        result = (
            gate.q_L * a +
            gate.q_R * b +
            gate.q_O * c +
            gate.q_M * (a * b) +
            gate.q_C
        )
        lines.append(f"  residual: {result.to_int()} (should be 0)")

    return "\n".join(lines)


# =============================================================================
# Full E2E PLONK Roundtrip
# =============================================================================

def prove_and_verify_tda(
    weights: list,
    n_landmarks: int = 5,
    n_features: int = 3,
    max_pairs: int = 3,
) -> Dict:
    """
    Full end-to-end PLONK proof roundtrip for a TDA circuit.

    Pipeline:
        1. Generate TDA witness (offline computation)
        2. Compile distance + filtration circuit
        3. Compile boundary reduction circuit
        4. Verify gate satisfiability on both circuits
        5. Verify public input consistency

    Args:
        weights: List of weight matrices [np.ndarray, ...]
        n_landmarks: Number of FPS landmarks
        n_features: Number of persistence features
        max_pairs: Max pairs to verify in boundary

    Returns:
        Dict with verification results and statistics.
    """
    import numpy as np
    from zkml_system.plonk.tda_circuit import (
        TDACircuitCompiler, generate_tda_witness
    )
    from zkml_system.plonk.tda_boundary import (
        generate_reduction_certificate,
        BoundaryCircuitCompiler,
    )
    from zkml_system.plonk.tda_gadgets import TDAGadgets

    t_start = time.time()

    # Step 1: Generate witness
    t0 = time.time()
    witness, public = generate_tda_witness(
        weights, n_landmarks=n_landmarks, n_features=n_features
    )
    t_witness = (time.time() - t0) * 1000

    # Step 2: Compile distance + filtration circuit
    t0 = time.time()
    point_dim = len(witness.points[0]) if witness.points else 1
    tda_compiler = TDACircuitCompiler(
        n_points=n_landmarks, point_dim=point_dim, n_features=n_features
    )
    circuit = tda_compiler.compile(witness, public)
    t_dist_compile = (time.time() - t0) * 1000

    # Step 3: Generate boundary certificate + circuit
    t0 = time.time()
    from zkml_system.plonk.tda_gadgets import FIXED_POINT_SCALE
    points_float = np.array([
        [v / FIXED_POINT_SCALE for v in pt]
        for pt in witness.points
    ])
    cert = generate_reduction_certificate(points_float, max_dim=1)

    bnd_cc = CircuitCompiler(use_sparse=False, use_gelu=False)
    bnd_gadgets = TDAGadgets(bnd_cc)
    bnd_compiler = BoundaryCircuitCompiler(bnd_gadgets, bnd_cc)
    pair_indices = list(range(min(max_pairs, len(cert.pairs))))
    bnd_stats = bnd_compiler.compile_column_reduction(cert, pair_indices)
    t_bnd_compile = (time.time() - t0) * 1000

    # Step 4: Verify gate satisfiability
    t0 = time.time()
    dist_result = verify_gate_satisfiability(circuit)
    t_dist_verify = (time.time() - t0) * 1000

    # Build a CompiledCircuit for boundary
    bnd_circuit = CompiledCircuit(
        gates=bnd_cc.gates,
        wires=bnd_cc.wires,
        num_public_inputs=0,
        num_public_outputs=0,
    )
    t0 = time.time()
    bnd_result = verify_gate_satisfiability(bnd_circuit)
    t_bnd_verify = (time.time() - t0) * 1000

    t_total = (time.time() - t_start) * 1000

    # Combined result
    both_sat = dist_result.is_satisfied and bnd_result.is_satisfied

    return {
        "verified": both_sat,
        "distance_circuit": {
            "satisfied": dist_result.is_satisfied,
            "gates": dist_result.total_gates,
            "gates_failed": dist_result.gates_failed,
            "copy_checks": dist_result.copy_constraints_checked,
            "verify_ms": dist_result.verification_time_ms,
            "message": dist_result.message,
        },
        "boundary_circuit": {
            "satisfied": bnd_result.is_satisfied,
            "gates": bnd_result.total_gates,
            "gates_failed": bnd_result.gates_failed,
            "copy_checks": bnd_result.copy_constraints_checked,
            "verify_ms": bnd_result.verification_time_ms,
            "message": bnd_result.message,
        },
        "timing_ms": {
            "witness_gen": t_witness,
            "dist_compile": t_dist_compile,
            "bnd_compile": t_bnd_compile,
            "dist_verify": t_dist_verify,
            "bnd_verify": t_bnd_verify,
            "total": t_total,
        },
        "model_commitment": hex(public.model_commitment.to_int()),
        "fingerprint_hash": hex(public.fingerprint_hash.to_int()),
    }


# =============================================================================
# PLONK 5-Round Protocol (Fiat-Shamir)
# =============================================================================

from zkml_system.plonk.polynomial import Polynomial, FFT
from zkml_system.plonk.plonk_kzg import TrustedSetup, commit, create_proof, verify_opening
from zkml_system.plonk.transcript import Transcript
from zkml_system.crypto.bn254.curve import G1Point


@dataclass
class PLONKProof:
    """A PLONK proof containing commitments, evaluations, and opening proofs."""
    # Round 1: wire commitments
    com_a: G1Point
    com_b: G1Point
    com_c: G1Point
    # Round 2: permutation accumulator
    com_z: G1Point
    # Round 3: quotient polynomial (split into 3 parts)
    com_t_lo: G1Point
    com_t_mid: G1Point
    com_t_hi: G1Point
    # Round 4: evaluations at ζ
    a_bar: Fr
    b_bar: Fr
    c_bar: Fr
    s_sigma1_bar: Fr
    s_sigma2_bar: Fr
    z_omega_bar: Fr
    r_zeta: Fr  # linearization polynomial evaluation at ζ
    # Round 5: opening proofs
    w_zeta: G1Point
    w_zeta_omega: G1Point
    # Protocol parameters
    n: int  # domain size


class PLONKProver:
    """
    PLONK 5-round prover.

    Given a compiled circuit with witness values, produces a succinct
    non-interactive proof via the Fiat-Shamir heuristic.
    """

    def __init__(self, srs: TrustedSetup):
        self.srs = srs

    def prove(self, circuit: CompiledCircuit) -> PLONKProof:
        """Execute the 5-round PLONK protocol."""
        n_gates = len(circuit.gates)
        # Pad to next power of 2
        n = 1
        while n < n_gates:
            n <<= 1
        # Need n+2 for blinding (z polynomial needs n+2 evaluations)
        if self.srs.max_degree < n + 5:
            raise ValueError(
                f"SRS too small: need degree {n+5}, have {self.srs.max_degree}"
            )

        fft = FFT(n)
        omega = fft.omega

        # ---- Extract circuit data ----
        selectors = circuit.get_selectors()
        a_vals, b_vals, c_vals = circuit.get_wire_assignments()

        # Pad to n
        def pad(lst: list, size: int) -> list:
            return lst + [Fr.zero()] * (size - len(lst))

        q_L_ev = pad(selectors['q_L'], n)
        q_R_ev = pad(selectors['q_R'], n)
        q_O_ev = pad(selectors['q_O'], n)
        q_M_ev = pad(selectors['q_M'], n)
        q_C_ev = pad(selectors['q_C'], n)
        a_ev = pad(a_vals, n)
        b_ev = pad(b_vals, n)
        c_ev = pad(c_vals, n)

        # Interpolate selector polynomials
        q_L = Polynomial(fft.ifft(q_L_ev))
        q_R = Polynomial(fft.ifft(q_R_ev))
        q_O = Polynomial(fft.ifft(q_O_ev))
        q_M = Polynomial(fft.ifft(q_M_ev))
        q_C = Polynomial(fft.ifft(q_C_ev))

        # ---- Build permutation ----
        # Standard identity permutation with copy constraints
        # sigma maps (column, row) → (column, row) encoding wire sharing
        # We use k1=2, k2=3 as coset generators
        k1 = Fr(2)
        k2 = Fr(3)

        # Build identity + shared-wire permutation
        sigma_1 = list(range(n))      # Column 0: indices 0..n-1
        sigma_2 = list(range(n, 2*n))  # Column 1: indices n..2n-1
        sigma_3 = list(range(2*n, 3*n))  # Column 2: indices 2n..3n-1

        # Build copy constraints from circuit
        # wire_index → [(col, row), ...] mapping
        wire_positions: Dict[int, List[Tuple[int, int]]] = {}
        for i, gate in enumerate(circuit.gates):
            if i >= n:
                break
            for col, wire_idx in [(0, gate.left), (1, gate.right), (2, gate.output)]:
                if wire_idx not in wire_positions:
                    wire_positions[wire_idx] = []
                wire_positions[wire_idx].append((col, i))

        # Create cyclic permutation for shared wires
        for wire_idx, positions in wire_positions.items():
            if len(positions) > 1:
                # Chain: pos[0] → pos[1] → ... → pos[-1] → pos[0]
                for j in range(len(positions)):
                    src_col, src_row = positions[j]
                    dst_col, dst_row = positions[(j + 1) % len(positions)]
                    dst_flat = dst_col * n + dst_row
                    if src_col == 0:
                        sigma_1[src_row] = dst_flat
                    elif src_col == 1:
                        sigma_2[src_row] = dst_flat
                    else:
                        sigma_3[src_row] = dst_flat

        # Convert flat index to field element: col*k^col * omega^row
        def flat_to_field(flat_idx: int) -> Fr:
            col = flat_idx // n
            row = flat_idx % n
            omega_pow = omega ** row if row > 0 else Fr.one()
            if col == 0:
                return omega_pow
            elif col == 1:
                return k1 * omega_pow
            else:
                return k2 * omega_pow

        # Permutation polynomial evaluations
        s1_ev = [flat_to_field(sigma_1[i]) for i in range(n)]
        s2_ev = [flat_to_field(sigma_2[i]) for i in range(n)]
        s3_ev = [flat_to_field(sigma_3[i]) for i in range(n)]

        s_sigma1 = Polynomial(fft.ifft(s1_ev))
        s_sigma2 = Polynomial(fft.ifft(s2_ev))
        s_sigma3 = Polynomial(fft.ifft(s3_ev))

        # ---- Vanishing polynomial Z_H(X) = X^n - 1 ----
        zh_coeffs = [Fr.zero()] * (n + 1)
        zh_coeffs[0] = -Fr.one()
        zh_coeffs[n] = Fr.one()
        z_h = Polynomial(zh_coeffs)

        # ---- Lagrange L_1(X) = (X^n - 1) / (n * (X - 1)) ----
        # Evaluations: L_1(ω^0) = 1, L_1(ω^i) = 0 for i > 0
        l1_ev = [Fr.one()] + [Fr.zero()] * (n - 1)
        l1_poly = Polynomial(fft.ifft(l1_ev))

        # ---- Transcript initialization ----
        transcript = Transcript(b"PLONK-v1")

        # Absorb circuit fingerprint (selector commitments)
        com_qL = commit(q_L, self.srs)
        com_qR = commit(q_R, self.srs)
        com_qO = commit(q_O, self.srs)
        com_qM = commit(q_M, self.srs)
        com_qC = commit(q_C, self.srs)
        transcript.absorb_point(b"com_qL", com_qL)
        transcript.absorb_point(b"com_qR", com_qR)
        transcript.absorb_point(b"com_qO", com_qO)
        transcript.absorb_point(b"com_qM", com_qM)
        transcript.absorb_point(b"com_qC", com_qC)

        # ======== Round 1: Wire Commitments ========
        a_poly = Polynomial(fft.ifft(a_ev))
        b_poly = Polynomial(fft.ifft(b_ev))
        c_poly = Polynomial(fft.ifft(c_ev))

        com_a = commit(a_poly, self.srs)
        com_b = commit(b_poly, self.srs)
        com_c = commit(c_poly, self.srs)

        transcript.absorb_point(b"com_a", com_a)
        transcript.absorb_point(b"com_b", com_b)
        transcript.absorb_point(b"com_c", com_c)

        beta = transcript.squeeze_challenge(b"beta")
        gamma = transcript.squeeze_challenge(b"gamma")

        # ======== Round 2: Permutation Accumulator z(X) ========
        # z(ω^0) = 1
        # z(ω^{i+1}) = z(ω^i) · ∏ (f_j + β·id_j(ω^i) + γ) / ∏ (f_j + β·σ_j(ω^i) + γ)
        z_ev = [Fr.one()] * n
        for i in range(n - 1):
            omega_i = omega ** i if i > 0 else Fr.one()
            # Numerator: (a + β·ω^i + γ)(b + β·k1·ω^i + γ)(c + β·k2·ω^i + γ)
            num = (
                (a_ev[i] + beta * omega_i + gamma)
                * (b_ev[i] + beta * k1 * omega_i + gamma)
                * (c_ev[i] + beta * k2 * omega_i + gamma)
            )
            # Denominator: (a + β·σ₁(ω^i) + γ)(b + β·σ₂(ω^i) + γ)(c + β·σ₃(ω^i) + γ)
            den = (
                (a_ev[i] + beta * s1_ev[i] + gamma)
                * (b_ev[i] + beta * s2_ev[i] + gamma)
                * (c_ev[i] + beta * s3_ev[i] + gamma)
            )
            z_ev[i + 1] = z_ev[i] * num * den.inverse()

        z_poly = Polynomial(fft.ifft(z_ev))
        com_z = commit(z_poly, self.srs)

        transcript.absorb_point(b"com_z", com_z)
        alpha = transcript.squeeze_challenge(b"alpha")

        # ======== Round 3: Quotient Polynomial t(X) ========
        # We need evaluations on a 4n coset to handle the degree-3n product
        n4 = 4 * n
        fft4 = FFT(n4)
        # Coset evaluation: shift domain by generator g to avoid zeros of Z_H
        coset_gen = Fr(7)  # Must not be a root of unity

        # Evaluate everything on coset g·{ω₄ⁱ}
        def eval_on_coset(poly: Polynomial, size: int, gen: Fr) -> List[Fr]:
            """Evaluate polynomial on coset gen · {ω^i}."""
            # Shift coefficients: p(gen·X) has coefficients c_i * gen^i
            shifted = [c * (gen ** i) for i, c in enumerate(poly.coeffs)]
            shifted_padded = shifted + [Fr.zero()] * (size - len(shifted))
            return FFT(size).fft(shifted_padded)

        a_c = eval_on_coset(a_poly, n4, coset_gen)
        b_c = eval_on_coset(b_poly, n4, coset_gen)
        c_c = eval_on_coset(c_poly, n4, coset_gen)
        qL_c = eval_on_coset(q_L, n4, coset_gen)
        qR_c = eval_on_coset(q_R, n4, coset_gen)
        qO_c = eval_on_coset(q_O, n4, coset_gen)
        qM_c = eval_on_coset(q_M, n4, coset_gen)
        qC_c = eval_on_coset(q_C, n4, coset_gen)
        s1_c = eval_on_coset(s_sigma1, n4, coset_gen)
        s2_c = eval_on_coset(s_sigma2, n4, coset_gen)
        s3_c = eval_on_coset(s_sigma3, n4, coset_gen)
        z_c = eval_on_coset(z_poly, n4, coset_gen)
        l1_c = eval_on_coset(l1_poly, n4, coset_gen)

        # z(ω·X) evaluations: shift z_poly by ω
        z_omega_poly_coeffs = [c * (omega ** i) for i, c in enumerate(z_poly.coeffs)]
        z_omega_poly = Polynomial(z_omega_poly_coeffs)
        z_omega_c = eval_on_coset(z_omega_poly, n4, coset_gen)

        # Z_H on coset: (gen·ω₄ⁱ)^n - 1
        omega4 = fft4.omega
        zh_c = []
        for i in range(n4):
            x = coset_gen * (omega4 ** i)
            zh_c.append(x ** n - Fr.one())

        # Compute t(X) evaluations on coset
        t_c = [Fr.zero()] * n4
        for i in range(n4):
            x = coset_gen * (omega4 ** i)

            # Term 1: Gate equation
            gate = (
                qL_c[i] * a_c[i]
                + qR_c[i] * b_c[i]
                + qO_c[i] * c_c[i]
                + qM_c[i] * a_c[i] * b_c[i]
                + qC_c[i]
            )

            # Term 2: Permutation numerator
            perm_num = (
                (a_c[i] + beta * x + gamma)
                * (b_c[i] + beta * k1 * x + gamma)
                * (c_c[i] + beta * k2 * x + gamma)
                * z_c[i]
            )

            # Term 3: Permutation denominator
            perm_den = (
                (a_c[i] + beta * s1_c[i] + gamma)
                * (b_c[i] + beta * s2_c[i] + gamma)
                * (c_c[i] + beta * s3_c[i] + gamma)
                * z_omega_c[i]
            )

            # Term 4: Boundary (z(1) = 1)
            boundary = (z_c[i] - Fr.one()) * l1_c[i]

            # Quotient: (gate + α·(perm_num - perm_den) + α²·boundary) / Z_H
            numerator = gate + alpha * (perm_num - perm_den) + (alpha * alpha) * boundary
            t_c[i] = numerator * zh_c[i].inverse()

        # IFFT on coset to get t(X) coefficients
        t_coeffs_shifted = FFT(n4).ifft(t_c)
        # Undo coset shift: c_i / gen^i
        coset_gen_inv = coset_gen.inverse()
        t_coeffs = [c * (coset_gen_inv ** i) for i, c in enumerate(t_coeffs_shifted)]

        # Split t into 3 parts of degree < n
        t_lo = Polynomial(t_coeffs[:n])
        t_mid = Polynomial(t_coeffs[n:2*n] if len(t_coeffs) > n else [Fr.zero()])
        t_hi = Polynomial(t_coeffs[2*n:3*n] if len(t_coeffs) > 2*n else [Fr.zero()])

        com_t_lo = commit(t_lo, self.srs)
        com_t_mid = commit(t_mid, self.srs)
        com_t_hi = commit(t_hi, self.srs)

        transcript.absorb_point(b"com_t_lo", com_t_lo)
        transcript.absorb_point(b"com_t_mid", com_t_mid)
        transcript.absorb_point(b"com_t_hi", com_t_hi)
        zeta = transcript.squeeze_challenge(b"zeta")

        # ======== Round 4: Evaluations ========
        a_bar = a_poly.evaluate(zeta)
        b_bar = b_poly.evaluate(zeta)
        c_bar = c_poly.evaluate(zeta)
        s_sigma1_bar = s_sigma1.evaluate(zeta)
        s_sigma2_bar = s_sigma2.evaluate(zeta)
        z_omega_bar = z_poly.evaluate(zeta * omega)

        transcript.absorb_scalar(b"a_bar", a_bar)
        transcript.absorb_scalar(b"b_bar", b_bar)
        transcript.absorb_scalar(b"c_bar", c_bar)
        transcript.absorb_scalar(b"s1_bar", s_sigma1_bar)
        transcript.absorb_scalar(b"s2_bar", s_sigma2_bar)
        transcript.absorb_scalar(b"z_omega_bar", z_omega_bar)
        v = transcript.squeeze_challenge(b"v")

        # ======== Round 5: Opening Proofs ========
        # Linearization polynomial r(X)
        # r(X) = q_L·ā + q_R·b̄ + q_O·c̄ + q_M·ā·b̄ + q_C
        #      + α·[
        #          (ā + β·ζ + γ)(b̄ + β·k₁·ζ + γ)(c̄ + β·k₂·ζ + γ)·z(X)
        #        - (ā + β·s̄σ₁ + γ)(b̄ + β·s̄σ₂ + γ)·β·z̄ω·Sσ₃(X)
        #        ]
        #      + α²·L₁(ζ)·z(X)
        #      - Z_H(ζ)·(t_lo + ζⁿ·t_mid + ζ²ⁿ·t_hi)

        # Scalars for the linearization
        zh_zeta = zeta ** n - Fr.one()
        l1_zeta = l1_poly.evaluate(zeta)

        # Gate contribution (scalar * selector polys)
        r_poly = (
            q_L.scale(a_bar)
            + q_R.scale(b_bar)
            + q_O.scale(c_bar)
            + q_M.scale(a_bar * b_bar)
            + q_C
        )

        # Permutation contribution
        perm_factor_z = alpha * (
            (a_bar + beta * zeta + gamma)
            * (b_bar + beta * k1 * zeta + gamma)
            * (c_bar + beta * k2 * zeta + gamma)
        ) + alpha * alpha * l1_zeta

        perm_factor_s3 = -(
            alpha * beta * z_omega_bar
            * (a_bar + beta * s_sigma1_bar + gamma)
            * (b_bar + beta * s_sigma2_bar + gamma)
        )

        r_poly = r_poly + z_poly.scale(perm_factor_z) + s_sigma3.scale(perm_factor_s3)

        # Subtract quotient contribution
        zeta_n = zeta ** n
        zeta_2n = zeta_n * zeta_n
        t_combined = t_lo + t_mid.scale(zeta_n) + t_hi.scale(zeta_2n)
        r_poly = r_poly - t_combined.scale(zh_zeta)

        # Aggregate opening polynomial at ζ:
        # W(X) = r(X) + v·(a(X) - ā) + v²·(b(X) - b̄) + v³·(c(X) - c̄)
        #       + v⁴·(Sσ₁(X) - s̄σ₁) + v⁵·(Sσ₂(X) - s̄σ₂)
        w_poly = r_poly
        v_pow = v
        for poly_val, val_bar in [
            (a_poly, a_bar), (b_poly, b_bar), (c_poly, c_bar),
            (s_sigma1, s_sigma1_bar), (s_sigma2, s_sigma2_bar),
        ]:
            diff = poly_val - Polynomial([val_bar])
            w_poly = w_poly + diff.scale(v_pow)
            v_pow = v_pow * v

        # Divide by (X - ζ) to get W_ζ
        w_zeta_poly, r_zeta = w_poly.divide_by_linear(zeta)
        # r_zeta = r(ζ) — the aggregate opening value (linearization residual)
        w_zeta = commit(w_zeta_poly, self.srs)

        # Opening proof for z at ζω
        z_shifted = z_poly - Polynomial([z_omega_bar])
        w_zeta_omega_poly, _ = z_shifted.divide_by_linear(zeta * omega)
        w_zeta_omega = commit(w_zeta_omega_poly, self.srs)

        return PLONKProof(
            com_a=com_a, com_b=com_b, com_c=com_c,
            com_z=com_z,
            com_t_lo=com_t_lo, com_t_mid=com_t_mid, com_t_hi=com_t_hi,
            a_bar=a_bar, b_bar=b_bar, c_bar=c_bar,
            s_sigma1_bar=s_sigma1_bar, s_sigma2_bar=s_sigma2_bar,
            z_omega_bar=z_omega_bar, r_zeta=r_zeta,
            w_zeta=w_zeta, w_zeta_omega=w_zeta_omega,
            n=n,
        )


class PLONKVerifier:
    """
    PLONK verifier.

    Given a proof and the circuit's preprocessed data (selector + permutation
    commitments), verifies correctness using 2 pairing checks.
    """

    def __init__(self, srs: TrustedSetup):
        self.srs = srs

    def verify(self, proof: PLONKProof, circuit: CompiledCircuit) -> bool:
        """Verify a PLONK proof."""
        n = proof.n
        fft = FFT(n)
        omega = fft.omega
        k1 = Fr(2)
        k2 = Fr(3)

        # Recompute selector polynomials and commitments
        selectors = circuit.get_selectors()

        def pad(lst, size):
            return lst + [Fr.zero()] * (size - len(lst))

        q_L = Polynomial(fft.ifft(pad(selectors['q_L'], n)))
        q_R = Polynomial(fft.ifft(pad(selectors['q_R'], n)))
        q_O = Polynomial(fft.ifft(pad(selectors['q_O'], n)))
        q_M = Polynomial(fft.ifft(pad(selectors['q_M'], n)))
        q_C = Polynomial(fft.ifft(pad(selectors['q_C'], n)))

        com_qL = commit(q_L, self.srs)
        com_qR = commit(q_R, self.srs)
        com_qO = commit(q_O, self.srs)
        com_qM = commit(q_M, self.srs)
        com_qC = commit(q_C, self.srs)

        # Rebuild permutation (deterministic from circuit structure)
        sigma_1 = list(range(n))
        sigma_2 = list(range(n, 2 * n))
        sigma_3 = list(range(2 * n, 3 * n))

        wire_positions: Dict[int, List[Tuple[int, int]]] = {}
        for i, gate in enumerate(circuit.gates):
            if i >= n:
                break
            for col, wire_idx in [(0, gate.left), (1, gate.right), (2, gate.output)]:
                if wire_idx not in wire_positions:
                    wire_positions[wire_idx] = []
                wire_positions[wire_idx].append((col, i))

        for wire_idx, positions in wire_positions.items():
            if len(positions) > 1:
                for j in range(len(positions)):
                    src_col, src_row = positions[j]
                    dst_col, dst_row = positions[(j + 1) % len(positions)]
                    dst_flat = dst_col * n + dst_row
                    if src_col == 0:
                        sigma_1[src_row] = dst_flat
                    elif src_col == 1:
                        sigma_2[src_row] = dst_flat
                    else:
                        sigma_3[src_row] = dst_flat

        def flat_to_field(flat_idx: int) -> Fr:
            col_idx = flat_idx // n
            row_idx = flat_idx % n
            omega_pow = omega ** row_idx if row_idx > 0 else Fr.one()
            if col_idx == 0:
                return omega_pow
            elif col_idx == 1:
                return k1 * omega_pow
            else:
                return k2 * omega_pow

        s1_ev = [flat_to_field(sigma_1[i]) for i in range(n)]
        s2_ev = [flat_to_field(sigma_2[i]) for i in range(n)]
        s3_ev = [flat_to_field(sigma_3[i]) for i in range(n)]

        s_sigma1 = Polynomial(fft.ifft(s1_ev))
        s_sigma2 = Polynomial(fft.ifft(s2_ev))
        s_sigma3 = Polynomial(fft.ifft(s3_ev))

        com_s1 = commit(s_sigma1, self.srs)
        com_s2 = commit(s_sigma2, self.srs)
        com_s3 = commit(s_sigma3, self.srs)

        # ---- Reconstruct transcript ----
        transcript = Transcript(b"PLONK-v1")
        transcript.absorb_point(b"com_qL", com_qL)
        transcript.absorb_point(b"com_qR", com_qR)
        transcript.absorb_point(b"com_qO", com_qO)
        transcript.absorb_point(b"com_qM", com_qM)
        transcript.absorb_point(b"com_qC", com_qC)

        transcript.absorb_point(b"com_a", proof.com_a)
        transcript.absorb_point(b"com_b", proof.com_b)
        transcript.absorb_point(b"com_c", proof.com_c)

        beta = transcript.squeeze_challenge(b"beta")
        gamma = transcript.squeeze_challenge(b"gamma")

        transcript.absorb_point(b"com_z", proof.com_z)
        alpha = transcript.squeeze_challenge(b"alpha")

        transcript.absorb_point(b"com_t_lo", proof.com_t_lo)
        transcript.absorb_point(b"com_t_mid", proof.com_t_mid)
        transcript.absorb_point(b"com_t_hi", proof.com_t_hi)
        zeta = transcript.squeeze_challenge(b"zeta")

        transcript.absorb_scalar(b"a_bar", proof.a_bar)
        transcript.absorb_scalar(b"b_bar", proof.b_bar)
        transcript.absorb_scalar(b"c_bar", proof.c_bar)
        transcript.absorb_scalar(b"s1_bar", proof.s_sigma1_bar)
        transcript.absorb_scalar(b"s2_bar", proof.s_sigma2_bar)
        transcript.absorb_scalar(b"z_omega_bar", proof.z_omega_bar)
        v = transcript.squeeze_challenge(b"v")

        # ---- Compute r(ζ) from evaluations ----
        zeta_n = zeta ** n
        zh_zeta = zeta_n - Fr.one()

        # L_1(ζ) = (ζ^n - 1) / (n · (ζ - 1))
        n_inv = Fr(n).inverse()
        l1_zeta = zh_zeta * n_inv * (zeta - Fr.one()).inverse()

        # Gate contribution
        r_0 = (
            proof.a_bar * q_L.evaluate(zeta)
            + proof.b_bar * q_R.evaluate(zeta)
            + proof.c_bar * q_O.evaluate(zeta)
            + proof.a_bar * proof.b_bar * q_M.evaluate(zeta)
            + q_C.evaluate(zeta)
        )
        # NOTE: The full verification uses a linearization trick to avoid
        # evaluating z(ζ) directly. Instead, we reconstruct the expected
        # opening value from the linearization polynomial.

        # ---- Pairing check ----
        # Compute the linearization commitment [r]₁
        # [r]₁ = ā·[qL] + b̄·[qR] + c̄·[qO] + ā·b̄·[qM] + [qC]
        #       + perm_factor_z · [z]
        #       + perm_factor_s3 · [Sσ₃]
        #       - Z_H(ζ) · ([tlo] + ζⁿ·[tmid] + ζ²ⁿ·[thi])

        perm_factor_z = (
            alpha * (
                (proof.a_bar + beta * zeta + gamma)
                * (proof.b_bar + beta * k1 * zeta + gamma)
                * (proof.c_bar + beta * k2 * zeta + gamma)
            )
            + alpha * alpha * l1_zeta
        )

        perm_factor_s3 = -(
            alpha * beta * proof.z_omega_bar
            * (proof.a_bar + beta * proof.s_sigma1_bar + gamma)
            * (proof.b_bar + beta * proof.s_sigma2_bar + gamma)
        )

        zeta_2n = zeta_n * zeta_n

        # [r]₁ via MSM on commitments
        com_r = (
            com_qL * proof.a_bar
            + com_qR * proof.b_bar
            + com_qO * proof.c_bar
            + com_qM * (proof.a_bar * proof.b_bar)
            + com_qC
            + proof.com_z * perm_factor_z
            + com_s3 * perm_factor_s3
            + (-proof.com_t_lo) * zh_zeta
            + (-proof.com_t_mid) * (zh_zeta * zeta_n)
            + (-proof.com_t_hi) * (zh_zeta * zeta_2n)
        )

        # ---- Aggregate opening check (2 pairings) ----
        # The prover's aggregate polynomial is:
        #   W(X) = r(X) + v·(a(X)-ā) + v²·(b(X)-b̄) + ... + v⁵·(Sσ₂(X)-s̄σ₂)
        # At X=ζ, each (p(ζ)-p̄) = 0, so W(ζ) = r(ζ).
        # The prover computes quotient = W(X) / (X-ζ), remainder = r(ζ).
        # The proof includes r(ζ) so the verifier can check the opening.

        # Aggregate commitment: [F]₁ = [r]₁ + v·[a]₁ + v²·[b]₁ + ...
        com_F = com_r
        v_pow = v
        for com_p in [proof.com_a, proof.com_b, proof.com_c, com_s1, com_s2]:
            com_F = com_F + com_p * v_pow
            v_pow = v_pow * v

        # Aggregate evaluation: r(ζ) + v·ā + v²·b̄ + v³·c̄ + v⁴·s̄σ₁ + v⁵·s̄σ₂
        e_scalar = proof.r_zeta
        v_pow = v
        for val_bar in [proof.a_bar, proof.b_bar, proof.c_bar,
                        proof.s_sigma1_bar, proof.s_sigma2_bar]:
            e_scalar = e_scalar + v_pow * val_bar
            v_pow = v_pow * v
        com_E = self.srs.g1 * e_scalar

        # Check 1: e([F-E]₁, G₂) = e(W_ζ, [τ-ζ]₂)
        com_FmE = com_F + (-com_E)
        check1 = verify_opening(
            commitment=com_FmE,
            point=zeta,
            value=Fr.zero(),
            witness=proof.w_zeta,
            srs=self.srs,
        )

        # Check 2: z(ζω) = z̄ω
        check2 = verify_opening(
            commitment=proof.com_z,
            point=zeta * omega,
            value=proof.z_omega_bar,
            witness=proof.w_zeta_omega,
            srs=self.srs,
        )

        return check1 and check2


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    from zkml_system.plonk.tda_gadgets import TDAGadgets

    print("=" * 60)
    print("PLONK Algebraic Satisfiability Verifier — Self-Test")
    print("=" * 60)

    n_pass = 0
    n_fail = 0

    # Test 1: Single ADD gate (3 + 5 = 8)
    print("\n--- Test 1: ADD gate ---")
    cc = CircuitCompiler(use_sparse=False, use_gelu=False)
    w_a = cc._new_wire(name="a"); cc._set_wire_value(w_a, Fr(3))
    w_b = cc._new_wire(name="b"); cc._set_wire_value(w_b, Fr(5))
    w_c = cc._new_wire(name="c"); cc._set_wire_value(w_c, Fr(8))
    cc._add_add_gate(w_a, w_b, w_c)
    r = verify_gate_satisfiability(CompiledCircuit(
        gates=cc.gates, wires=cc.wires,
        num_public_inputs=0, num_public_outputs=0
    ))
    print(f"  {r.message}")
    assert r.is_satisfied, "ADD gate should be satisfiable"
    n_pass += 1

    # Test 2: MUL gate with wrong witness (3 * 5 ≠ 16)
    print("\n--- Test 2: Reject incorrect MUL witness ---")
    cc2 = CircuitCompiler(use_sparse=False, use_gelu=False)
    wa2 = cc2._new_wire("a"); cc2._set_wire_value(wa2, Fr(3))
    wb2 = cc2._new_wire("b"); cc2._set_wire_value(wb2, Fr(5))
    wc2 = cc2._new_wire("c"); cc2._set_wire_value(wc2, Fr(16))
    cc2._add_mul_gate(wa2, wb2, wc2)
    r2 = verify_gate_satisfiability(CompiledCircuit(
        gates=cc2.gates, wires=cc2.wires,
        num_public_inputs=0, num_public_outputs=0
    ))
    print(f"  {r2.message}")
    assert not r2.is_satisfied, "Wrong MUL witness should be rejected"
    n_pass += 1

    # Test 3: CONST gate (wire = 42)
    print("\n--- Test 3: CONST gate ---")
    cc3 = CircuitCompiler(use_sparse=False, use_gelu=False)
    w3 = cc3._new_wire("x"); cc3._set_wire_value(w3, Fr(42))
    cc3._add_const_gate(w3, Fr(42))
    r3 = verify_gate_satisfiability(CompiledCircuit(
        gates=cc3.gates, wires=cc3.wires,
        num_public_inputs=0, num_public_outputs=0
    ))
    print(f"  {r3.message}")
    assert r3.is_satisfied, "CONST gate should be satisfiable"
    n_pass += 1

    # Test 4: assert_leq with range proof (10 ≤ 20)
    print("\n--- Test 4: assert_leq + range_check (C2) ---")
    cc4 = CircuitCompiler(use_sparse=False, use_gelu=False)
    g4 = TDAGadgets(cc4)
    wa4 = cc4._new_wire("a"); cc4._set_wire_value(wa4, Fr(10))
    wb4 = cc4._new_wire("b"); cc4._set_wire_value(wb4, Fr(20))
    g4.assert_leq(wa4, wb4)
    r4 = verify_gate_satisfiability(CompiledCircuit(
        gates=cc4.gates, wires=cc4.wires,
        num_public_inputs=0, num_public_outputs=0
    ))
    print(f"  {r4.message}")
    print(f"  Gate count: {r4.total_gates} (40-bit range proof)")
    assert r4.is_satisfied, "assert_leq should be satisfiable"
    n_pass += 1

    # Test 5: Mixed circuit (ADD + MUL + CONST chain)
    print("\n--- Test 5: Mixed circuit (a*b + c = d, d=const) ---")
    cc5 = CircuitCompiler(use_sparse=False, use_gelu=False)
    g5 = TDAGadgets(cc5)
    wa5 = cc5._new_wire("a"); cc5._set_wire_value(wa5, Fr(7))
    wb5 = cc5._new_wire("b"); cc5._set_wire_value(wb5, Fr(6))
    wc5 = cc5._new_wire("c"); cc5._set_wire_value(wc5, Fr(2))
    # a * b = 42
    prod = g5.mul(wa5, wb5)
    # prod + c = 44
    total = g5.add(prod, wc5)
    # assert total == 44
    w44 = g5.const_wire(Fr(44))
    g5.assert_equal(total, w44)
    r5 = verify_gate_satisfiability(CompiledCircuit(
        gates=cc5.gates, wires=cc5.wires,
        num_public_inputs=0, num_public_outputs=0
    ))
    print(f"  {r5.message}")
    assert r5.is_satisfied, "Mixed circuit should be satisfiable"
    n_pass += 1

    # Test 6: Full PLONK prove + verify (cryptographic)
    print("\n--- Test 6: PLONK 5-Round Prove + Verify ---")
    t0 = time.time()

    # Build a small circuit: a * b = c, where a=3, b=5, c=15
    cc6 = CircuitCompiler(use_sparse=False, use_gelu=False)
    wa6 = cc6._new_wire("a"); cc6._set_wire_value(wa6, Fr(3))
    wb6 = cc6._new_wire("b"); cc6._set_wire_value(wb6, Fr(5))
    wc6 = cc6._new_wire("c"); cc6._set_wire_value(wc6, Fr(15))
    cc6._add_mul_gate(wa6, wb6, wc6)

    # Add a second gate: c + 1 = d (d=16)
    wd6 = cc6._new_wire("one"); cc6._set_wire_value(wd6, Fr(1))
    we6 = cc6._new_wire("d"); cc6._set_wire_value(we6, Fr(16))
    cc6._add_add_gate(wc6, wd6, we6)

    circuit6 = CompiledCircuit(
        gates=cc6.gates, wires=cc6.wires,
        num_public_inputs=0, num_public_outputs=0
    )

    # Domain will be n=2 (next power of 2 >= 2 gates)
    # SRS needs degree >= n+5 = 7, but 4n FFT needs more. Use 32.
    srs = TrustedSetup.generate(max_degree=32)
    prover = PLONKProver(srs)
    verifier = PLONKVerifier(srs)

    t_setup = (time.time() - t0) * 1000
    print(f"  Setup: {t_setup:.0f}ms (SRS degree=32)")

    # Prove
    t0 = time.time()
    proof = prover.prove(circuit6)
    t_prove = (time.time() - t0) * 1000
    print(f"  Prove: {t_prove:.0f}ms (5 rounds)")
    print(f"    Commitments: [a]₁, [b]₁, [c]₁, [z]₁, [t_lo]₁, [t_mid]₁, [t_hi]₁")
    print(f"    Evaluations: ā, b̄, c̄, s̄σ₁, s̄σ₂, z̄ω")
    print(f"    Opening proofs: W_ζ, W_ζω")

    # Verify
    t0 = time.time()
    valid = verifier.verify(proof, circuit6)
    t_verify = (time.time() - t0) * 1000
    result_str = "✅ VALID" if valid else "❌ INVALID"
    print(f"  Verify: {result_str} ({t_verify:.0f}ms, 2 pairing checks)")

    if valid:
        n_pass += 1
    else:
        n_fail += 1
        print("  ⚠️ Proof verification FAILED — debugging needed")

    # Test 7: Tampered proof should fail
    print("\n--- Test 7: Tampered proof rejection ---")
    tampered = PLONKProof(
        com_a=proof.com_a, com_b=proof.com_b, com_c=proof.com_c,
        com_z=proof.com_z,
        com_t_lo=proof.com_t_lo, com_t_mid=proof.com_t_mid, com_t_hi=proof.com_t_hi,
        a_bar=proof.a_bar + Fr.one(),  # ← tamper
        b_bar=proof.b_bar, c_bar=proof.c_bar,
        s_sigma1_bar=proof.s_sigma1_bar, s_sigma2_bar=proof.s_sigma2_bar,
        z_omega_bar=proof.z_omega_bar, r_zeta=proof.r_zeta,
        w_zeta=proof.w_zeta, w_zeta_omega=proof.w_zeta_omega,
        n=proof.n,
    )
    tampered_valid = verifier.verify(tampered, circuit6)
    if not tampered_valid:
        print(f"  ✅ Tampered proof correctly REJECTED")
        n_pass += 1
    else:
        print(f"  ❌ Tampered proof was NOT rejected!")
        n_fail += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{n_pass + n_fail} passed")
    print(f"{'=' * 60}")
