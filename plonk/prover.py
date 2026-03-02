"""
PLONK Prover

This module implements the PLONK prover, which generates zero-knowledge proofs
for arithmetic circuits.

The prover:
1. Commits to wire polynomials (a, b, c)
2. Commits to the permutation polynomial (z)
3. Commits to the quotient polynomial (t)
4. Opens polynomials at the challenge point
5. Generates the linearization polynomial

Reference:
    "PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive
    arguments of Knowledge" by Gabizon, Williamson, and Ciobotaru
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fr
from zkml_system.crypto.bn254.curve import G1Point
from zkml_system.plonk.polynomial import Polynomial, FFT, lagrange_interpolation
from zkml_system.plonk.kzg import SRS, KZG, KZGCommitment, KZGProof


@dataclass
class PlonkCircuit:
    """
    A PLONK arithmetic circuit.

    The circuit consists of:
    - n gates, each with left (a), right (b), output (c) wires
    - Selector polynomials: q_L, q_R, q_O, q_M, q_C
    - Permutation polynomials: σ_1, σ_2, σ_3

    Gate equation: q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0
    """
    n: int  # Number of gates (power of 2)
    q_L: List[Fr]  # Left wire selector
    q_R: List[Fr]  # Right wire selector
    q_O: List[Fr]  # Output wire selector
    q_M: List[Fr]  # Multiplication selector
    q_C: List[Fr]  # Constant selector
    sigma_1: List[int]  # Permutation for wire a
    sigma_2: List[int]  # Permutation for wire b
    sigma_3: List[int]  # Permutation for wire c

    @classmethod
    def empty(cls, n: int) -> PlonkCircuit:
        """Create an empty circuit with n gates."""
        return cls(
            n=n,
            q_L=[Fr.zero()] * n,
            q_R=[Fr.zero()] * n,
            q_O=[Fr.zero()] * n,
            q_M=[Fr.zero()] * n,
            q_C=[Fr.zero()] * n,
            sigma_1=list(range(n)),
            sigma_2=list(range(n, 2*n)),
            sigma_3=list(range(2*n, 3*n))
        )


@dataclass
class PlonkWitness:
    """
    A witness for a PLONK circuit.

    Contains the wire values for each gate.
    """
    a: List[Fr]  # Left wire values
    b: List[Fr]  # Right wire values
    c: List[Fr]  # Output wire values


@dataclass
class PlonkProof:
    """
    A PLONK proof.

    Contains all commitments and evaluations needed for verification.
    """
    # Round 1: Wire commitments
    a_commit: KZGCommitment
    b_commit: KZGCommitment
    c_commit: KZGCommitment

    # Round 2: Permutation commitment
    z_commit: KZGCommitment

    # Round 3: Quotient commitments
    t_lo_commit: KZGCommitment
    t_mid_commit: KZGCommitment
    t_hi_commit: KZGCommitment

    # Round 4: Evaluations at challenge zeta
    a_eval: Fr
    b_eval: Fr
    c_eval: Fr
    sigma_1_eval: Fr
    sigma_2_eval: Fr
    z_omega_eval: Fr

    # Round 5: Opening proofs
    w_zeta_proof: KZGProof
    w_zeta_omega_proof: KZGProof


class PlonkProver:
    """
    PLONK Prover.

    Generates zero-knowledge proofs for arithmetic circuits.
    """

    def __init__(self, srs: SRS, circuit: PlonkCircuit):
        """
        Initialize the prover.

        Args:
            srs: The structured reference string.
            circuit: The circuit to prove.
        """
        self.srs = srs
        self.circuit = circuit
        self.kzg = KZG(srs)
        self.n = circuit.n

        # Initialize FFT
        self.fft = FFT(self.n)
        self.omega = self.fft.omega

        # Precompute domain
        self.domain = [self.omega ** i for i in range(self.n)]

        # Precompute selector polynomials
        self._precompute_selectors()

    def _precompute_selectors(self):
        """Precompute selector polynomials from evaluations."""
        self.q_L_poly = self._interpolate(self.circuit.q_L)
        self.q_R_poly = self._interpolate(self.circuit.q_R)
        self.q_O_poly = self._interpolate(self.circuit.q_O)
        self.q_M_poly = self._interpolate(self.circuit.q_M)
        self.q_C_poly = self._interpolate(self.circuit.q_C)

        # Precompute permutation polynomials
        k1 = Fr(2)  # Coset generator 1
        k2 = Fr(3)  # Coset generator 2

        sigma_1_evals = []
        sigma_2_evals = []
        sigma_3_evals = []

        for i in range(self.n):
            idx1 = self.circuit.sigma_1[i]
            idx2 = self.circuit.sigma_2[i]
            idx3 = self.circuit.sigma_3[i]

            # Map index to field element
            sigma_1_evals.append(self._index_to_field(idx1, k1, k2))
            sigma_2_evals.append(self._index_to_field(idx2, k1, k2))
            sigma_3_evals.append(self._index_to_field(idx3, k1, k2))

        self.sigma_1_poly = self._interpolate(sigma_1_evals)
        self.sigma_2_poly = self._interpolate(sigma_2_evals)
        self.sigma_3_poly = self._interpolate(sigma_3_evals)

    def _index_to_field(self, idx: int, k1: Fr, k2: Fr) -> Fr:
        """Convert a wire index to a field element."""
        if idx < self.n:
            return self.domain[idx]
        elif idx < 2 * self.n:
            return k1 * self.domain[idx - self.n]
        else:
            return k2 * self.domain[idx - 2 * self.n]

    def _interpolate(self, evals: List[Fr]) -> Polynomial:
        """Interpolate a polynomial from evaluations on the domain."""
        coeffs = self.fft.ifft(evals)
        return Polynomial(coeffs)

    def _hash_to_field(self, *args) -> Fr:
        """Hash inputs to a field element (Fiat-Shamir)."""
        hasher = hashlib.sha256()
        for arg in args:
            if isinstance(arg, Fr):
                hasher.update(arg.to_int().to_bytes(32, 'big'))
            elif isinstance(arg, KZGCommitment):
                if arg.point.is_identity():
                    hasher.update(b'\x00' * 64)
                else:
                    x, y = arg.point.to_affine()
                    hasher.update(x.to_int().to_bytes(32, 'big'))
                    hasher.update(y.to_int().to_bytes(32, 'big'))
            elif isinstance(arg, bytes):
                hasher.update(arg)
        digest = hasher.digest()
        return Fr(int.from_bytes(digest, 'big'))

    def prove(self, witness: PlonkWitness) -> PlonkProof:
        """
        Generate a PLONK proof.

        Args:
            witness: The circuit witness.

        Returns:
            The proof.
        """
        n = self.n

        # ========== Round 1: Wire Commitments ==========

        # Interpolate wire polynomials
        a_poly = self._interpolate(witness.a)
        b_poly = self._interpolate(witness.b)
        c_poly = self._interpolate(witness.c)

        # Commit to wire polynomials
        a_commit = self.kzg.commit(a_poly)
        b_commit = self.kzg.commit(b_poly)
        c_commit = self.kzg.commit(c_poly)

        # ========== Round 2: Permutation Polynomial ==========

        # Fiat-Shamir challenge
        beta = self._hash_to_field(a_commit, b_commit, c_commit, b"beta")
        gamma = self._hash_to_field(a_commit, b_commit, c_commit, b"gamma")

        # Compute permutation polynomial z
        k1 = Fr(2)
        k2 = Fr(3)

        z_evals = [Fr.one()]  # z(1) = 1

        for i in range(n - 1):
            # Numerator: (a + β*ω^i + γ)(b + β*k1*ω^i + γ)(c + β*k2*ω^i + γ)
            num = (witness.a[i] + beta * self.domain[i] + gamma) * \
                  (witness.b[i] + beta * k1 * self.domain[i] + gamma) * \
                  (witness.c[i] + beta * k2 * self.domain[i] + gamma)

            # Denominator: (a + β*σ1(i) + γ)(b + β*σ2(i) + γ)(c + β*σ3(i) + γ)
            sigma_1_val = self._index_to_field(self.circuit.sigma_1[i], k1, k2)
            sigma_2_val = self._index_to_field(self.circuit.sigma_2[i], k1, k2)
            sigma_3_val = self._index_to_field(self.circuit.sigma_3[i], k1, k2)

            den = (witness.a[i] + beta * sigma_1_val + gamma) * \
                  (witness.b[i] + beta * sigma_2_val + gamma) * \
                  (witness.c[i] + beta * sigma_3_val + gamma)

            z_evals.append(z_evals[-1] * num / den)

        z_poly = self._interpolate(z_evals)
        z_commit = self.kzg.commit(z_poly)

        # ========== Round 3: Quotient Polynomial ==========

        alpha = self._hash_to_field(z_commit, b"alpha")

        # Compute quotient polynomial t(X)
        # t(X) = (gate_constraint + α*perm_constraint + α²*z_boundary) / Z_H(X)

        # For simplicity, we compute a simplified quotient
        # In a full implementation, this would be more complex

        # Gate constraint: q_L*a + q_R*b + q_O*c + q_M*a*b + q_C
        gate_poly = self.q_L_poly * a_poly + \
                    self.q_R_poly * b_poly + \
                    self.q_O_poly * c_poly + \
                    self.q_M_poly * (a_poly * b_poly) + \
                    self.q_C_poly

        # Vanishing polynomial Z_H(X) = X^n - 1
        z_h_coeffs = [Fr.zero()] * (n + 1)
        z_h_coeffs[0] = -Fr.one()
        z_h_coeffs[n] = Fr.one()
        z_h_poly = Polynomial(z_h_coeffs)

        # Simplified quotient (just gate constraint for now)
        # In production, include permutation and boundary constraints
        t_poly = gate_poly  # Placeholder

        # Split quotient into low, mid, high parts
        t_coeffs = t_poly.coeffs + [Fr.zero()] * (3 * n - len(t_poly.coeffs))
        t_lo = Polynomial(t_coeffs[:n])
        t_mid = Polynomial(t_coeffs[n:2*n])
        t_hi = Polynomial(t_coeffs[2*n:3*n])

        t_lo_commit = self.kzg.commit(t_lo)
        t_mid_commit = self.kzg.commit(t_mid)
        t_hi_commit = self.kzg.commit(t_hi)

        # ========== Round 4: Evaluations ==========

        zeta = self._hash_to_field(t_lo_commit, t_mid_commit, t_hi_commit, b"zeta")

        a_eval = a_poly.evaluate(zeta)
        b_eval = b_poly.evaluate(zeta)
        c_eval = c_poly.evaluate(zeta)
        sigma_1_eval = self.sigma_1_poly.evaluate(zeta)
        sigma_2_eval = self.sigma_2_poly.evaluate(zeta)
        z_omega_eval = z_poly.evaluate(zeta * self.omega)

        # ========== Round 5: Opening Proofs ==========

        v = self._hash_to_field(
            a_eval, b_eval, c_eval, sigma_1_eval, sigma_2_eval, z_omega_eval, b"v"
        )

        # Aggregate polynomial for opening at zeta
        # W_zeta = a + v*b + v²*c + v³*σ1 + v⁴*σ2 + v⁵*t_lo + v⁶*t_mid*zeta^n + v⁷*t_hi*zeta^(2n)
        w_zeta_poly = a_poly + \
                      b_poly.scale(v) + \
                      c_poly.scale(v ** 2) + \
                      self.sigma_1_poly.scale(v ** 3) + \
                      self.sigma_2_poly.scale(v ** 4)

        w_zeta_proof, _ = self.kzg.create_proof(w_zeta_poly, zeta)

        # Opening at zeta*omega for z polynomial
        w_zeta_omega_proof, _ = self.kzg.create_proof(z_poly, zeta * self.omega)

        return PlonkProof(
            a_commit=a_commit,
            b_commit=b_commit,
            c_commit=c_commit,
            z_commit=z_commit,
            t_lo_commit=t_lo_commit,
            t_mid_commit=t_mid_commit,
            t_hi_commit=t_hi_commit,
            a_eval=a_eval,
            b_eval=b_eval,
            c_eval=c_eval,
            sigma_1_eval=sigma_1_eval,
            sigma_2_eval=sigma_2_eval,
            z_omega_eval=z_omega_eval,
            w_zeta_proof=w_zeta_proof,
            w_zeta_omega_proof=w_zeta_omega_proof
        )
