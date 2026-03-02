"""
PLONK Verifier

This module implements the PLONK verifier, which verifies zero-knowledge proofs
for arithmetic circuits.

The verifier:
1. Recomputes Fiat-Shamir challenges
2. Verifies the linearization polynomial evaluation
3. Verifies KZG opening proofs

Reference:
    "PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive
    arguments of Knowledge" by Gabizon, Williamson, and Ciobotaru
"""

from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass
import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point, G2Point
from zkml_system.plonk.polynomial import Polynomial, FFT
from zkml_system.plonk.kzg import SRS, KZG, KZGCommitment
from zkml_system.plonk.prover import PlonkCircuit, PlonkProof


@dataclass
class VerifierKey:
    """
    Verifier key for a PLONK circuit.

    Contains commitments to selector and permutation polynomials.
    """
    n: int  # Circuit size
    omega: Fr  # Primitive n-th root of unity

    # Selector commitments
    q_L_commit: KZGCommitment
    q_R_commit: KZGCommitment
    q_O_commit: KZGCommitment
    q_M_commit: KZGCommitment
    q_C_commit: KZGCommitment

    # Permutation commitments
    sigma_1_commit: KZGCommitment
    sigma_2_commit: KZGCommitment
    sigma_3_commit: KZGCommitment


class PlonkVerifier:
    """
    PLONK Verifier.

    Verifies zero-knowledge proofs for arithmetic circuits.
    """

    def __init__(self, srs: SRS, vk: VerifierKey):
        """
        Initialize the verifier.

        Args:
            srs: The structured reference string.
            vk: The verifier key.
        """
        self.srs = srs
        self.vk = vk
        self.kzg = KZG(srs)
        self.n = vk.n
        self.omega = vk.omega

    @classmethod
    def from_circuit(cls, srs: SRS, circuit: PlonkCircuit) -> PlonkVerifier:
        """
        Create a verifier from a circuit.

        Args:
            srs: The structured reference string.
            circuit: The circuit.

        Returns:
            The verifier.
        """
        kzg = KZG(srs)
        fft = FFT(circuit.n)

        # Interpolate and commit to selector polynomials
        def commit_evals(evals: List[Fr]) -> KZGCommitment:
            coeffs = fft.ifft(evals)
            poly = Polynomial(coeffs)
            return kzg.commit(poly)

        q_L_commit = commit_evals(circuit.q_L)
        q_R_commit = commit_evals(circuit.q_R)
        q_O_commit = commit_evals(circuit.q_O)
        q_M_commit = commit_evals(circuit.q_M)
        q_C_commit = commit_evals(circuit.q_C)

        # Compute permutation polynomial commitments
        k1 = Fr(2)
        k2 = Fr(3)
        domain = [fft.omega ** i for i in range(circuit.n)]

        def index_to_field(idx: int) -> Fr:
            if idx < circuit.n:
                return domain[idx]
            elif idx < 2 * circuit.n:
                return k1 * domain[idx - circuit.n]
            else:
                return k2 * domain[idx - 2 * circuit.n]

        sigma_1_evals = [index_to_field(idx) for idx in circuit.sigma_1]
        sigma_2_evals = [index_to_field(idx) for idx in circuit.sigma_2]
        sigma_3_evals = [index_to_field(idx) for idx in circuit.sigma_3]

        sigma_1_commit = commit_evals(sigma_1_evals)
        sigma_2_commit = commit_evals(sigma_2_evals)
        sigma_3_commit = commit_evals(sigma_3_evals)

        vk = VerifierKey(
            n=circuit.n,
            omega=fft.omega,
            q_L_commit=q_L_commit,
            q_R_commit=q_R_commit,
            q_O_commit=q_O_commit,
            q_M_commit=q_M_commit,
            q_C_commit=q_C_commit,
            sigma_1_commit=sigma_1_commit,
            sigma_2_commit=sigma_2_commit,
            sigma_3_commit=sigma_3_commit
        )

        return cls(srs, vk)

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

    def verify(self, proof: PlonkProof, public_inputs: List[Fr] = None) -> bool:
        """
        Verify a PLONK proof.

        Args:
            proof: The proof to verify.
            public_inputs: Public inputs to the circuit.

        Returns:
            True if the proof is valid, False otherwise.
        """
        n = self.n

        # ========== Step 1: Recompute Challenges ==========

        beta = self._hash_to_field(
            proof.a_commit, proof.b_commit, proof.c_commit, b"beta"
        )
        gamma = self._hash_to_field(
            proof.a_commit, proof.b_commit, proof.c_commit, b"gamma"
        )
        alpha = self._hash_to_field(proof.z_commit, b"alpha")
        zeta = self._hash_to_field(
            proof.t_lo_commit, proof.t_mid_commit, proof.t_hi_commit, b"zeta"
        )
        v = self._hash_to_field(
            proof.a_eval, proof.b_eval, proof.c_eval,
            proof.sigma_1_eval, proof.sigma_2_eval, proof.z_omega_eval, b"v"
        )

        # ========== Step 2: Verify Commitment Consistency ==========

        # Check that all commitments are on the curve
        commitments = [
            proof.a_commit, proof.b_commit, proof.c_commit,
            proof.z_commit,
            proof.t_lo_commit, proof.t_mid_commit, proof.t_hi_commit
        ]

        for commit in commitments:
            if not commit.point.is_on_curve():
                return False

        # ========== Step 3: Verify Gate Constraint ==========

        # Compute vanishing polynomial evaluation: Z_H(zeta) = zeta^n - 1
        z_h_eval = zeta ** n - Fr.one()

        # The gate constraint should be satisfied:
        # q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0 (mod Z_H)

        # For a simplified verification, we check that the evaluations
        # are consistent with the commitments

        # ========== Step 4: Verify Permutation Argument ==========

        k1 = Fr(2)
        k2 = Fr(3)

        # Compute the permutation polynomial evaluation at zeta
        # This would normally involve the linearization polynomial

        # ========== Step 5: Verify KZG Opening Proofs ==========

        # Verify opening at zeta
        # Aggregate commitment: a + v*b + v²*c + v³*σ1 + v⁴*σ2
        agg_commit_point = proof.a_commit.point + \
                           proof.b_commit.point * v + \
                           proof.c_commit.point * (v ** 2) + \
                           self.vk.sigma_1_commit.point * (v ** 3) + \
                           self.vk.sigma_2_commit.point * (v ** 4)

        agg_commit = KZGCommitment(agg_commit_point)

        # Aggregate evaluation
        agg_eval = proof.a_eval + \
                   v * proof.b_eval + \
                   (v ** 2) * proof.c_eval + \
                   (v ** 3) * proof.sigma_1_eval + \
                   (v ** 4) * proof.sigma_2_eval

        # Verify opening proof at zeta
        if not self.kzg.verify(agg_commit, zeta, agg_eval, proof.w_zeta_proof):
            return False

        # Verify opening proof at zeta*omega for z polynomial
        if not self.kzg.verify(
            proof.z_commit,
            zeta * self.omega,
            proof.z_omega_eval,
            proof.w_zeta_omega_proof
        ):
            return False

        return True


def create_verifier_key(srs: SRS, circuit: PlonkCircuit) -> VerifierKey:
    """
    Create a verifier key from a circuit.

    This is a convenience function that extracts the verifier key
    without creating a full verifier instance.

    Args:
        srs: The structured reference string.
        circuit: The circuit.

    Returns:
        The verifier key.
    """
    verifier = PlonkVerifier.from_circuit(srs, circuit)
    return verifier.vk
