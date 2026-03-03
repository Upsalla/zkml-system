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
import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point
from zkml_system.plonk.polynomial import Polynomial, FFT, lagrange_interpolation
from zkml_system.plonk.kzg import SRS, KZG, KZGCommitment, KZGProof

warnings.warn(
    "plonk.prover is DEPRECATED. PlonkProver generates incomplete proofs "
    "(missing permutation and boundary constraints in quotient polynomial). "
    "Use plonk.plonk_prover.PLONKProver for production-grade proofs.",
    DeprecationWarning,
    stacklevel=2,
)


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
    PLONK Prover (DEPRECATED).

    WARNING: This prover generates incomplete proofs. The quotient polynomial
    only includes the gate constraint — permutation and boundary constraints
    are missing. Use PLONKProver from plonk_prover.py for sound proofs.
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

        BLOCKED: This prover generates cryptographically unsound proofs.
        The quotient polynomial only includes gate constraints — permutation
        and boundary constraints are missing. Use PLONKProver from
        plonk_prover.py for sound proofs.

        Raises:
            NotImplementedError: Always. This prover is deprecated.
        """
        raise NotImplementedError(
            "PlonkProver.prove() is BLOCKED — quotient polynomial is incomplete "
            "(missing permutation + boundary constraints). "
            "Use plonk.plonk_prover.PLONKProver for cryptographically sound proofs."
        )
