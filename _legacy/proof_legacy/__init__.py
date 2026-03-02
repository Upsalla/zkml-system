"""
Proof Module für zkML
=====================

Enthält Prover und Verifier für Zero-Knowledge Proofs.
"""

from .prover import (
    ProofComponents, Proof, Prover, ProofStats
)

from .verifier import (
    VerificationResult, Verifier, BatchVerifier
)

__all__ = [
    'ProofComponents', 'Proof', 'Prover', 'ProofStats',
    'VerificationResult', 'Verifier', 'BatchVerifier'
]
