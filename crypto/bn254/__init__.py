"""
BN254 Cryptographic Primitives

This module provides all cryptographic primitives for the BN254 elliptic curve:
- Field arithmetic (Fp, Fr)
- Extension field arithmetic (Fp2, Fp6, Fp12)
- Elliptic curve points (G1, G2)
- Pairing operations

BN254 is Ethereum-compatible and provides ~128-bit security.
"""

from .field import Fp, Fr
from .extension_field import Fp2, Fp6, Fp12
from .curve import G1Point, G2Point
from .pairing import pairing, miller_loop, final_exponentiation, multi_pairing
from .constants import (
    FIELD_MODULUS,
    CURVE_ORDER,
    CURVE_B,
    G1_X, G1_Y,
    G2_X, G2_Y,
)

__all__ = [
    # Base fields
    'Fp', 'Fr',
    # Extension fields
    'Fp2', 'Fp6', 'Fp12',
    # Curve points
    'G1Point', 'G2Point',
    # Pairing
    'pairing', 'miller_loop', 'final_exponentiation', 'multi_pairing',
    # Constants
    'FIELD_MODULUS', 'CURVE_ORDER', 'CURVE_B',
    'G1_X', 'G1_Y', 'G2_X', 'G2_Y',
]
