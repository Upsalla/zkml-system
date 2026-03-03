"""
Fiat-Shamir Transcript for PLONK.

Converts the interactive PLONK protocol to non-interactive by
deriving verifier challenges from a cryptographic hash of all
prior prover messages.

Uses SHA-256 as the random oracle. Each absorb/squeeze includes
a domain-separation label to ensure distinct challenges.

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
import hashlib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point


class Transcript:
    """
    Fiat-Shamir transcript using SHA-256.

    Usage:
        t = Transcript(b"PlonkV1")
        t.absorb_scalar(b"a_eval", a_bar)
        t.absorb_point(b"com_a", com_a)
        beta = t.squeeze_challenge(b"beta")
    """

    def __init__(self, label: bytes):
        """Initialize with a domain-separation label."""
        self._state = hashlib.sha256(label).digest()

    def _update(self, data: bytes):
        """Mix data into the running hash state."""
        h = hashlib.sha256()
        h.update(self._state)
        h.update(data)
        self._state = h.digest()

    def absorb_scalar(self, label: bytes, scalar: Fr):
        """Absorb a field element into the transcript."""
        val = scalar.to_int()
        self._update(label + val.to_bytes(32, 'big'))

    def absorb_point(self, label: bytes, point: G1Point):
        """Absorb a G1 commitment into the transcript."""
        if point.is_identity():
            self._update(label + b'\x00' * 64)
        else:
            x, y = point.to_affine()
            self._update(
                label
                + x.to_int().to_bytes(32, 'big')
                + y.to_int().to_bytes(32, 'big')
            )

    def squeeze_challenge(self, label: bytes) -> Fr:
        """Derive a challenge from the current transcript state."""
        h = hashlib.sha256()
        h.update(self._state)
        h.update(label)
        digest = h.digest()
        # Update state so subsequent squeezes are different
        self._state = digest
        # Reduce mod r to get a field element
        val = int.from_bytes(digest, 'big') % Fr.MODULUS
        return Fr(val)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Fiat-Shamir Transcript Self-Test")
    print("=" * 60)

    # Test 1: Determinism
    t1 = Transcript(b"test")
    t1.absorb_scalar(b"x", Fr(42))
    c1 = t1.squeeze_challenge(b"ch")

    t2 = Transcript(b"test")
    t2.absorb_scalar(b"x", Fr(42))
    c2 = t2.squeeze_challenge(b"ch")

    assert c1 == c2, "Determinism failed"
    print(f"  1. Determinism: ✅ (challenge = {c1.to_int() % 10000}...)")

    # Test 2: Different inputs → different challenges
    t3 = Transcript(b"test")
    t3.absorb_scalar(b"x", Fr(43))
    c3 = t3.squeeze_challenge(b"ch")

    assert c1 != c3, "Collision on different inputs"
    print(f"  2. Sensitivity: ✅")

    # Test 3: Absorb point
    g = G1Point.generator()
    t4 = Transcript(b"test")
    t4.absorb_point(b"P", g)
    c4 = t4.squeeze_challenge(b"ch")

    t5 = Transcript(b"test")
    t5.absorb_point(b"P", g + g)
    c5 = t5.squeeze_challenge(b"ch")

    assert c4 != c5, "Collision on different points"
    print(f"  3. Point absorption: ✅")

    # Test 4: Sequential squeezes are different
    t6 = Transcript(b"test")
    t6.absorb_scalar(b"x", Fr(1))
    s1 = t6.squeeze_challenge(b"a")
    s2 = t6.squeeze_challenge(b"b")
    assert s1 != s2, "Sequential squeezes should differ"
    print(f"  4. Sequential squeezes: ✅")

    print(f"\n{'=' * 60}")
    print(f"All transcript tests passed!")
    print(f"{'=' * 60}")
