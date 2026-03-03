"""
Transcript domain separation & security tests — GA Blocker 6C.

Tests that the Fiat-Shamir transcript correctly produces distinct
challenges under different domain separations, message orderings,
and edge cases.

A soundness-critical property: if two different protocol executions
produce the same Fiat-Shamir challenges, the system is vulnerable
to proof replay and forgery.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.crypto.bn254.curve import G1Point
from zkml_system.plonk.transcript import Transcript


class TestTranscriptDomainSeparation(unittest.TestCase):
    """Fiat-Shamir transcript must produce distinct challenges for distinct inputs."""

    # ---- Label sensitivity ----

    def test_same_data_different_labels_produce_different_challenges(self):
        """Same scalar, different absorb labels → different challenges."""
        t1 = Transcript(b"PLONK-v1")
        t1.absorb_scalar(b"alpha", Fr(42))
        c1 = t1.squeeze_challenge(b"ch")

        t2 = Transcript(b"PLONK-v1")
        t2.absorb_scalar(b"beta", Fr(42))
        c2 = t2.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2, "Label must affect digest")

    def test_same_data_different_squeeze_labels(self):
        """Same absorbed data, different squeeze labels → different challenges."""
        t1 = Transcript(b"PLONK-v1")
        t1.absorb_scalar(b"x", Fr(99))
        c1 = t1.squeeze_challenge(b"alpha")

        t2 = Transcript(b"PLONK-v1")
        t2.absorb_scalar(b"x", Fr(99))
        c2 = t2.squeeze_challenge(b"beta")

        self.assertNotEqual(c1, c2, "Squeeze label must affect challenge")

    # ---- Domain separation (init label) ----

    def test_different_init_labels_produce_different_challenges(self):
        """Transcript(b"PLONK-v1") vs Transcript(b"PLONK-v2") → different."""
        t1 = Transcript(b"PLONK-v1")
        t1.absorb_scalar(b"x", Fr(1))
        c1 = t1.squeeze_challenge(b"ch")

        t2 = Transcript(b"PLONK-v2")
        t2.absorb_scalar(b"x", Fr(1))
        c2 = t2.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2, "Init label divergence required")

    # ---- Order dependence (non-commutativity) ----

    def test_absorb_order_matters_scalars(self):
        """absorb(a) then absorb(b) ≠ absorb(b) then absorb(a)."""
        t1 = Transcript(b"test")
        t1.absorb_scalar(b"x", Fr(1))
        t1.absorb_scalar(b"y", Fr(2))
        c1 = t1.squeeze_challenge(b"ch")

        t2 = Transcript(b"test")
        t2.absorb_scalar(b"y", Fr(2))
        t2.absorb_scalar(b"x", Fr(1))
        c2 = t2.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2, "Absorb order must affect challenge")

    def test_absorb_order_matters_points(self):
        """Point absorption order must matter."""
        g = G1Point.generator()
        g2 = g + g

        t1 = Transcript(b"test")
        t1.absorb_point(b"P", g)
        t1.absorb_point(b"Q", g2)
        c1 = t1.squeeze_challenge(b"ch")

        t2 = Transcript(b"test")
        t2.absorb_point(b"Q", g2)
        t2.absorb_point(b"P", g)
        c2 = t2.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2)

    # ---- Replay/re-squeeze prevention ----

    def test_sequential_squeezes_produce_different_values(self):
        """Each squeeze must update internal state → no repeats."""
        t = Transcript(b"test")
        t.absorb_scalar(b"x", Fr(1))
        c1 = t.squeeze_challenge(b"first")
        c2 = t.squeeze_challenge(b"second")
        c3 = t.squeeze_challenge(b"third")

        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c2, c3)
        self.assertNotEqual(c1, c3)

    def test_same_label_squeeze_still_differs(self):
        """Even squeezing the same label twice → different (state advances)."""
        t = Transcript(b"test")
        t.absorb_scalar(b"x", Fr(1))
        c1 = t.squeeze_challenge(b"ch")
        c2 = t.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2, "Same-label re-squeeze must still differ")

    # ---- Determinism ----

    def test_identical_transcripts_produce_identical_challenges(self):
        """Exact same sequence → exact same challenge (randomness is deterministic)."""
        for _ in range(3):
            t = Transcript(b"PLONK-v1")
            t.absorb_scalar(b"x", Fr(42))
            t.absorb_point(b"com", G1Point.generator())
            c = t.squeeze_challenge(b"alpha")

        # Re-run to compare
        t1 = Transcript(b"PLONK-v1")
        t1.absorb_scalar(b"x", Fr(42))
        t1.absorb_point(b"com", G1Point.generator())
        c1 = t1.squeeze_challenge(b"alpha")

        t2 = Transcript(b"PLONK-v1")
        t2.absorb_scalar(b"x", Fr(42))
        t2.absorb_point(b"com", G1Point.generator())
        c2 = t2.squeeze_challenge(b"alpha")

        self.assertEqual(c1, c2, "Determinism broken")

    # ---- Edge cases ----

    def test_identity_vs_generator_point(self):
        """Identity point (point at infinity) vs generator → different challenges."""
        t1 = Transcript(b"test")
        t1.absorb_point(b"P", G1Point.identity())
        c1 = t1.squeeze_challenge(b"ch")

        t2 = Transcript(b"test")
        t2.absorb_point(b"P", G1Point.generator())
        c2 = t2.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2)

    def test_zero_scalar_vs_one(self):
        """Fr(0) vs Fr(1) must produce different challenges."""
        t1 = Transcript(b"test")
        t1.absorb_scalar(b"x", Fr.zero())
        c1 = t1.squeeze_challenge(b"ch")

        t2 = Transcript(b"test")
        t2.absorb_scalar(b"x", Fr.one())
        c2 = t2.squeeze_challenge(b"ch")

        self.assertNotEqual(c1, c2)

    def test_challenge_is_in_field(self):
        """Squeezed challenge must be a valid Fr element."""
        t = Transcript(b"test")
        t.absorb_scalar(b"x", Fr(123456789))
        c = t.squeeze_challenge(b"ch")

        self.assertIsInstance(c, Fr)
        # Must be in [0, MODULUS)
        val = c.to_int()
        self.assertGreaterEqual(val, 0)
        self.assertLess(val, Fr.MODULUS)

    def test_empty_transcript_squeeze(self):
        """Squeezing without any absorb must still produce valid challenge."""
        t = Transcript(b"test")
        c = t.squeeze_challenge(b"ch")
        self.assertIsInstance(c, Fr)


if __name__ == "__main__":
    unittest.main()
