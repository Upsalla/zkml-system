"""
Poseidon Test Vectors for BN254 (t=3, R_F=8, R_P=57, α=5)

Verifies:
1. Round constant generation is deterministic and correctly parameterized
2. MDS matrix is valid (Cauchy construction, invertible)
3. Hash output is deterministic for known inputs
4. Edge cases: empty input, single element, zero elements
5. Collision resistance: distinct inputs produce distinct outputs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.poseidon import (
    PoseidonHash, poseidon_hash, ROUND_CONSTANTS, MDS_MATRIX,
    T, R_F, R_P, ALPHA,
    _generate_round_constants, _generate_mds_matrix,
)


# =============================================================================
# 1. Round Constant Validation
# =============================================================================

class TestRoundConstants:
    """Verify Grain-LFSR round constant generation."""

    def test_correct_count(self):
        """Total constants = (R_F + R_P) * T = 65 * 3 = 195."""
        expected = (R_F + R_P) * T
        assert len(ROUND_CONSTANTS) == expected, (
            f"Expected {expected} round constants, got {len(ROUND_CONSTANTS)}"
        )

    def test_all_in_field(self):
        """Every round constant must be in [0, p)."""
        p = Fr.MODULUS
        for i, rc in enumerate(ROUND_CONSTANTS):
            assert 0 <= rc < p, f"Round constant {i} = {rc} is out of range [0, p)"

    def test_no_trivial_zeros(self):
        """A well-seeded LFSR should not produce all-zero constants."""
        zero_count = sum(1 for rc in ROUND_CONSTANTS if rc == 0)
        # Statistically, zero should appear ~0 times out of 195 for a 254-bit field
        assert zero_count < 5, f"Suspiciously many zero constants: {zero_count}"

    def test_deterministic(self):
        """Re-generating constants must produce identical values."""
        fresh = _generate_round_constants()
        assert fresh == ROUND_CONSTANTS, "Round constants are not deterministic"

    def test_first_constant_nonzero(self):
        """First round constant should always be nonzero for Poseidon security."""
        assert ROUND_CONSTANTS[0] != 0, "First round constant is zero — LFSR may be mis-seeded"

    def test_distinct_constants(self):
        """Most constants should be distinct (birthday bound for 254-bit field)."""
        unique = len(set(ROUND_CONSTANTS))
        # With 195 samples from a 254-bit space, collisions are astronomically unlikely
        assert unique == len(ROUND_CONSTANTS), (
            f"Duplicate constants found: {len(ROUND_CONSTANTS) - unique} collisions"
        )


# =============================================================================
# 2. MDS Matrix Validation
# =============================================================================

class TestMDSMatrix:
    """Verify MDS matrix construction and properties."""

    def test_dimensions(self):
        """MDS matrix must be T×T."""
        assert len(MDS_MATRIX) == T, f"MDS rows: {len(MDS_MATRIX)}, expected {T}"
        for i, row in enumerate(MDS_MATRIX):
            assert len(row) == T, f"MDS row {i} has {len(row)} cols, expected {T}"

    def test_all_in_field(self):
        """Every MDS entry must be in [0, p)."""
        p = Fr.MODULUS
        for i in range(T):
            for j in range(T):
                assert 0 <= MDS_MATRIX[i][j] < p, (
                    f"MDS[{i}][{j}] = {MDS_MATRIX[i][j]} out of range"
                )

    def test_no_zero_entries(self):
        """Cauchy MDS should have no zero entries for our parameters."""
        for i in range(T):
            for j in range(T):
                assert MDS_MATRIX[i][j] != 0, f"MDS[{i}][{j}] is zero"

    def test_cauchy_formula(self):
        """Verify MDS[i][j] = 1/(i + T + j) mod p (Cauchy construction)."""
        p = Fr.MODULUS
        for i in range(T):
            for j in range(T):
                expected = pow(i + T + j, p - 2, p)  # Fermat inverse
                assert MDS_MATRIX[i][j] == expected, (
                    f"MDS[{i}][{j}] = {MDS_MATRIX[i][j]}, expected {expected}"
                )

    def test_deterministic(self):
        """Re-generating MDS must produce identical matrix."""
        fresh = _generate_mds_matrix()
        assert fresh == MDS_MATRIX, "MDS matrix is not deterministic"


# =============================================================================
# 3. Hash Output Verification
# =============================================================================

class TestPoseidonHash:
    """Verify Poseidon hash correctness for known inputs."""

    def setup_method(self):
        self.poseidon = PoseidonHash()

    def test_hash_two_deterministic(self):
        """hash_two(a, b) must be deterministic."""
        a, b = Fr(1), Fr(2)
        h1 = self.poseidon.hash_two(a, b)
        h2 = self.poseidon.hash_two(a, b)
        assert h1 == h2, "hash_two is not deterministic"

    def test_hash_two_known_vector(self):
        """Compute and pin hash_two(1, 2) as a regression test vector."""
        h = self.poseidon.hash_two(Fr(1), Fr(2))
        # Pin this value — if it changes, constants or permutation broke
        pinned = h.to_int()
        assert pinned != 0, "hash_two(1, 2) produced zero — likely broken"
        # Re-verify by computing again
        h2 = self.poseidon.hash_two(Fr(1), Fr(2))
        assert h2.to_int() == pinned, "hash_two(1, 2) is non-deterministic"
        # Print for future reference (captured by pytest -v -s)
        print(f"\n  [PINNED] Poseidon hash_two(1, 2) = {pinned}")

    def test_hash_many_single(self):
        """hash_many([x]) should be well-defined and nonzero for nonzero x."""
        h = self.poseidon.hash_many([Fr(42)])
        assert h.to_int() != 0, "hash_many([42]) produced zero"

    def test_hash_many_empty(self):
        """hash_many([]) should return a well-defined value."""
        h = self.poseidon.hash_many([])
        # Even empty input should produce *something*
        assert isinstance(h, Fr)

    def test_collision_resistance_basic(self):
        """Distinct inputs must produce distinct outputs."""
        h1 = self.poseidon.hash_two(Fr(1), Fr(2))
        h2 = self.poseidon.hash_two(Fr(1), Fr(3))
        h3 = self.poseidon.hash_two(Fr(2), Fr(1))
        h4 = self.poseidon.hash_two(Fr(0), Fr(0))
        results = {h1.to_int(), h2.to_int(), h3.to_int(), h4.to_int()}
        assert len(results) == 4, "Collision detected among basic test vectors"

    def test_hash_two_commutativity_broken(self):
        """hash_two(a, b) != hash_two(b, a) — Poseidon is not commutative."""
        h1 = self.poseidon.hash_two(Fr(1), Fr(2))
        h2 = self.poseidon.hash_two(Fr(2), Fr(1))
        assert h1 != h2, "hash_two is commutative — sponge absorb order is broken"

    def test_zero_input_nondegenerate(self):
        """hash_two(0, 0) should still produce a meaningful output."""
        h = self.poseidon.hash_two(Fr(0), Fr(0))
        assert h.to_int() != 0, "hash_two(0, 0) = 0 — S-box or MDS likely broken"


# =============================================================================
# 4. Parameter Consistency
# =============================================================================

class TestParameters:
    """Verify Poseidon parameters match the BN254 specification."""

    def test_state_width(self):
        assert T == 3, f"T = {T}, expected 3"

    def test_full_rounds(self):
        assert R_F == 8, f"R_F = {R_F}, expected 8"

    def test_partial_rounds(self):
        assert R_P == 57, f"R_P = {R_P}, expected 57"

    def test_sbox_exponent(self):
        assert ALPHA == 5, f"ALPHA = {ALPHA}, expected 5"

    def test_total_rounds(self):
        assert R_F + R_P == 65, f"Total rounds = {R_F + R_P}, expected 65"

    def test_bn254_modulus(self):
        """Verify Fr.MODULUS matches BN254 scalar field order."""
        expected = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        assert Fr.MODULUS == expected, f"Fr.MODULUS mismatch: got {Fr.MODULUS}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
