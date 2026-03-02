"""
Poseidon Hash for BN254 Scalar Field (Fr).

Implements the Poseidon algebraic hash function as specified in:
  "POSEIDON: A New Hash Function for Zero-Knowledge Proof Systems"
  Grassi, Khovratovich, Rechberger, Roy, Schofnegger (USENIX 2021)

Configuration:
  - Field: BN254 Fr (p = 21888...001)
  - Width: t = 3 (2-to-1 compression, rate r = 2, capacity c = 1)
  - S-box exponent: α = 5  (x^5, since gcd(5, p-1) = 1)
  - Full rounds: R_F = 8  (4 at start + 4 at end)
  - Partial rounds: R_P = 57  (for 128-bit security at 254-bit field)
  - Round constants: Generated via Grain-LFSR per the specification
  - MDS matrix: 3×3 Cauchy matrix over Fr

This module provides:
  1. PoseidonHash — pure-Python offline hash (for witness generation)
  2. PoseidonGadget — in-circuit Poseidon (adds PLONK gates)
  3. poseidon_sponge — variable-length input via sponge construction

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr

# =============================================================================
# Poseidon Parameters for BN254 Fr, t=3, α=5
# =============================================================================

# Security: 128 bits
# t = 3  (state width)
# R_F = 8  (full rounds, 4+4)
# R_P = 57  (partial rounds)
# Total rounds = 65, each needing t=3 round constants → 195 constants

T = 3       # State width
R_F = 8     # Full rounds
R_P = 57    # Partial rounds
ALPHA = 5   # S-box exponent


def _generate_round_constants() -> List[int]:
    """
    Generate Poseidon round constants via Grain-LFSR.

    Implements the Grain-LFSR construction specified in:
      "POSEIDON: A New Hash Function for Zero-Knowledge Proof Systems"
      Section 4.1 - Round Constant Generation

    Uses a packed 80-bit integer LFSR for performance (avoids Python
    list overhead). Constants are cached to a JSON file after first
    generation to avoid recomputation on subsequent module loads.
    """
    import json
    
    # --- Check file cache first ---
    cache_path = os.path.join(os.path.dirname(__file__), '_poseidon_constants_cache.json')
    try:
        with open(cache_path, 'r') as f:
            cached = json.load(f)
            if (cached.get('t') == T and cached.get('rf') == R_F and
                cached.get('rp') == R_P and cached.get('alpha') == ALPHA and
                len(cached.get('constants', [])) == (R_F + R_P) * T):
                return cached['constants']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    p = Fr.MODULUS
    field_size = 254  # BN254 scalar field
    n_constants = (R_F + R_P) * T  # 65 * 3 = 195

    # --- Initialize 80-bit LFSR as packed integer ---
    # Bit layout (MSB first): [field_type:2][sbox:4][field_size:12][t:12][rf:10][rp:10][pad:30]
    # Packed into an 80-bit integer: bit 79 = MSB, bit 0 = LSB
    init_bits = []
    init_bits += [1, 0]  # prime field
    init_bits += [1, 0, 0, 0]  # S-box x^alpha
    init_bits += [(field_size >> (11 - i)) & 1 for i in range(12)]
    init_bits += [(T >> (11 - i)) & 1 for i in range(12)]
    init_bits += [(R_F >> (9 - i)) & 1 for i in range(10)]
    init_bits += [(R_P >> (9 - i)) & 1 for i in range(10)]
    init_bits += [1] * 30

    # Pack into 80-bit integer: bit 79 = init_bits[0], bit 0 = init_bits[79]
    state = 0
    for b in init_bits:
        state = (state << 1) | b
    MASK_80 = (1 << 80) - 1

    def clock():
        """Clock LFSR once, return output bit (bit 79 / MSB)."""
        nonlocal state
        # Output = MSB (bit 79)
        out = (state >> 79) & 1
        # Feedback: taps at positions 79, 66, 56, 41, 28, 17 (counting from LSB=0)
        # These correspond to original list indices 0, 13, 23, 38, 51, 62
        fb = (
            ((state >> 79) ^ (state >> 66) ^ (state >> 56)
             ^ (state >> 41) ^ (state >> 28) ^ (state >> 17)) & 1
        )
        state = ((state << 1) | fb) & MASK_80
        return out

    # Warm-up: discard first 160 bits
    for _ in range(160):
        clock()

    # Generate constants
    constants = []
    while len(constants) < n_constants:
        # Rejection: if "new" bit is 0, discard field_size bits
        while clock() == 0:
            for _ in range(field_size):
                clock()

        # Accumulate field_size bits
        val = 0
        for _ in range(field_size):
            val = (val << 1) | clock()

        # Accept only if val < p
        if val < p:
            constants.append(val)

    # --- Cache to file ---
    try:
        with open(cache_path, 'w') as f:
            json.dump({'t': T, 'rf': R_F, 'rp': R_P, 'alpha': ALPHA,
                       'constants': constants}, f)
    except OSError:
        pass  # Non-fatal: cache write failure

    return constants


def _generate_mds_matrix() -> List[List[int]]:
    """
    Generate t×t MDS (Cauchy) matrix over Fr.
    
    M[i][j] = 1 / (x_i + y_j)  where x_i = i, y_j = t + j
    
    This construction guarantees MDS property (all square 
    submatrices are invertible) as long as all x_i + y_j are distinct
    and non-zero, which holds for our small t.
    """
    p = Fr.MODULUS
    matrix = []
    for i in range(T):
        row = []
        for j in range(T):
            # x_i = i, y_j = T + j → sum = i + T + j  (always > 0 for our range)
            val = pow(i + T + j, p - 2, p)  # Modular inverse via Fermat
            row.append(val)
        matrix.append(row)
    return matrix


# Pre-compute constants at module load
ROUND_CONSTANTS: List[int] = _generate_round_constants()
MDS_MATRIX: List[List[int]] = _generate_mds_matrix()


# =============================================================================
# Pure-Python Poseidon Hash (offline, for witness generation)
# =============================================================================

class PoseidonHash:
    """
    Pure-Python Poseidon hash over BN254 Fr.
    
    Used by the prover to compute hashes offline. The same computation
    is replicated in-circuit by PoseidonGadget.
    """
    
    @staticmethod
    def permutation(state: List[Fr]) -> List[Fr]:
        """
        Apply the Poseidon permutation to a state of t field elements.
        
        Structure:
          - 4 full rounds (all t S-boxes active)
          - 57 partial rounds (only first S-box active)
          - 4 full rounds
        
        Each round:
          1. Add round constants
          2. Apply S-box (x^5) to all/first element
          3. Multiply by MDS matrix
        """
        assert len(state) == T, f"State must have {T} elements, got {len(state)}"
        
        state = list(state)  # Copy
        rc_offset = 0
        
        # --- First R_F/2 full rounds ---
        for r in range(R_F // 2):
            # Add round constants
            for i in range(T):
                state[i] = state[i] + Fr(ROUND_CONSTANTS[rc_offset + i])
            rc_offset += T
            
            # Full S-box: apply x^5 to all elements
            for i in range(T):
                state[i] = _sbox(state[i])
            
            # MDS multiplication
            state = _mds_multiply(state)
        
        # --- R_P partial rounds ---
        for r in range(R_P):
            # Add round constants
            for i in range(T):
                state[i] = state[i] + Fr(ROUND_CONSTANTS[rc_offset + i])
            rc_offset += T
            
            # Partial S-box: only first element
            state[0] = _sbox(state[0])
            
            # MDS multiplication
            state = _mds_multiply(state)
        
        # --- Last R_F/2 full rounds ---
        for r in range(R_F // 2):
            # Add round constants
            for i in range(T):
                state[i] = state[i] + Fr(ROUND_CONSTANTS[rc_offset + i])
            rc_offset += T
            
            # Full S-box
            for i in range(T):
                state[i] = _sbox(state[i])
            
            # MDS multiplication
            state = _mds_multiply(state)
        
        return state
    
    @staticmethod
    def hash_two(a: Fr, b: Fr) -> Fr:
        """Hash two field elements (2-to-1 compression)."""
        state = [Fr.zero(), a, b]  # capacity=0, rate elements a,b
        state = PoseidonHash.permutation(state)
        return state[0]
    
    @staticmethod
    def hash_many(inputs: List[Fr]) -> Fr:
        """
        Hash variable-length input via sponge construction.
        
        Absorb rate=2 elements at a time, squeeze output from state[0].
        """
        # Pad to multiple of rate (r=2)
        padded = list(inputs)
        if len(padded) % 2 != 0:
            padded.append(Fr.zero())
        
        # Initialize state: all zeros
        state = [Fr.zero()] * T
        
        # Absorb phase
        for i in range(0, len(padded), 2):
            state[1] = state[1] + padded[i]
            state[2] = state[2] + padded[i + 1]
            state = PoseidonHash.permutation(state)
        
        # Squeeze: return first element
        return state[0]


def _sbox(x: Fr) -> Fr:
    """Poseidon S-box: x → x^5."""
    x2 = x * x
    x4 = x2 * x2
    return x4 * x


def _mds_multiply(state: List[Fr]) -> List[Fr]:
    """Multiply state vector by MDS matrix."""
    result = []
    for i in range(T):
        acc = Fr.zero()
        for j in range(T):
            acc = acc + state[j] * Fr(MDS_MATRIX[i][j])
        result.append(acc)
    return result


# =============================================================================
# In-Circuit Poseidon Gadget (adds PLONK gates)
# =============================================================================

class PoseidonGadget:
    """
    Poseidon hash as circuit gadgets.
    
    Each call adds gates to the underlying CircuitCompiler.
    The in-circuit computation mirrors PoseidonHash exactly,
    ensuring offline hash == in-circuit hash.
    """
    
    def __init__(self, tda_gadgets):
        """
        Args:
            tda_gadgets: TDAGadgets instance (provides sub, mul, add, const_wire)
        """
        self.g = tda_gadgets
        self.cc = tda_gadgets.cc
    
    def permutation(self, state_wires: List[int]) -> List[int]:
        """
        Apply Poseidon permutation in-circuit.
        
        Args:
            state_wires: List of T wire indices representing the state.
        
        Returns:
            List of T wire indices for the output state.
        
        Gate cost per round:
          - Add round constants: T const_wires + T adds = 2T gates
          - S-box (x^5): 2 muls per element = 2T (full) or 2 (partial)
          - MDS: T² muls + T(T-1) adds ≈ 2T² gates
          Full round: ~2T + 2T + 2T² = 2T² + 4T  (for T=3: 30 gates)
          Partial round: ~2T + 2 + 2T² = 2T² + 2T + 2  (for T=3: 26 gates)
          
          Total: 4 × 30 + 57 × 26 + 4 × 30 = 120 + 1482 + 120 = 1722 gates
        """
        assert len(state_wires) == T
        
        state = list(state_wires)
        rc_offset = 0
        
        # --- First R_F/2 full rounds ---
        for r in range(R_F // 2):
            state = self._add_round_constants(state, rc_offset)
            rc_offset += T
            
            state = self._full_sbox(state)
            state = self._mds_multiply(state)
        
        # --- R_P partial rounds ---
        for r in range(R_P):
            state = self._add_round_constants(state, rc_offset)
            rc_offset += T
            
            state[0] = self._sbox_wire(state[0])
            state = self._mds_multiply(state)
        
        # --- Last R_F/2 full rounds ---
        for r in range(R_F // 2):
            state = self._add_round_constants(state, rc_offset)
            rc_offset += T
            
            state = self._full_sbox(state)
            state = self._mds_multiply(state)
        
        return state
    
    def hash_two(self, w_a: int, w_b: int) -> int:
        """Hash two field elements in-circuit."""
        w_zero = self.g.const_wire(Fr.zero())
        state = self.permutation([w_zero, w_a, w_b])
        return state[0]
    
    def hash_many(self, input_wires: List[int]) -> int:
        """
        Sponge construction for variable-length input.
        
        Gate cost: ~1722 × ceil(len/2) gates.
        """
        # Pad to multiple of rate
        padded = list(input_wires)
        if len(padded) % 2 != 0:
            padded.append(self.g.const_wire(Fr.zero()))
        
        # Initialize state
        state = [self.g.const_wire(Fr.zero()) for _ in range(T)]
        
        # Absorb
        for i in range(0, len(padded), 2):
            state[1] = self.g.add(state[1], padded[i])
            state[2] = self.g.add(state[2], padded[i + 1])
            state = self.permutation(state)
        
        return state[0]
    
    # --- Internal methods ---
    
    def _add_round_constants(self, state: List[int], rc_offset: int) -> List[int]:
        """Add round constants to state."""
        result = []
        for i in range(T):
            w_rc = self.g.const_wire(Fr(ROUND_CONSTANTS[rc_offset + i]))
            result.append(self.g.add(state[i], w_rc))
        return result
    
    def _sbox_wire(self, w: int) -> int:
        """Apply x^5 S-box to a single wire. Cost: 2 MUL gates."""
        w2 = self.g.mul(w, w)        # x²
        w4 = self.g.mul(w2, w2)      # x⁴
        w5 = self.g.mul(w4, w)       # x⁵
        return w5
    
    def _full_sbox(self, state: List[int]) -> List[int]:
        """Apply S-box to all state elements."""
        return [self._sbox_wire(w) for w in state]
    
    def _mds_multiply(self, state: List[int]) -> List[int]:
        """Multiply state by MDS matrix in-circuit."""
        result = []
        for i in range(T):
            # acc = Σ M[i][j] * state[j]
            w_coeff_0 = self.g.const_wire(Fr(MDS_MATRIX[i][0]))
            acc = self.g.mul(w_coeff_0, state[0])
            
            for j in range(1, T):
                w_coeff = self.g.const_wire(Fr(MDS_MATRIX[i][j]))
                w_term = self.g.mul(w_coeff, state[j])
                acc = self.g.add(acc, w_term)
            
            result.append(acc)
        return result


# =============================================================================
# Convenience Function
# =============================================================================

def poseidon_hash(inputs: List[Fr]) -> Fr:
    """Compute Poseidon hash of a list of field elements."""
    if len(inputs) == 2:
        return PoseidonHash.hash_two(inputs[0], inputs[1])
    return PoseidonHash.hash_many(inputs)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Poseidon Hash Self-Test (BN254 Fr)")
    print("=" * 60)
    
    # Test 1: Basic hash
    a = Fr(1)
    b = Fr(2)
    h = PoseidonHash.hash_two(a, b)
    print(f"\n  hash(1, 2) = 0x{h.to_int():064x}")
    
    # Test 2: Determinism
    h2 = PoseidonHash.hash_two(Fr(1), Fr(2))
    assert h == h2, "Non-deterministic!"
    print(f"  Determinism: ✅")
    
    # Test 3: Different inputs → different outputs
    h3 = PoseidonHash.hash_two(Fr(1), Fr(3))
    assert h != h3, "Collision!"
    print(f"  Collision resistance: ✅ (h(1,2) ≠ h(1,3))")
    
    # Test 4: Sponge (many inputs)
    inputs = [Fr(i) for i in range(10)]
    h4 = PoseidonHash.hash_many(inputs)
    print(f"  hash(0..9) = 0x{h4.to_int():064x}")
    
    # Test 5: In-circuit matches offline
    from zkml_system.plonk.circuit_compiler import CircuitCompiler
    from zkml_system.plonk.tda_gadgets import TDAGadgets
    
    cc = CircuitCompiler(use_sparse=False, use_gelu=False)
    gadgets = TDAGadgets(cc)
    pg = PoseidonGadget(gadgets)
    
    w_a = cc._new_wire(name="a")
    cc._set_wire_value(w_a, Fr(1))
    w_b = cc._new_wire(name="b")
    cc._set_wire_value(w_b, Fr(2))
    
    w_h = pg.hash_two(w_a, w_b)
    circuit_value = cc.wires[w_h].value
    
    assert circuit_value == h, \
        f"Mismatch! offline={h.to_int()} circuit={circuit_value.to_int()}"
    print(f"  In-circuit == offline: ✅")
    
    print(f"\n  Total gates for hash_two: {len(cc.gates)}")
    print(f"  Total wires: {len(cc.wires)}")
    
    print("\n" + "=" * 60)
    print("All Poseidon self-tests passed!")
    print("=" * 60)

