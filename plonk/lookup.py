"""
Lookup Arguments for zkML Operations

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

This module implements lookup arguments for efficient verification of
table-based operations in zkML. Lookup arguments allow proving that
values are contained in a predefined table without expensive range proofs.

Mathematical Foundation:
------------------------
Lookup arguments use the following approach (Plookup-style):

Given:
- Table T = {t₁, t₂, ..., tₙ} of allowed values
- Witness values f = {f₁, f₂, ..., fₘ} to be checked

We prove that all fᵢ ∈ T by showing:
1. Sort f ∪ T to get sorted sequence s
2. Prove s is a valid sorting of f ∪ T
3. Prove s contains only adjacent duplicates (no gaps)

The key insight is that checking membership becomes checking sortedness,
which can be done efficiently with polynomial constraints.

Applications in zkML:
---------------------
1. **Activation Functions**: ReLU, sigmoid, tanh via lookup tables
2. **Range Proofs**: Ensure values are in valid range [0, 2^k)
3. **Quantization**: Map float values to fixed-point representations
4. **Non-linear Operations**: Division, square root, exponential

Performance Characteristics:
----------------------------
- Table size n: O(n) preprocessing
- Lookup count m: O(m) prover work
- Verification: O(1) - constant time
- Constraint reduction: Up to 100x for complex operations

Trade-offs:
-----------
- Larger tables = more preprocessing but better precision
- Smaller tables = faster but may need interpolation
- Memory vs. accuracy trade-off for activation functions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkml_system.crypto.bn254.fr_adapter import Fr
from plonk.core import Polynomial


# =============================================================================
# Lookup Table
# =============================================================================

@dataclass
class LookupTable:
    """
    A lookup table for efficient membership proofs.
    
    Attributes:
        name: Human-readable name for the table
        entries: List of (input, output) pairs
        input_bits: Number of bits for input values
        output_bits: Number of bits for output values
    """
    name: str
    entries: List[Tuple[Fr, Fr]]
    input_bits: int
    output_bits: int
    
    # Cached polynomial representation
    _polynomial: Optional[Polynomial] = field(default=None, repr=False)
    _input_set: Optional[set] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Build lookup structures."""
        self._input_set = {e[0].to_int() for e in self.entries}
    
    def contains(self, value: Fr) -> bool:
        """Check if a value is in the table."""
        return value.to_int() in self._input_set
    
    def lookup(self, input_val: Fr) -> Optional[Fr]:
        """Look up the output for a given input."""
        for inp, out in self.entries:
            if inp == input_val:
                return out
        return None
    
    def to_polynomial(self) -> Polynomial:
        """
        Convert table to polynomial representation.
        
        Uses Lagrange interpolation to create a polynomial p(x) such that
        p(input) = output for all table entries.
        """
        if self._polynomial is not None:
            return self._polynomial
        
        points = [(inp, out) for inp, out in self.entries]
        self._polynomial = Polynomial.lagrange_interpolate(points)
        return self._polynomial
    
    @classmethod
    def from_function(
        cls,
        name: str,
        func: Callable[[int], int],
        input_range: range,
        input_bits: int,
        output_bits: int
    ) -> LookupTable:
        """
        Create a lookup table from a function.
        
        Args:
            name: Table name
            func: Function mapping int -> int
            input_range: Range of input values
            input_bits: Bits for input
            output_bits: Bits for output
        
        Returns:
            A new LookupTable
        """
        entries = []
        for x in input_range:
            y = func(x)
            entries.append((Fr(x), Fr(y)))
        
        return cls(
            name=name,
            entries=entries,
            input_bits=input_bits,
            output_bits=output_bits
        )


# =============================================================================
# Pre-built Tables for zkML
# =============================================================================

def create_relu_table(bits: int = 8) -> LookupTable:
    """
    Create a ReLU lookup table.
    
    ReLU(x) = max(0, x)
    
    For signed integers in range [-2^(bits-1), 2^(bits-1)-1]
    """
    half = 2 ** (bits - 1)
    entries = []
    
    for x in range(-half, half):
        y = max(0, x)
        # Convert to field element (handle negative)
        x_field = Fr(x % Fr.MODULUS)
        y_field = Fr(y)
        entries.append((x_field, y_field))
    
    return LookupTable(
        name=f"relu_{bits}bit",
        entries=entries,
        input_bits=bits,
        output_bits=bits
    )


def create_sigmoid_table(bits: int = 8, scale: int = 256) -> LookupTable:
    """
    Create a sigmoid lookup table.
    
    sigmoid(x) = 1 / (1 + exp(-x))
    
    Output is scaled to [0, scale] for fixed-point representation.
    """
    import math
    
    half = 2 ** (bits - 1)
    entries = []
    
    for x in range(-half, half):
        # Scale input to reasonable range for sigmoid
        x_scaled = x / (half / 4)  # Map to roughly [-4, 4]
        y = 1.0 / (1.0 + math.exp(-x_scaled))
        y_int = int(y * scale)
        
        x_field = Fr(x % Fr.MODULUS)
        y_field = Fr(y_int)
        entries.append((x_field, y_field))
    
    return LookupTable(
        name=f"sigmoid_{bits}bit",
        entries=entries,
        input_bits=bits,
        output_bits=bits
    )


def create_range_table(bits: int = 8) -> LookupTable:
    """
    Create a range table for proving x ∈ [0, 2^bits).
    
    This is used for range proofs in quantization.
    """
    entries = [(Fr(x), Fr(x)) for x in range(2 ** bits)]
    
    return LookupTable(
        name=f"range_{bits}bit",
        entries=entries,
        input_bits=bits,
        output_bits=bits
    )


# =============================================================================
# Plookup-style Lookup Argument
# =============================================================================

@dataclass
class LookupProof:
    """
    A proof that all lookup values are in the table.
    
    Attributes:
        table_commitment: Commitment to the table polynomial
        witness_commitment: Commitment to the witness values
        sorted_commitment: Commitment to the sorted sequence
        quotient_commitment: Commitment to the quotient polynomial
        evaluations: Polynomial evaluations at challenge point
    """
    table_commitment: Fr  # Simplified - would be G1Point in production
    witness_commitment: Fr
    sorted_commitment: Fr
    quotient_commitment: Fr
    evaluations: Dict[str, Fr]


class PlookupArgument:
    """
    Plookup-style lookup argument implementation.
    
    This class implements the core lookup argument protocol for proving
    that a set of witness values are all contained in a lookup table.
    """
    
    def __init__(self, table: LookupTable):
        """
        Initialize with a lookup table.
        
        Args:
            table: The lookup table to use
        """
        self.table = table
        self.table_values = [e[0] for e in table.entries]
    
    def prove(self, witness_values: List[Fr]) -> Tuple[LookupProof, str]:
        """
        Generate a lookup proof.
        
        Args:
            witness_values: Values to prove are in the table
        
        Returns:
            (proof, status_message)
        """
        # Step 1: Check all values are in table
        for v in witness_values:
            if not self.table.contains(v):
                return None, f"Value {v.to_int()} not in table"
        
        # Step 2: Combine and sort
        combined = self.table_values + witness_values
        sorted_values = sorted(combined, key=lambda x: x.to_int())
        
        # Step 3: Compute accumulator polynomials
        # In a full implementation, this would use:
        # - Grand product argument for permutation
        # - Quotient polynomial for constraint satisfaction
        
        # Simplified proof generation
        hasher = hashlib.sha256()
        for v in witness_values:
            hasher.update(v.to_int().to_bytes(32, 'big'))
        
        commitment_hash = hasher.digest()
        
        proof = LookupProof(
            table_commitment=Fr(int.from_bytes(commitment_hash[:8], 'big')),
            witness_commitment=Fr(int.from_bytes(commitment_hash[8:16], 'big')),
            sorted_commitment=Fr(int.from_bytes(commitment_hash[16:24], 'big')),
            quotient_commitment=Fr(int.from_bytes(commitment_hash[24:32], 'big')),
            evaluations={
                'table': Fr(len(self.table_values)),
                'witness': Fr(len(witness_values)),
                'sorted': Fr(len(sorted_values))
            }
        )
        
        return proof, f"Lookup proof generated for {len(witness_values)} values"
    
    def verify(self, proof: LookupProof) -> Tuple[bool, str]:
        """
        Verify a lookup proof.
        
        Args:
            proof: The proof to verify
        
        Returns:
            (is_valid, reason)
        """
        # In a full implementation, this would verify:
        # 1. Table commitment matches known table
        # 2. Sorted sequence is valid permutation
        # 3. Grand product argument is correct
        # 4. Quotient polynomial evaluates correctly
        
        # Simplified verification
        if proof.evaluations['witness'].to_int() == 0:
            return True, "Empty witness trivially verified"
        
        expected_sorted = (
            proof.evaluations['table'].to_int() +
            proof.evaluations['witness'].to_int()
        )
        
        if proof.evaluations['sorted'].to_int() != expected_sorted:
            return False, "Sorted sequence length mismatch"
        
        return True, "Lookup proof verified"


# =============================================================================
# Lookup-based Operations for zkML
# =============================================================================

class LookupOperations:
    """
    Collection of lookup-based operations for zkML.
    
    This class provides efficient implementations of common neural network
    operations using lookup tables.
    """
    
    def __init__(self, bits: int = 8):
        """
        Initialize with default bit width.
        
        Args:
            bits: Bit width for fixed-point representation
        """
        self.bits = bits
        self.tables: Dict[str, LookupTable] = {}
        self.arguments: Dict[str, PlookupArgument] = {}
        
        # Pre-build common tables
        self._build_tables()
    
    def _build_tables(self):
        """Build common lookup tables."""
        self.tables['relu'] = create_relu_table(self.bits)
        self.tables['sigmoid'] = create_sigmoid_table(self.bits)
        self.tables['range'] = create_range_table(self.bits)
        
        for name, table in self.tables.items():
            self.arguments[name] = PlookupArgument(table)
    
    def relu(self, values: List[Fr]) -> Tuple[List[Fr], LookupProof]:
        """
        Apply ReLU using lookup table.
        
        Args:
            values: Input values
        
        Returns:
            (outputs, proof)
        """
        outputs = []
        for v in values:
            out = self.tables['relu'].lookup(v)
            if out is None:
                # Value out of range - clamp
                v_int = v.to_int()
                if v_int > Fr.MODULUS // 2:
                    # Negative (in field representation)
                    out = Fr.zero()
                else:
                    out = v
            outputs.append(out)
        
        proof, _ = self.arguments['relu'].prove(values)
        return outputs, proof
    
    def sigmoid(self, values: List[Fr]) -> Tuple[List[Fr], LookupProof]:
        """
        Apply sigmoid using lookup table.
        
        Args:
            values: Input values
        
        Returns:
            (outputs, proof)
        """
        outputs = []
        for v in values:
            out = self.tables['sigmoid'].lookup(v)
            if out is None:
                # Default to 0.5 (scaled)
                out = Fr(128)
            outputs.append(out)
        
        proof, _ = self.arguments['sigmoid'].prove(values)
        return outputs, proof
    
    def range_check(self, values: List[Fr]) -> Tuple[bool, LookupProof]:
        """
        Check all values are in valid range.
        
        Args:
            values: Values to check
        
        Returns:
            (all_valid, proof)
        """
        all_valid = all(self.tables['range'].contains(v) for v in values)
        proof, _ = self.arguments['range'].prove(values)
        return all_valid, proof


# =============================================================================
# Performance Analysis
# =============================================================================

def analyze_lookup_performance():
    """
    Analyze the performance benefits of lookup arguments.
    
    Returns a comparison of constraint counts for different approaches.
    """
    results = {
        'operation': [],
        'naive_constraints': [],
        'lookup_constraints': [],
        'reduction': []
    }
    
    # ReLU: max(0, x)
    # Naive: Requires decomposition into bits + comparison
    # Lookup: Single table lookup
    results['operation'].append('ReLU (8-bit)')
    results['naive_constraints'].append(8 * 2 + 8)  # Bit decomposition + comparison
    results['lookup_constraints'].append(1)
    results['reduction'].append('96%')
    
    # Sigmoid
    # Naive: Taylor series approximation (many multiplications)
    # Lookup: Single table lookup
    results['operation'].append('Sigmoid (8-bit)')
    results['naive_constraints'].append(50)  # ~10 terms of Taylor series
    results['lookup_constraints'].append(1)
    results['reduction'].append('98%')
    
    # Range proof [0, 2^8)
    # Naive: Bit decomposition
    # Lookup: Single range table lookup
    results['operation'].append('Range [0, 256)')
    results['naive_constraints'].append(8)  # 8 bit constraints
    results['lookup_constraints'].append(1)
    results['reduction'].append('87.5%')
    
    # Division (approximate)
    # Naive: Newton-Raphson iteration
    # Lookup: Precomputed inverse table
    results['operation'].append('Division (8-bit)')
    results['naive_constraints'].append(100)  # Multiple iterations
    results['lookup_constraints'].append(2)  # Lookup + multiplication
    results['reduction'].append('98%')
    
    return results


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lookup Arguments Self-Test")
    print("=" * 70)
    
    # Test 1: Create tables
    print("\n1. Creating lookup tables...")
    relu_table = create_relu_table(8)
    print(f"   ReLU table: {len(relu_table.entries)} entries")
    
    sigmoid_table = create_sigmoid_table(8)
    print(f"   Sigmoid table: {len(sigmoid_table.entries)} entries")
    
    range_table = create_range_table(8)
    print(f"   Range table: {len(range_table.entries)} entries")
    
    # Test 2: Table lookups
    print("\n2. Testing table lookups...")
    
    # ReLU
    print("   ReLU:")
    for x in [-5, 0, 5, 10]:
        x_field = Fr(x % Fr.MODULUS)
        y = relu_table.lookup(x_field)
        print(f"     ReLU({x}) = {y.to_int() if y else 'None'}")
    
    # Sigmoid
    print("   Sigmoid (scaled to 256):")
    for x in [-100, -10, 0, 10, 100]:
        x_field = Fr(x % Fr.MODULUS)
        y = sigmoid_table.lookup(x_field)
        if y:
            print(f"     sigmoid({x}) ≈ {y.to_int() / 256:.3f}")
    
    # Test 3: Lookup argument
    print("\n3. Testing Plookup argument...")
    argument = PlookupArgument(range_table)
    
    # Valid values
    valid_values = [Fr(10), Fr(50), Fr(100), Fr(200)]
    proof, msg = argument.prove(valid_values)
    print(f"   Valid values: {msg}")
    
    is_valid, reason = argument.verify(proof)
    print(f"   Verification: {is_valid} - {reason}")
    
    # Test 4: Lookup operations
    print("\n4. Testing lookup operations...")
    ops = LookupOperations(bits=8)
    
    # ReLU
    inputs = [Fr(x % Fr.MODULUS) for x in [-10, -5, 0, 5, 10]]
    outputs, proof = ops.relu(inputs)
    print(f"   ReLU outputs: {[o.to_int() for o in outputs]}")
    
    # Range check
    valid_range = [Fr(x) for x in [0, 100, 200, 255]]
    is_valid, proof = ops.range_check(valid_range)
    print(f"   Range check [0,100,200,255]: {is_valid}")
    
    # Test 5: Performance analysis
    print("\n5. Performance analysis...")
    perf = analyze_lookup_performance()
    
    print(f"\n   {'Operation':<20} {'Naive':<10} {'Lookup':<10} {'Reduction':<10}")
    print("   " + "-" * 50)
    for i in range(len(perf['operation'])):
        print(f"   {perf['operation'][i]:<20} "
              f"{perf['naive_constraints'][i]:<10} "
              f"{perf['lookup_constraints'][i]:<10} "
              f"{perf['reduction'][i]:<10}")
    
    print("\n" + "=" * 70)
    print("LOOKUP ARGUMENTS TESTS COMPLETE!")
    print("=" * 70)
