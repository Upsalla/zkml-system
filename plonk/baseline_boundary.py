"""
Baseline Boundary Verification via Bit-Decomposition.

This provides a naive alternative to the mod-2 witness technique
used in tda_boundary.py. The purpose is to demonstrate the
constraint-count advantage of the mod-2 approach.

Naive approach: For each entry in the reduced column, verify 
that the sum is even by decomposing it into 254 bits and checking
the LSB. This costs 254 boolean constraints + 254 addition 
constraints per entry, compared to ~3 constraints for mod-2.

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.circuit_compiler import (
    CircuitCompiler, Gate, GateType, Wire, CompiledCircuit
)
from zkml_system.plonk.tda_gadgets import TDAGadgets
from zkml_system.plonk.tda_boundary import (
    BoundaryReductionCertificate, ColumnReduction
)


# Number of bits needed for BN254 Fr
FIELD_BITS = 254


@dataclass
class BaselineStats:
    """Statistics from baseline boundary verification."""
    total_gates: int
    boolean_gates: int      # b_i ∈ {0,1} constraints
    decomp_gates: int       # Σ b_i * 2^i == value reconstruction
    lsb_check_gates: int    # LSB == 0 constraints
    entries_verified: int   # Total entries checked
    columns_verified: int   # Number of columns


class BaselineBoundaryCompiler:
    """
    Naive bit-decomposition boundary verification.
    
    For each entry in a reduced boundary column, verify that the
    sum (original + reducers) is congruent to the reduced value mod 2
    by fully decomposing the sum into 254 bits and checking.
    
    This is the "textbook" approach that the mod-2 witness technique
    improves upon. Used only for benchmarking.
    """

    def __init__(self, cert: BoundaryReductionCertificate, max_pairs: int = 10):
        self.cert = cert
        self.max_pairs = max_pairs

    def compile(self) -> Tuple[CompiledCircuit, BaselineStats]:
        """
        Compile the baseline boundary verification circuit.
        
        Returns:
            Tuple of (CompiledCircuit, BaselineStats)
        """
        cc = CircuitCompiler(use_sparse=False, use_gelu=False)
        g = TDAGadgets(cc)

        stats = BaselineStats(
            total_gates=0,
            boolean_gates=0,
            decomp_gates=0,
            lsb_check_gates=0,
            entries_verified=0,
            columns_verified=0,
        )

        # Verify a subset of columns (same as mod-2 for fair comparison)
        pair_count = 0
        for col in self.cert.columns:
            if col.pivot is None or col.pivot < 0:
                continue
            if pair_count >= self.max_pairs:
                break
            
            self._verify_column_naive(cc, g, col, stats)
            pair_count += 1
            stats.columns_verified += 1

        stats.total_gates = len(cc.gates)
        return stats

    def _verify_column_naive(
        self,
        cc: CircuitCompiler,
        g: TDAGadgets,
        col: ColumnReduction,
        stats: BaselineStats,
    ):
        """
        Verify a single column using naive bit-decomposition.
        
        For each active row:
          1. Compute sum = boundary[row] + Σ reducer_boundary[row]
          2. Decompose sum into 254 bits: sum = Σ b_i * 2^i
          3. Check each b_i ∈ {0,1}  (254 constraints)
          4. Check reconstruction: Σ b_i * 2^i == sum  (254 muls + 253 adds)
          5. Check LSB matches reduced value  (1 constraint)
          
        Total per entry: ~254 boolean + 254 mul + 253 add + 1 = 762 gates
        """
        boundary_set = set(col.boundary_entries)
        reduced_set = set(col.reduced_entries)

        # Determine active rows
        active_rows = set(boundary_set)
        for k in col.reducer_indices:
            k_boundary = set(self.cert.columns[k].boundary_entries)
            active_rows.update(k_boundary)

        for row in sorted(active_rows):
            # Compute sum value
            sum_val = 1 if row in boundary_set else 0
            for k in col.reducer_indices:
                k_boundary = set(self.cert.columns[k].boundary_entries)
                if row in k_boundary:
                    sum_val += 1

            reduced_val = 1 if row in reduced_set else 0

            # Allocate sum wire
            w_sum = cc._new_wire(name=f"bl_sum_{col.column_idx}_{row}")
            cc._set_wire_value(w_sum, Fr(sum_val))

            # Bit decomposition: 254 bits
            bits = []
            remaining = sum_val
            for bit_idx in range(FIELD_BITS):
                b = remaining & 1
                remaining >>= 1

                w_bit = cc._new_wire(
                    name=f"bl_bit_{col.column_idx}_{row}_{bit_idx}"
                )
                cc._set_wire_value(w_bit, Fr(b))

                # Boolean constraint: b * (1 - b) = 0
                g.assert_boolean(w_bit)
                stats.boolean_gates += 1

                bits.append(w_bit)

            # Reconstruction: Σ b_i * 2^i == sum
            power_of_two = Fr.one()
            two = Fr(2)

            # Start accumulator with first bit * 1
            w_coeff = g.const_wire(power_of_two)
            acc = g.mul(w_coeff, bits[0])
            stats.decomp_gates += 2  # const + mul

            for bit_idx in range(1, FIELD_BITS):
                power_of_two = power_of_two * two
                w_coeff = g.const_wire(power_of_two)
                w_term = g.mul(w_coeff, bits[bit_idx])
                acc = g.add(acc, w_term)
                stats.decomp_gates += 3  # const + mul + add

            # Assert reconstruction matches sum
            g.assert_equal(acc, w_sum)
            stats.decomp_gates += 1

            # LSB check: bits[0] should equal reduced_val
            w_reduced = g.const_wire(Fr(reduced_val))
            g.assert_equal(bits[0], w_reduced)
            stats.lsb_check_gates += 2  # const + assert

            stats.entries_verified += 1


# =============================================================================
# Benchmark: mod-2 vs naive
# =============================================================================

def compare_approaches(n_points: int = 8, max_pairs: int = 3) -> Dict[str, int]:
    """
    Compare gate counts between mod-2 witness and naive bit-decomposition.
    
    Returns dict with gate counts for both approaches.
    """
    from zkml_system.plonk.tda_boundary import (
        generate_reduction_certificate,
        BoundaryCircuitCompiler,
    )
    import numpy as np

    # Generate a point cloud
    np.random.seed(42)
    points = np.random.randn(n_points, 2)

    # Generate certificate
    cert = generate_reduction_certificate(points, max_edge_length=2.0)

    # Select pair indices
    pair_indices = list(range(min(max_pairs, len(cert.pairs))))

    # Mod-2 approach
    cc_mod2 = CircuitCompiler(use_sparse=False, use_gelu=False)
    g_mod2 = TDAGadgets(cc_mod2)
    mod2_compiler = BoundaryCircuitCompiler(g_mod2, cc_mod2)
    mod2_stats = mod2_compiler.compile_column_reduction(cert, pair_indices)
    mod2_gate_count = len(cc_mod2.gates)

    # Naive approach
    baseline_compiler = BaselineBoundaryCompiler(cert, max_pairs=max_pairs)
    baseline_stats = baseline_compiler.compile()

    return {
        "n_points": n_points,
        "max_pairs": max_pairs,
        "mod2_gates": mod2_gate_count,
        "baseline_gates": baseline_stats.total_gates,
        "ratio": baseline_stats.total_gates / max(mod2_gate_count, 1),
        "mod2_stats": {
            "total": mod2_gate_count,
        },
        "baseline_stats": {
            "total": baseline_stats.total_gates,
            "boolean": baseline_stats.boolean_gates,
            "decomp": baseline_stats.decomp_gates,
            "lsb_checks": baseline_stats.lsb_check_gates,
            "entries": baseline_stats.entries_verified,
            "columns": baseline_stats.columns_verified,
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline vs Mod-2 Boundary Verification Comparison")
    print("=" * 60)

    for n_points, pairs in [(6, 2), (8, 3), (10, 5)]:
        result = compare_approaches(n_points, pairs)
        print(f"\n  Points={n_points}, Pairs={pairs}:")
        print(f"    Mod-2 gates:    {result['mod2_gates']:>8,}")
        print(f"    Baseline gates: {result['baseline_gates']:>8,}")
        print(f"    Ratio:          {result['ratio']:>8.1f}x")
        print(f"    Entries checked: {result['baseline_stats']['entries']}")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)
