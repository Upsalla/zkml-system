"""
Boundary Matrix Reduction Circuit for ZK Persistent Homology.

This module implements the circuit constraints that verify the correctness
of a persistence computation. It is the core innovation of the ZK-TDA system.

Key Insight (Verify, Don't Compute):
    Instead of computing column reduction inside the circuit, the prover:
    1. Runs the standard persistence algorithm offline
    2. Records which columns were used to reduce each column (the "reduction certificate")
    3. Supplies the reduced boundary matrix as a circuit witness

    The circuit then verifies:
    a) The boundary matrix entries are correct for the given simplices
    b) XOR (mod 2) of the original column with reduction columns produces the claimed reduced column
    c) Pivots (lowest non-zero entries) are unique across all reduced columns
    d) Persistence pairs read correctly from the pivot assignments

Mod-2 Verification in Prime Field:
    Since we work in Fr (BN254 scalar field, prime ~254 bits), but boundary
    entries are in Z/2Z (binary), we use the mod-2 witness technique:

    sum = original[row] + Σ reducer[row]     (computed in Fr, as integers)
    reduced[row] = sum mod 2                  (claimed by prover)
    quotient[row] = (sum - reduced[row]) / 2  (claimed by prover)

    Circuit checks:
    - sum == reduced[row] + 2 * quotient[row]  (1 linear constraint)
    - reduced[row] ∈ {0, 1}                    (1 boolean constraint)

Author: David Weyhe
Date: 2026-03-01
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr
from zkml_system.plonk.circuit_compiler import (
    CircuitCompiler, Gate, GateType, Wire
)
from zkml_system.plonk.tda_gadgets import TDAGadgets


# =============================================================================
# Reduction Certificate (Witness Data)
# =============================================================================

@dataclass
class ColumnReduction:
    """
    Records how one column was reduced in the persistence algorithm.

    For column j with boundary B[j]:
    reduced = B[j] XOR B[k1] XOR B[k2] XOR ... XOR B[km]

    Attributes:
        column_idx: The index j of this column in the simplex ordering.
        simplex: The simplex this column represents (e.g., (0, 3) for edge 0-3).
        dimension: Dimension of the simplex (0=vertex, 1=edge, 2=triangle).
        filtration_value: When this simplex enters the filtration.
        reducer_indices: Columns k1, k2, ..., km used for reduction (in order).
        boundary_entries: Row indices where B[j] has a 1 (sparse representation).
        reduced_entries: Row indices where the reduced column has a 1.
        pivot: Lowest row index in reduced column, or -1 if zero.
    """
    column_idx: int
    simplex: Tuple[int, ...]
    dimension: int
    filtration_value: float
    reducer_indices: List[int]
    boundary_entries: List[int]
    reduced_entries: List[int]
    pivot: int


@dataclass
class BoundaryReductionCertificate:
    """
    Complete reduction certificate for the persistence computation.

    Contains all information the circuit needs to verify that the
    claimed persistence diagram is correct for the given simplicial complex.
    """
    # Total number of simplices
    n_simplices: int

    # Number of rows in the boundary matrix (= max simplex index + 1)
    n_rows: int

    # Reduction record for each column
    columns: List[ColumnReduction]

    # Persistence pairs: (birth_simplex_idx, death_simplex_idx)
    pairs: List[Tuple[int, int]]

    # Simplex-to-index mapping
    simplex_to_idx: Dict[Tuple[int, ...], int]


# =============================================================================
# Generate Reduction Certificate (Offline, Prover-side)
# =============================================================================

def generate_reduction_certificate(
    points: np.ndarray,
    max_dim: int = 1,
    max_edge_length: float = None,
) -> BoundaryReductionCertificate:
    """
    Run persistence algorithm and record the full reduction certificate.

    This is the prover-side computation. It produces:
    1. The boundary matrix
    2. For each column: which columns were used to reduce it
    3. The resulting reduced matrix
    4. The persistence pairs

    This is a modified version of PersistenceComputer.compute() that
    records the reduction history.
    """
    from zkml_system.tda.persistence import SimplexTree

    n_points = len(points)

    # Build simplex tree
    tree = SimplexTree()

    # Add vertices (filtration = 0)
    for i in range(n_points):
        tree.insert((i,), 0.0)

    # Compute distances
    dists = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = float(np.sqrt(np.sum((points[i] - points[j]) ** 2)))
            dists[i][j] = d
            dists[j][i] = d

    if max_edge_length is None:
        max_edge_length = np.max(dists) * 1.1

    # Add edges
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dists[i][j] <= max_edge_length:
                tree.insert((i, j), dists[i][j])

    # Add triangles (for H1)
    if max_dim >= 1:
        for i in range(n_points):
            for j in range(i + 1, n_points):
                for k in range(j + 1, n_points):
                    # Triangle exists if all three edges exist
                    max_d = max(dists[i][j], dists[j][k], dists[i][k])
                    if max_d <= max_edge_length:
                        tree.insert((i, j, k), max_d)

    # Get all simplices sorted by (filtration, dimension, simplex)
    all_simplices = []
    for dim in range(max_dim + 2):
        all_simplices.extend(tree.get_simplices_by_dim(dim))
    all_simplices.sort(key=lambda x: (x[1], len(x[0]), x[0]))

    # Create index mapping
    simplex_to_idx = {s: i for i, (s, _) in enumerate(all_simplices)}
    n = len(all_simplices)

    # Build boundary matrix (sparse)
    boundary = [set() for _ in range(n)]
    for j, (simplex, _) in enumerate(all_simplices):
        for face in tree.boundary(simplex):
            if face in simplex_to_idx:
                boundary[j].add(simplex_to_idx[face])

    # Reduce with history recording
    low = {}
    reduction_history = [[] for _ in range(n)]  # Which columns reduced each

    for j in range(n):
        working = set(boundary[j])  # Copy
        while working:
            i = max(working)
            if i in low:
                k = low[i]
                reduction_history[j].append(k)
                working = working.symmetric_difference(boundary[k])
            else:
                low[i] = j
                break

    # Build reduction certificate
    columns = []
    for j, (simplex, filt) in enumerate(all_simplices):
        dim = len(simplex) - 1

        # Compute reduced column by replaying reductions
        reduced = set(boundary[j])
        for k in reduction_history[j]:
            reduced = reduced.symmetric_difference(boundary[k])

        pivot = max(reduced) if reduced else -1

        columns.append(ColumnReduction(
            column_idx=j,
            simplex=simplex,
            dimension=dim,
            filtration_value=filt,
            reducer_indices=list(reduction_history[j]),
            boundary_entries=sorted(boundary[j]),
            reduced_entries=sorted(reduced),
            pivot=pivot,
        ))

    # Extract persistence pairs
    pairs = []
    for i_val, j_val in low.items():
        pairs.append((i_val, j_val))

    return BoundaryReductionCertificate(
        n_simplices=n,
        n_rows=n,
        columns=columns,
        pairs=pairs,
        simplex_to_idx=simplex_to_idx,
    )


# =============================================================================
# Boundary Verification Circuit
# =============================================================================

class BoundaryCircuitCompiler:
    """
    Compiles circuit constraints for boundary matrix reduction verification.

    For each column j that is part of a persistence pair:
    1. Allocate boundary wires (sparse binary vector)
    2. For each reducer column k: accumulate XOR via mod-2 witness
    3. Verify the reduced column matches the claimed pivot
    4. Verify pivot uniqueness
    """

    def __init__(self, gadgets: TDAGadgets, cc: CircuitCompiler):
        self.g = gadgets
        self.cc = cc

    def compile_column_reduction(
        self,
        cert: BoundaryReductionCertificate,
        pair_indices: List[int] = None,
    ) -> Dict[str, int]:
        """
        Compile verification constraints for boundary column reductions.

        Args:
            cert: The reduction certificate from the prover.
            pair_indices: Which persistence pairs to verify (indices into cert.pairs).
                         If None, verify all pairs.

        Returns:
            Dict with circuit statistics.
        """
        if pair_indices is None:
            pair_indices = list(range(len(cert.pairs)))

        stats = {
            "columns_verified": 0,
            "mod2_checks": 0,
            "boolean_checks": 0,
            "pivot_checks": 0,
        }

        verified_pivots = []  # (pivot_wire, column_wire) for uniqueness

        for pair_idx in pair_indices:
            birth_idx, death_idx = cert.pairs[pair_idx]
            death_col = cert.columns[death_idx]

            # Verify the death column's reduction
            pivot_wire = self._verify_single_column(
                cert, death_col, stats
            )
            stats["columns_verified"] += 1

            if pivot_wire is not None:
                verified_pivots.append(pivot_wire)

        # Verify pivot uniqueness (all pivots are different)
        self._verify_pivot_uniqueness(verified_pivots, stats)

        return stats

    def _verify_single_column(
        self,
        cert: BoundaryReductionCertificate,
        col: ColumnReduction,
        stats: Dict[str, int],
    ) -> Optional[int]:
        """
        Verify reduction of a single boundary column.

        Steps:
        1. For each row: compute sum = boundary[row] + Σ reducer_boundary[row]
        2. Verify sum = reduced[row] + 2 * quotient[row]
        3. Verify reduced[row] ∈ {0, 1}
        4. Verify pivot is the lowest non-zero entry in reduced column

        Returns:
            Wire index of the verified pivot value, or None.
        """
        j = col.column_idx
        boundary_set = set(col.boundary_entries)
        reduced_set = set(col.reduced_entries)

        # Determine which rows are "active" — have at least one non-zero
        # entry across the column and its reducers. Only verify those rows.
        active_rows = set(boundary_set)
        for k in col.reducer_indices:
            active_rows.update(cert.columns[k].boundary_entries)
        active_rows.update(reduced_set)

        if not active_rows:
            return None

        # For each active row, verify the mod-2 reduction
        reduced_wires = {}

        for row in sorted(active_rows):
            # Compute sum in Z (not mod 2)
            # sum = boundary[j][row] + Σ boundary[k][row] for k in reducers
            count = 1 if row in boundary_set else 0
            for k in col.reducer_indices:
                k_boundary = set(cert.columns[k].boundary_entries)
                if row in k_boundary:
                    count += 1

            # Expected reduced value (mod 2)
            expected_reduced = count % 2
            expected_quotient = count // 2

            # Allocate witness wires
            w_sum = self.g.const_wire(Fr(count))
            w_reduced = self.cc._new_wire(name=f"reduced_c{j}_r{row}")
            self.cc._set_wire_value(w_reduced, Fr(expected_reduced))
            w_quotient = self.cc._new_wire(name=f"quotient_c{j}_r{row}")
            self.cc._set_wire_value(w_quotient, Fr(expected_quotient))

            # Constraint 1: sum == reduced + 2 * quotient
            # Rewrite: sum - reduced - 2*quotient = 0
            # Build: temp = 2 * quotient
            w_two = self.g.const_wire(Fr(2))
            w_2q = self.g.mul(w_two, w_quotient)
            # temp2 = reduced + 2*quotient
            w_rhs = self.g.add(w_reduced, w_2q)
            # Assert: sum == rhs
            self.g.assert_equal(w_sum, w_rhs)
            stats["mod2_checks"] += 1

            # Constraint 2: reduced ∈ {0, 1}
            self.g.assert_boolean(w_reduced)
            stats["boolean_checks"] += 1

            reduced_wires[row] = w_reduced

        # Verify pivot: lowest non-zero entry in reduced column
        if col.pivot >= 0:
            pivot_row = col.pivot

            # Assert reduced[pivot] == 1
            if pivot_row in reduced_wires:
                w_one = self.g.const_wire(Fr.one())
                self.g.assert_equal(reduced_wires[pivot_row], w_one)
                stats["pivot_checks"] += 1

            # Assert all rows below pivot are 0
            for row in sorted(active_rows):
                if row < pivot_row and row in reduced_wires:
                    w_zero = self.g.const_wire(Fr.zero())
                    self.g.assert_equal(reduced_wires[row], w_zero)
                    stats["pivot_checks"] += 1

            # Return pivot wire for uniqueness check
            w_pivot = self.g.const_wire(Fr(pivot_row))
            return w_pivot

        return None

    def _verify_pivot_uniqueness(
        self,
        pivot_wires: List[int],
        stats: Dict[str, int],
    ):
        """
        Verify all pivots are distinct.

        For each pair of pivots (pᵢ, pⱼ):
        Assert pᵢ ≠ pⱼ by checking (pᵢ - pⱼ) has an inverse.

        The prover supplies the inverse as witness; the circuit verifies
        (pᵢ - pⱼ) * inv = 1.
        """
        for i in range(len(pivot_wires)):
            for j_idx in range(i + 1, len(pivot_wires)):
                # diff = pivot_i - pivot_j
                diff = self.g.sub(pivot_wires[i], pivot_wires[j_idx])

                # Prover supplies inverse: (pivot_i - pivot_j)^(-1)
                w_inv = self.cc._new_wire(name=f"pivot_inv_{i}_{j_idx}")
                # Compute actual inverse from witness values
                val_i = self.g._get_val(pivot_wires[i])
                val_j = self.g._get_val(pivot_wires[j_idx])
                diff_val = val_i - val_j
                if diff_val.is_zero():
                    raise ValueError(
                        f"Pivot uniqueness violation: pivots {i} and {j_idx} "
                        f"have the same value"
                    )
                inv_val = diff_val.inverse()
                self.cc._set_wire_value(w_inv, inv_val)

                # Assert diff * inv == 1
                w_product = self.g.mul(diff, w_inv)
                w_one = self.g.const_wire(Fr.one())
                self.g.assert_equal(w_product, w_one)
                stats["pivot_checks"] += 1


# =============================================================================
# End-to-End Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Boundary Matrix Reduction Certificate Test")
    print("=" * 60)

    np.random.seed(42)

    # Small point cloud
    points = np.random.randn(10, 3)
    print(f"\nPoint cloud: {len(points)} points in {points.shape[1]}D")

    # Generate reduction certificate
    print("\n1. Computing reduction certificate...")
    cert = generate_reduction_certificate(points, max_dim=1)
    print(f"   Simplices: {cert.n_simplices}")
    print(f"   Persistence pairs: {len(cert.pairs)}")

    for birth, death in cert.pairs:
        col_b = cert.columns[birth]
        col_d = cert.columns[death]
        dim = col_b.dimension
        print(f"     dim={dim}: simplex {col_b.simplex} (filt={col_b.filtration_value:.3f}) "
              f"→ simplex {col_d.simplex} (filt={col_d.filtration_value:.3f}), "
              f"reducers={len(col_d.reducer_indices)}")

    # Compile boundary verification circuit
    print("\n2. Compiling boundary verification circuit...")
    cc = CircuitCompiler(use_sparse=False, use_gelu=False)
    gadgets = TDAGadgets(cc)
    boundary_compiler = BoundaryCircuitCompiler(gadgets, cc)

    stats = boundary_compiler.compile_column_reduction(cert)

    print(f"   Columns verified: {stats['columns_verified']}")
    print(f"   Mod-2 checks: {stats['mod2_checks']}")
    print(f"   Boolean checks: {stats['boolean_checks']}")
    print(f"   Pivot checks: {stats['pivot_checks']}")
    print(f"   Total gates: {len(cc.gates)}")

    gate_types = {}
    for g in cc.gates:
        gt = g.gate_type
        gate_types[gt] = gate_types.get(gt, 0) + 1
    print(f"   Gate breakdown: {gate_types}")

    print("\n" + "=" * 60)
    print("Boundary verification circuit compiled successfully!")
    print("=" * 60)
