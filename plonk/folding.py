"""
Nova-based Folding Scheme for Incremental Verifiable Computation (IVC)

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

This module implements a simplified Nova-style folding scheme for zkML.
Folding schemes allow us to compress multiple proof instances into one,
enabling efficient verification of iterative computations like neural
network inference.

Mathematical Foundation:
------------------------
Nova uses a relaxed R1CS representation where:
- Standard R1CS: A·z ∘ B·z = C·z
- Relaxed R1CS: A·z ∘ B·z = u·C·z + E

Where:
- z = (W, x, 1) is the extended witness
- u is a scalar (u=1 for standard R1CS)
- E is an error vector (E=0 for standard R1CS)

Folding combines two instances (u₁, E₁, W₁) and (u₂, E₂, W₂) into:
- u' = u₁ + r·u₂
- E' = E₁ + r·T + r²·E₂
- W' = W₁ + r·W₂

Where T is a cross-term and r is a random challenge.

Key Benefits:
-------------
1. Constant proof size regardless of computation depth
2. Efficient recursive composition
3. No trusted setup per-circuit (uses universal SRS)

Performance Characteristics:
----------------------------
- Folding: O(n) field operations where n = witness size
- Verification: O(1) - constant time regardless of depth
- Memory: O(n) - stores only current accumulated instance

Limitations:
------------
1. Requires relaxed R1CS representation
2. Cross-term computation can be expensive
3. Final proof still requires a SNARK for the accumulated instance
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkml_system.crypto.bn254.fr_adapter import Fr
from crypto.bn254.curve import G1Point
from plonk.core import Polynomial, KZGCommitment


# =============================================================================
# Relaxed R1CS Representation
# =============================================================================

@dataclass
class R1CSMatrix:
    """
    Sparse representation of an R1CS matrix.
    
    Stores only non-zero entries as (row, col, value) tuples.
    """
    rows: int
    cols: int
    entries: List[Tuple[int, int, Fr]] = field(default_factory=list)
    
    def add_entry(self, row: int, col: int, value: Fr):
        """Add a non-zero entry."""
        if value != Fr.zero():
            self.entries.append((row, col, value))
    
    def multiply_vector(self, v: List[Fr]) -> List[Fr]:
        """Multiply matrix by vector."""
        result = [Fr.zero() for _ in range(self.rows)]
        for row, col, val in self.entries:
            result[row] = result[row] + val * v[col]
        return result


@dataclass
class R1CS:
    """
    R1CS constraint system: A·z ∘ B·z = C·z
    
    Where z = (1, x, W) is the extended witness vector.
    """
    A: R1CSMatrix
    B: R1CSMatrix
    C: R1CSMatrix
    num_constraints: int
    num_variables: int
    num_public_inputs: int
    
    def is_satisfied(self, witness: List[Fr], public_inputs: List[Fr]) -> bool:
        """Check if the R1CS is satisfied by the given witness."""
        # Construct z = (1, public_inputs, witness)
        z = [Fr.one()] + public_inputs + witness
        
        # Compute A·z, B·z, C·z
        az = self.A.multiply_vector(z)
        bz = self.B.multiply_vector(z)
        cz = self.C.multiply_vector(z)
        
        # Check A·z ∘ B·z = C·z
        for i in range(self.num_constraints):
            if az[i] * bz[i] != cz[i]:
                return False
        return True


@dataclass
class RelaxedR1CSInstance:
    """
    A relaxed R1CS instance.
    
    Relaxed R1CS: A·z ∘ B·z = u·C·z + E
    
    Attributes:
        commitment_W: Commitment to the witness W
        commitment_E: Commitment to the error vector E
        u: Scalar (u=1 for standard R1CS)
        x: Public inputs
    """
    commitment_W: G1Point
    commitment_E: G1Point
    u: Fr
    x: List[Fr]
    
    @classmethod
    def from_standard(cls, commitment_W: G1Point, x: List[Fr]) -> RelaxedR1CSInstance:
        """Create a relaxed instance from a standard R1CS instance."""
        return cls(
            commitment_W=commitment_W,
            commitment_E=G1Point.identity(),  # E = 0 for standard
            u=Fr.one(),  # u = 1 for standard
            x=x
        )


@dataclass
class RelaxedR1CSWitness:
    """
    A relaxed R1CS witness.
    
    Attributes:
        W: The witness vector
        E: The error vector
    """
    W: List[Fr]
    E: List[Fr]
    
    @classmethod
    def from_standard(cls, W: List[Fr], num_constraints: int) -> RelaxedR1CSWitness:
        """Create a relaxed witness from a standard witness."""
        return cls(
            W=W,
            E=[Fr.zero() for _ in range(num_constraints)]
        )


# =============================================================================
# Nova Folding Scheme
# =============================================================================

class NovaFolding:
    """
    Nova-style folding scheme for incremental verifiable computation.
    
    This class implements the core folding operation that combines two
    relaxed R1CS instances into one.
    """
    
    def __init__(self, r1cs: R1CS):
        """
        Initialize the folding scheme with an R1CS.
        
        Args:
            r1cs: The R1CS constraint system
        """
        self.r1cs = r1cs
    
    def compute_cross_term(
        self,
        instance1: RelaxedR1CSInstance,
        witness1: RelaxedR1CSWitness,
        instance2: RelaxedR1CSInstance,
        witness2: RelaxedR1CSWitness
    ) -> List[Fr]:
        """
        Compute the cross-term T for folding.
        
        T = A·z₁ ∘ B·z₂ + A·z₂ ∘ B·z₁ - u₁·C·z₂ - u₂·C·z₁
        
        This is the key computation in Nova that enables folding.
        """
        # Construct z vectors
        z1 = [Fr.one()] + instance1.x + witness1.W
        z2 = [Fr.one()] + instance2.x + witness2.W
        
        # Compute matrix-vector products
        az1 = self.r1cs.A.multiply_vector(z1)
        az2 = self.r1cs.A.multiply_vector(z2)
        bz1 = self.r1cs.B.multiply_vector(z1)
        bz2 = self.r1cs.B.multiply_vector(z2)
        cz1 = self.r1cs.C.multiply_vector(z1)
        cz2 = self.r1cs.C.multiply_vector(z2)
        
        # Compute T = A·z₁ ∘ B·z₂ + A·z₂ ∘ B·z₁ - u₁·C·z₂ - u₂·C·z₁
        T = []
        for i in range(self.r1cs.num_constraints):
            t_i = (
                az1[i] * bz2[i] +
                az2[i] * bz1[i] -
                instance1.u * cz2[i] -
                instance2.u * cz1[i]
            )
            T.append(t_i)
        
        return T
    
    def fold_instances(
        self,
        instance1: RelaxedR1CSInstance,
        instance2: RelaxedR1CSInstance,
        commitment_T: G1Point,
        r: Fr
    ) -> RelaxedR1CSInstance:
        """
        Fold two instances into one.
        
        Args:
            instance1: First relaxed R1CS instance
            instance2: Second relaxed R1CS instance
            commitment_T: Commitment to the cross-term T
            r: Random folding challenge
        
        Returns:
            The folded instance
        """
        # u' = u₁ + r·u₂
        u_prime = instance1.u + r * instance2.u
        
        # x' = x₁ + r·x₂
        x_prime = [
            instance1.x[i] + r * instance2.x[i]
            for i in range(len(instance1.x))
        ]
        
        # commitment_W' = commitment_W₁ + r·commitment_W₂
        commitment_W_prime = instance1.commitment_W + (instance2.commitment_W * r)
        
        # commitment_E' = commitment_E₁ + r·commitment_T + r²·commitment_E₂
        r_sq = r * r
        commitment_E_prime = (
            instance1.commitment_E +
            (commitment_T * r) +
            (instance2.commitment_E * r_sq)
        )
        
        return RelaxedR1CSInstance(
            commitment_W=commitment_W_prime,
            commitment_E=commitment_E_prime,
            u=u_prime,
            x=x_prime
        )
    
    def fold_witnesses(
        self,
        witness1: RelaxedR1CSWitness,
        witness2: RelaxedR1CSWitness,
        T: List[Fr],
        r: Fr
    ) -> RelaxedR1CSWitness:
        """
        Fold two witnesses into one.
        
        Args:
            witness1: First relaxed R1CS witness
            witness2: Second relaxed R1CS witness
            T: The cross-term
            r: Random folding challenge
        
        Returns:
            The folded witness
        """
        # W' = W₁ + r·W₂
        W_prime = [
            witness1.W[i] + r * witness2.W[i]
            for i in range(len(witness1.W))
        ]
        
        # E' = E₁ + r·T + r²·E₂
        r_sq = r * r
        E_prime = [
            witness1.E[i] + r * T[i] + r_sq * witness2.E[i]
            for i in range(len(witness1.E))
        ]
        
        return RelaxedR1CSWitness(W=W_prime, E=E_prime)
    
    def generate_challenge(
        self,
        instance1: RelaxedR1CSInstance,
        instance2: RelaxedR1CSInstance,
        commitment_T: G1Point
    ) -> Fr:
        """
        Generate the folding challenge using Fiat-Shamir.
        
        Args:
            instance1: First instance
            instance2: Second instance
            commitment_T: Commitment to cross-term
        
        Returns:
            The random challenge r
        """
        hasher = hashlib.sha256()
        
        # Hash instance1
        hasher.update(instance1.u.to_int().to_bytes(32, 'big'))
        for x in instance1.x:
            hasher.update(x.to_int().to_bytes(32, 'big'))
        
        # Hash instance2
        hasher.update(instance2.u.to_int().to_bytes(32, 'big'))
        for x in instance2.x:
            hasher.update(x.to_int().to_bytes(32, 'big'))
        
        # Hash commitment_T
        if not commitment_T.is_identity():
            tx, ty = commitment_T.to_affine()
            hasher.update(tx.to_int().to_bytes(32, 'big'))
            hasher.update(ty.to_int().to_bytes(32, 'big'))
        
        r_bytes = hasher.digest()
        return Fr(int.from_bytes(r_bytes, 'big') % Fr.MODULUS)


# =============================================================================
# IVC (Incremental Verifiable Computation) for zkML
# =============================================================================

@dataclass
class IVCProof:
    """
    An IVC proof for a sequence of computations.
    
    Attributes:
        accumulated_instance: The accumulated relaxed R1CS instance
        accumulated_witness: The accumulated witness (prover-side only)
        num_steps: Number of computation steps folded
        final_output: The final computation output
    """
    accumulated_instance: RelaxedR1CSInstance
    accumulated_witness: Optional[RelaxedR1CSWitness]
    num_steps: int
    final_output: List[Fr]


class ZkMLIVC:
    """
    IVC wrapper for zkML inference.
    
    This class enables incremental proof generation for neural network
    inference, where each layer is a computation step.
    """
    
    def __init__(self, layer_r1cs: R1CS):
        """
        Initialize IVC for zkML.
        
        Args:
            layer_r1cs: The R1CS for a single layer computation
        """
        self.layer_r1cs = layer_r1cs
        self.folding = NovaFolding(layer_r1cs)
        self.current_proof: Optional[IVCProof] = None
    
    def initialize(
        self,
        initial_input: List[Fr],
        initial_witness: List[Fr],
        commitment_W: G1Point
    ) -> IVCProof:
        """
        Initialize IVC with the first layer.
        
        Args:
            initial_input: Public input (e.g., input hash)
            initial_witness: Witness for first layer
            commitment_W: Commitment to the witness
        
        Returns:
            The initial IVC proof
        """
        instance = RelaxedR1CSInstance.from_standard(commitment_W, initial_input)
        witness = RelaxedR1CSWitness.from_standard(
            initial_witness,
            self.layer_r1cs.num_constraints
        )
        
        self.current_proof = IVCProof(
            accumulated_instance=instance,
            accumulated_witness=witness,
            num_steps=1,
            final_output=initial_input  # First output is input
        )
        
        return self.current_proof
    
    def step(
        self,
        new_input: List[Fr],
        new_witness: List[Fr],
        commitment_W: G1Point,
        commitment_T: G1Point
    ) -> IVCProof:
        """
        Add a computation step (layer) to the IVC.
        
        Args:
            new_input: Public input for this step
            new_witness: Witness for this step
            commitment_W: Commitment to new witness
            commitment_T: Commitment to cross-term
        
        Returns:
            The updated IVC proof
        """
        if self.current_proof is None:
            raise ValueError("IVC not initialized. Call initialize() first.")
        
        # Create new instance and witness
        new_instance = RelaxedR1CSInstance.from_standard(commitment_W, new_input)
        new_witness_relaxed = RelaxedR1CSWitness.from_standard(
            new_witness,
            self.layer_r1cs.num_constraints
        )
        
        # Compute cross-term (prover)
        T = self.folding.compute_cross_term(
            self.current_proof.accumulated_instance,
            self.current_proof.accumulated_witness,
            new_instance,
            new_witness_relaxed
        )
        
        # Generate challenge
        r = self.folding.generate_challenge(
            self.current_proof.accumulated_instance,
            new_instance,
            commitment_T
        )
        
        # Fold instances and witnesses
        folded_instance = self.folding.fold_instances(
            self.current_proof.accumulated_instance,
            new_instance,
            commitment_T,
            r
        )
        
        folded_witness = self.folding.fold_witnesses(
            self.current_proof.accumulated_witness,
            new_witness_relaxed,
            T,
            r
        )
        
        # Update proof
        self.current_proof = IVCProof(
            accumulated_instance=folded_instance,
            accumulated_witness=folded_witness,
            num_steps=self.current_proof.num_steps + 1,
            final_output=new_input
        )
        
        return self.current_proof
    
    def finalize(self) -> IVCProof:
        """
        Finalize the IVC proof.
        
        Returns:
            The final IVC proof (witness removed for verification)
        """
        if self.current_proof is None:
            raise ValueError("IVC not initialized")
        
        # Remove witness for verification
        return IVCProof(
            accumulated_instance=self.current_proof.accumulated_instance,
            accumulated_witness=None,  # Remove for verification
            num_steps=self.current_proof.num_steps,
            final_output=self.current_proof.final_output
        )


# =============================================================================
# Verification
# =============================================================================

def verify_ivc_proof(
    proof: IVCProof,
    r1cs: R1CS,
    expected_output: List[Fr]
) -> Tuple[bool, str]:
    """
    Verify an IVC proof.
    
    Note: This is a simplified verification. A full implementation would
    require a final SNARK proof for the accumulated instance.
    
    Args:
        proof: The IVC proof to verify
        r1cs: The R1CS constraint system
        expected_output: The expected final output
    
    Returns:
        (is_valid, reason)
    """
    # Check output matches
    if proof.final_output != expected_output:
        return False, "Output mismatch"
    
    # Check instance is well-formed
    instance = proof.accumulated_instance
    
    # In a full implementation, we would verify:
    # 1. The commitment to W is correct
    # 2. The commitment to E is correct
    # 3. The accumulated instance satisfies the relaxed R1CS
    
    # For now, we do basic checks
    if instance.u == Fr.zero():
        return False, "Invalid u (cannot be zero)"
    
    return True, f"IVC proof verified ({proof.num_steps} steps)"


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Nova Folding Scheme Self-Test")
    print("=" * 70)
    
    # Create a simple R1CS: x² = y (one constraint)
    print("\n1. Creating simple R1CS...")
    A = R1CSMatrix(rows=1, cols=3)
    B = R1CSMatrix(rows=1, cols=3)
    C = R1CSMatrix(rows=1, cols=3)
    
    # Constraint: x * x = y
    # z = (1, x, y) where x is public input, y is witness
    A.add_entry(0, 1, Fr.one())  # A[0,1] = 1 (select x)
    B.add_entry(0, 1, Fr.one())  # B[0,1] = 1 (select x)
    C.add_entry(0, 2, Fr.one())  # C[0,2] = 1 (select y)
    
    r1cs = R1CS(
        A=A, B=B, C=C,
        num_constraints=1,
        num_variables=3,
        num_public_inputs=1
    )
    print("   R1CS: x² = y")
    
    # Test R1CS satisfaction
    print("\n2. Testing R1CS satisfaction...")
    x = Fr(5)
    y = x * x  # y = 25
    is_satisfied = r1cs.is_satisfied([y], [x])
    print(f"   x=5, y=25: satisfied={is_satisfied}")
    
    # Test folding
    print("\n3. Testing Nova folding...")
    folding = NovaFolding(r1cs)
    
    # Create two instances
    g1 = G1Point.generator()
    
    instance1 = RelaxedR1CSInstance.from_standard(g1 * Fr(1), [Fr(5)])
    witness1 = RelaxedR1CSWitness.from_standard([Fr(25)], 1)
    
    instance2 = RelaxedR1CSInstance.from_standard(g1 * Fr(2), [Fr(7)])
    witness2 = RelaxedR1CSWitness.from_standard([Fr(49)], 1)
    
    # Compute cross-term
    T = folding.compute_cross_term(instance1, witness1, instance2, witness2)
    print(f"   Cross-term T computed: {len(T)} elements")
    
    # Generate challenge
    commitment_T = g1 * Fr(3)  # Dummy commitment
    r = folding.generate_challenge(instance1, instance2, commitment_T)
    print(f"   Challenge r: {r.to_int() % 1000}... (truncated)")
    
    # Fold
    folded_instance = folding.fold_instances(instance1, instance2, commitment_T, r)
    folded_witness = folding.fold_witnesses(witness1, witness2, T, r)
    print(f"   Folded u: {folded_instance.u.to_int()}")
    print(f"   Folded witness size: {len(folded_witness.W)}")
    
    # Test IVC
    print("\n4. Testing IVC for zkML...")
    ivc = ZkMLIVC(r1cs)
    
    # Initialize
    proof = ivc.initialize(
        initial_input=[Fr(5)],
        initial_witness=[Fr(25)],
        commitment_W=g1 * Fr(1)
    )
    print(f"   Initialized: {proof.num_steps} step(s)")
    
    # Add steps
    for i in range(3):
        proof = ivc.step(
            new_input=[Fr(i + 10)],
            new_witness=[Fr((i + 10) ** 2)],
            commitment_W=g1 * Fr(i + 2),
            commitment_T=g1 * Fr(i + 100)
        )
        print(f"   Step {i + 2}: {proof.num_steps} steps accumulated")
    
    # Finalize
    final_proof = ivc.finalize()
    print(f"   Finalized: {final_proof.num_steps} steps, witness removed")
    
    # Verify
    print("\n5. Verifying IVC proof...")
    is_valid, reason = verify_ivc_proof(final_proof, r1cs, [Fr(12)])
    print(f"   Valid: {is_valid}")
    print(f"   Reason: {reason}")
    
    print("\n" + "=" * 70)
    print("FOLDING SCHEME TESTS COMPLETE!")
    print("=" * 70)
