"""
R1CS (Rank-1 Constraint System) for zkML
=========================================

R1CS is the standard representation for arithmetic circuits in
zero-knowledge proofs. Each constraint has the form:

    A(w) * B(w) = C(w)

where A, B, C are linear combinations of the witness variables w.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# Import field helpers from the local module
from .field import FIELD_DEV, field_mul


@dataclass
class LinearCombination:
    """
    A linear combination of witness variables.

    Representation: sum(coefficient_i * variable_i)
    """
    terms: Dict[int, int] = field(default_factory=dict)  # {variable_index: coefficient}

    @classmethod
    def zero(cls) -> 'LinearCombination':
        """Return the zero linear combination."""
        return cls()

    @classmethod
    def single(cls, index: int, coeff: int = 1) -> 'LinearCombination':
        """Create a linear combination with a single variable."""
        return cls({index: coeff})

    @classmethod
    def constant(cls, value: int) -> 'LinearCombination':
        """Create a constant (coefficient of variable 0, which is always 1)."""
        return cls({0: value})

    def evaluate(self, witness: List[int], prime: int) -> int:
        """Evaluate the linear combination with the given witness."""
        result = 0
        for idx, coeff in self.terms.items():
            if idx < len(witness):
                result = (result + coeff * witness[idx]) % prime
        return result

    def __add__(self, other: 'LinearCombination') -> 'LinearCombination':
        result = dict(self.terms)
        for idx, coeff in other.terms.items():
            result[idx] = result.get(idx, 0) + coeff
        return LinearCombination(result)

    def __mul__(self, scalar: int) -> 'LinearCombination':
        return LinearCombination({idx: coeff * scalar for idx, coeff in self.terms.items()})

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for idx, coeff in sorted(self.terms.items()):
            if coeff == 1:
                parts.append(f"w[{idx}]")
            else:
                parts.append(f"{coeff}*w[{idx}]")
        return " + ".join(parts)


@dataclass
class R1CSConstraint:
    """
    A single R1CS constraint: A * B = C.

    Each of A, B, C is a linear combination of witness variables.
    """
    a: LinearCombination
    b: LinearCombination
    c: LinearCombination
    description: str = ""
    neuron_id: Optional[int] = None

    def is_satisfied(self, witness: List[int], prime: int) -> bool:
        """Check whether this constraint is satisfied by the witness."""
        a_val = self.a.evaluate(witness, prime)
        b_val = self.b.evaluate(witness, prime)
        c_val = self.c.evaluate(witness, prime)
        return (a_val * b_val) % prime == c_val

    def __repr__(self) -> str:
        desc = f" [{self.description}]" if self.description else ""
        return f"({self.a}) * ({self.b}) = ({self.c}){desc}"


@dataclass
class R1CS:
    """
    A complete R1CS constraint system.

    Attributes:
        constraints: List of all constraints
        num_variables: Number of variables in the witness
        num_public: Number of public inputs
        prime: The prime field used
    """
    constraints: List[R1CSConstraint] = field(default_factory=list)
    num_variables: int = 0
    num_public: int = 0
    prime: int = FIELD_DEV.prime

    # Variable tracking
    _variable_names: Dict[int, str] = field(default_factory=dict)
    _next_variable: int = 1  # 0 is reserved for the constant 1

    def __post_init__(self):
        # Variable 0 is always the constant 1
        self._variable_names[0] = "ONE"

    def allocate_variable(self, name: str = "") -> int:
        """
        Allocate a new variable in the witness.

        Returns:
            The index of the new variable
        """
        idx = self._next_variable
        self._next_variable += 1
        self.num_variables = self._next_variable
        if name:
            self._variable_names[idx] = name
        return idx

    def allocate_public_input(self, name: str = "") -> int:
        """
        Allocate a public input.

        Public inputs come directly after the constant 1.
        """
        idx = self.allocate_variable(name)
        self.num_public += 1
        return idx

    def add_constraint(
        self,
        a: LinearCombination,
        b: LinearCombination,
        c: LinearCombination,
        description: str = "",
        neuron_id: Optional[int] = None,
    ) -> None:
        """Add a new constraint."""
        self.constraints.append(R1CSConstraint(a, b, c, description, neuron_id))

    def add_multiplication_constraint(
        self,
        left_idx: int,
        right_idx: int,
        result_idx: int,
        description: str = "",
    ) -> None:
        """
        Add a simple multiplication constraint.

        Constraint: witness[left_idx] * witness[right_idx] = witness[result_idx]
        """
        a = LinearCombination.single(left_idx)
        b = LinearCombination.single(right_idx)
        c = LinearCombination.single(result_idx)
        self.add_constraint(a, b, c, description)

    def verify(self, witness: List[int]) -> Tuple[bool, List[int]]:
        """
        Verify whether the witness satisfies all constraints.

        Returns:
            (all_satisfied, list_of_violated_constraint_indices)
        """
        violated = []
        for i, constraint in enumerate(self.constraints):
            if not constraint.is_satisfied(witness, self.prime):
                violated.append(i)
        return len(violated) == 0, violated

    def check_witness(self, witness: List[int]) -> bool:
        """Check whether the witness satisfies all constraints."""
        is_valid, _ = self.verify(witness)
        return is_valid

    def num_constraints(self) -> int:
        """Return the number of constraints."""
        return len(self.constraints)

    def get_statistics(self) -> Dict[str, int]:
        """Return statistics about the R1CS system."""
        return {
            "num_constraints": len(self.constraints),
            "num_variables": self.num_variables,
            "num_public": self.num_public,
            "num_private": self.num_variables - self.num_public - 1,  # -1 for constant
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"R1CS(constraints={stats['num_constraints']}, "
            f"variables={stats['num_variables']}, "
            f"public={stats['num_public']})"
        )


class R1CSBuilder:
    """
    Builder class for easy R1CS system creation.

    Provides high-level methods for common operations.
    """

    def __init__(self, prime: int = FIELD_DEV.prime):
        self.r1cs = R1CS(prime=prime)
        self.witness_values: Dict[int, int] = {0: 1}  # Constant 1

    def input(self, name: str, value: int, public: bool = False) -> int:
        """Create an input variable."""
        if public:
            idx = self.r1cs.allocate_public_input(name)
        else:
            idx = self.r1cs.allocate_variable(name)
        self.witness_values[idx] = value % self.r1cs.prime
        return idx

    def constant(self, value: int) -> LinearCombination:
        """Create a constant as a linear combination."""
        return LinearCombination.constant(value)

    def var(self, idx: int) -> LinearCombination:
        """Create a linear combination for a single variable."""
        return LinearCombination.single(idx)

    def mul(self, left_idx: int, right_idx: int, name: str = "") -> int:
        """
        Multiply two variables and return the index of the result.

        Automatically adds the corresponding constraint.
        """
        result_idx = self.r1cs.allocate_variable(name)

        # Compute the value
        left_val = self.witness_values.get(left_idx, 0)
        right_val = self.witness_values.get(right_idx, 0)
        result_val = field_mul(left_val, right_val, self.r1cs.prime)
        self.witness_values[result_idx] = result_val

        # Add constraint
        self.r1cs.add_multiplication_constraint(left_idx, right_idx, result_idx, name)

        return result_idx

    def add(self, *indices: int, name: str = "") -> int:
        """
        Add multiple variables.

        Addition does not require a constraint, only a new variable.
        """
        result_idx = self.r1cs.allocate_variable(name)

        # Compute the value
        result_val = sum(self.witness_values.get(idx, 0) for idx in indices) % self.r1cs.prime
        self.witness_values[result_idx] = result_val

        return result_idx

    def square(self, idx: int, name: str = "") -> int:
        """Square a variable (x * x)."""
        return self.mul(idx, idx, name)

    def get_witness(self) -> List[int]:
        """Return the witness as a list."""
        max_idx = max(self.witness_values.keys())
        return [self.witness_values.get(i, 0) for i in range(max_idx + 1)]

    def add_constraint(
        self,
        a: LinearCombination,
        b: LinearCombination,
        c: LinearCombination,
        description: str = "",
        neuron_id: Optional[int] = None,
    ) -> None:
        """Add a constraint directly."""
        self.r1cs.add_constraint(a, b, c, description, neuron_id)

    def build(self, num_vars: Optional[int] = None) -> 'R1CS':
        """Return the completed R1CS system."""
        if num_vars is not None:
            self.r1cs.num_variables = num_vars
        return self.r1cs

    def build_with_witness(self) -> Tuple[R1CS, List[int]]:
        """Return the completed R1CS system and the witness."""
        return self.r1cs, self.get_witness()
