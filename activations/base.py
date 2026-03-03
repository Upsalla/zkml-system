"""
Base Interface for Activation Functions in zkML
================================================

Every activation function must provide two capabilities:
1. Compute the value (for the witness)
2. Generate constraints (for the R1CS)

The constraint count is the critical factor for zkML performance.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ActivationResult:
    """Result of an activation function."""
    output_index: int           # Index of the output value in the witness
    intermediate_indices: List[int]  # Indices of intermediate values
    num_constraints: int        # Number of generated constraints


class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.

    Every activation function must implement:
    - compute(): Computes the output value
    - generate_constraints(): Generates R1CS constraints
    - constraint_count: Returns the number of constraints
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the activation function."""
        pass

    @property
    @abstractmethod
    def constraint_count(self) -> int:
        """Number of constraints per activation."""
        pass

    @abstractmethod
    def compute(self, x: int, prime: int) -> int:
        """
        Compute the output value of the activation function.

        Args:
            x: Input value (in the field)
            prime: The prime field

        Returns:
            The output value (in the field)
        """
        pass

    @abstractmethod
    def generate_constraints(
        self,
        input_index: int,
        r1cs: Any,      # R1CS object
        witness: Any,   # Witness object
        neuron_id: int = 0,
    ) -> ActivationResult:
        """
        Generate R1CS constraints for this activation.

        Args:
            input_index: Index of the input value in the witness
            r1cs: The R1CS system
            witness: The witness
            neuron_id: ID of the neuron (for sparse tracking)

        Returns:
            ActivationResult with output index and constraint info
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(constraints={self.constraint_count})"


class PolynomialActivation(ActivationFunction):
    """
    Base class for polynomial activation functions.

    Polynomials are especially efficient in R1CS since they only
    require multiplications and additions.
    """

    @property
    @abstractmethod
    def coefficients(self) -> List[int]:
        """
        Coefficients of the polynomial.

        For p(x) = a_0 + a_1*x + a_2*x^2 + ...
        Returns [a_0, a_1, a_2, ...]
        """
        pass

    @property
    def degree(self) -> int:
        """Degree of the polynomial."""
        return len(self.coefficients) - 1

    def compute(self, x: int, prime: int) -> int:
        """Evaluate the polynomial at point x."""
        result = 0
        x_power = 1
        for coeff in self.coefficients:
            result = (result + coeff * x_power) % prime
            x_power = (x_power * x) % prime
        return result

    @property
    def constraint_count(self) -> int:
        """
        Number of constraints for a polynomial of degree n.

        We need:
        - (n-1) constraints for x^2, x^3, ..., x^n
        - 1 constraint for the final result (if needed)
        """
        return max(0, self.degree - 1) + 1


# Helper functions for fixed-point arithmetic

FIXED_POINT_SCALE = 2**16  # 16 fractional bits


def float_to_fixed(x: float) -> int:
    """Convert float to fixed-point."""
    return int(x * FIXED_POINT_SCALE)


def fixed_to_float(x: int) -> float:
    """Convert fixed-point to float."""
    return x / FIXED_POINT_SCALE


def fixed_mul(a: int, b: int, prime: int) -> int:
    """Multiplication in fixed-point arithmetic."""
    # a * b / SCALE
    product = (a * b) % prime
    # Division by SCALE (multiplication by its inverse)
    scale_inv = pow(FIXED_POINT_SCALE, prime - 2, prime)
    return (product * scale_inv) % prime
