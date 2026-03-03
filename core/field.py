"""
Finite Field Arithmetic for zkML
================================

This module implements arithmetic over finite fields (prime fields).
All computations in ZK proofs take place in such fields.

Mathematical background:
- A finite field F_p consists of the numbers {0, 1, 2, ..., p-1}
- All operations are performed modulo p
- p must be a prime number for division to be possible
"""

from dataclasses import dataclass
from typing import Union, List
import random


# BN254 Scalar Field — standard for Ethereum/Groth16
BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Smaller field for development and testing
DEV_PRIME = 101


@dataclass
class FieldConfig:
    """Configuration for a finite field."""
    prime: int
    name: str

    # Known primes (for which we skip the primality test)
    KNOWN_PRIMES = {
        21888242871839275222246405745257275088548364400416034343698204186575808495617,  # BN254
        52435875175126190479447740508185965837690552500527637822603658699938581184513,  # BLS12-381
        101,  # Dev
    }

    def __post_init__(self):
        # Skip test for known primes
        if self.prime in self.KNOWN_PRIMES:
            return
        if not self._is_prime(self.prime):
            raise ValueError(f"{self.prime} is not a prime number")

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Simple primality test (only reliable for small numbers < 10^6)."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        # For large numbers: assume prime (or verify via KNOWN_PRIMES)
        if n > 10**6:
            return True
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True


# Predefined field configurations
FIELD_BN254 = FieldConfig(BN254_PRIME, "BN254")
FIELD_DEV = FieldConfig(DEV_PRIME, "Development")


class FieldElement:
    """
    An element of a finite field.

    Supports all arithmetic operations with automatic modular reduction.
    """

    __slots__ = ['value', 'field']

    def __init__(self, value: int, field: FieldConfig = FIELD_DEV):
        """
        Create a field element.

        Args:
            value: The value (automatically reduced mod p)
            field: The field configuration
        """
        self.field = field
        self.value = value % field.prime

    def __repr__(self) -> str:
        return f"FieldElement({self.value}, mod {self.field.prime})"

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Union['FieldElement', int]) -> bool:
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Cannot compare field elements from different fields")
            return self.value == other.value
        return self.value == (other % self.field.prime)

    def __hash__(self) -> int:
        return hash((self.value, self.field.prime))

    # Arithmetic operations

    def __add__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Addition: (a + b) mod p"""
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Field elements from different fields")
            return FieldElement((self.value + other.value) % self.field.prime, self.field)
        return FieldElement((self.value + other) % self.field.prime, self.field)

    def __radd__(self, other: int) -> 'FieldElement':
        return self.__add__(other)

    def __sub__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Subtraction: (a - b) mod p"""
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Field elements from different fields")
            return FieldElement((self.value - other.value) % self.field.prime, self.field)
        return FieldElement((self.value - other) % self.field.prime, self.field)

    def __rsub__(self, other: int) -> 'FieldElement':
        return FieldElement((other - self.value) % self.field.prime, self.field)

    def __mul__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Multiplication: (a * b) mod p"""
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Field elements from different fields")
            return FieldElement((self.value * other.value) % self.field.prime, self.field)
        return FieldElement((self.value * other) % self.field.prime, self.field)

    def __rmul__(self, other: int) -> 'FieldElement':
        return self.__mul__(other)

    def __pow__(self, exp: int) -> 'FieldElement':
        """Exponentiation: a^exp mod p (using fast exponentiation)"""
        if exp < 0:
            # Negative exponents: a^(-n) = (a^(-1))^n
            return self.inverse() ** (-exp)
        return FieldElement(pow(self.value, exp, self.field.prime), self.field)

    def __neg__(self) -> 'FieldElement':
        """Negation: -a mod p"""
        return FieldElement((-self.value) % self.field.prime, self.field)

    def __truediv__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Division: a / b = a * b^(-1) mod p"""
        if isinstance(other, FieldElement):
            return self * other.inverse()
        return self * FieldElement(other, self.field).inverse()

    def inverse(self) -> 'FieldElement':
        """
        Multiplicative inverse: a^(-1) mod p

        Computed using the extended Euclidean algorithm.
        Satisfies: a * a^(-1) = 1 mod p
        """
        if self.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse")

        # Extended Euclidean algorithm
        def extended_gcd(a: int, b: int) -> tuple:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        _, x, _ = extended_gcd(self.value % self.field.prime, self.field.prime)
        return FieldElement(x % self.field.prime, self.field)

    def is_zero(self) -> bool:
        """Check whether the element is 0."""
        return self.value == 0

    def is_one(self) -> bool:
        """Check whether the element is 1."""
        return self.value == 1

    @classmethod
    def zero(cls, field: FieldConfig = FIELD_DEV) -> 'FieldElement':
        """Return the zero element of the field."""
        return cls(0, field)

    @classmethod
    def one(cls, field: FieldConfig = FIELD_DEV) -> 'FieldElement':
        """Return the one element of the field."""
        return cls(1, field)

    @classmethod
    def random(cls, field: FieldConfig = FIELD_DEV) -> 'FieldElement':
        """Return a random field element."""
        return cls(random.randint(0, field.prime - 1), field)


class FixedPoint:
    """
    Fixed-point arithmetic for decimal numbers in finite fields.

    We represent a decimal number x as x * SCALE, where SCALE = 2^PRECISION.

    Example with PRECISION=16:
    - 1.5 is stored as 1.5 * 65536 = 98304
    - 0.25 is stored as 0.25 * 65536 = 16384
    """

    PRECISION = 16  # Number of fractional bits
    SCALE = 1 << PRECISION  # 2^16 = 65536

    __slots__ = ['element']

    def __init__(self, value: Union[float, int, FieldElement], field: FieldConfig = FIELD_DEV):
        """
        Create a fixed-point number.

        Args:
            value: Float, int, or already-scaled FieldElement
            field: The field configuration
        """
        if isinstance(value, FieldElement):
            self.element = value
        elif isinstance(value, float):
            scaled = int(value * self.SCALE)
            self.element = FieldElement(scaled, field)
        else:
            scaled = value * self.SCALE
            self.element = FieldElement(scaled, field)

    def __repr__(self) -> str:
        return f"FixedPoint({self.to_float():.6f})"

    def to_float(self) -> float:
        """Convert back to float (only for debugging/display)."""
        val = self.element.value
        # Handle negative numbers (upper half of the field)
        if val > self.element.field.prime // 2:
            val = val - self.element.field.prime
        return val / self.SCALE

    def __add__(self, other: 'FixedPoint') -> 'FixedPoint':
        """Addition of fixed-point numbers."""
        result = FixedPoint.__new__(FixedPoint)
        result.element = self.element + other.element
        return result

    def __sub__(self, other: 'FixedPoint') -> 'FixedPoint':
        """Subtraction of fixed-point numbers."""
        result = FixedPoint.__new__(FixedPoint)
        result.element = self.element - other.element
        return result

    def __mul__(self, other: 'FixedPoint') -> 'FixedPoint':
        """
        Multiplication of fixed-point numbers.

        (a * SCALE) * (b * SCALE) = a * b * SCALE^2
        We must divide by SCALE to get back to a * b * SCALE.
        """
        # Multiplication
        product = self.element * other.element
        # Division by SCALE (multiplication by SCALE^(-1))
        scale_inv = FieldElement(self.SCALE, self.element.field).inverse()
        result = FixedPoint.__new__(FixedPoint)
        result.element = product * scale_inv
        return result

    @classmethod
    def from_field_element(cls, element: FieldElement) -> 'FixedPoint':
        """Create a FixedPoint number from an already-scaled FieldElement."""
        result = cls.__new__(cls)
        result.element = element
        return result


class PrimeField:
    """
    Simple wrapper class for a prime field.

    Provides helper methods for field operations.
    """

    def __init__(self, prime: int):
        self.prime = prime
        self.config = FieldConfig(prime, f"F_{prime}")

    def element(self, value: int) -> FieldElement:
        """Create a field element."""
        return FieldElement(value, self.config)

    def add(self, a: int, b: int) -> int:
        """Addition mod p."""
        return (a + b) % self.prime

    def sub(self, a: int, b: int) -> int:
        """Subtraction mod p."""
        return (a - b) % self.prime

    def mul(self, a: int, b: int) -> int:
        """Multiplication mod p."""
        return (a * b) % self.prime

    def inv(self, a: int) -> int:
        """Multiplicative inverse mod p."""
        return pow(a, self.prime - 2, self.prime)

    def pow(self, a: int, exp: int) -> int:
        """Exponentiation mod p."""
        return pow(a, exp, self.prime)

    def neg(self, a: int) -> int:
        """Negation mod p."""
        return (-a) % self.prime


# Helper functions for simple usage

def field_add(a: int, b: int, prime: int = DEV_PRIME) -> int:
    """Simple addition mod p."""
    return (a + b) % prime


def field_sub(a: int, b: int, prime: int = DEV_PRIME) -> int:
    """Simple subtraction mod p."""
    return (a - b) % prime


def field_mul(a: int, b: int, prime: int = DEV_PRIME) -> int:
    """Simple multiplication mod p."""
    return (a * b) % prime


def field_pow(a: int, exp: int, prime: int = DEV_PRIME) -> int:
    """Simple exponentiation mod p."""
    return pow(a, exp, prime)


def field_inv(a: int, prime: int = DEV_PRIME) -> int:
    """Simple multiplicative inverse mod p (via Fermat's little theorem)."""
    return pow(a, prime - 2, prime)
