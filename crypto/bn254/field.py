"""
BN254 Finite Field Arithmetic

This module provides implementations for the base field Fp and the scalar field Fr
of the BN254 elliptic curve. Both fields use Montgomery representation internally
for efficient modular arithmetic.

Classes:
    Fp: Elements of the base field (modulus p).
    Fr: Elements of the scalar field (modulus r, the curve order).
"""

from __future__ import annotations
from typing import Union, Tuple
from ..utils.montgomery import (
    to_montgomery,
    from_montgomery,
    montgomery_mul,
    montgomery_square,
    montgomery_pow,
)
from .constants import (
    FIELD_MODULUS,
    CURVE_ORDER,
    FP_R,
    FP_R_SQUARED,
    FP_N_PRIME,
    FR_R,
    FR_R_SQUARED,
    FR_N_PRIME,
)


class Fp:
    """
    An element of the BN254 base field Fp.

    Internally stores the value in Montgomery representation for efficient
    arithmetic. Conversion to/from standard representation is handled
    automatically on input/output.

    Attributes:
        value: The value in Montgomery representation (a * R mod p).
    """

    MODULUS = FIELD_MODULUS
    R = FP_R
    R_SQUARED = FP_R_SQUARED
    N_PRIME = FP_N_PRIME

    # Precomputed Montgomery representation of common values
    _ZERO_MONT: int = 0
    _ONE_MONT: int = FP_R  # 1 * R mod p

    __slots__ = ("value",)

    def __init__(self, value: Union[int, "Fp"], _is_montgomery: bool = False):
        """
        Create a new Fp element.

        Args:
            value: The value as an integer or another Fp element.
            _is_montgomery: Internal flag. If True, `value` is already in
                            Montgomery representation. Do not use directly.
        """
        if isinstance(value, Fp):
            self.value = value.value
        elif _is_montgomery:
            self.value = value
        else:
            # Ensure value is in range [0, p)
            value = value % self.MODULUS
            self.value = to_montgomery(value, self.R_SQUARED, self.MODULUS, self.N_PRIME)

    @classmethod
    def zero(cls) -> Fp:
        """Return the additive identity (0)."""
        return cls(cls._ZERO_MONT, _is_montgomery=True)

    @classmethod
    def one(cls) -> Fp:
        """Return the multiplicative identity (1)."""
        return cls(cls._ONE_MONT, _is_montgomery=True)

    def to_int(self) -> int:
        """Convert from Montgomery representation to a standard integer."""
        return from_montgomery(self.value, self.MODULUS, self.N_PRIME)

    def __repr__(self) -> str:
        return f"Fp({self.to_int()})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fp):
            return self.value == other.value
        if isinstance(other, int):
            return self.to_int() == (other % self.MODULUS)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __neg__(self) -> Fp:
        """Return the additive inverse (-self)."""
        if self.value == 0:
            return Fp.zero()
        return Fp(self.MODULUS - self.value, _is_montgomery=True)

    def __add__(self, other: Union[Fp, int]) -> Fp:
        """Add two field elements."""
        if isinstance(other, int):
            other = Fp(other)
        result = self.value + other.value
        if result >= self.MODULUS:
            result -= self.MODULUS
        return Fp(result, _is_montgomery=True)

    def __radd__(self, other: int) -> Fp:
        return self.__add__(other)

    def __sub__(self, other: Union[Fp, int]) -> Fp:
        """Subtract two field elements."""
        if isinstance(other, int):
            other = Fp(other)
        if self.value >= other.value:
            result = self.value - other.value
        else:
            result = self.MODULUS + self.value - other.value
        return Fp(result, _is_montgomery=True)

    def __rsub__(self, other: int) -> Fp:
        return Fp(other).__sub__(self)

    def __mul__(self, other: Union[Fp, int]) -> Fp:
        """Multiply two field elements."""
        if isinstance(other, int):
            other = Fp(other)
        result = montgomery_mul(self.value, other.value, self.MODULUS, self.N_PRIME)
        return Fp(result, _is_montgomery=True)

    def __rmul__(self, other: int) -> Fp:
        return self.__mul__(other)

    def square(self) -> Fp:
        """Return the square of this element."""
        result = montgomery_square(self.value, self.MODULUS, self.N_PRIME)
        return Fp(result, _is_montgomery=True)

    def __pow__(self, exp: int) -> Fp:
        """Compute self^exp using square-and-multiply."""
        if exp < 0:
            return self.inverse() ** (-exp)
        if exp == 0:
            return Fp.one()
        if exp == 1:
            return Fp(self.value, _is_montgomery=True)
        if exp == 2:
            return self.square()

        result = montgomery_pow(self.value, exp, self.MODULUS, self.N_PRIME, self.R)
        return Fp(result, _is_montgomery=True)

    def inverse(self) -> Fp:
        """
        Compute the multiplicative inverse using Fermat's Little Theorem.

        a^(-1) = a^(p-2) mod p

        Raises:
            ZeroDivisionError: If self is zero.
        """
        if self.value == 0:
            raise ZeroDivisionError("Cannot invert zero in Fp")
        # a^(-1) = a^(p-2) mod p
        return self ** (self.MODULUS - 2)

    def __truediv__(self, other: Union[Fp, int]) -> Fp:
        """Divide two field elements (self / other)."""
        if isinstance(other, int):
            other = Fp(other)
        return self * other.inverse()

    def is_zero(self) -> bool:
        """Check if this element is zero."""
        return self.value == 0

    def is_one(self) -> bool:
        """Check if this element is one."""
        return self.value == self._ONE_MONT

    def legendre(self) -> int:
        """
        Compute the Legendre symbol (self / p).

        Returns:
            1 if self is a quadratic residue (non-zero square).
            -1 if self is a quadratic non-residue.
            0 if self is zero.
        """
        if self.is_zero():
            return 0
        # a^((p-1)/2) mod p
        exp = (self.MODULUS - 1) // 2
        result = self ** exp
        if result.is_one():
            return 1
        return -1

    def sqrt(self) -> Fp:
        """
        Compute the square root of this element using Tonelli-Shanks.

        For BN254, p ≡ 3 (mod 4), so we can use the simpler formula:
        sqrt(a) = a^((p+1)/4) mod p

        Returns:
            A square root of self.

        Raises:
            ValueError: If self is not a quadratic residue.
        """
        if self.is_zero():
            return Fp.zero()

        # Check if a is a quadratic residue
        if self.legendre() != 1:
            raise ValueError(f"{self} is not a quadratic residue in Fp")

        # For p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4)
        # BN254 p mod 4 = 3, so this applies.
        exp = (self.MODULUS + 1) // 4
        return self ** exp


class Fr:
    """
    An element of the BN254 scalar field Fr.

    This is the field of integers modulo the curve order r. It is used for
    scalar multiplication and for R1CS witness values in SNARKs.

    Internally stores the value in Montgomery representation.

    Attributes:
        value: The value in Montgomery representation (a * R mod r).
    """

    MODULUS = CURVE_ORDER
    R = FR_R
    R_SQUARED = FR_R_SQUARED
    N_PRIME = FR_N_PRIME

    _ZERO_MONT: int = 0
    _ONE_MONT: int = FR_R

    __slots__ = ("value",)

    def __init__(self, value: Union[int, "Fr"], _is_montgomery: bool = False):
        """
        Create a new Fr element.

        Args:
            value: The value as an integer or another Fr element.
            _is_montgomery: Internal flag. If True, `value` is already in
                            Montgomery representation.
        """
        if isinstance(value, Fr):
            self.value = value.value
        elif _is_montgomery:
            self.value = value
        else:
            value = value % self.MODULUS
            self.value = to_montgomery(value, self.R_SQUARED, self.MODULUS, self.N_PRIME)

    @classmethod
    def zero(cls) -> Fr:
        """Return the additive identity (0)."""
        return cls(cls._ZERO_MONT, _is_montgomery=True)

    @classmethod
    def one(cls) -> Fr:
        """Return the multiplicative identity (1)."""
        return cls(cls._ONE_MONT, _is_montgomery=True)

    def to_int(self) -> int:
        """Convert from Montgomery representation to a standard integer."""
        return from_montgomery(self.value, self.MODULUS, self.N_PRIME)

    def __repr__(self) -> str:
        return f"Fr({self.to_int()})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fr):
            return self.value == other.value
        if isinstance(other, int):
            return self.to_int() == (other % self.MODULUS)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __neg__(self) -> Fr:
        if self.value == 0:
            return Fr.zero()
        return Fr(self.MODULUS - self.value, _is_montgomery=True)

    def __add__(self, other: Union[Fr, int]) -> Fr:
        if isinstance(other, int):
            other = Fr(other)
        result = self.value + other.value
        if result >= self.MODULUS:
            result -= self.MODULUS
        return Fr(result, _is_montgomery=True)

    def __radd__(self, other: int) -> Fr:
        return self.__add__(other)

    def __sub__(self, other: Union[Fr, int]) -> Fr:
        if isinstance(other, int):
            other = Fr(other)
        if self.value >= other.value:
            result = self.value - other.value
        else:
            result = self.MODULUS + self.value - other.value
        return Fr(result, _is_montgomery=True)

    def __rsub__(self, other: int) -> Fr:
        return Fr(other).__sub__(self)

    def __mul__(self, other: Union[Fr, int]) -> Fr:
        if isinstance(other, int):
            other = Fr(other)
        result = montgomery_mul(self.value, other.value, self.MODULUS, self.N_PRIME)
        return Fr(result, _is_montgomery=True)

    def __rmul__(self, other: int) -> Fr:
        return self.__mul__(other)

    def square(self) -> Fr:
        result = montgomery_square(self.value, self.MODULUS, self.N_PRIME)
        return Fr(result, _is_montgomery=True)

    def __pow__(self, exp: int) -> Fr:
        if exp < 0:
            return self.inverse() ** (-exp)
        if exp == 0:
            return Fr.one()
        if exp == 1:
            return Fr(self.value, _is_montgomery=True)
        if exp == 2:
            return self.square()

        result = montgomery_pow(self.value, exp, self.MODULUS, self.N_PRIME, self.R)
        return Fr(result, _is_montgomery=True)

    def inverse(self) -> Fr:
        if self.value == 0:
            raise ZeroDivisionError("Cannot invert zero in Fr")
        return self ** (self.MODULUS - 2)

    def __truediv__(self, other: Union[Fr, int]) -> Fr:
        if isinstance(other, int):
            other = Fr(other)
        return self * other.inverse()

    def is_zero(self) -> bool:
        return self.value == 0

    def is_one(self) -> bool:
        return self.value == self._ONE_MONT
