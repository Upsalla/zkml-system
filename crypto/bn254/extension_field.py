"""
BN254 Extension Field Arithmetic

This module provides implementations for the tower of extension fields required
for BN254 pairing operations:
    - Fp2: Quadratic extension Fp[u]/(u^2 + 1)
    - Fp6: Cubic extension Fp2[v]/(v^3 - (u+1))
    - Fp12: Quadratic extension Fp6[w]/(w^2 - v)

The tower structure is:
    Fp -> Fp2 -> Fp6 -> Fp12

Reference:
    "High-Speed Software Implementation of the Optimal Ate Pairing over
    Barreto-Naehrig Curves" by Beuchat et al.
"""

from __future__ import annotations
from typing import Union, Tuple
from .field import Fp
from .constants import FIELD_MODULUS


# =============================================================================
# Precomputed Frobenius Coefficients (lazy-initialized)
# =============================================================================
# These are ξ^((p-1)/k) for k = 2, 3, 6 where ξ = 1+u is the Fp6 non-residue.
# Computed once at first use to avoid expensive Fp2 exponentiation at module load.

_FROBENIUS_COEFFS_CACHE = {}


class Fp2:
    """
    An element of the quadratic extension field Fp2 = Fp[u]/(u^2 + 1).

    Elements are represented as c0 + c1 * u where c0, c1 ∈ Fp.
    The irreducible polynomial is u^2 + 1 = 0, so u^2 = -1.

    Attributes:
        c0: The constant coefficient (Fp element).
        c1: The coefficient of u (Fp element).
    """

    __slots__ = ("c0", "c1")

    def __init__(self, c0: Union[Fp, int], c1: Union[Fp, int] = 0):
        """
        Create a new Fp2 element.

        Args:
            c0: The constant coefficient.
            c1: The coefficient of u (default 0).
        """
        self.c0 = c0 if isinstance(c0, Fp) else Fp(c0)
        self.c1 = c1 if isinstance(c1, Fp) else Fp(c1)

    @classmethod
    def zero(cls) -> Fp2:
        """Return the additive identity."""
        return cls(Fp.zero(), Fp.zero())

    @classmethod
    def one(cls) -> Fp2:
        """Return the multiplicative identity."""
        return cls(Fp.one(), Fp.zero())

    def __repr__(self) -> str:
        return f"Fp2({self.c0.to_int()}, {self.c1.to_int()})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fp2):
            return self.c0 == other.c0 and self.c1 == other.c1
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.c0.value, self.c1.value))

    def is_zero(self) -> bool:
        """Check if this element is zero."""
        return self.c0.is_zero() and self.c1.is_zero()

    def is_one(self) -> bool:
        """Check if this element is one."""
        return self.c0.is_one() and self.c1.is_zero()

    def __neg__(self) -> Fp2:
        """Return the additive inverse."""
        return Fp2(-self.c0, -self.c1)

    def __add__(self, other: Fp2) -> Fp2:
        """Add two Fp2 elements."""
        return Fp2(self.c0 + other.c0, self.c1 + other.c1)

    def __sub__(self, other: Fp2) -> Fp2:
        """Subtract two Fp2 elements."""
        return Fp2(self.c0 - other.c0, self.c1 - other.c1)

    def __mul__(self, other: Union[Fp2, Fp, int]) -> Fp2:
        """
        Multiply two Fp2 elements using Karatsuba-like optimization.

        (a0 + a1*u) * (b0 + b1*u)
        = a0*b0 + a0*b1*u + a1*b0*u + a1*b1*u^2
        = a0*b0 - a1*b1 + (a0*b1 + a1*b0)*u    (since u^2 = -1)

        Optimized with Karatsuba:
        c0 = a0*b0 - a1*b1
        c1 = (a0+a1)*(b0+b1) - a0*b0 - a1*b1
        """
        if isinstance(other, (Fp, int)):
            # Scalar multiplication
            if isinstance(other, int):
                other = Fp(other)
            return Fp2(self.c0 * other, self.c1 * other)

        # Karatsuba multiplication
        a0b0 = self.c0 * other.c0
        a1b1 = self.c1 * other.c1

        c0 = a0b0 - a1b1  # a0*b0 - a1*b1 (since u^2 = -1)
        c1 = (self.c0 + self.c1) * (other.c0 + other.c1) - a0b0 - a1b1

        return Fp2(c0, c1)

    def __rmul__(self, other: Union[Fp, int]) -> Fp2:
        return self.__mul__(other)

    def square(self) -> Fp2:
        """
        Compute the square of this element.

        (a + b*u)^2 = a^2 + 2ab*u + b^2*u^2
                    = a^2 - b^2 + 2ab*u    (since u^2 = -1)

        Optimized:
        c0 = (a+b)*(a-b) = a^2 - b^2
        c1 = 2*a*b
        """
        c0 = (self.c0 + self.c1) * (self.c0 - self.c1)
        c1 = self.c0 * self.c1
        c1 = c1 + c1  # 2*a*b

        return Fp2(c0, c1)

    def inverse(self) -> Fp2:
        """
        Compute the multiplicative inverse.

        (a + b*u)^(-1) = (a - b*u) / (a^2 + b^2)

        Since u^2 = -1, the norm is a^2 + b^2.
        """
        if self.is_zero():
            raise ZeroDivisionError("Cannot invert zero in Fp2")

        # Norm: a^2 + b^2
        norm = self.c0.square() + self.c1.square()
        norm_inv = norm.inverse()

        return Fp2(self.c0 * norm_inv, -self.c1 * norm_inv)

    def __truediv__(self, other: Fp2) -> Fp2:
        """Divide two Fp2 elements."""
        return self * other.inverse()

    def __pow__(self, exp: int) -> Fp2:
        """Compute self^exp using square-and-multiply."""
        if exp < 0:
            return self.inverse() ** (-exp)
        if exp == 0:
            return Fp2.one()
        if exp == 1:
            return Fp2(self.c0, self.c1)
        if exp == 2:
            return self.square()

        result = Fp2.one()
        base = Fp2(self.c0, self.c1)

        while exp > 0:
            if exp & 1:
                result = result * base
            base = base.square()
            exp >>= 1

        return result

    def conjugate(self) -> Fp2:
        """
        Compute the conjugate (Frobenius endomorphism).

        For Fp2, the conjugate of (a + b*u) is (a - b*u).
        """
        return Fp2(self.c0, -self.c1)

    def mul_by_nonresidue(self) -> Fp2:
        """
        Multiply by the non-residue (1 + u) used in the Fp6 construction.

        (a + b*u) * (1 + u) = a + a*u + b*u + b*u^2
                            = (a - b) + (a + b)*u
        """
        return Fp2(self.c0 - self.c1, self.c0 + self.c1)


class Fp6:
    """
    An element of the sextic extension field Fp6 = Fp2[v]/(v^3 - ξ).

    Here ξ = (1 + u) is a non-residue in Fp2.
    Elements are represented as c0 + c1*v + c2*v^2 where c0, c1, c2 ∈ Fp2.

    Attributes:
        c0, c1, c2: The coefficients (Fp2 elements).
    """

    __slots__ = ("c0", "c1", "c2")

    def __init__(self, c0: Fp2, c1: Fp2 = None, c2: Fp2 = None):
        """
        Create a new Fp6 element.

        Args:
            c0: The constant coefficient.
            c1: The coefficient of v (default Fp2.zero()).
            c2: The coefficient of v^2 (default Fp2.zero()).
        """
        self.c0 = c0
        self.c1 = c1 if c1 is not None else Fp2.zero()
        self.c2 = c2 if c2 is not None else Fp2.zero()

    @classmethod
    def zero(cls) -> Fp6:
        """Return the additive identity."""
        return cls(Fp2.zero(), Fp2.zero(), Fp2.zero())

    @classmethod
    def one(cls) -> Fp6:
        """Return the multiplicative identity."""
        return cls(Fp2.one(), Fp2.zero(), Fp2.zero())

    def __repr__(self) -> str:
        return f"Fp6({self.c0}, {self.c1}, {self.c2})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fp6):
            return self.c0 == other.c0 and self.c1 == other.c1 and self.c2 == other.c2
        return NotImplemented

    def is_zero(self) -> bool:
        return self.c0.is_zero() and self.c1.is_zero() and self.c2.is_zero()

    def is_one(self) -> bool:
        return self.c0.is_one() and self.c1.is_zero() and self.c2.is_zero()

    def __neg__(self) -> Fp6:
        return Fp6(-self.c0, -self.c1, -self.c2)

    def __add__(self, other: Fp6) -> Fp6:
        return Fp6(self.c0 + other.c0, self.c1 + other.c1, self.c2 + other.c2)

    def __sub__(self, other: Fp6) -> Fp6:
        return Fp6(self.c0 - other.c0, self.c1 - other.c1, self.c2 - other.c2)

    def __mul__(self, other: Union[Fp6, Fp2]) -> Fp6:
        """
        Multiply two Fp6 elements.

        Uses the formula from "Multiplication and Squaring on Pairing-Friendly Fields"
        by Devegili et al.
        """
        if isinstance(other, Fp2):
            return Fp6(self.c0 * other, self.c1 * other, self.c2 * other)

        # Karatsuba-like multiplication
        # (a0 + a1*v + a2*v^2) * (b0 + b1*v + b2*v^2)
        a0, a1, a2 = self.c0, self.c1, self.c2
        b0, b1, b2 = other.c0, other.c1, other.c2

        v0 = a0 * b0
        v1 = a1 * b1
        v2 = a2 * b2

        # c0 = v0 + ξ * ((a1 + a2)(b1 + b2) - v1 - v2)
        c0 = v0 + ((a1 + a2) * (b1 + b2) - v1 - v2).mul_by_nonresidue()

        # c1 = (a0 + a1)(b0 + b1) - v0 - v1 + ξ * v2
        c1 = (a0 + a1) * (b0 + b1) - v0 - v1 + v2.mul_by_nonresidue()

        # c2 = (a0 + a2)(b0 + b2) - v0 - v2 + v1
        c2 = (a0 + a2) * (b0 + b2) - v0 - v2 + v1

        return Fp6(c0, c1, c2)

    def square(self) -> Fp6:
        """
        Compute the square of this element.

        Uses optimized squaring formula.
        """
        a0, a1, a2 = self.c0, self.c1, self.c2

        s0 = a0.square()
        ab = a0 * a1
        s1 = ab + ab  # 2 * a0 * a1
        s2 = (a0 - a1 + a2).square()
        bc = a1 * a2
        s3 = bc + bc  # 2 * a1 * a2
        s4 = a2.square()

        c0 = s0 + s3.mul_by_nonresidue()
        c1 = s1 + s4.mul_by_nonresidue()
        c2 = s1 + s2 + s3 - s0 - s4

        return Fp6(c0, c1, c2)

    def inverse(self) -> Fp6:
        """
        Compute the multiplicative inverse.

        Uses the formula from "Implementing Cryptographic Pairings over
        Barreto-Naehrig Curves" by Naehrig et al.
        """
        if self.is_zero():
            raise ZeroDivisionError("Cannot invert zero in Fp6")

        a0, a1, a2 = self.c0, self.c1, self.c2

        # Compute the cofactors
        c0 = a0.square() - (a1 * a2).mul_by_nonresidue()
        c1 = a2.square().mul_by_nonresidue() - a0 * a1
        c2 = a1.square() - a0 * a2

        # Compute the norm
        t = ((a2 * c1 + a1 * c2).mul_by_nonresidue() + a0 * c0).inverse()

        return Fp6(c0 * t, c1 * t, c2 * t)

    def __truediv__(self, other: Fp6) -> Fp6:
        return self * other.inverse()

    def __pow__(self, exp: int) -> Fp6:
        if exp < 0:
            return self.inverse() ** (-exp)
        if exp == 0:
            return Fp6.one()
        if exp == 1:
            return Fp6(self.c0, self.c1, self.c2)
        if exp == 2:
            return self.square()

        result = Fp6.one()
        base = Fp6(self.c0, self.c1, self.c2)

        while exp > 0:
            if exp & 1:
                result = result * base
            base = base.square()
            exp >>= 1

        return result

    def mul_by_nonresidue(self) -> Fp6:
        """
        Multiply by v (the generator of Fp6 over Fp2).

        v * (c0 + c1*v + c2*v^2) = c0*v + c1*v^2 + c2*v^3
                                = c2*ξ + c0*v + c1*v^2
        """
        return Fp6(self.c2.mul_by_nonresidue(), self.c0, self.c1)


class Fp12:
    """
    An element of the dodecic extension field Fp12 = Fp6[w]/(w^2 - v).

    Elements are represented as c0 + c1*w where c0, c1 ∈ Fp6.

    This is the target field for the BN254 pairing.

    Attributes:
        c0, c1: The coefficients (Fp6 elements).
    """

    __slots__ = ("c0", "c1")

    def __init__(self, c0: Fp6, c1: Fp6 = None):
        """
        Create a new Fp12 element.

        Args:
            c0: The constant coefficient.
            c1: The coefficient of w (default Fp6.zero()).
        """
        self.c0 = c0
        self.c1 = c1 if c1 is not None else Fp6.zero()

    @classmethod
    def zero(cls) -> Fp12:
        """Return the additive identity."""
        return cls(Fp6.zero(), Fp6.zero())

    @classmethod
    def one(cls) -> Fp12:
        """Return the multiplicative identity."""
        return cls(Fp6.one(), Fp6.zero())

    def __repr__(self) -> str:
        return f"Fp12({self.c0}, {self.c1})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fp12):
            return self.c0 == other.c0 and self.c1 == other.c1
        return NotImplemented

    def is_zero(self) -> bool:
        return self.c0.is_zero() and self.c1.is_zero()

    def is_one(self) -> bool:
        return self.c0.is_one() and self.c1.is_zero()

    def __neg__(self) -> Fp12:
        return Fp12(-self.c0, -self.c1)

    def __add__(self, other: Fp12) -> Fp12:
        return Fp12(self.c0 + other.c0, self.c1 + other.c1)

    def __sub__(self, other: Fp12) -> Fp12:
        return Fp12(self.c0 - other.c0, self.c1 - other.c1)

    def __mul__(self, other: Fp12) -> Fp12:
        """
        Multiply two Fp12 elements.

        (a0 + a1*w) * (b0 + b1*w) = a0*b0 + a0*b1*w + a1*b0*w + a1*b1*w^2
                                  = a0*b0 + a1*b1*v + (a0*b1 + a1*b0)*w

        Since w^2 = v.
        """
        a0, a1 = self.c0, self.c1
        b0, b1 = other.c0, other.c1

        # Karatsuba
        v0 = a0 * b0
        v1 = a1 * b1

        c0 = v0 + v1.mul_by_nonresidue()  # a0*b0 + a1*b1*v
        c1 = (a0 + a1) * (b0 + b1) - v0 - v1

        return Fp12(c0, c1)

    def square(self) -> Fp12:
        """
        Compute the square of this element.

        (a + b*w)^2 = a^2 + 2ab*w + b^2*w^2
                    = a^2 + b^2*v + 2ab*w
        """
        a, b = self.c0, self.c1

        ab = a * b
        c0 = (a + b) * (a + b.mul_by_nonresidue()) - ab - ab.mul_by_nonresidue()
        c1 = ab + ab

        return Fp12(c0, c1)

    def inverse(self) -> Fp12:
        """
        Compute the multiplicative inverse.

        (a + b*w)^(-1) = (a - b*w) / (a^2 - b^2*v)
        """
        if self.is_zero():
            raise ZeroDivisionError("Cannot invert zero in Fp12")

        a, b = self.c0, self.c1

        # Norm: a^2 - b^2 * v
        t = (a.square() - b.square().mul_by_nonresidue()).inverse()

        return Fp12(a * t, -b * t)

    def __truediv__(self, other: Fp12) -> Fp12:
        return self * other.inverse()

    def __pow__(self, exp: int) -> Fp12:
        if exp < 0:
            return self.inverse() ** (-exp)
        if exp == 0:
            return Fp12.one()
        if exp == 1:
            return Fp12(self.c0, self.c1)
        if exp == 2:
            return self.square()

        result = Fp12.one()
        base = Fp12(self.c0, self.c1)

        while exp > 0:
            if exp & 1:
                result = result * base
            base = base.square()
            exp >>= 1

        return result

    def conjugate(self) -> Fp12:
        """
        Compute the conjugate (Frobenius endomorphism of degree 6).

        For Fp12, the conjugate of (a + b*w) is (a - b*w).
        """
        return Fp12(self.c0, -self.c1)

    def frobenius(self) -> Fp12:
        """
        Compute the Frobenius endomorphism (raising to the p-th power).

        Uses precomputed coefficients for the tower decomposition:
          Fp12(c0, c1)^p = (c0^p, c1^p · ξ^((p-1)/6))
        where c0^p, c1^p are Fp6 Frobenius and ξ = 1+u.

        Performance: O(1) Fp multiplications instead of O(254) Fp12 squarings.
        """
        gamma = _get_frobenius_coeff_fp12()
        # Fp6 Frobenius for each component
        c0_frob = _fp6_frobenius(self.c0)
        c1_frob = _fp6_frobenius(self.c1)
        # Multiply c1 by ξ^((p-1)/6)
        c1_frob = _fp6_mul_by_fp2(c1_frob, gamma)
        return Fp12(c0_frob, c1_frob)

    def cyclotomic_square(self) -> Fp12:
        """
        Compute the square in the cyclotomic subgroup.

        This is an optimized squaring for elements in the cyclotomic subgroup
        of Fp12, which is where pairing results live after final exponentiation.

        Uses the method from "Faster Squaring in the Cyclotomic Subgroup of
        Sixth Degree Extensions" by Granger and Scott.
        """
        # For simplicity, we use the standard square here.
        # A full implementation would use the optimized cyclotomic squaring.
        return self.square()

    def cyclotomic_exp(self, exp: int) -> Fp12:
        """
        Compute exponentiation in the cyclotomic subgroup.

        Uses cyclotomic squaring for efficiency.
        """
        if exp == 0:
            return Fp12.one()

        result = Fp12.one()
        base = Fp12(self.c0, self.c1)

        while exp > 0:
            if exp & 1:
                result = result * base
            base = base.cyclotomic_square()
            exp >>= 1

        return result


# =============================================================================
# Frobenius Helper Functions
# =============================================================================

def _fp2_pow(base: Fp2, exp: int) -> Fp2:
    """Fp2 exponentiation via square-and-multiply (used for coefficient precomputation)."""
    if exp == 0:
        return Fp2.one()
    result = Fp2.one()
    b = Fp2(base.c0, base.c1)
    while exp > 0:
        if exp & 1:
            result = result * b
        b = b.square()
        exp >>= 1
    return result


def _get_frobenius_coeffs_fp6():
    """
    Get precomputed Frobenius coefficients for Fp6.
    
    gamma1 = ξ^((p-1)/3), gamma2 = ξ^(2(p-1)/3)
    where ξ = 1+u is the Fp6 non-residue.
    """
    if 'fp6' not in _FROBENIUS_COEFFS_CACHE:
        p = FIELD_MODULUS
        xi = Fp2(Fp(1), Fp(1))  # 1 + u
        e = (p - 1) // 3
        gamma1 = _fp2_pow(xi, e)
        gamma2 = _fp2_pow(xi, 2 * e)
        _FROBENIUS_COEFFS_CACHE['fp6'] = (gamma1, gamma2)
    return _FROBENIUS_COEFFS_CACHE['fp6']


def _get_frobenius_coeff_fp12() -> Fp2:
    """
    Get precomputed Frobenius coefficient for Fp12.
    
    gamma = ξ^((p-1)/6) where ξ = 1+u.
    """
    if 'fp12' not in _FROBENIUS_COEFFS_CACHE:
        p = FIELD_MODULUS
        xi = Fp2(Fp(1), Fp(1))  # 1 + u
        e = (p - 1) // 6
        _FROBENIUS_COEFFS_CACHE['fp12'] = _fp2_pow(xi, e)
    return _FROBENIUS_COEFFS_CACHE['fp12']


def _fp6_frobenius(x: Fp6) -> Fp6:
    """
    Compute Frobenius endomorphism on Fp6.
    
    For Fp6(c0, c1, c2) = c0 + c1·v + c2·v²:
      x^p = c0^p + c1^p · v^p + c2^p · v^(2p)
    
    Fp2 Frobenius is conjugation: (a+bu)^p = a-bu.
    v^p = ξ^((p-1)/3) · v, v^(2p) = ξ^(2(p-1)/3) · v².
    """
    gamma1, gamma2 = _get_frobenius_coeffs_fp6()
    c0_frob = x.c0.conjugate()
    c1_frob = x.c1.conjugate() * gamma1
    c2_frob = x.c2.conjugate() * gamma2
    return Fp6(c0_frob, c1_frob, c2_frob)


def _fp6_mul_by_fp2(x: Fp6, scalar: Fp2) -> Fp6:
    """Multiply each Fp2 component of an Fp6 element by an Fp2 scalar."""
    return Fp6(x.c0 * scalar, x.c1 * scalar, x.c2 * scalar)
