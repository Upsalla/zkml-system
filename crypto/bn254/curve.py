"""
BN254 Elliptic Curve Arithmetic

This module provides implementations for elliptic curve point operations on
the BN254 curve:
    - G1: Points on the curve over Fp
    - G2: Points on the twisted curve over Fp2

The curve equation is y² = x³ + 3 for G1.
For G2, the twist is y² = x³ + 3/(u+1) over Fp2.

Points are represented in Jacobian coordinates (X, Y, Z) where the affine
point is (X/Z², Y/Z³). This avoids expensive field inversions during
point addition and doubling.

Reference:
    "Guide to Elliptic Curve Cryptography" by Hankerson, Menezes, Vanstone
"""

from __future__ import annotations
from typing import Union, Tuple, Optional
from .field import Fp, Fr
from .extension_field import Fp2
from .constants import (
    FIELD_MODULUS,
    CURVE_ORDER,
    CURVE_B,
    G1_X,
    G1_Y,
    G2_X,
    G2_Y,
    TWIST_B_C0,
    TWIST_B_C1,
)


class G1Point:
    """
    A point on the BN254 G1 curve (over Fp).

    Points are stored in Jacobian coordinates (X, Y, Z) where:
    - The affine point is (X/Z², Y/Z³)
    - The point at infinity is represented by Z = 0

    Attributes:
        x, y, z: Jacobian coordinates (Fp elements).
    """

    __slots__ = ("x", "y", "z")

    # Curve parameter b = 3
    B = Fp(CURVE_B)

    def __init__(self, x: Fp, y: Fp, z: Fp = None):
        """
        Create a new G1 point.

        Args:
            x: X coordinate (Fp element).
            y: Y coordinate (Fp element).
            z: Z coordinate (Fp element, default Fp.one() for affine input).
        """
        self.x = x
        self.y = y
        self.z = z if z is not None else Fp.one()

    @classmethod
    def identity(cls) -> G1Point:
        """Return the point at infinity (identity element)."""
        return cls(Fp.zero(), Fp.one(), Fp.zero())

    @classmethod
    def generator(cls) -> G1Point:
        """Return the generator point of G1."""
        return cls(Fp(G1_X), Fp(G1_Y), Fp.one())

    @classmethod
    def from_affine(cls, x: int, y: int) -> G1Point:
        """Create a G1 point from affine coordinates."""
        return cls(Fp(x), Fp(y), Fp.one())

    def is_identity(self) -> bool:
        """Check if this is the point at infinity."""
        return self.z.is_zero()

    def to_affine(self) -> Tuple[Fp, Fp]:
        """
        Convert from Jacobian to affine coordinates.

        Returns:
            Tuple (x, y) in affine coordinates.

        Raises:
            ValueError: If the point is at infinity.
        """
        if self.is_identity():
            raise ValueError("Cannot convert point at infinity to affine")

        z_inv = self.z.inverse()
        z_inv_sq = z_inv.square()
        z_inv_cu = z_inv_sq * z_inv

        return (self.x * z_inv_sq, self.y * z_inv_cu)

    def __repr__(self) -> str:
        if self.is_identity():
            return "G1Point(infinity)"
        x, y = self.to_affine()
        return f"G1Point({x.to_int()}, {y.to_int()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, G1Point):
            return NotImplemented

        # Handle identity cases
        if self.is_identity() and other.is_identity():
            return True
        if self.is_identity() or other.is_identity():
            return False

        # Compare in Jacobian: (X1/Z1², Y1/Z1³) == (X2/Z2², Y2/Z2³)
        # Cross-multiply to avoid inversions:
        # X1 * Z2² == X2 * Z1² and Y1 * Z2³ == Y2 * Z1³
        z1_sq = self.z.square()
        z2_sq = other.z.square()
        z1_cu = z1_sq * self.z
        z2_cu = z2_sq * other.z

        return (self.x * z2_sq == other.x * z1_sq and
                self.y * z2_cu == other.y * z1_cu)

    def __neg__(self) -> G1Point:
        """Return the negation of this point."""
        return G1Point(self.x, -self.y, self.z)

    def double(self) -> G1Point:
        """
        Double this point using the optimized Jacobian doubling formula.

        Uses the formula from "Guide to Elliptic Curve Cryptography".
        Cost: 1M + 5S + 1*a + 7add + 2*2 + 1*3 + 1*8
        For a = 0 (BN254): 1M + 5S + 7add + 2*2 + 1*3 + 1*8
        """
        if self.is_identity():
            return G1Point.identity()

        # For y = 0, the tangent is vertical, so 2P = O
        if self.y.is_zero():
            return G1Point.identity()

        x, y, z = self.x, self.y, self.z

        # A = X²
        a = x.square()
        # B = Y²
        b = y.square()
        # C = B²
        c = b.square()
        # D = 2 * ((X + B)² - A - C)
        d = (x + b).square() - a - c
        d = d + d
        # E = 3 * A (for a = 0)
        e = a + a + a
        # F = E²
        f = e.square()

        # X3 = F - 2*D
        x3 = f - d - d
        # Y3 = E * (D - X3) - 8*C
        y3 = e * (d - x3) - (c + c + c + c + c + c + c + c)
        # Z3 = 2 * Y * Z
        z3 = y * z
        z3 = z3 + z3

        return G1Point(x3, y3, z3)

    def __add__(self, other: G1Point) -> G1Point:
        """
        Add two G1 points using the mixed addition formula when possible.

        Uses optimized Jacobian addition formulas.
        """
        if self.is_identity():
            return G1Point(other.x, other.y, other.z)
        if other.is_identity():
            return G1Point(self.x, self.y, self.z)

        x1, y1, z1 = self.x, self.y, self.z
        x2, y2, z2 = other.x, other.y, other.z

        # U1 = X1 * Z2²
        z2_sq = z2.square()
        u1 = x1 * z2_sq

        # U2 = X2 * Z1²
        z1_sq = z1.square()
        u2 = x2 * z1_sq

        # S1 = Y1 * Z2³
        s1 = y1 * z2_sq * z2

        # S2 = Y2 * Z1³
        s2 = y2 * z1_sq * z1

        # H = U2 - U1
        h = u2 - u1

        # R = S2 - S1
        r = s2 - s1

        # If H == 0 and R == 0, the points are equal -> double
        if h.is_zero():
            if r.is_zero():
                return self.double()
            else:
                # Points are inverses of each other
                return G1Point.identity()

        # H² and H³
        h_sq = h.square()
        h_cu = h_sq * h

        # X3 = R² - H³ - 2*U1*H²
        x3 = r.square() - h_cu - (u1 * h_sq + u1 * h_sq)

        # Y3 = R * (U1*H² - X3) - S1*H³
        y3 = r * (u1 * h_sq - x3) - s1 * h_cu

        # Z3 = H * Z1 * Z2
        z3 = h * z1 * z2

        return G1Point(x3, y3, z3)

    def __sub__(self, other: G1Point) -> G1Point:
        """Subtract two G1 points."""
        return self + (-other)

    def __mul__(self, scalar: Union[Fr, int]) -> G1Point:
        """
        Scalar multiplication using double-and-add.

        Args:
            scalar: The scalar (Fr element or int).

        Returns:
            self * scalar
        """
        if isinstance(scalar, Fr):
            scalar = scalar.to_int()

        if scalar == 0:
            return G1Point.identity()
        if scalar < 0:
            return (-self) * (-scalar)

        result = G1Point.identity()
        base = G1Point(self.x, self.y, self.z)

        while scalar > 0:
            if scalar & 1:
                result = result + base
            base = base.double()
            scalar >>= 1

        return result

    def __rmul__(self, scalar: Union[Fr, int]) -> G1Point:
        return self.__mul__(scalar)

    def is_on_curve(self) -> bool:
        """
        Check if this point is on the curve y² = x³ + 3.

        For Jacobian coordinates: Y² = X³ + b*Z⁶
        """
        if self.is_identity():
            return True

        x, y, z = self.x, self.y, self.z

        # Y² == X³ + b*Z⁶
        lhs = y.square()
        z_sq = z.square()
        z_cu = z_sq * z
        z_6 = z_cu.square()
        rhs = x.square() * x + self.B * z_6

        return lhs == rhs


class G2Point:
    """
    A point on the BN254 G2 twisted curve (over Fp2).

    The twist curve is y² = x³ + b' where b' = 3/(u+1).
    Points are stored in Jacobian coordinates.

    Attributes:
        x, y, z: Jacobian coordinates (Fp2 elements).
    """

    __slots__ = ("x", "y", "z")

    # Twisted curve parameter b' = 3/(u+1)
    B = Fp2(Fp(TWIST_B_C0), Fp(TWIST_B_C1))

    def __init__(self, x: Fp2, y: Fp2, z: Fp2 = None):
        """
        Create a new G2 point.

        Args:
            x: X coordinate (Fp2 element).
            y: Y coordinate (Fp2 element).
            z: Z coordinate (Fp2 element, default Fp2.one() for affine input).
        """
        self.x = x
        self.y = y
        self.z = z if z is not None else Fp2.one()

    @classmethod
    def identity(cls) -> G2Point:
        """Return the point at infinity (identity element)."""
        return cls(Fp2.zero(), Fp2.one(), Fp2.zero())

    @classmethod
    def generator(cls) -> G2Point:
        """Return the generator point of G2."""
        x = Fp2(Fp(G2_X[0]), Fp(G2_X[1]))
        y = Fp2(Fp(G2_Y[0]), Fp(G2_Y[1]))
        return cls(x, y, Fp2.one())

    def is_identity(self) -> bool:
        """Check if this is the point at infinity."""
        return self.z.is_zero()

    def to_affine(self) -> Tuple[Fp2, Fp2]:
        """Convert from Jacobian to affine coordinates."""
        if self.is_identity():
            raise ValueError("Cannot convert point at infinity to affine")

        z_inv = self.z.inverse()
        z_inv_sq = z_inv.square()
        z_inv_cu = z_inv_sq * z_inv

        return (self.x * z_inv_sq, self.y * z_inv_cu)

    def __repr__(self) -> str:
        if self.is_identity():
            return "G2Point(infinity)"
        return f"G2Point({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, G2Point):
            return NotImplemented

        if self.is_identity() and other.is_identity():
            return True
        if self.is_identity() or other.is_identity():
            return False

        z1_sq = self.z.square()
        z2_sq = other.z.square()
        z1_cu = z1_sq * self.z
        z2_cu = z2_sq * other.z

        return (self.x * z2_sq == other.x * z1_sq and
                self.y * z2_cu == other.y * z1_cu)

    def __neg__(self) -> G2Point:
        """Return the negation of this point."""
        return G2Point(self.x, -self.y, self.z)

    def double(self) -> G2Point:
        """Double this point using the optimized Jacobian doubling formula."""
        if self.is_identity():
            return G2Point.identity()

        if self.y.is_zero():
            return G2Point.identity()

        x, y, z = self.x, self.y, self.z

        a = x.square()
        b = y.square()
        c = b.square()
        d = (x + b).square() - a - c
        d = d + d
        e = a + a + a
        f = e.square()

        x3 = f - d - d
        eight_c = c + c
        eight_c = eight_c + eight_c
        eight_c = eight_c + eight_c
        y3 = e * (d - x3) - eight_c
        z3 = y * z
        z3 = z3 + z3

        return G2Point(x3, y3, z3)

    def __add__(self, other: G2Point) -> G2Point:
        """Add two G2 points."""
        if self.is_identity():
            return G2Point(other.x, other.y, other.z)
        if other.is_identity():
            return G2Point(self.x, self.y, self.z)

        x1, y1, z1 = self.x, self.y, self.z
        x2, y2, z2 = other.x, other.y, other.z

        z2_sq = z2.square()
        u1 = x1 * z2_sq
        z1_sq = z1.square()
        u2 = x2 * z1_sq
        s1 = y1 * z2_sq * z2
        s2 = y2 * z1_sq * z1

        h = u2 - u1
        r = s2 - s1

        if h.is_zero():
            if r.is_zero():
                return self.double()
            else:
                return G2Point.identity()

        h_sq = h.square()
        h_cu = h_sq * h

        x3 = r.square() - h_cu - (u1 * h_sq + u1 * h_sq)
        y3 = r * (u1 * h_sq - x3) - s1 * h_cu
        z3 = h * z1 * z2

        return G2Point(x3, y3, z3)

    def __sub__(self, other: G2Point) -> G2Point:
        """Subtract two G2 points."""
        return self + (-other)

    def __mul__(self, scalar: Union[Fr, int]) -> G2Point:
        """Scalar multiplication using double-and-add."""
        if isinstance(scalar, Fr):
            scalar = scalar.to_int()

        if scalar == 0:
            return G2Point.identity()
        if scalar < 0:
            return (-self) * (-scalar)

        result = G2Point.identity()
        base = G2Point(self.x, self.y, self.z)

        while scalar > 0:
            if scalar & 1:
                result = result + base
            base = base.double()
            scalar >>= 1

        return result

    def __rmul__(self, scalar: Union[Fr, int]) -> G2Point:
        return self.__mul__(scalar)

    def is_on_curve(self) -> bool:
        """Check if this point is on the twisted curve y² = x³ + b'."""
        if self.is_identity():
            return True

        x, y, z = self.x, self.y, self.z

        lhs = y.square()
        z_sq = z.square()
        z_cu = z_sq * z
        z_6 = z_cu.square()
        rhs = x.square() * x + self.B * z_6

        return lhs == rhs
