"""
PLONK Core Module - Standardized API

This module provides the core abstractions for the PLONK proof system:
- Field elements (using BN254 scalar field Fr)
- Polynomials with FFT support
- KZG commitments
- Circuit representation

All components use a consistent API and are designed to work together.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import hashlib
import sys
import os

# Import BN254 primitives
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto.bn254.field import Fr, Fp
from crypto.bn254.curve import G1Point, G2Point


# =============================================================================
# Field Abstraction Layer
# =============================================================================

class Field:
    """
    Unified field interface for the PLONK system.
    Uses BN254 scalar field (Fr) as the underlying implementation.
    """
    
    # BN254 scalar field modulus
    MODULUS = Fr.MODULUS
    
    @staticmethod
    def element(value: Union[int, Fr]) -> Fr:
        """Create a field element."""
        if isinstance(value, Fr):
            return value
        return Fr(value % Fr.MODULUS)
    
    @staticmethod
    def zero() -> Fr:
        """Return the additive identity."""
        return Fr.zero()
    
    @staticmethod
    def one() -> Fr:
        """Return the multiplicative identity."""
        return Fr.one()
    
    @staticmethod
    def random() -> Fr:
        """Generate a random field element."""
        import random
        return Fr(random.randint(1, Fr.MODULUS - 1))
    
    @staticmethod
    def from_bytes(data: bytes) -> Fr:
        """Create field element from bytes (hash)."""
        h = hashlib.sha256(data).digest()
        value = int.from_bytes(h, 'big') % Fr.MODULUS
        return Fr(value)


# =============================================================================
# Polynomial with FFT Support
# =============================================================================

@dataclass
class Polynomial:
    """
    Polynomial over the scalar field with FFT support.
    
    Coefficients are stored in ascending order: [a0, a1, a2, ...] = a0 + a1*x + a2*x^2 + ...
    """
    coeffs: List[Fr]
    
    @classmethod
    def zero(cls) -> Polynomial:
        """Return the zero polynomial."""
        return cls([Fr.zero()])
    
    @classmethod
    def one(cls) -> Polynomial:
        """Return the constant polynomial 1."""
        return cls([Fr.one()])
    
    @classmethod
    def from_ints(cls, values: List[int]) -> Polynomial:
        """Create polynomial from integer coefficients."""
        return cls([Fr(v) for v in values])
    
    @classmethod
    def from_roots(cls, roots: List[Fr]) -> Polynomial:
        """Create polynomial from its roots: (x - r1)(x - r2)..."""
        result = cls.one()
        for root in roots:
            # Multiply by (x - root)
            factor = cls([Fr.zero() - root, Fr.one()])
            result = result * factor
        return result
    
    @classmethod
    def lagrange_interpolate(cls, points: List[Tuple[Fr, Fr]]) -> Polynomial:
        """Lagrange interpolation from (x, y) points."""
        n = len(points)
        if n == 0:
            return cls.zero()
        
        result = cls.zero()
        
        for i in range(n):
            xi, yi = points[i]
            
            # Build Lagrange basis polynomial L_i(x)
            numerator = cls.one()
            denominator = Fr.one()
            
            for j in range(n):
                if i != j:
                    xj, _ = points[j]
                    # numerator *= (x - xj)
                    numerator = numerator * cls([Fr.zero() - xj, Fr.one()])
                    # denominator *= (xi - xj)
                    denominator = denominator * (xi - xj)
            
            # L_i(x) = numerator / denominator
            inv_denom = denominator.inverse()
            basis = numerator.scalar_mul(inv_denom)
            
            # result += yi * L_i(x)
            result = result + basis.scalar_mul(yi)
        
        return result
    
    def degree(self) -> int:
        """Return the degree of the polynomial."""
        # Remove leading zeros
        d = len(self.coeffs) - 1
        while d > 0 and self.coeffs[d] == Fr.zero():
            d -= 1
        return d
    
    def evaluate(self, x: Fr) -> Fr:
        """Evaluate polynomial at point x using Horner's method."""
        if not self.coeffs:
            return Fr.zero()
        
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]
        return result
    
    def scalar_mul(self, scalar: Fr) -> Polynomial:
        """Multiply polynomial by a scalar."""
        return Polynomial([c * scalar for c in self.coeffs])
    
    def __add__(self, other: Polynomial) -> Polynomial:
        """Add two polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else Fr.zero()
            b = other.coeffs[i] if i < len(other.coeffs) else Fr.zero()
            result.append(a + b)
        return Polynomial(result)
    
    def __sub__(self, other: Polynomial) -> Polynomial:
        """Subtract two polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else Fr.zero()
            b = other.coeffs[i] if i < len(other.coeffs) else Fr.zero()
            result.append(a - b)
        return Polynomial(result)
    
    def __mul__(self, other: Polynomial) -> Polynomial:
        """Multiply two polynomials (convolution)."""
        if not self.coeffs or not other.coeffs:
            return Polynomial.zero()
        
        result_len = len(self.coeffs) + len(other.coeffs) - 1
        result = [Fr.zero() for _ in range(result_len)]
        
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = result[i + j] + a * b
        
        return Polynomial(result)
    
    def __neg__(self) -> Polynomial:
        """Negate polynomial."""
        return Polynomial([Fr.zero() - c for c in self.coeffs])
    
    def divide_by_linear(self, root: Fr) -> Tuple[Polynomial, Fr]:
        """
        Divide by (x - root) using synthetic division.
        Returns (quotient, remainder).
        """
        n = len(self.coeffs)
        if n == 0:
            return Polynomial.zero(), Fr.zero()
        
        quotient = [Fr.zero() for _ in range(n - 1)]
        remainder = self.coeffs[-1]
        
        for i in range(n - 2, -1, -1):
            quotient[i] = remainder
            remainder = self.coeffs[i] + remainder * root
        
        return Polynomial(quotient) if quotient else Polynomial.zero(), remainder


# =============================================================================
# FFT for Polynomial Operations
# =============================================================================

class FFT:
    """
    Fast Fourier Transform over finite fields.
    Uses roots of unity in the BN254 scalar field.
    """
    
    # Primitive root of unity for BN254 (2^28-th root)
    PRIMITIVE_ROOT = Fr(5)
    TWO_ADICITY = 28
    
    @classmethod
    def get_root_of_unity(cls, n: int) -> Fr:
        """Get n-th root of unity (n must be power of 2)."""
        if n & (n - 1) != 0:
            raise ValueError("n must be a power of 2")
        
        log_n = n.bit_length() - 1
        if log_n > cls.TWO_ADICITY:
            raise ValueError(f"n too large, max is 2^{cls.TWO_ADICITY}")
        
        # omega = primitive_root^(2^(TWO_ADICITY - log_n))
        exp = 1 << (cls.TWO_ADICITY - log_n)
        omega = cls.PRIMITIVE_ROOT
        for _ in range(exp - 1):
            omega = omega * cls.PRIMITIVE_ROOT
        
        return omega
    
    @classmethod
    def fft(cls, coeffs: List[Fr], inverse: bool = False) -> List[Fr]:
        """
        Compute FFT or inverse FFT.
        
        Args:
            coeffs: Polynomial coefficients (length must be power of 2)
            inverse: If True, compute inverse FFT
        
        Returns:
            Evaluations at roots of unity (or coefficients if inverse)
        """
        n = len(coeffs)
        if n == 1:
            return coeffs[:]
        
        if n & (n - 1) != 0:
            raise ValueError("Length must be a power of 2")
        
        # Get root of unity
        omega = cls.get_root_of_unity(n)
        if inverse:
            omega = omega.inverse()
        
        # Cooley-Tukey FFT
        result = coeffs[:]
        
        # Bit-reverse permutation
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                result[i], result[j] = result[j], result[i]
        
        # FFT
        length = 2
        while length <= n:
            w = cls.get_root_of_unity(length)
            if inverse:
                w = w.inverse()
            
            for i in range(0, n, length):
                wj = Fr.one()
                for j in range(length // 2):
                    u = result[i + j]
                    v = result[i + j + length // 2] * wj
                    result[i + j] = u + v
                    result[i + j + length // 2] = u - v
                    wj = wj * w
            
            length *= 2
        
        # Scale for inverse FFT
        if inverse:
            n_inv = Fr(n).inverse()
            result = [x * n_inv for x in result]
        
        return result
    
    @classmethod
    def polynomial_mul(cls, a: Polynomial, b: Polynomial) -> Polynomial:
        """Multiply polynomials using FFT."""
        # Pad to next power of 2
        n = 1
        target_len = len(a.coeffs) + len(b.coeffs) - 1
        while n < target_len:
            n *= 2
        
        # Pad coefficients
        a_padded = a.coeffs + [Fr.zero()] * (n - len(a.coeffs))
        b_padded = b.coeffs + [Fr.zero()] * (n - len(b.coeffs))
        
        # FFT
        a_fft = cls.fft(a_padded)
        b_fft = cls.fft(b_padded)
        
        # Point-wise multiplication
        c_fft = [a_fft[i] * b_fft[i] for i in range(n)]
        
        # Inverse FFT
        c_coeffs = cls.fft(c_fft, inverse=True)
        
        return Polynomial(c_coeffs[:target_len])


# =============================================================================
# KZG Commitment Scheme
# =============================================================================

@dataclass
class SRS:
    """
    Structured Reference String for KZG commitments.
    
    Contains powers of tau in G1 and G2:
    - g1_powers: [G1, tau*G1, tau^2*G1, ..., tau^n*G1]
    - g2_powers: [G2, tau*G2] (only need first two for verification)
    """
    g1_powers: List[G1Point]
    g2_powers: List[G2Point]
    max_degree: int
    
    @classmethod
    def generate_insecure(cls, max_degree: int, tau: Optional[Fr] = None) -> SRS:
        """
        Generate SRS for testing (INSECURE - tau is known).
        
        In production, use a trusted setup ceremony.
        """
        if tau is None:
            tau = Field.random()
        
        g1 = G1Point.generator()
        g2 = G2Point.generator()
        
        # Compute powers of tau
        g1_powers = []
        g2_powers = []
        
        tau_power = Fr.one()
        for i in range(max_degree + 1):
            g1_powers.append(g1 * tau_power)
            if i < 2:
                g2_powers.append(g2 * tau_power)
            tau_power = tau_power * tau
        
        return cls(g1_powers, g2_powers, max_degree)


@dataclass
class KZGCommitment:
    """A KZG polynomial commitment (a point on G1)."""
    point: G1Point
    
    def to_bytes(self) -> bytes:
        """Serialize commitment to bytes."""
        if self.point.is_identity():
            return b'\x00' * 64
        x, y = self.point.to_affine()
        return x.to_int().to_bytes(32, 'big') + y.to_int().to_bytes(32, 'big')


@dataclass
class KZGProof:
    """A KZG opening proof."""
    commitment: KZGCommitment
    point: Fr
    value: Fr
    proof: G1Point
    
    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        point_bytes = self.point.to_int().to_bytes(32, 'big')
        value_bytes = self.value.to_int().to_bytes(32, 'big')
        if self.proof.is_identity():
            proof_bytes = b'\x00' * 64
        else:
            px, py = self.proof.to_affine()
            proof_bytes = px.to_int().to_bytes(32, 'big') + py.to_int().to_bytes(32, 'big')
        return self.commitment.to_bytes() + point_bytes + value_bytes + proof_bytes


class KZG:
    """
    KZG Polynomial Commitment Scheme.
    
    Provides:
    - commit(polynomial) -> commitment
    - create_proof(polynomial, point) -> proof
    - verify(commitment, point, value, proof) -> bool
    """
    
    def __init__(self, srs: SRS):
        self.srs = srs
    
    def commit(self, poly: Polynomial) -> KZGCommitment:
        """
        Commit to a polynomial.
        
        C = sum(coeffs[i] * g1_powers[i])
        """
        if poly.degree() > self.srs.max_degree:
            raise ValueError(f"Polynomial degree {poly.degree()} exceeds SRS max {self.srs.max_degree}")
        
        result = G1Point.identity()
        for i, coeff in enumerate(poly.coeffs):
            if coeff != Fr.zero():
                result = result + (self.srs.g1_powers[i] * coeff)
        
        return KZGCommitment(result)
    
    def create_proof(self, poly: Polynomial, point: Fr) -> KZGProof:
        """
        Create an opening proof for polynomial at point.
        
        Proves that poly(point) = value by computing:
        q(x) = (poly(x) - value) / (x - point)
        proof = commit(q)
        """
        value = poly.evaluate(point)
        
        # Compute quotient polynomial
        # q(x) = (p(x) - p(point)) / (x - point)
        shifted = Polynomial([poly.coeffs[0] - value] + poly.coeffs[1:])
        quotient, remainder = shifted.divide_by_linear(point)
        
        # The remainder should be zero (or very close due to field arithmetic)
        proof_point = self.commit(quotient).point
        
        return KZGProof(
            commitment=self.commit(poly),
            point=point,
            value=value,
            proof=proof_point
        )
    
    def verify(self, proof: KZGProof) -> bool:
        """
        Verify a KZG opening proof using pairing check.
        
        Uses the production-grade verification with BN254 pairings.
        
        Checks: e(π, [τ]₂) = e(z·π + C - v·G₁, G₂)
        """
        from crypto.bn254.pairing_pyecc import verify_kzg_opening
        
        return verify_kzg_opening(
            commitment=proof.commitment.point,
            point=proof.point,
            value=proof.value,
            proof=proof.proof,
            tau_g2=self.srs.g2_powers[1]
        )


# =============================================================================
# Circuit Representation
# =============================================================================

class GateType(Enum):
    """Types of gates in a PLONK circuit."""
    ADD = "add"
    MUL = "mul"
    CONST = "const"
    PUBLIC_INPUT = "public_input"
    CUSTOM = "custom"


@dataclass
class Wire:
    """A wire in the circuit (variable reference)."""
    index: int
    name: Optional[str] = None


@dataclass
class Gate:
    """A gate in the circuit."""
    gate_type: GateType
    left: Optional[Wire] = None
    right: Optional[Wire] = None
    output: Optional[Wire] = None
    constant: Optional[Fr] = None
    selector: Dict[str, Fr] = field(default_factory=dict)


@dataclass
class Circuit:
    """
    A PLONK arithmetic circuit.
    
    Represents a computation as a series of gates operating on wires.
    """
    gates: List[Gate] = field(default_factory=list)
    wires: Dict[int, Wire] = field(default_factory=dict)
    public_inputs: List[int] = field(default_factory=list)
    
    _next_wire: int = field(default=0, repr=False)
    
    def new_wire(self, name: Optional[str] = None) -> Wire:
        """Allocate a new wire."""
        wire = Wire(self._next_wire, name)
        self.wires[self._next_wire] = wire
        self._next_wire += 1
        return wire
    
    def add_public_input(self, name: Optional[str] = None) -> Wire:
        """Add a public input wire."""
        wire = self.new_wire(name)
        self.public_inputs.append(wire.index)
        self.gates.append(Gate(GateType.PUBLIC_INPUT, output=wire))
        return wire
    
    def add_constant(self, value: Fr, name: Optional[str] = None) -> Wire:
        """Add a constant wire."""
        wire = self.new_wire(name)
        self.gates.append(Gate(GateType.CONST, output=wire, constant=value))
        return wire
    
    def add(self, left: Wire, right: Wire, name: Optional[str] = None) -> Wire:
        """Add two wires."""
        output = self.new_wire(name)
        self.gates.append(Gate(GateType.ADD, left=left, right=right, output=output))
        return output
    
    def mul(self, left: Wire, right: Wire, name: Optional[str] = None) -> Wire:
        """Multiply two wires."""
        output = self.new_wire(name)
        self.gates.append(Gate(GateType.MUL, left=left, right=right, output=output))
        return output
    
    def num_constraints(self) -> int:
        """Return the number of constraints (gates)."""
        return len(self.gates)
    
    def num_wires(self) -> int:
        """Return the number of wires."""
        return self._next_wire


@dataclass
class Witness:
    """
    A witness (assignment) for a circuit.
    
    Maps wire indices to field element values.
    """
    values: Dict[int, Fr] = field(default_factory=dict)
    
    def set(self, wire: Wire, value: Fr):
        """Set the value of a wire."""
        self.values[wire.index] = value
    
    def get(self, wire: Wire) -> Fr:
        """Get the value of a wire."""
        return self.values.get(wire.index, Fr.zero())
    
    def to_list(self, num_wires: int) -> List[Fr]:
        """Convert to a list of values."""
        return [self.values.get(i, Fr.zero()) for i in range(num_wires)]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Field
    'Field', 'Fr', 'Fp',
    # Polynomial
    'Polynomial', 'FFT',
    # KZG
    'SRS', 'KZGCommitment', 'KZGProof', 'KZG',
    # Circuit
    'GateType', 'Wire', 'Gate', 'Circuit', 'Witness',
    # Curve points
    'G1Point', 'G2Point',
]
