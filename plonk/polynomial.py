"""
Polynomial Arithmetic for PLONK

This module provides polynomial operations over finite fields, which are
fundamental to the PLONK proof system.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.fr_adapter import Fr


class Polynomial:
    """A polynomial over the scalar field Fr."""

    def __init__(self, coeffs: List[Fr]):
        while len(coeffs) > 1 and coeffs[-1].is_zero():
            coeffs.pop()
        self.coeffs = coeffs if coeffs else [Fr.zero()]

    @classmethod
    def zero(cls) -> Polynomial:
        return cls([Fr.zero()])

    @classmethod
    def one(cls) -> Polynomial:
        return cls([Fr.one()])

    @classmethod
    def from_ints(cls, coeffs: List[int]) -> Polynomial:
        return cls([Fr(c) for c in coeffs])

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def is_zero(self) -> bool:
        return len(self.coeffs) == 1 and self.coeffs[0].is_zero()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        if len(self.coeffs) != len(other.coeffs):
            return False
        return all(a == b for a, b in zip(self.coeffs, other.coeffs))

    def __neg__(self) -> Polynomial:
        return Polynomial([-c for c in self.coeffs])

    def __add__(self, other: Polynomial) -> Polynomial:
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []
        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else Fr.zero()
            b = other.coeffs[i] if i < len(other.coeffs) else Fr.zero()
            result.append(a + b)
        return Polynomial(result)

    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + (-other)

    def __mul__(self, other: Polynomial) -> Polynomial:
        if self.is_zero() or other.is_zero():
            return Polynomial.zero()
        result_len = len(self.coeffs) + len(other.coeffs) - 1
        result = [Fr.zero()] * result_len
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = result[i + j] + a * b
        return Polynomial(result)

    def scale(self, scalar: Fr) -> Polynomial:
        return Polynomial([c * scalar for c in self.coeffs])

    def evaluate(self, x: Fr) -> Fr:
        if self.is_zero():
            return Fr.zero()
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]
        return result

    def divide_by_linear(self, root: Fr) -> Tuple[Polynomial, Fr]:
        n = len(self.coeffs)
        if n == 1:
            return Polynomial.zero(), self.coeffs[0]
        quotient = [Fr.zero()] * (n - 1)
        quotient[-1] = self.coeffs[-1]
        for i in range(n - 2, 0, -1):
            quotient[i - 1] = self.coeffs[i] + quotient[i] * root
        remainder = self.coeffs[0] + quotient[0] * root
        return Polynomial(quotient), remainder


class FFT:
    """Fast Fourier Transform over finite fields."""

    def __init__(self, n: int):
        if n & (n - 1) != 0:
            raise ValueError(f"FFT size must be a power of 2, got {n}")
        self.n = n
        self.omega = self._find_root_of_unity(n)
        self.omega_inv = self.omega.inverse()

    def _find_root_of_unity(self, n: int) -> Fr:
        g = Fr(5)
        r_minus_1 = Fr.MODULUS - 1
        exp = r_minus_1 // n
        omega = g ** exp
        return omega

    def fft(self, coeffs: List[Fr]) -> List[Fr]:
        a = coeffs + [Fr.zero()] * (self.n - len(coeffs))
        return self._fft_recursive(a, self.omega)

    def ifft(self, evals: List[Fr]) -> List[Fr]:
        result = self._fft_recursive(evals, self.omega_inv)
        n_inv = Fr(self.n).inverse()
        return [x * n_inv for x in result]

    def _fft_recursive(self, a: List[Fr], omega: Fr) -> List[Fr]:
        n = len(a)
        if n == 1:
            return a
        a_even = a[0::2]
        a_odd = a[1::2]
        omega_sq = omega.square()
        y_even = self._fft_recursive(a_even, omega_sq)
        y_odd = self._fft_recursive(a_odd, omega_sq)
        y = [Fr.zero()] * n
        omega_k = Fr.one()
        for k in range(n // 2):
            t = omega_k * y_odd[k]
            y[k] = y_even[k] + t
            y[k + n // 2] = y_even[k] - t
            omega_k = omega_k * omega
        return y


def lagrange_interpolation(points: List[Tuple[Fr, Fr]]) -> Polynomial:
    n = len(points)
    if n == 0:
        return Polynomial.zero()
    result = Polynomial.zero()
    for i in range(n):
        xi, yi = points[i]
        numerator = Polynomial.one()
        denominator = Fr.one()
        for j in range(n):
            if i != j:
                xj, _ = points[j]
                numerator = numerator * Polynomial([-xj, Fr.one()])
                denominator = denominator * (xi - xj)
        basis = numerator.scale(yi / denominator)
        result = result + basis
    return result
