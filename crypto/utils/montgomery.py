"""
Montgomery Modular Arithmetic

This module provides functions for Montgomery reduction and multiplication,
which are essential for efficient modular arithmetic in cryptographic applications.

Montgomery multiplication replaces division by the modulus with cheaper operations
(multiplication and bit shifts), making it significantly faster for repeated
modular multiplications.

Reference:
    Montgomery, P. L. (1985). Modular Multiplication Without Trial Division.
"""

from typing import Tuple


def montgomery_reduce(t: int, modulus: int, n_prime: int, num_limbs: int = 4) -> int:
    """
    Perform Montgomery reduction on a value T.

    Given T < R * N, computes T * R^(-1) mod N.

    This implementation uses the CIOS (Coarsely Integrated Operand Scanning)
    method for efficiency.

    Args:
        t: The value to reduce (must be < R * N where R = 2^(64*num_limbs)).
        modulus: The modulus N.
        n_prime: The precomputed value -N^(-1) mod 2^64.
        num_limbs: Number of 64-bit limbs in the modulus (default 4 for 256-bit).

    Returns:
        The Montgomery-reduced value T * R^(-1) mod N.
    """
    mask = (1 << 64) - 1
    result = t

    for _ in range(num_limbs):
        # m = (result mod 2^64) * n_prime mod 2^64
        m = ((result & mask) * n_prime) & mask
        # result = (result + m * modulus) / 2^64
        result = (result + m * modulus) >> 64

    # Final reduction if result >= modulus
    if result >= modulus:
        result -= modulus

    return result


def to_montgomery(a: int, r_squared: int, modulus: int, n_prime: int) -> int:
    """
    Convert a standard integer to Montgomery representation.

    The Montgomery representation of `a` is `a * R mod N`.
    This is computed as `montgomery_reduce(a * R^2, N, n_prime)`.

    Args:
        a: The integer to convert (must be in range [0, modulus)).
        r_squared: The precomputed value R^2 mod N.
        modulus: The modulus N.
        n_prime: The precomputed value -N^(-1) mod 2^64.

    Returns:
        The Montgomery representation a * R mod N.
    """
    return montgomery_reduce(a * r_squared, modulus, n_prime)


def from_montgomery(a_mont: int, modulus: int, n_prime: int) -> int:
    """
    Convert a value from Montgomery representation back to standard form.

    Given `a_mont = a * R mod N`, computes `a mod N`.
    This is simply `montgomery_reduce(a_mont, N, n_prime)`.

    Args:
        a_mont: The value in Montgomery representation.
        modulus: The modulus N.
        n_prime: The precomputed value -N^(-1) mod 2^64.

    Returns:
        The standard representation a mod N.
    """
    return montgomery_reduce(a_mont, modulus, n_prime)


def montgomery_mul(a_mont: int, b_mont: int, modulus: int, n_prime: int) -> int:
    """
    Multiply two values in Montgomery representation.

    Given `a_mont = a * R` and `b_mont = b * R`, computes `(a * b) * R mod N`.
    This is `montgomery_reduce(a_mont * b_mont, N, n_prime)`.

    Args:
        a_mont: First operand in Montgomery representation.
        b_mont: Second operand in Montgomery representation.
        modulus: The modulus N.
        n_prime: The precomputed value -N^(-1) mod 2^64.

    Returns:
        The product (a * b) * R mod N in Montgomery representation.
    """
    return montgomery_reduce(a_mont * b_mont, modulus, n_prime)


def montgomery_square(a_mont: int, modulus: int, n_prime: int) -> int:
    """
    Square a value in Montgomery representation.

    This is a specialized version of montgomery_mul for squaring, which can
    be slightly optimized.

    Args:
        a_mont: The operand in Montgomery representation.
        modulus: The modulus N.
        n_prime: The precomputed value -N^(-1) mod 2^64.

    Returns:
        The square a^2 * R mod N in Montgomery representation.
    """
    return montgomery_reduce(a_mont * a_mont, modulus, n_prime)


def montgomery_pow(base_mont: int, exp: int, modulus: int, n_prime: int, r_mod_n: int) -> int:
    """
    Compute modular exponentiation in Montgomery representation.

    Given `base_mont = base * R` and exponent `exp`, computes `base^exp * R mod N`.

    Uses the square-and-multiply algorithm.

    Args:
        base_mont: The base in Montgomery representation.
        exp: The exponent (a non-negative integer).
        modulus: The modulus N.
        n_prime: The precomputed value -N^(-1) mod 2^64.
        r_mod_n: The precomputed value R mod N (Montgomery representation of 1).

    Returns:
        The result base^exp * R mod N in Montgomery representation.
    """
    if exp == 0:
        return r_mod_n  # Return Montgomery representation of 1

    result = r_mod_n  # Start with 1 in Montgomery form
    base = base_mont

    while exp > 0:
        if exp & 1:
            result = montgomery_mul(result, base, modulus, n_prime)
        base = montgomery_square(base, modulus, n_prime)
        exp >>= 1

    return result
