//! BN254 Scalar Field (Fr) Arithmetic — Montgomery Form
//!
//! This module implements Fr (the scalar field of BN254) using
//! Montgomery multiplication for efficient modular arithmetic.
//!
//! Field modulus r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//!
//! Montgomery parameters:
//!   R = 2^256 mod r
//!   R^2 mod r (for Montgomery encoding)
//!   r' = -r^(-1) mod 2^64 (for REDC)

use std::fmt;
use std::ops::{Add, Sub, Mul, Neg};

/// The BN254 scalar field modulus r, stored as 4 × u64 limbs (little-endian).
///
/// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const MODULUS: [u64; 4] = [
    0x43E1F593F0000001,
    0x2833E84879B97091,
    0xB85045B68181585D,
    0x30644E72E131A029,
];

/// R = 2^256 mod r (Montgomery form of 1)
const R: [u64; 4] = [
    0xAC96341C4FFFFFFB,
    0x36FC76959F60CD29,
    0x666EA36F7879462E,
    0x0E0A77C19A07DF2F,
];

/// R^2 mod r (used to convert to Montgomery form)
const R_SQUARED: [u64; 4] = [
    0x1BB8E645AE216DA7,
    0x53FE3AB1E35C59E3,
    0x8C49833D53BB8085,
    0x0216D0B17F4E44A5,
];

/// r' = -r^(-1) mod 2^64
const INV: u64 = 0xC2E1F593EFFFFFFF;

/// A field element in BN254 Fr, stored in Montgomery form.
///
/// Internal representation: `a * R mod r` where R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fr {
    pub limbs: [u64; 4],
}

impl Fr {
    /// The zero element.
    pub const ZERO: Fr = Fr { limbs: [0, 0, 0, 0] };

    /// The multiplicative identity (1 in Montgomery form = R mod r).
    pub const ONE: Fr = Fr { limbs: R };

    /// Create a field element from a u64 value.
    pub fn from_u64(val: u64) -> Self {
        let mut a = Fr { limbs: [val, 0, 0, 0] };
        a = a.mont_mul(&Fr { limbs: R_SQUARED });
        a
    }

    /// Create a field element from a raw integer (big-endian bytes or limbs).
    /// Input: 4 × u64 limbs in little-endian order, NOT in Montgomery form.
    pub fn from_raw(limbs: [u64; 4]) -> Self {
        let a = Fr { limbs };
        a.mont_mul(&Fr { limbs: R_SQUARED })
    }

    /// Convert back from Montgomery form to standard form.
    pub fn to_raw(&self) -> [u64; 4] {
        let one = Fr { limbs: [1, 0, 0, 0] };
        self.mont_mul(&one).limbs
    }

    /// Return the raw Montgomery-encoded limbs.
    /// Used for the Python `.value` property (internal representation).
    pub fn to_mont_raw(&self) -> [u64; 4] {
        self.limbs
    }

    /// Convert to a Python-compatible big integer (as a string for large values).
    pub fn to_int_string(&self) -> String {
        let raw = self.to_raw();
        // Convert 4 × u64 little-endian limbs to a big integer
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            let b = raw[i].to_le_bytes();
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&b);
        }
        // Convert little-endian bytes to big integer string
        // Use u128 arithmetic to build the full 256-bit value
        let lo = raw[0] as u128 | ((raw[1] as u128) << 64);
        let hi = raw[2] as u128 | ((raw[3] as u128) << 64);
        if hi == 0 {
            format!("{lo}")
        } else {
            // For full 256-bit values, format via big-endian hex then parse
            let mut hex = String::new();
            for &limb in raw.iter().rev() {
                hex.push_str(&format!("{limb:016x}"));
            }
            // Strip leading zeros
            let hex = hex.trim_start_matches('0');
            if hex.is_empty() {
                "0".to_string()
            } else {
                format!("0x{hex}")
            }
        }
    }

    /// Check if this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs == [0, 0, 0, 0]
    }

    /// Montgomery multiplication: compute `a * b * R^(-1) mod r`.
    ///
    /// Uses the CIOS (Coarsely Integrated Operand Scanning) algorithm.
    /// Per Koç et al., the intermediate value t[4]+carry after Step 3 can
    /// reach 2*(2^64-1), requiring a wider accumulator.
    #[inline]
    #[allow(unused_assignments)]
    pub fn mont_mul(&self, other: &Fr) -> Fr {
        let a = &self.limbs;
        let b = &other.limbs;
        let mut t = [0u64; 5]; // Accumulator (4 limbs + carry)
        let mut t5: u64 = 0;   // Extra carry bit from t[4] overflow (may appear unused on last iter)

        for i in 0..4 {
            // Step 1: t += a[i] * b
            let mut carry: u64 = 0;
            for j in 0..4 {
                let (hi, lo) = mul_add_carry(a[i], b[j], t[j], carry);
                t[j] = lo;
                carry = hi;
            }
            // t[4] was zeroed by previous shift; safe to overwrite
            let wide = t[4] as u128 + carry as u128 + t5 as u128;
            t[4] = wide as u64;
            t5 = (wide >> 64) as u64;

            // Step 2: Reduction — m = t[0] * r' mod 2^64
            let m = t[0].wrapping_mul(INV);

            // Step 3: t += m * MODULUS
            let mut carry: u64 = 0;
            for j in 0..4 {
                let (hi, lo) = mul_add_carry(m, MODULUS[j], t[j], carry);
                t[j] = lo;
                carry = hi;
            }
            // Safe wide addition: t[4]+carry can reach 2^65-2 (Koç bound)
            let wide = t[4] as u128 + carry as u128;
            t[4] = wide as u64;
            t5 = (wide >> 64) as u64;

            // Step 4: Shift right by one limb (divide by 2^64)
            t[0] = t[1];
            t[1] = t[2];
            t[2] = t[3];
            t[3] = t[4];
            t[4] = t5;
            t5 = 0;
        }

        // Final reduction: if t >= MODULUS, subtract MODULUS
        // After all 4 iterations, t[4] and t5 should be 0 for valid inputs,
        // but we check t[4] to handle edge cases.
        let mut result = [t[0], t[1], t[2], t[3]];
        if t[4] > 0 || !lt(&result, &MODULUS) {
            sub_mod_inner(&mut result, &MODULUS);
        }

        Fr { limbs: result }
    }

    /// Montgomery squaring (specialized for a * a).
    #[inline]
    pub fn square(&self) -> Fr {
        self.mont_mul(self)
    }

    /// Field addition: (a + b) mod r.
    #[inline]
    pub fn add(&self, other: &Fr) -> Fr {
        let mut result = [0u64; 4];
        let mut carry = 0u64;

        for i in 0..4 {
            let (s1, c1) = self.limbs[i].overflowing_add(other.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }

        // Reduce if result >= MODULUS
        if carry > 0 || !lt(&result, &MODULUS) {
            sub_mod_inner(&mut result, &MODULUS);
        }

        Fr { limbs: result }
    }

    /// Field subtraction: (a - b) mod r.
    #[inline]
    pub fn sub(&self, other: &Fr) -> Fr {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;

        for i in 0..4 {
            let (s1, b1) = self.limbs[i].overflowing_sub(other.limbs[i]);
            let (s2, b2) = s1.overflowing_sub(borrow);
            result[i] = s2;
            borrow = (b1 as u64) + (b2 as u64);
        }

        // If underflow, add MODULUS
        if borrow > 0 {
            let mut carry = 0u64;
            for i in 0..4 {
                let (s1, c1) = result[i].overflowing_add(MODULUS[i]);
                let (s2, c2) = s1.overflowing_add(carry);
                result[i] = s2;
                carry = (c1 as u64) + (c2 as u64);
            }
        }

        Fr { limbs: result }
    }

    /// Field negation: -a mod r.
    #[inline]
    pub fn neg(&self) -> Fr {
        if self.is_zero() {
            Fr::ZERO
        } else {
            let mut result = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (s1, b1) = MODULUS[i].overflowing_sub(self.limbs[i]);
                let (s2, b2) = s1.overflowing_sub(borrow);
                result[i] = s2;
                borrow = (b1 as u64) + (b2 as u64);
            }
            Fr { limbs: result }
        }
    }

    /// Modular exponentiation via square-and-multiply.
    pub fn pow(&self, exp: [u64; 4]) -> Fr {
        let mut result = Fr::ONE;
        let mut base = *self;

        for i in 0..4 {
            let mut e = exp[i];
            for _ in 0..64 {
                if e & 1 == 1 {
                    result = result.mont_mul(&base);
                }
                base = base.square();
                e >>= 1;
            }
        }
        result
    }

    /// Modular inverse via Fermat's little theorem: a^(r-2) mod r.
    pub fn inverse(&self) -> Option<Fr> {
        if self.is_zero() {
            return None;
        }
        // r - 2
        let mut exp = MODULUS;
        // Subtract 2 from the modulus (little-endian)
        let (s, borrow) = exp[0].overflowing_sub(2);
        exp[0] = s;
        if borrow {
            for i in 1..4 {
                let (s2, b2) = exp[i].overflowing_sub(1);
                exp[i] = s2;
                if !b2 { break; }
            }
        }
        Some(self.pow(exp))
    }

    /// Batch inversion using Montgomery's trick.
    /// Returns None if any element is zero.
    pub fn batch_inverse(elements: &[Fr]) -> Option<Vec<Fr>> {
        let n = elements.len();
        if n == 0 {
            return Some(vec![]);
        }

        // Compute prefix products
        let mut products = Vec::with_capacity(n);
        let mut acc = Fr::ONE;
        for elem in elements {
            if elem.is_zero() {
                return None;
            }
            acc = acc.mont_mul(elem);
            products.push(acc);
        }

        // Invert the accumulated product once
        let mut inv = acc.inverse()?;

        // Walk backwards, recovering individual inverses
        let mut result = vec![Fr::ZERO; n];
        for i in (1..n).rev() {
            result[i] = products[i - 1].mont_mul(&inv);
            inv = inv.mont_mul(&elements[i]);
        }
        result[0] = inv;

        Some(result)
    }
}

// --- Operator overloads ---

impl Add for Fr {
    type Output = Fr;
    #[inline]
    fn add(self, rhs: Fr) -> Fr {
        Fr::add(&self, &rhs)
    }
}

impl Sub for Fr {
    type Output = Fr;
    #[inline]
    fn sub(self, rhs: Fr) -> Fr {
        Fr::sub(&self, &rhs)
    }
}

impl Mul for Fr {
    type Output = Fr;
    #[inline]
    fn mul(self, rhs: Fr) -> Fr {
        self.mont_mul(&rhs)
    }
}

impl Neg for Fr {
    type Output = Fr;
    #[inline]
    fn neg(self) -> Fr {
        Fr::neg(&self)
    }
}

impl fmt::Debug for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raw = self.to_raw();
        write!(f, "Fr(0x{:016x}{:016x}{:016x}{:016x})", raw[3], raw[2], raw[1], raw[0])
    }
}

impl fmt::Display for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_int_string())
    }
}

// --- Helper functions ---

/// Multiply-and-add with carry: returns (hi, lo) of a*b + c + carry.
#[inline(always)]
fn mul_add_carry(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let wide = a as u128 * b as u128 + c as u128 + carry as u128;
    (wide.wrapping_shr(64) as u64, wide as u64)
}

/// Compare two 4-limb numbers (little-endian): returns true if a < b.
#[inline]
fn lt(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

/// Subtract b from a in-place (assumes a >= b).
#[inline]
fn sub_mod_inner(a: &mut [u64; 4], b: &[u64; 4]) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (s1, b1) = a[i].overflowing_sub(b[i]);
        let (s2, b2) = s1.overflowing_sub(borrow);
        a[i] = s2;
        borrow = (b1 as u64) + (b2 as u64);
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_and_one() {
        assert!(Fr::ZERO.is_zero());
        assert!(!Fr::ONE.is_zero());
    }

    #[test]
    fn test_from_u64() {
        let a = Fr::from_u64(42);
        let raw = a.to_raw();
        assert_eq!(raw[0], 42);
        assert_eq!(raw[1], 0);
        assert_eq!(raw[2], 0);
        assert_eq!(raw[3], 0);
    }

    #[test]
    fn test_add_sub() {
        let a = Fr::from_u64(100);
        let b = Fr::from_u64(200);
        let c = a + b;
        let raw = c.to_raw();
        assert_eq!(raw[0], 300);

        let d = c - b;
        assert_eq!(d, a);
    }

    #[test]
    fn test_mul() {
        let a = Fr::from_u64(7);
        let b = Fr::from_u64(11);
        let c = a * b;
        let raw = c.to_raw();
        assert_eq!(raw[0], 77);
    }

    #[test]
    fn test_square() {
        let a = Fr::from_u64(13);
        let s = a.square();
        let raw = s.to_raw();
        assert_eq!(raw[0], 169);
    }

    #[test]
    fn test_negation() {
        let a = Fr::from_u64(42);
        let neg_a = -a;
        let sum = a + neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_inverse() {
        let a = Fr::from_u64(7);
        let inv = a.inverse().unwrap();
        let product = a * inv;
        assert_eq!(product, Fr::ONE);
    }

    #[test]
    fn test_zero_inverse() {
        assert!(Fr::ZERO.inverse().is_none());
    }

    #[test]
    fn test_batch_inverse() {
        let elements: Vec<Fr> = (1..=10).map(Fr::from_u64).collect();
        let inverses = Fr::batch_inverse(&elements).unwrap();
        for (elem, inv) in elements.iter().zip(inverses.iter()) {
            assert_eq!(*elem * *inv, Fr::ONE);
        }
    }

    #[test]
    fn test_modular_reduction() {
        // Test that r ≡ 0 mod r
        let r = Fr::from_raw(MODULUS);
        assert!(r.is_zero());
    }

    #[test]
    fn test_sub_underflow() {
        let a = Fr::from_u64(1);
        let b = Fr::from_u64(2);
        let c = a - b; // Should be r - 1
        let d = c + b;
        assert_eq!(d, a);
    }
}
