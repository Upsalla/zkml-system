//! Polynomial Operations & Number Theoretic Transform (NTT/FFT)
//!
//! Provides:
//! - Polynomial type over Fr with standard operations
//! - Radix-2 NTT (Number Theoretic Transform) for O(n log n) multiplication
//! - Inverse NTT (INTT)
//! - Polynomial evaluation, interpolation (Lagrange), division

use crate::field::Fr;

// =============================================================================
// Polynomial
// =============================================================================

/// A polynomial over Fr, stored as coefficients [a₀, a₁, ..., aₙ].
/// p(x) = a₀ + a₁·x + a₂·x² + ... + aₙ·xⁿ
#[derive(Clone, Debug)]
pub struct Polynomial {
    pub coeffs: Vec<Fr>,
}

impl Polynomial {
    /// Zero polynomial.
    pub fn zero() -> Self {
        Polynomial { coeffs: vec![] }
    }

    /// Constant polynomial.
    pub fn constant(c: Fr) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Polynomial { coeffs: vec![c] }
        }
    }

    /// Create from coefficients.
    pub fn from_coeffs(coeffs: Vec<Fr>) -> Self {
        let mut p = Polynomial { coeffs };
        p.trim();
        p
    }

    /// Degree of the polynomial (-1 for zero polynomial → returns None).
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() { None } else { Some(self.coeffs.len() - 1) }
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Remove trailing zero coefficients.
    fn trim(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Evaluate p(x) using Horner's method.
    pub fn evaluate(&self, x: &Fr) -> Fr {
        if self.coeffs.is_empty() {
            return Fr::ZERO;
        }
        let mut result = *self.coeffs.last().unwrap();
        for i in (0..self.coeffs.len() - 1).rev() {
            result = result.mont_mul(x).add(&self.coeffs[i]);
        }
        result
    }

    /// Polynomial addition.
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeffs.get(i).copied().unwrap_or(Fr::ZERO);
            let b = other.coeffs.get(i).copied().unwrap_or(Fr::ZERO);
            coeffs.push(a.add(&b));
        }
        Polynomial::from_coeffs(coeffs)
    }

    /// Polynomial subtraction.
    pub fn sub(&self, other: &Polynomial) -> Polynomial {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeffs.get(i).copied().unwrap_or(Fr::ZERO);
            let b = other.coeffs.get(i).copied().unwrap_or(Fr::ZERO);
            coeffs.push(a.sub(&b));
        }
        Polynomial::from_coeffs(coeffs)
    }

    /// Polynomial multiplication via NTT (O(n log n)).
    pub fn mul(&self, other: &Polynomial) -> Polynomial {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }

        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let n = result_len.next_power_of_two();

        let mut a = self.coeffs.clone();
        a.resize(n, Fr::ZERO);
        let mut b = other.coeffs.clone();
        b.resize(n, Fr::ZERO);

        ntt(&mut a);
        ntt(&mut b);

        // Pointwise multiplication
        for i in 0..n {
            a[i] = a[i].mont_mul(&b[i]);
        }

        intt(&mut a);
        a.truncate(result_len);
        Polynomial::from_coeffs(a)
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: &Fr) -> Polynomial {
        let coeffs: Vec<Fr> = self.coeffs.iter()
            .map(|c| c.mont_mul(scalar))
            .collect();
        Polynomial::from_coeffs(coeffs)
    }

    /// Polynomial division with remainder: self = quotient * divisor + remainder.
    /// Returns (quotient, remainder).
    pub fn div_rem(&self, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        // L4: Defensive trim in case coeffs were modified directly
        let mut dividend = self.clone();
        dividend.trim();
        let mut div = divisor.clone();
        div.trim();
        assert!(!div.is_zero(), "Division by zero polynomial");

        if dividend.is_zero() || dividend.coeffs.len() < div.coeffs.len() {
            return (Polynomial::zero(), dividend);
        }

        let mut remainder = dividend.coeffs.clone();
        let divisor_lead_inv = div.coeffs.last().unwrap().inverse().unwrap();
        let quot_len = remainder.len() - div.coeffs.len() + 1;
        let mut quotient = vec![Fr::ZERO; quot_len];

        for i in (0..quot_len).rev() {
            let coeff = remainder[i + div.coeffs.len() - 1].mont_mul(&divisor_lead_inv);
            quotient[i] = coeff;
            for j in 0..div.coeffs.len() {
                let sub = coeff.mont_mul(&div.coeffs[j]);
                remainder[i + j] = remainder[i + j].sub(&sub);
            }
        }

        (Polynomial::from_coeffs(quotient), Polynomial::from_coeffs(remainder))
    }

    /// Lagrange interpolation: given points (xᵢ, yᵢ), return the unique
    /// polynomial of degree ≤ n-1 through all points.
    pub fn lagrange_interpolate(xs: &[Fr], ys: &[Fr]) -> Polynomial {
        assert_eq!(xs.len(), ys.len());
        let n = xs.len();

        let mut result = Polynomial::zero();

        for i in 0..n {
            // Compute Lagrange basis polynomial Lᵢ(x)
            let mut basis = Polynomial::constant(Fr::ONE);
            let mut denom = Fr::ONE;

            for j in 0..n {
                if i == j { continue; }
                // basis *= (x - xⱼ)
                let linear = Polynomial::from_coeffs(vec![xs[j].neg(), Fr::ONE]);
                basis = basis.mul(&linear);
                // denom *= (xᵢ - xⱼ)
                denom = denom.mont_mul(&xs[i].sub(&xs[j]));
            }

            let denom_inv = denom.inverse().unwrap();
            let term = basis.scale(&ys[i].mont_mul(&denom_inv));
            result = result.add(&term);
        }

        result
    }

    /// Evaluate polynomial at the n-th roots of unity using NTT.
    pub fn evaluate_domain(&self, n: usize) -> Vec<Fr> {
        assert!(n.is_power_of_two(), "Domain size must be power of 2");
        assert!(
            n >= self.coeffs.len(),
            "Domain size {n} smaller than polynomial degree+1 {}; would silently truncate",
            self.coeffs.len()
        );
        let mut coeffs = self.coeffs.clone();
        coeffs.resize(n, Fr::ZERO);
        ntt(&mut coeffs);
        coeffs
    }
}


// =============================================================================
// Number Theoretic Transform (NTT)
// =============================================================================

/// Find a primitive n-th root of unity in Fr.
///
/// For BN254 Fr, the 2-adicity is 28 (r-1 = 2^28 · m where m is odd).
/// The maximum NTT size is 2^28.
///
/// Computes ω = 5^((r-1)/2^28) at first call, then caches.
fn get_root_of_unity(n: usize) -> Fr {
    use std::sync::OnceLock;
    use crate::field::MODULUS;

    static ROOT_2_28: OnceLock<Fr> = OnceLock::new();

    assert!(n.is_power_of_two(), "NTT size must be power of 2");
    let log_n = n.trailing_zeros();
    assert!(log_n <= 28, "NTT size exceeds 2^28 (2-adicity of BN254 Fr)");

    let root_2_28 = ROOT_2_28.get_or_init(|| {
        // Compute (r-1) >> 28 as 4×u64 limbs
        let r_minus_1 = [
            MODULUS[0].wrapping_sub(1), // r is odd, so no borrow
            MODULUS[1],
            MODULUS[2],
            MODULUS[3],
        ];
        let shift = 28u32;
        let exp = [
            (r_minus_1[0] >> shift) | (r_minus_1[1] << (64 - shift)),
            (r_minus_1[1] >> shift) | (r_minus_1[2] << (64 - shift)),
            (r_minus_1[2] >> shift) | (r_minus_1[3] << (64 - shift)),
            r_minus_1[3] >> shift,
        ];
        // ω = 5^exp mod r
        Fr::from_u64(5).pow(exp)
    });

    // Reduce to size n: ω^(2^(28-log_n))
    let mut root = *root_2_28;
    for _ in 0..(28 - log_n) {
        root = root.square();
    }
    root
}

/// In-place radix-2 NTT (Cooley-Tukey, DIT).
pub fn ntt(a: &mut [Fr]) {
    let n = a.len();
    assert!(n.is_power_of_two());
    if n == 1 { return; }

    // Bit-reversal permutation
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let root = get_root_of_unity(len);

        for start in (0..n).step_by(len) {
            let mut w = Fr::ONE;
            for j in 0..half {
                let u = a[start + j];
                let v = a[start + j + half].mont_mul(&w);
                a[start + j] = u.add(&v);
                a[start + j + half] = u.sub(&v);
                w = w.mont_mul(&root);
            }
        }
        len <<= 1;
    }
}

/// In-place inverse NTT.
///
/// Uses the identity: INTT(a) = (1/n) · reverse(NTT(a)), where reverse
/// swaps elements a[k] and a[n-k] for k=1..n-1.
/// This avoids needing explicit inverse roots of unity.
pub fn intt(a: &mut [Fr]) {
    let n = a.len();
    assert!(n.is_power_of_two());
    if n == 1 { return; }

    // Step 1: Forward NTT (same roots as forward transform)
    ntt(a);

    // Step 2: Reverse elements a[1..n-1] to get conjugate DFT
    a[1..].reverse();

    // Scale by 1/n
    let n_inv = Fr::from_u64(n as u64).inverse().unwrap();
    for val in a.iter_mut() {
        *val = val.mont_mul(&n_inv);
    }
}

/// Bit-reverse an index for NTT butterfly.
fn bit_reverse(x: usize, log_n: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - log_n)
}


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_evaluate() {
        // p(x) = 1 + 2x + 3x²
        let p = Polynomial::from_coeffs(vec![
            Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3),
        ]);
        // p(0) = 1
        assert_eq!(p.evaluate(&Fr::ZERO), Fr::from_u64(1));
        // p(1) = 1 + 2 + 3 = 6
        assert_eq!(p.evaluate(&Fr::ONE), Fr::from_u64(6));
        // p(2) = 1 + 4 + 12 = 17
        assert_eq!(p.evaluate(&Fr::from_u64(2)), Fr::from_u64(17));
    }

    #[test]
    fn test_polynomial_add() {
        let a = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let b = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(4), Fr::from_u64(5)]);
        let c = a.add(&b);
        // [4, 6, 5]
        assert_eq!(c.coeffs.len(), 3);
        assert_eq!(c.evaluate(&Fr::ZERO), Fr::from_u64(4));
    }

    #[test]
    fn test_polynomial_mul() {
        // (1 + x) * (1 + x) = 1 + 2x + x²
        let a = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(1)]);
        let c = a.mul(&a);
        assert_eq!(c.coeffs.len(), 3);
        assert_eq!(c.evaluate(&Fr::from_u64(2)), Fr::from_u64(9)); // (1+2)² = 9
    }

    #[test]
    fn test_polynomial_div() {
        // (x² - 1) / (x - 1) = (x + 1)
        let p = Polynomial::from_coeffs(vec![
            Fr::from_u64(1).neg(), Fr::ZERO, Fr::from_u64(1) // -1 + x²
        ]);
        let d = Polynomial::from_coeffs(vec![
            Fr::from_u64(1).neg(), Fr::from_u64(1) // -1 + x
        ]);
        let (q, r) = p.div_rem(&d);
        assert!(r.is_zero());
        // q should be 1 + x
        assert_eq!(q.evaluate(&Fr::from_u64(3)), Fr::from_u64(4)); // 1+3 = 4
    }

    #[test]
    fn test_ntt_intt_roundtrip() {
        let n = 8;
        let original: Vec<Fr> = (1..=n).map(|i| Fr::from_u64(i as u64)).collect();
        let mut data = original.clone();

        ntt(&mut data);
        intt(&mut data);

        assert_eq!(data.len(), original.len());
        for (a, b) in data.iter().zip(original.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_ntt_mul_consistency() {
        // Compare NTT-based mul with schoolbook
        let a = Polynomial::from_coeffs(vec![
            Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)
        ]);
        let b = Polynomial::from_coeffs(vec![
            Fr::from_u64(4), Fr::from_u64(5)
        ]);
        let c = a.mul(&b);

        // Schoolbook: (1 + 2x + 3x²)(4 + 5x) = 4 + 13x + 22x² + 15x³
        assert_eq!(c.evaluate(&Fr::ZERO), Fr::from_u64(4));
        assert_eq!(c.evaluate(&Fr::ONE), Fr::from_u64(54)); // 4+13+22+15 = 54
    }

    #[test]
    fn test_lagrange_interpolation() {
        // Interpolate through (0,1), (1,3), (2,7) → p(x) = 1 + x + x²
        let xs = vec![Fr::ZERO, Fr::ONE, Fr::from_u64(2)];
        let ys = vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(7)];
        let p = Polynomial::lagrange_interpolate(&xs, &ys);
        assert_eq!(p.evaluate(&Fr::from_u64(3)), Fr::from_u64(13)); // 1+3+9 = 13
    }

    #[test]
    fn test_root_of_unity() {
        // ω^n should equal 1
        let n = 16;
        let omega = get_root_of_unity(n);
        let omega_n = omega.pow([n as u64, 0, 0, 0]);
        assert_eq!(omega_n, Fr::ONE);

        // ω^(n/2) should not equal 1 (primitive root)
        let omega_half = omega.pow([(n / 2) as u64, 0, 0, 0]);
        assert_ne!(omega_half, Fr::ONE);
    }
}
