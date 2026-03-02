//! BN254 Elliptic Curve Operations — G1 (Jacobian Coordinates)
//!
//! Curve: y² = x³ + 3  over Fp
//! Generator: G1 = (1, 2)
//!
//! Points are stored in Jacobian coordinates (X, Y, Z) where
//! the affine point is (X/Z², Y/Z³). The point at infinity
//! is represented by Z = 0.

use crate::field::Fr;
use std::fmt;

/// The BN254 base field modulus p (for Fp arithmetic in curve operations).
///
/// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
///
/// NOTE: We reuse the Fr Montgomery infrastructure for Fp by defining
/// Fp-specific constants. For Phase A, we implement Fp as a separate
/// set of constants using the same 4-limb Montgomery machinery.
pub const FP_MODULUS: [u64; 4] = [
    0x3C208C16D87CFD47,
    0x97816A916871CA8D,
    0xB85045B68181585D,
    0x30644E72E131A029,
];

/// R mod p (Montgomery form of 1 in Fp)
const FP_R: [u64; 4] = [
    0xD35D438DC58F0D9D,
    0x0A78EB28F5C70B3D,
    0x666EA36F7879462C,
    0x0E0A77C19A07DF2F,
];

/// R² mod p (for converting to Montgomery form in Fp)
const FP_R_SQUARED: [u64; 4] = [
    0xF32CFC5B538AFA89,
    0xB5E71911D44501FB,
    0x47AB1EFF0A417FF6,
    0x06D89F71CAB8351F,
];

/// -p⁻¹ mod 2⁶⁴
const FP_INV: u64 = 0x87D20782E4866389;

/// Curve parameter b = 3 (y² = x³ + b)
const CURVE_B: u64 = 3;

/// A field element in BN254 Fp (base field), Montgomery form.
/// Separate from Fr (scalar field) — different modulus.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fp {
    pub limbs: [u64; 4],
}

impl Fp {
    pub const ZERO: Fp = Fp { limbs: [0, 0, 0, 0] };
    pub const ONE: Fp = Fp { limbs: FP_R };

    pub fn from_u64(val: u64) -> Self {
        let a = Fp { limbs: [val, 0, 0, 0] };
        a.mont_mul(&Fp { limbs: FP_R_SQUARED })
    }

    pub fn from_raw(limbs: [u64; 4]) -> Self {
        let a = Fp { limbs };
        a.mont_mul(&Fp { limbs: FP_R_SQUARED })
    }

    pub fn to_raw(&self) -> [u64; 4] {
        let one = Fp { limbs: [1, 0, 0, 0] };
        self.mont_mul(&one).limbs
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs == [0, 0, 0, 0]
    }

    /// Montgomery multiplication for Fp using CIOS.
    /// Per Koç et al., t[4]+carry can reach 2^65-2; using u128 wide addition.
    #[inline]
    #[allow(unused_assignments)]
    pub fn mont_mul(&self, other: &Fp) -> Fp {
        let a = &self.limbs;
        let b = &other.limbs;
        let mut t = [0u64; 5];
        let mut t5: u64 = 0;

        for i in 0..4 {
            let mut carry: u64 = 0;
            for j in 0..4 {
                let wide = a[i] as u128 * b[j] as u128 + t[j] as u128 + carry as u128;
                t[j] = wide as u64;
                carry = (wide >> 64) as u64;
            }
            let wide = t[4] as u128 + carry as u128 + t5 as u128;
            t[4] = wide as u64;
            t5 = (wide >> 64) as u64;

            let m = t[0].wrapping_mul(FP_INV);
            let mut carry: u64 = 0;
            for j in 0..4 {
                let wide = m as u128 * FP_MODULUS[j] as u128 + t[j] as u128 + carry as u128;
                t[j] = wide as u64;
                carry = (wide >> 64) as u64;
            }
            let wide = t[4] as u128 + carry as u128;
            t[4] = wide as u64;
            t5 = (wide >> 64) as u64;

            t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = t5; t5 = 0;
        }

        let mut result = [t[0], t[1], t[2], t[3]];
        if t[4] > 0 || !fp_lt(&result, &FP_MODULUS) {
            fp_sub_inner(&mut result, &FP_MODULUS);
        }
        Fp { limbs: result }
    }

    #[inline]
    pub fn square(&self) -> Fp { self.mont_mul(self) }

    #[inline]
    pub fn add(&self, other: &Fp) -> Fp {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let (s1, c1) = self.limbs[i].overflowing_add(other.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        if carry > 0 || !fp_lt(&result, &FP_MODULUS) {
            fp_sub_inner(&mut result, &FP_MODULUS);
        }
        Fp { limbs: result }
    }

    #[inline]
    pub fn sub(&self, other: &Fp) -> Fp {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (s1, b1) = self.limbs[i].overflowing_sub(other.limbs[i]);
            let (s2, b2) = s1.overflowing_sub(borrow);
            result[i] = s2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        if borrow > 0 {
            let mut carry = 0u64;
            for i in 0..4 {
                let (s1, c1) = result[i].overflowing_add(FP_MODULUS[i]);
                let (s2, c2) = s1.overflowing_add(carry);
                result[i] = s2;
                carry = (c1 as u64) + (c2 as u64);
            }
        }
        Fp { limbs: result }
    }

    #[inline]
    pub fn neg(&self) -> Fp {
        if self.is_zero() { Fp::ZERO }
        else {
            let mut result = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (s1, b1) = FP_MODULUS[i].overflowing_sub(self.limbs[i]);
                let (s2, b2) = s1.overflowing_sub(borrow);
                result[i] = s2;
                borrow = (b1 as u64) + (b2 as u64);
            }
            Fp { limbs: result }
        }
    }

    /// Double this element (add to self).
    #[inline]
    pub fn double(&self) -> Fp { self.add(self) }

    /// Modular inverse via Fermat: a^(p-2) mod p.
    pub fn inverse(&self) -> Option<Fp> {
        if self.is_zero() { return None; }
        let mut exp = FP_MODULUS;
        let (s, borrow) = exp[0].overflowing_sub(2);
        exp[0] = s;
        if borrow {
            for i in 1..4 {
                let (s2, b2) = exp[i].overflowing_sub(1);
                exp[i] = s2;
                if !b2 { break; }
            }
        }
        let mut result = Fp::ONE;
        let mut base = *self;
        for i in 0..4 {
            let mut e = exp[i];
            for _ in 0..64 {
                if e & 1 == 1 { result = result.mont_mul(&base); }
                base = base.square();
                e >>= 1;
            }
        }
        Some(result)
    }
}

impl std::ops::Add for Fp {
    type Output = Fp;
    fn add(self, rhs: Fp) -> Fp { Fp::add(&self, &rhs) }
}
impl std::ops::Sub for Fp {
    type Output = Fp;
    fn sub(self, rhs: Fp) -> Fp { Fp::sub(&self, &rhs) }
}
impl std::ops::Mul for Fp {
    type Output = Fp;
    fn mul(self, rhs: Fp) -> Fp { self.mont_mul(&rhs) }
}
impl std::ops::Neg for Fp {
    type Output = Fp;
    fn neg(self) -> Fp { Fp::neg(&self) }
}

fn fp_lt(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

fn fp_sub_inner(a: &mut [u64; 4], b: &[u64; 4]) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (s1, b1) = a[i].overflowing_sub(b[i]);
        let (s2, b2) = s1.overflowing_sub(borrow);
        a[i] = s2;
        borrow = (b1 as u64) + (b2 as u64);
    }
}


// =============================================================================
// G1 Point — Jacobian Coordinates
// =============================================================================

/// A point on the BN254 G1 curve in Jacobian coordinates.
///
/// Affine = (X/Z², Y/Z³). Identity: Z = 0.
#[derive(Clone, Copy)]
pub struct G1Point {
    pub x: Fp,
    pub y: Fp,
    pub z: Fp,
}

impl G1Point {
    /// Point at infinity.
    pub const IDENTITY: G1Point = G1Point {
        x: Fp::ONE,
        y: Fp::ONE,
        z: Fp::ZERO,
    };

    /// The standard generator G1 = (1, 2).
    pub fn generator() -> G1Point {
        G1Point {
            x: Fp::from_u64(1),
            y: Fp::from_u64(2),
            z: Fp::ONE,
        }
    }

    /// Check if this is the point at infinity.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    /// Verify this point lies on BN254 G1: y² = x³ + 3.
    ///
    /// In Jacobian coordinates: Y² = X³ + 3·Z⁶.
    /// For BN254 G1 (cofactor = 1), on-curve ⟹ in subgroup.
    pub fn is_on_curve(&self) -> bool {
        if self.is_identity() {
            return true; // Point at infinity is on the curve
        }
        let y2 = self.y.square();               // Y²
        let x3 = self.x.square().mont_mul(&self.x); // X³
        let z2 = self.z.square();
        let z4 = z2.square();
        let z6 = z4.mont_mul(&z2);              // Z⁶
        let b = Fp::from_u64(CURVE_B);          // b = 3
        let rhs = x3.add(&b.mont_mul(&z6));     // X³ + 3·Z⁶
        y2 == rhs
    }

    /// Convert to affine coordinates (x, y). Returns None for identity.
    pub fn to_affine(&self) -> Option<(Fp, Fp)> {
        if self.is_identity() {
            return None;
        }
        let z_inv = self.z.inverse()?;
        let z2 = z_inv.square();
        let z3 = z2.mont_mul(&z_inv);
        Some((self.x.mont_mul(&z2), self.y.mont_mul(&z3)))
    }

    /// Point doubling in Jacobian coordinates.
    ///
    /// Cost: 1S + 3M + 8add (using a=0 optimization for BN254).
    pub fn double(&self) -> G1Point {
        if self.is_identity() || self.y.is_zero() {
            return G1Point::IDENTITY;
        }

        // Algorithm: dbl-2009-l (a = 0)
        let a = self.x.square();           // X1²
        let b = self.y.square();           // Y1²
        let c = b.square();                // Y1⁴

        let d = {
            let t = self.x.add(&b);
            t.square().sub(&a).sub(&c).double()
        };

        let e = a.double().add(&a);        // 3X1²  (a=0, so no aZ1⁴ term)
        let f = e.square();                // (3X1²)²

        let x3 = f.sub(&d.double());
        let y3 = e.mont_mul(&d.sub(&x3)).sub(&c.double().double().double());
        let z3 = self.y.mont_mul(&self.z).double();

        G1Point { x: x3, y: y3, z: z3 }
    }

    /// Point addition in Jacobian coordinates.
    ///
    /// Mixed addition (rhs.z = 1) would be faster but this handles the general case.
    pub fn add(&self, other: &G1Point) -> G1Point {
        if self.is_identity() { return *other; }
        if other.is_identity() { return *self; }

        let z1z1 = self.z.square();
        let z2z2 = other.z.square();

        let u1 = self.x.mont_mul(&z2z2);
        let u2 = other.x.mont_mul(&z1z1);

        let s1 = self.y.mont_mul(&z2z2).mont_mul(&other.z);
        let s2 = other.y.mont_mul(&z1z1).mont_mul(&self.z);

        if u1 == u2 {
            if s1 == s2 {
                return self.double();
            } else {
                return G1Point::IDENTITY;
            }
        }

        let h = u2.sub(&u1);
        let i = h.double().square();
        let j = h.mont_mul(&i);
        let r = s2.sub(&s1).double();

        let v = u1.mont_mul(&i);

        let x3 = r.square().sub(&j).sub(&v.double());
        let y3 = r.mont_mul(&v.sub(&x3)).sub(&s1.mont_mul(&j).double());
        let z3 = {
            let t = self.z.add(&other.z);
            t.square().sub(&z1z1).sub(&z2z2).mont_mul(&h)
        };

        G1Point { x: x3, y: y3, z: z3 }
    }

    /// Scalar multiplication using double-and-add (Fr scalar).
    ///
    /// ⚠️ VARIABLE-TIME: branches on scalar bits. Do NOT use with secret scalars.
    /// In this ZK-TDA system, scalars are public polynomial coefficients.
    /// For secret scalars, use a constant-time algorithm (Montgomery ladder).
    pub fn scalar_mul(&self, scalar: &Fr) -> G1Point {
        let bits = scalar.to_raw();
        let mut result = G1Point::IDENTITY;
        let mut base = *self;

        for i in 0..4 {
            let mut e = bits[i];
            for _ in 0..64 {
                if e & 1 == 1 {
                    result = result.add(&base);
                }
                base = base.double();
                e >>= 1;
            }
        }
        result
    }

    /// Negation: -(X, Y, Z) = (X, -Y, Z).
    pub fn neg(&self) -> G1Point {
        G1Point {
            x: self.x,
            y: self.y.neg(),
            z: self.z,
        }
    }
}

impl PartialEq for G1Point {
    fn eq(&self, other: &G1Point) -> bool {
        if self.is_identity() && other.is_identity() { return true; }
        if self.is_identity() || other.is_identity() { return false; }
        // Compare in affine: X1*Z2² == X2*Z1² and Y1*Z2³ == Y2*Z1³
        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let lhs_x = self.x.mont_mul(&z2z2);
        let rhs_x = other.x.mont_mul(&z1z1);
        if lhs_x != rhs_x { return false; }
        let lhs_y = self.y.mont_mul(&z2z2.mont_mul(&other.z));
        let rhs_y = other.y.mont_mul(&z1z1.mont_mul(&self.z));
        lhs_y == rhs_y
    }
}
impl Eq for G1Point {}

impl fmt::Debug for G1Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "G1(∞)")
        } else if let Some((ax, ay)) = self.to_affine() {
            write!(f, "G1({:?}, {:?})", ax.to_raw()[0], ay.to_raw()[0])
        } else {
            write!(f, "G1(err)")
        }
    }
}

// =============================================================================
// Multi-Scalar Multiplication (MSM) — Pippenger
// =============================================================================

/// Compute Σ sᵢ · Pᵢ using a simple windowed approach.
///
/// For production, this should use Pippenger's algorithm with bucket
/// accumulation. This version uses a basic windowed scalar-mul.
pub fn msm(scalars: &[Fr], points: &[G1Point]) -> G1Point {
    assert_eq!(scalars.len(), points.len(), "MSM: length mismatch");
    let mut result = G1Point::IDENTITY;
    for (s, p) in scalars.iter().zip(points.iter()) {
        result = result.add(&p.scalar_mul(s));
    }
    result
}


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_basic() {
        let a = Fp::from_u64(7);
        let b = Fp::from_u64(11);
        let c = a * b;
        assert_eq!(c.to_raw()[0], 77);
    }

    #[test]
    fn test_fp_inverse() {
        let a = Fp::from_u64(7);
        let inv = a.inverse().unwrap();
        let product = a * inv;
        assert_eq!(product, Fp::ONE);
    }

    #[test]
    fn test_generator_not_identity() {
        let g = G1Point::generator();
        assert!(!g.is_identity());
    }

    #[test]
    fn test_identity_add() {
        let g = G1Point::generator();
        let sum = g.add(&G1Point::IDENTITY);
        assert_eq!(sum, g);
        let sum2 = G1Point::IDENTITY.add(&g);
        assert_eq!(sum2, g);
    }

    #[test]
    fn test_double_equals_add() {
        let g = G1Point::generator();
        let dbl = g.double();
        let add = g.add(&g);
        assert_eq!(dbl, add);
    }

    #[test]
    fn test_scalar_mul_zero() {
        let g = G1Point::generator();
        let result = g.scalar_mul(&Fr::ZERO);
        assert!(result.is_identity());
    }

    #[test]
    fn test_scalar_mul_one() {
        let g = G1Point::generator();
        let result = g.scalar_mul(&Fr::ONE);
        assert_eq!(result, g);
    }

    #[test]
    fn test_scalar_mul_two() {
        let g = G1Point::generator();
        let result = g.scalar_mul(&Fr::from_u64(2));
        let expected = g.double();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_negation() {
        let g = G1Point::generator();
        let neg_g = g.neg();
        let sum = g.add(&neg_g);
        assert!(sum.is_identity());
    }

    #[test]
    fn test_associativity() {
        let g = G1Point::generator();
        let g2 = g.double();
        let g3 = g2.add(&g);
        // (G + G) + G == G + (G + G)
        let alt = g.add(&g2);
        assert_eq!(g3, alt);
    }

    #[test]
    fn test_scalar_mul_distributive() {
        let g = G1Point::generator();
        let a = Fr::from_u64(3);
        let b = Fr::from_u64(5);
        // 3G + 5G == 8G
        let lhs = g.scalar_mul(&a).add(&g.scalar_mul(&b));
        let rhs = g.scalar_mul(&Fr::from_u64(8));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_msm_basic() {
        let g = G1Point::generator();
        let g2 = g.double();
        // 3*G + 5*2G = 3G + 10G = 13G
        let result = msm(
            &[Fr::from_u64(3), Fr::from_u64(5)],
            &[g, g2],
        );
        let expected = g.scalar_mul(&Fr::from_u64(13));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_on_curve_generator() {
        let g = G1Point::generator();
        assert!(g.is_on_curve());
    }

    #[test]
    fn test_on_curve_identity() {
        assert!(G1Point::IDENTITY.is_on_curve());
    }

    #[test]
    fn test_on_curve_after_ops() {
        let g = G1Point::generator();
        assert!(g.double().is_on_curve());
        assert!(g.scalar_mul(&Fr::from_u64(42)).is_on_curve());
        assert!(g.add(&g.double()).is_on_curve());
    }

    #[test]
    fn test_off_curve_rejected() {
        // Arbitrary point (1, 3) is NOT on y²=x³+3 since 9 ≠ 1+3=4
        let bad = G1Point {
            x: Fp::from_u64(1),
            y: Fp::from_u64(3),
            z: Fp::ONE,
        };
        assert!(!bad.is_on_curve());
    }
}
