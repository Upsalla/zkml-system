//! KZG Polynomial Commitment Scheme
//!
//! Implements the Kate-Zaverucha-Goldberg polynomial commitment scheme
//! over BN254 G1. This provides:
//!
//! - Trusted setup (SRS generation from toxic waste τ)
//! - Commitment: C = f(τ)·G  (MSM over SRS)
//! - Opening: π = q(τ)·G where q(x) = (f(x) - f(z)) / (x - z)
//! - Verification: e(C - v·G, G2) == e(π, τ·G2 - z·G2)
//!   (Simplified here as G1-only check since we don't have G2/pairing in Rust)
//!
//! NOTE: Full pairing-based verification requires G2 and the optimal ate
//! pairing, which is delegated to py_ecc in the Python layer. This module
//! provides the G1 operations needed for proving.

use crate::field::Fr;
use crate::curve::{G1Point, msm};
use crate::polynomial::Polynomial;

/// Structured Reference String (SRS) for KZG.
/// Contains [G, τG, τ²G, ..., τⁿG] in G1.
#[derive(Clone)]
pub struct SRS {
    pub g1_powers: Vec<G1Point>,
    pub max_degree: usize,
}

impl SRS {
    /// Generate SRS from a secret τ (toxic waste).
    ///
    /// In production, this would be from a trusted setup ceremony.
    /// Here we use a deterministic τ for testing.
    pub fn generate(max_degree: usize, tau: &Fr) -> Self {
        let g = G1Point::generator();
        let mut g1_powers = Vec::with_capacity(max_degree + 1);
        let mut tau_power = Fr::ONE;

        for _ in 0..=max_degree {
            g1_powers.push(g.scalar_mul(&tau_power));
            tau_power = tau_power.mont_mul(tau);
        }

        SRS { g1_powers, max_degree }
    }
}

/// A KZG commitment (a G1 point).
#[derive(Clone, Debug)]
pub struct Commitment {
    pub point: G1Point,
}

/// A KZG opening proof.
#[derive(Clone, Debug)]
pub struct OpeningProof {
    pub quotient_commit: G1Point,  // π = q(τ)·G
    pub evaluation: Fr,            // v = f(z)
}

/// Commit to a polynomial using the SRS.
///
/// C = Σ aᵢ · [τⁱ]₁
pub fn commit(srs: &SRS, poly: &Polynomial) -> Commitment {
    assert!(
        poly.coeffs.len() <= srs.g1_powers.len(),
        "Polynomial degree exceeds SRS size"
    );

    let scalars = &poly.coeffs;
    let bases = &srs.g1_powers[..scalars.len()];
    let point = msm(scalars, bases);

    Commitment { point }
}

/// Create an opening proof for f(z) = v.
///
/// Computes the quotient polynomial q(x) = (f(x) - v) / (x - z)
/// and commits to it: π = commit(q).
pub fn create_opening_proof(
    srs: &SRS,
    poly: &Polynomial,
    z: &Fr,
) -> OpeningProof {
    let v = poly.evaluate(z);

    // Numerator: f(x) - v
    let numerator = {
        let v_poly = Polynomial::constant(v);
        poly.sub(&v_poly)
    };

    // Divisor: (x - z)
    let divisor = Polynomial::from_coeffs(vec![z.neg(), Fr::ONE]);

    // Quotient: q(x) = (f(x) - v) / (x - z)
    let (quotient, remainder) = numerator.div_rem(&divisor);
    assert!(
        remainder.is_zero(),
        "f(z) != v: polynomial division has non-zero remainder"
    );

    let quotient_commit = commit(srs, &quotient).point;

    OpeningProof {
        quotient_commit,
        evaluation: v,
    }
}

/// Verify an opening proof (G1-only partial check).
///
/// Full verification requires the pairing check:
///   e(C - v·G, G2) == e(π, [τ]₂ - z·G2)
///
/// This function performs the G1-side computation:
/// checks that C - v·[1]₁ == (τ - z) · π
/// using the SRS (we know τ, which is only valid in test scenarios).
///
/// In production, this check is done via the pairing in Python.
pub fn verify_opening_proof_with_tau(
    _srs: &SRS,
    commitment: &Commitment,
    proof: &OpeningProof,
    z: &Fr,
    tau: &Fr,
) -> bool {
    // C - v·G
    let g = G1Point::generator();
    let v_g = g.scalar_mul(&proof.evaluation);
    let lhs = commitment.point.add(&v_g.neg());

    // (τ - z) · π
    let tau_minus_z = tau.sub(z);
    let rhs = proof.quotient_commit.scalar_mul(&tau_minus_z);

    lhs == rhs
}

/// Batch verification: verify multiple opening proofs at once.
///
/// Uses random linear combination to batch-verify:
/// given (Cᵢ, zᵢ, vᵢ, πᵢ), pick random γ and check:
///   Σ γⁱ(Cᵢ - vᵢ·G) == Σ γⁱ(τ - zᵢ)·πᵢ
pub fn batch_verify_with_tau(
    _srs: &SRS,
    commitments: &[Commitment],
    proofs: &[OpeningProof],
    points: &[Fr],
    tau: &Fr,
    random_challenge: &Fr,
) -> bool {
    assert_eq!(commitments.len(), proofs.len());
    assert_eq!(commitments.len(), points.len());

    let g = G1Point::generator();
    let mut lhs = G1Point::IDENTITY;
    let mut rhs = G1Point::IDENTITY;
    let mut gamma_power = Fr::ONE;

    for i in 0..commitments.len() {
        // γⁱ(Cᵢ - vᵢ·G)
        let v_g = g.scalar_mul(&proofs[i].evaluation);
        let ci_minus_vg = commitments[i].point.add(&v_g.neg());
        lhs = lhs.add(&ci_minus_vg.scalar_mul(&gamma_power));

        // γⁱ(τ - zᵢ)·πᵢ
        let tau_minus_z = tau.sub(&points[i]);
        let scaled_pi = proofs[i].quotient_commit
            .scalar_mul(&tau_minus_z)
            .scalar_mul(&gamma_power);
        rhs = rhs.add(&scaled_pi);

        gamma_power = gamma_power.mont_mul(random_challenge);
    }

    lhs == rhs
}


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_srs() -> (SRS, Fr) {
        let tau = Fr::from_u64(42); // Deterministic "toxic waste" for testing
        let srs = SRS::generate(16, &tau);
        (srs, tau)
    }

    #[test]
    fn test_srs_generation() {
        let (srs, _) = test_srs();
        assert_eq!(srs.g1_powers.len(), 17);  // 0..=16
        assert!(!srs.g1_powers[0].is_identity());
        assert_eq!(srs.g1_powers[0], G1Point::generator());
    }

    #[test]
    fn test_commit_constant() {
        let (srs, _) = test_srs();
        let poly = Polynomial::constant(Fr::from_u64(5));
        let c = commit(&srs, &poly);
        // C = 5·G
        let expected = G1Point::generator().scalar_mul(&Fr::from_u64(5));
        assert_eq!(c.point, expected);
    }

    #[test]
    fn test_commit_linear() {
        let (srs, _tau) = test_srs();
        // f(x) = 3 + 7x
        let poly = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(7)]);
        let c = commit(&srs, &poly);
        // C = 3·G + 7·τG = (3 + 7·42)·G = 297·G
        let expected = G1Point::generator().scalar_mul(&Fr::from_u64(297));
        assert_eq!(c.point, expected);
    }

    #[test]
    fn test_opening_proof_constant() {
        let (srs, tau) = test_srs();
        let poly = Polynomial::constant(Fr::from_u64(42));
        let z = Fr::from_u64(7);

        let c = commit(&srs, &poly);
        let proof = create_opening_proof(&srs, &poly, &z);

        assert_eq!(proof.evaluation, Fr::from_u64(42)); // f(7) = 42
        assert!(verify_opening_proof_with_tau(&srs, &c, &proof, &z, &tau));
    }

    #[test]
    fn test_opening_proof_polynomial() {
        let (srs, tau) = test_srs();
        // f(x) = 1 + 2x + 3x²
        let poly = Polynomial::from_coeffs(vec![
            Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)
        ]);
        let z = Fr::from_u64(5);
        // f(5) = 1 + 10 + 75 = 86
        let c = commit(&srs, &poly);
        let proof = create_opening_proof(&srs, &poly, &z);

        assert_eq!(proof.evaluation, Fr::from_u64(86));
        assert!(verify_opening_proof_with_tau(&srs, &c, &proof, &z, &tau));
    }

    #[test]
    fn test_wrong_evaluation_fails() {
        let (srs, tau) = test_srs();
        let poly = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let z = Fr::from_u64(3);

        let c = commit(&srs, &poly);
        let mut proof = create_opening_proof(&srs, &poly, &z);
        // Tamper with the evaluation
        proof.evaluation = Fr::from_u64(999);

        assert!(!verify_opening_proof_with_tau(&srs, &c, &proof, &z, &tau));
    }

    #[test]
    fn test_batch_verify() {
        let (srs, tau) = test_srs();

        let poly1 = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let poly2 = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(4), Fr::from_u64(5)]);

        let z1 = Fr::from_u64(7);
        let z2 = Fr::from_u64(11);

        let c1 = commit(&srs, &poly1);
        let c2 = commit(&srs, &poly2);

        let proof1 = create_opening_proof(&srs, &poly1, &z1);
        let proof2 = create_opening_proof(&srs, &poly2, &z2);

        let gamma = Fr::from_u64(13); // Random challenge

        assert!(batch_verify_with_tau(
            &srs,
            &[c1, c2],
            &[proof1, proof2],
            &[z1, z2],
            &tau,
            &gamma,
        ));
    }

    #[test]
    fn test_batch_verify_tampered_fails() {
        let (srs, tau) = test_srs();

        let poly1 = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let poly2 = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(4)]);

        let z1 = Fr::from_u64(5);
        let z2 = Fr::from_u64(6);

        let c1 = commit(&srs, &poly1);
        let c2 = commit(&srs, &poly2);

        let proof1 = create_opening_proof(&srs, &poly1, &z1);
        let mut proof2 = create_opening_proof(&srs, &poly2, &z2);
        proof2.evaluation = Fr::from_u64(0); // Tamper

        let gamma = Fr::from_u64(17);

        assert!(!batch_verify_with_tau(
            &srs,
            &[c1, c2],
            &[proof1, proof2],
            &[z1, z2],
            &tau,
            &gamma,
        ));
    }
}
