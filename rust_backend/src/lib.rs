//! ZK-TDA Rust Backend — PyO3 Module
//!
//! Exposes BN254 field arithmetic to Python with the same API as
//! `zkml_system.crypto.bn254.field.Fr`.
//!
// L5: Public APIs for Phase E (PyO3 expansion) — unused until bindings grow
#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::{PyZeroDivisionError, PyValueError};
use pyo3::types::PyTuple;

mod field;
mod curve;
mod polynomial;
mod kzg;
use field::Fr;
use curve::Fp;

/// Python-visible wrapper for the Rust Fr field element.
#[pyclass(name = "RustFr")]
#[derive(Clone)]
struct PyFr {
    inner: Fr,
}

#[pymethods]
impl PyFr {
    /// Create a new field element from an integer.
    /// Accepts Python int (any size).
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Extract as u64 first (fast path)
        if let Ok(v) = value.extract::<u64>() {
            return Ok(PyFr { inner: Fr::from_u64(v) });
        }
        // Fall back to string conversion for large ints
        if let Ok(v) = value.extract::<i64>() {
            if v >= 0 {
                return Ok(PyFr { inner: Fr::from_u64(v as u64) });
            }
            // Negative: compute MODULUS - |v|
            // Safety: use i128 to avoid overflow on i64::MIN
            let abs = (v as i128).unsigned_abs() as u64;
            let pos = Fr::from_u64(abs);
            return Ok(PyFr { inner: pos.neg() });
        }
        // Large Python int: convert via string
        let s: String = value.str()?.to_string();
        let fr = fr_from_decimal_str(&s)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyFr { inner: fr })
    }

    /// Return the integer value as a Python int.
    fn to_int(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.to_int_string();
        // Use Python's int() to parse, which handles both decimal and hex
        let builtins = py.import("builtins")?;
        if s.starts_with("0x") {
            builtins.call_method1("int", (&s, 16))
                .map(|v| v.into())
        } else {
            builtins.call_method1("int", (&s,))
                .map(|v| v.into())
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyFr> {
        let rhs = if let Ok(pf) = other.extract::<PyFr>() { pf.inner } else { PyFr::new(other)?.inner };
        Ok(PyFr { inner: self.inner + rhs })
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyFr> {
        let rhs = if let Ok(pf) = other.extract::<PyFr>() { pf.inner } else { PyFr::new(other)?.inner };
        Ok(PyFr { inner: self.inner - rhs })
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyFr> {
        let rhs = if let Ok(pf) = other.extract::<PyFr>() { pf.inner } else { PyFr::new(other)?.inner };
        Ok(PyFr { inner: self.inner * rhs })
    }

    fn __neg__(&self) -> PyFr {
        PyFr { inner: -self.inner }
    }

    fn __eq__(&self, other: &PyFr) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        // FxHash-style mixing for better distribution
        let raw = self.inner.to_raw();
        let mut h: u64 = 0;
        for &limb in &raw {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(limb);
        }
        // Python requires hash != -1 (sentinel for errors)
        let result = h as isize;
        if result == -1 { -2 } else { result }
    }

    fn __repr__(&self) -> String {
        format!("RustFr({})", self.inner)
    }

    fn __bool__(&self) -> bool {
        !self.inner.is_zero()
    }

    /// Compute self^exp mod r.  Accepts arbitrary-size Python ints (up to 256 bits).
    fn __pow__(&self, exp: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyFr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err(
                "RustFr.__pow__: 3-arg pow(a, b, mod) is not supported; field modulus is implicit"
            ));
        }
        // Try fast path: fits in u64
        if let Ok(small) = exp.extract::<u64>() {
            let exp_limbs = [small, 0, 0, 0];
            return Ok(PyFr { inner: self.inner.pow(exp_limbs) });
        }
        // Slow path: extract 4 limbs from arbitrary Python int
        let mask: u64 = u64::MAX;
        let l0: u64 = exp.call_method1("__and__", (mask,))?.extract()?;
        let s1 = exp.call_method1("__rshift__", (64_u32,))?;
        let l1: u64 = s1.call_method1("__and__", (mask,))?.extract()?;
        let s2 = exp.call_method1("__rshift__", (128_u32,))?;
        let l2: u64 = s2.call_method1("__and__", (mask,))?.extract()?;
        let s3 = exp.call_method1("__rshift__", (192_u32,))?;
        let l3: u64 = s3.call_method1("__and__", (mask,))?.extract()?;
        let exp_limbs = [l0, l1, l2, l3];
        Ok(PyFr { inner: self.inner.pow(exp_limbs) })
    }

    /// Comparison: self < other.  Handles both RustFr and int operands.
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other_inner = if let Ok(pf) = other.extract::<PyFr>() {
            pf.inner
        } else {
            PyFr::new(other)?.inner
        };
        let a = self.inner.to_raw();
        let b = other_inner.to_raw();
        // Compare 256-bit values in big-endian limb order (MSB first)
        for i in (0..4).rev() {
            if a[i] != b[i] {
                return Ok(a[i] < b[i]);
            }
        }
        Ok(false) // equal
    }


    /// Multiplicative inverse. Raises ZeroDivisionError if zero.
    fn inverse(&self) -> PyResult<PyFr> {
        self.inner.inverse()
            .map(|inv| PyFr { inner: inv })
            .ok_or_else(|| PyZeroDivisionError::new_err("Cannot invert zero"))
    }

    /// Square this element.
    fn square(&self) -> PyFr {
        PyFr { inner: self.inner.square() }
    }

    /// Check if zero.
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// The additive identity.
    #[staticmethod]
    fn zero() -> PyFr {
        PyFr { inner: Fr::ZERO }
    }

    /// The multiplicative identity.
    #[staticmethod]
    fn one() -> PyFr {
        PyFr { inner: Fr::ONE }
    }

    /// Batch inversion using Montgomery's trick.
    #[staticmethod]
    fn batch_inverse(elements: Vec<PyFr>) -> PyResult<Vec<PyFr>> {
        let frs: Vec<Fr> = elements.iter().map(|e| e.inner).collect();
        Fr::batch_inverse(&frs)
            .map(|inv| inv.into_iter().map(|f| PyFr { inner: f }).collect())
            .ok_or_else(|| PyZeroDivisionError::new_err("Cannot invert zero in batch"))
    }

    // --- API Parity with Python Fr ---

    /// The BN254 scalar field modulus.
    /// Matches Python Fr.MODULUS class attribute.
    #[classattr]
    #[pyo3(name = "MODULUS")]
    fn modulus(py: Python<'_>) -> PyResult<PyObject> {
        let builtins = py.import("builtins")?;
        builtins.call_method1("int", ("21888242871839275222246405745257275088548364400416034343698204186575808495617",))
            .map(|v| v.into())
    }

    /// Internal Montgomery representation.
    /// Used by the PLONK pipeline for direct comparison/storage.
    /// Returns the Montgomery-encoded integer (matches Python Fr.value).
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Reconstruct the full Montgomery value from 4 limbs
        let raw = self.inner.to_mont_raw();
        let builtins = py.import("builtins")?;
        // Build Python int from 4 limbs: sum(limb[i] * 2^(64*i))
        let mut result = builtins.call_method1("int", (0_u64,))?;
        for (i, &limb) in raw.iter().enumerate() {
            let shift = 64 * i;
            let limb_py = builtins.call_method1("int", (limb,))?;
            let shifted = limb_py.call_method1("__lshift__", (shift,))?;
            result = result.call_method1("__or__", (shifted,))?;
        }
        Ok(result.into())
    }

    /// Check if this is the multiplicative identity.
    fn is_one(&self) -> bool {
        self.inner == Fr::ONE
    }

    /// Division: self / other = self * other.inverse()
    fn __truediv__(&self, other: &PyFr) -> PyResult<PyFr> {
        let inv = other.inner.inverse()
            .ok_or_else(|| PyZeroDivisionError::new_err("Cannot divide by zero"))?;
        Ok(PyFr { inner: self.inner * inv })
    }

    /// Reflected addition: int + RustFr
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyFr> {
        let lhs = PyFr::new(other)?;
        Ok(PyFr { inner: lhs.inner + self.inner })
    }

    /// Reflected subtraction: int - RustFr
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyFr> {
        let lhs = PyFr::new(other)?;
        Ok(PyFr { inner: lhs.inner - self.inner })
    }

    /// Reflected multiplication: int * RustFr
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyFr> {
        let lhs = PyFr::new(other)?;
        Ok(PyFr { inner: lhs.inner * self.inner })
    }
}

/// Parse a decimal string into an Fr element.
/// Handles negative values by parsing absolute value and negating mod r.
fn fr_from_decimal_str(s: &str) -> Result<Fr, String> {
    let s = s.trim();
    if s == "0" {
        return Ok(Fr::ZERO);
    }

    // M3: Handle negative numbers
    let (is_negative, s) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else {
        (false, s)
    };
    if s.is_empty() {
        return Err("Invalid integer: bare '-' sign".to_string());
    }
    if s == "0" {
        return Ok(Fr::ZERO);
    }

    // Parse as bytes of decimal digits
    let digits: Vec<u8> = s.bytes()
        .map(|b| {
            if b >= b'0' && b <= b'9' {
                Ok(b - b'0')
            } else {
                Err(format!("Invalid digit in '{s}'"))
            }
        })
        .collect::<Result<Vec<u8>, String>>()?;

    // Convert decimal digits to 4 × u64 limbs via repeated division
    let mut remaining = digits;
    let mut limbs = [0u64; 4];
    for limb_idx in 0..4 {
        // Divide by 2^64, keeping quotient and remainder
        let divisor = 1u128 << 64;
        let current = remaining;
        let mut remainder: u128 = 0;
        let mut quotient_digits = Vec::new();
        let mut leading = true;

        for &digit in &current {
            remainder = remainder * 10 + digit as u128;
            let q = (remainder / divisor) as u8;
            remainder %= divisor;
            if q != 0 || !leading {
                quotient_digits.push(q);
                leading = false;
            }
        }

        limbs[limb_idx] = remainder as u64;
        remaining = quotient_digits;

        if remaining.is_empty() {
            break;
        }
    }

    // H1: Reject values that exceed 256 bits (4 limbs)
    if !remaining.is_empty() {
        return Err(format!("Integer exceeds 256-bit field size: '{s}'"));
    }

    let result = Fr::from_raw(limbs);
    Ok(if is_negative { result.neg() } else { result })
}


// =============================================================================
// PyPolynomial — Polynomial over Fr
// =============================================================================

/// Python-visible wrapper for Polynomial operations.
#[pyclass(name = "RustPolynomial")]
#[derive(Clone)]
struct PyPolynomial {
    inner: polynomial::Polynomial,
}

#[pymethods]
impl PyPolynomial {
    /// Create from a list of Fr coefficient values.
    #[new]
    fn new(coeffs: Vec<PyFr>) -> Self {
        let frs: Vec<Fr> = coeffs.iter().map(|c| c.inner).collect();
        PyPolynomial { inner: polynomial::Polynomial::from_coeffs(frs) }
    }

    /// Zero polynomial.
    #[staticmethod]
    fn zero() -> Self {
        PyPolynomial { inner: polynomial::Polynomial::zero() }
    }

    /// Constant polynomial.
    #[staticmethod]
    fn constant(c: &PyFr) -> Self {
        PyPolynomial { inner: polynomial::Polynomial::constant(c.inner) }
    }

    /// Degree of the polynomial (None for zero).
    fn degree(&self) -> Option<usize> {
        self.inner.degree()
    }

    /// Evaluate p(x) using Horner's method.
    fn evaluate(&self, x: &PyFr) -> PyFr {
        PyFr { inner: self.inner.evaluate(&x.inner) }
    }

    /// Polynomial addition.
    fn __add__(&self, other: &PyPolynomial) -> PyPolynomial {
        PyPolynomial { inner: self.inner.add(&other.inner) }
    }

    /// Polynomial subtraction.
    fn __sub__(&self, other: &PyPolynomial) -> PyPolynomial {
        PyPolynomial { inner: self.inner.sub(&other.inner) }
    }

    /// Polynomial multiplication (NTT-accelerated).
    fn __mul__(&self, other: &PyPolynomial) -> PyPolynomial {
        PyPolynomial { inner: self.inner.mul(&other.inner) }
    }

    /// Scalar multiplication.
    fn scale(&self, scalar: &PyFr) -> PyPolynomial {
        PyPolynomial { inner: self.inner.scale(&scalar.inner) }
    }

    /// Division with remainder: (quotient, remainder).
    fn div_rem(&self, divisor: &PyPolynomial) -> (PyPolynomial, PyPolynomial) {
        let (q, r) = self.inner.div_rem(&divisor.inner);
        (PyPolynomial { inner: q }, PyPolynomial { inner: r })
    }

    /// Lagrange interpolation from points (xs, ys).
    #[staticmethod]
    fn lagrange_interpolate(xs: Vec<PyFr>, ys: Vec<PyFr>) -> PyPolynomial {
        let xs_fr: Vec<Fr> = xs.iter().map(|x| x.inner).collect();
        let ys_fr: Vec<Fr> = ys.iter().map(|y| y.inner).collect();
        PyPolynomial {
            inner: polynomial::Polynomial::lagrange_interpolate(&xs_fr, &ys_fr),
        }
    }

    /// Evaluate at n-th roots of unity using NTT.
    fn evaluate_domain(&self, n: usize) -> Vec<PyFr> {
        self.inner.evaluate_domain(n)
            .into_iter()
            .map(|f| PyFr { inner: f })
            .collect()
    }

    /// Number of coefficients.
    fn __len__(&self) -> usize {
        self.inner.coeffs.len()
    }

    /// Get coefficient at index.
    fn coeff(&self, idx: usize) -> PyResult<PyFr> {
        if idx < self.inner.coeffs.len() {
            Ok(PyFr { inner: self.inner.coeffs[idx] })
        } else {
            Ok(PyFr { inner: Fr::ZERO })
        }
    }

    fn __repr__(&self) -> String {
        format!("RustPolynomial(degree={:?}, {} coeffs)",
                self.inner.degree(), self.inner.coeffs.len())
    }
}


// =============================================================================
// Helpers: u64[4] <-> Python int
// =============================================================================

/// Convert 4 little-endian u64 limbs to a Python int.
fn limbs_to_pyint(py: Python<'_>, limbs: &[u64; 4]) -> PyResult<PyObject> {
    // Construct: l0 + l1*2^64 + l2*2^128 + l3*2^192
    let l0 = limbs[0].into_pyobject(py)?.into_any().unbind();
    let l1 = limbs[1].into_pyobject(py)?.into_any().unbind();
    let l2 = limbs[2].into_pyobject(py)?.into_any().unbind();
    let l3 = limbs[3].into_pyobject(py)?.into_any().unbind();
    let result = l0.bind(py)
        .call_method1("__add__", (l1.bind(py).call_method1("__lshift__", (64_u32,))?,))?
        .call_method1("__add__", (l2.bind(py).call_method1("__lshift__", (128_u32,))?,))?
        .call_method1("__add__", (l3.bind(py).call_method1("__lshift__", (192_u32,))?,))?;
    Ok(result.unbind())
}

/// Extract 4 little-endian u64 limbs from a Python int.
fn pyint_to_limbs(val: &Bound<'_, PyAny>) -> PyResult<[u64; 4]> {
    let mask: u64 = u64::MAX;
    let l0: u64 = val.call_method1("__and__", (mask,))?.extract()?;
    let s1 = val.call_method1("__rshift__", (64_u32,))?;
    let l1: u64 = s1.call_method1("__and__", (mask,))?.extract()?;
    let s2 = val.call_method1("__rshift__", (128_u32,))?;
    let l2: u64 = s2.call_method1("__and__", (mask,))?.extract()?;
    let s3 = val.call_method1("__rshift__", (192_u32,))?;
    let l3: u64 = s3.call_method1("__and__", (mask,))?.extract()?;
    Ok([l0, l1, l2, l3])
}


// =============================================================================
// PyG1Point — BN254 G1 Elliptic Curve Point
// =============================================================================

/// Python-visible wrapper for G1 curve points.
#[pyclass(name = "RustG1Point")]
#[derive(Clone)]
struct PyG1Point {
    inner: curve::G1Point,
}

#[pymethods]
impl PyG1Point {
    /// The generator point G1 = (1, 2).
    #[staticmethod]
    fn generator() -> Self {
        PyG1Point { inner: curve::G1Point::generator() }
    }

    /// The identity (point at infinity).
    #[staticmethod]
    fn identity() -> Self {
        PyG1Point { inner: curve::G1Point::IDENTITY }
    }

    /// Check if this is the identity point.
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// Verify the point lies on BN254 G1.
    fn is_on_curve(&self) -> bool {
        self.inner.is_on_curve()
    }

    /// Scalar multiplication: self * scalar.
    fn scalar_mul(&self, scalar: &PyFr) -> PyG1Point {
        PyG1Point { inner: self.inner.scalar_mul(&scalar.inner) }
    }

    /// Point addition.
    fn __add__(&self, other: &PyG1Point) -> PyG1Point {
        PyG1Point { inner: self.inner.add(&other.inner) }
    }

    /// Point negation.
    fn __neg__(&self) -> PyG1Point {
        PyG1Point { inner: self.inner.neg() }
    }

    /// Point equality (compares affine coordinates).
    fn __eq__(&self, other: &PyG1Point) -> bool {
        self.inner == other.inner
    }

    /// Multi-scalar multiplication (MSM): Σ scalars[i] * points[i].
    #[staticmethod]
    fn msm(points: Vec<PyG1Point>, scalars: Vec<PyFr>) -> PyG1Point {
        let scs: Vec<Fr> = scalars.iter().map(|s| s.inner).collect();
        let pts: Vec<curve::G1Point> = points.iter().map(|p| p.inner).collect();
        PyG1Point { inner: curve::msm(&scs, &pts) }
    }

    fn __repr__(&self) -> String {
        if self.inner.is_identity() {
            "RustG1Point(IDENTITY)".to_string()
        } else {
            "RustG1Point(...)".to_string()
        }
    }

    /// Convert to affine coordinates. Returns (x, y) as Python ints, or None for identity.
    fn to_affine(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.to_affine() {
            Some((x_fp, y_fp)) => {
                let x_limbs = x_fp.to_raw();
                let y_limbs = y_fp.to_raw();
                // Convert [u64; 4] to Python int (little-endian limb order)
                let x_int = limbs_to_pyint(py, &x_limbs)?;
                let y_int = limbs_to_pyint(py, &y_limbs)?;
                Ok(PyTuple::new(py, &[x_int, y_int])?.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }

    /// Create a G1 point from affine coordinates (x, y) as Python ints.
    #[staticmethod]
    fn from_affine(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<PyG1Point> {
        let x_limbs = pyint_to_limbs(x)?;
        let y_limbs = pyint_to_limbs(y)?;
        let x_fp = Fp::from_raw(x_limbs);
        let y_fp = Fp::from_raw(y_limbs);
        Ok(PyG1Point {
            inner: curve::G1Point {
                x: x_fp,
                y: y_fp,
                z: Fp::ONE,
            },
        })
    }
}


// =============================================================================
// KZG — Polynomial Commitment Scheme
// =============================================================================

/// Python-visible wrapper for KZG SRS (Structured Reference String).
#[pyclass(name = "RustSRS")]
#[derive(Clone)]
struct PySRS {
    inner: kzg::SRS,
}

#[pymethods]
impl PySRS {
    /// Generate SRS from a secret tau and max polynomial degree.
    #[new]
    fn new(max_degree: usize, tau: &PyFr) -> Self {
        PySRS { inner: kzg::SRS::generate(max_degree, &tau.inner) }
    }

    /// Number of SRS points.
    fn __len__(&self) -> usize {
        self.inner.g1_powers.len()
    }

    fn __repr__(&self) -> String {
        format!("RustSRS(max_degree={})", self.inner.g1_powers.len() - 1)
    }
}

/// Python-visible wrapper for a KZG commitment.
#[pyclass(name = "RustCommitment")]
#[derive(Clone)]
struct PyCommitment {
    inner: kzg::Commitment,
}

#[pymethods]
impl PyCommitment {
    /// Get the underlying G1 point.
    #[getter]
    fn point(&self) -> PyG1Point {
        PyG1Point { inner: self.inner.point }
    }

    fn __repr__(&self) -> String {
        "RustCommitment(...)".to_string()
    }
}

/// Python-visible wrapper for a KZG opening proof.
#[pyclass(name = "RustOpeningProof")]
#[derive(Clone)]
struct PyOpeningProof {
    inner: kzg::OpeningProof,
}

#[pymethods]
impl PyOpeningProof {
    /// The evaluation value f(z).
    #[getter]
    fn evaluation(&self) -> PyFr {
        PyFr { inner: self.inner.evaluation }
    }

    /// The proof G1 point π = commit(quotient).
    #[getter]
    fn quotient_commit(&self) -> PyG1Point {
        PyG1Point { inner: self.inner.quotient_commit }
    }

    fn __repr__(&self) -> String {
        "RustOpeningProof(...)".to_string()
    }
}

/// KZG operations as module-level functions.
#[pyfunction]
fn kzg_commit(srs: &PySRS, poly: &PyPolynomial) -> PyCommitment {
    PyCommitment { inner: kzg::commit(&srs.inner, &poly.inner) }
}

#[pyfunction]
fn kzg_create_proof(srs: &PySRS, poly: &PyPolynomial, z: &PyFr) -> PyOpeningProof {
    PyOpeningProof { inner: kzg::create_opening_proof(&srs.inner, &poly.inner, &z.inner) }
}

#[pyfunction]
fn kzg_verify(srs: &PySRS, commitment: &PyCommitment, proof: &PyOpeningProof,
              z: &PyFr, tau: &PyFr) -> bool {
    kzg::verify_opening_proof_with_tau(&srs.inner, &commitment.inner, &proof.inner,
                                        &z.inner, &tau.inner)
}

#[pyfunction]
fn kzg_batch_verify(srs: &PySRS, commitments: Vec<PyCommitment>, proofs: Vec<PyOpeningProof>,
                    points: Vec<PyFr>, tau: &PyFr, challenge: &PyFr) -> bool {
    let comms: Vec<kzg::Commitment> = commitments.iter().map(|c| c.inner.clone()).collect();
    let pfs: Vec<kzg::OpeningProof> = proofs.iter().map(|p| p.inner.clone()).collect();
    let pts: Vec<Fr> = points.iter().map(|p| p.inner).collect();
    kzg::batch_verify_with_tau(&srs.inner, &comms, &pfs, &pts, &tau.inner, &challenge.inner)
}


/// The Python module.
#[pymodule]
fn zkml_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Field
    m.add_class::<PyFr>()?;
    // Polynomial
    m.add_class::<PyPolynomial>()?;
    // Curve
    m.add_class::<PyG1Point>()?;
    // KZG
    m.add_class::<PySRS>()?;
    m.add_class::<PyCommitment>()?;
    m.add_class::<PyOpeningProof>()?;
    m.add_function(wrap_pyfunction!(kzg_commit, m)?)?;
    m.add_function(wrap_pyfunction!(kzg_create_proof, m)?)?;
    m.add_function(wrap_pyfunction!(kzg_verify, m)?)?;
    m.add_function(wrap_pyfunction!(kzg_batch_verify, m)?)?;
    Ok(())
}
