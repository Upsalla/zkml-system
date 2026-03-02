//! ZK-TDA Rust Backend — PyO3 Module
//!
//! Exposes BN254 field arithmetic to Python with the same API as
//! `zkml_system.crypto.bn254.field.Fr`.
//!
// L5: Public APIs for Phase E (PyO3 expansion) — unused until bindings grow
#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::{PyZeroDivisionError, PyValueError};

mod field;
mod curve;
mod polynomial;
mod kzg;
use field::Fr;

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

    fn __add__(&self, other: &PyFr) -> PyFr {
        PyFr { inner: self.inner + other.inner }
    }

    fn __sub__(&self, other: &PyFr) -> PyFr {
        PyFr { inner: self.inner - other.inner }
    }

    fn __mul__(&self, other: &PyFr) -> PyFr {
        PyFr { inner: self.inner * other.inner }
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

    /// Compute self^exp mod r.
    fn __pow__(&self, exp: u64, modulo: Option<u64>) -> PyResult<PyFr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err(
                "RustFr.__pow__: 3-arg pow(a, b, mod) is not supported; field modulus is implicit"
            ));
        }
        let exp_limbs = [exp, 0, 0, 0];
        Ok(PyFr { inner: self.inner.pow(exp_limbs) })
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


/// The Python module.
#[pymodule]
fn zkml_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFr>()?;
    Ok(())
}
