"""
Field Element Adapter — Drop-in RustFr ↔ Python Fr

Provides a unified `Fr` import that automatically selects the fastest
available backend:
  1. RustFr (from zkml_rust) — 10-100x faster field arithmetic
  2. Python Fr (from crypto.bn254.field) — pure Python fallback

Usage:
    from crypto.bn254.fr_adapter import Fr
    # Fr is now the fastest available implementation

Both backends expose the same API:
    Fr(value), Fr.zero(), Fr.one(), Fr.MODULUS
    +, -, *, /, **, ==, hash, repr, bool
    .to_int(), .value, .inverse(), .square()
    .is_zero(), .is_one()
    Fr.batch_inverse([...])
"""

import os

def _load_backend():
    """Select the fastest available Fr backend."""
    # Allow explicit override via environment variable
    force = os.environ.get("ZKML_FR_BACKEND", "auto").lower()
    
    if force == "python":
        from crypto.bn254.field import Fr
        return Fr, "python"
    
    if force == "rust":
        from zkml_rust import RustFr
        return RustFr, "rust"
    
    # Auto-detect: prefer Rust
    try:
        from zkml_rust import RustFr
        # Smoke test: basic arithmetic must work
        assert (RustFr(2) + RustFr(3)).to_int() == 5
        return RustFr, "rust"
    except (ImportError, AssertionError, Exception):
        from crypto.bn254.field import Fr
        return Fr, "python"


Fr, _BACKEND = _load_backend()

__all__ = ["Fr"]
