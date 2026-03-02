"""
Field Element Adapter — Drop-in RustFr ↔ Python Fr

Auto-selects the fastest Fr backend:
  1. RustFr (from zkml_rust) — 10-100x faster field arithmetic
  2. Python Fr (from crypto.bn254.field) — pure Python fallback

Override: ZKML_FR_BACKEND=rust|python|auto

Usage:
    from zkml_system.crypto.bn254.fr_adapter import Fr
"""

import os

_BACKEND = None
Fr = None


def _load_backend():
    """Select the fastest available Fr backend."""
    force = os.environ.get("ZKML_FR_BACKEND", "auto").lower()

    if force == "python":
        from zkml_system.crypto.bn254.field import Fr as _Fr
        return _Fr, "python"

    if force == "rust":
        from zkml_rust import RustFr
        return RustFr, "rust"

    # Auto-detect: prefer Rust
    try:
        from zkml_rust import RustFr
        assert (RustFr(2) + RustFr(3)).to_int() == 5
        return RustFr, "rust"
    except Exception:
        from zkml_system.crypto.bn254.field import Fr as _Fr
        return _Fr, "python"


Fr, _BACKEND = _load_backend()

__all__ = ["Fr", "_BACKEND"]
