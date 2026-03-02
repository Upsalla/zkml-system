"""Compatibility shim: re-exports from _legacy/core_legacy/field.py"""
import sys
import os

# Add parent directory so _legacy is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from _legacy.core_legacy.field import *  # noqa: F401,F403
