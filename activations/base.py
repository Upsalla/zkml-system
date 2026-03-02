"""Compatibility shim: re-exports from _legacy/activations_legacy/base.py"""
import sys
import os

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from _legacy.activations_legacy.base import *  # noqa: F401,F403
