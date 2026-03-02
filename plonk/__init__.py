"""
PLONK zkML System v5.0

Author: David Weyhe
Date: 27. Januar 2026

Production-grade PLONK-based zero-knowledge proof system for machine learning.
"""

__version__ = "5.0.0"
__author__ = "David Weyhe"

# Lazy imports — many optional dependencies (py_ecc, etc.) may not be
# installed in all environments. Wrap each block so individual
# submodule imports (e.g. `from zkml_system.plonk.circuit_compiler import ...`)
# still work.

try:
    from .core import (
        Field, Fr, Fp,
        Polynomial,
        SRS,
        KZG, KZGCommitment, KZGProof,
        Wire, Gate, GateType,
        Circuit, Witness
    )
except ImportError:
    pass

try:
    from .optimizations import PLONKOptimizer
except ImportError:
    pass

try:
    from .zkml import ZkML, NetworkConfig, ZkMLProof
except ImportError:
    pass

try:
    from .kzg_verifier import KZGVerifier, VerificationResult
except ImportError:
    pass

try:
    from .trusted_setup import (
        TrustedSetupCeremony,
        validate_srs,
        serialize_srs,
        deserialize_srs,
        generate_test_srs
    )
except ImportError:
    pass

try:
    from .folding import (
        R1CS, R1CSMatrix,
        RelaxedR1CSInstance, RelaxedR1CSWitness,
        NovaFolding, ZkMLIVC, IVCProof,
        verify_ivc_proof
    )
except ImportError:
    pass

try:
    from .lookup import (
        LookupTable,
        LookupOperations,
        PlookupArgument,
        create_relu_table,
        create_sigmoid_table,
        create_range_table
    )
except ImportError:
    pass

try:
    from .batch import BatchAggregator, ProofAggregator
except ImportError:
    pass

try:
    from .operators import Conv2D, Conv2DConfig, Attention, AttentionConfig
except ImportError:
    pass
