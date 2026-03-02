"""
zkML System - Zero-Knowledge Machine Learning from Scratch
==========================================================

Ein vollständig von Grund auf implementiertes Zero-Knowledge Machine Learning System.
"""

__version__ = "2.0.0"
__author__ = "zkML Research"

# Legacy imports wrapped in try/except — the original module structure
# (core, activations, sparse, network, proof) was replaced by
# (crypto, plonk, tda, network) in v2.0. Submodule imports
# like `from zkml_system.crypto.bn254.field import Fr` still work.

try:
    from .core import (
        FieldElement, FieldConfig, FixedPoint, PrimeField,
        R1CS, R1CSBuilder, R1CSConstraint, LinearCombination,
        Witness, WitnessBuilder
    )
except ImportError:
    pass

try:
    from .activations import (
        ActivationFunction, FIXED_POINT_SCALE,
        GELUApproxActivation as GELUApprox,
        SwishApproxActivation as SwishApprox,
        QuadraticActivation,
        get_activation
    )
except ImportError:
    pass

try:
    from .sparse import (
        SparsityStats, SparseConstraintSet, SparseProofBuilder,
        ZeroProofGenerator, analyze_sparsity
    )
except ImportError:
    pass

try:
    from .network import (
        LayerConfig, LayerWeights, LayerOutput,
        DenseLayer, InputLayer, OutputLayer,
        NetworkStats, LayerSpec, Network, NetworkBuilder
    )
except ImportError:
    pass

try:
    from .proof import (
        ProofComponents, Proof, Prover, ProofStats,
        VerificationResult, Verifier, BatchVerifier
    )
except ImportError:
    pass
