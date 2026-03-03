"""
zkml-system — Zero-Knowledge Machine Learning from Scratch.

PLONK prover/verifier on BN254 with a hybrid TDA+ZK bridge
for privacy-preserving model similarity proofs.
"""

__version__ = "3.1.0"

# Direct submodule imports:
#   from zkml_system.crypto.bn254.fr_adapter import Fr
#   from zkml_system.plonk.plonk_prover import PLONKProver, PLONKVerifier
#   from zkml_system.hybrid_bridge import HybridBridge

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
