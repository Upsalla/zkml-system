from .base import ActivationFunction, ActivationResult, FIXED_POINT_SCALE
from .optimized import (
    QuadraticActivation, GELUApproxActivation, SwishApproxActivation, 
    ReLUActivation, get_activation, print_activation_comparison
)
