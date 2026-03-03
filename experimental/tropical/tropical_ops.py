"""
Tropical Arithmetic Operations for zkML

This module implements tropical (min-plus) arithmetic operations optimized
for zero-knowledge circuits. The key insight is that certain operations
(max-pooling, softmax, argmax) can be expressed more efficiently using
tropical arithmetic.

Mathematical Background:
- Tropical addition: a ⊕ b = min(a, b)
- Tropical multiplication: a ⊙ b = a + b
- The tropical semiring (R ∪ {+∞}, ⊕, ⊙) is idempotent: a ⊕ a = a

For ZK circuits:
- Standard multiplication costs 1 constraint
- Tropical multiplication (= addition) costs 0 constraints
- Tropical addition (= min) costs ~1 constraint with proper encoding
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math


@dataclass
class TropicalConstraint:
    """
    A constraint in the tropical circuit.
    
    Types:
    - 'linear': a₁x₁ + a₂x₂ + ... = b (0 multiplicative constraints)
    - 'comparison': x ≤ y with indicator (1 constraint for the indicator)
    - 'selection': z = x if indicator else y (1 constraint)
    """
    constraint_type: str
    variables: List[str]
    coefficients: List[int]
    constant: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def multiplicative_cost(self) -> int:
        """Return the number of multiplicative constraints."""
        if self.constraint_type == 'linear':
            return 0
        elif self.constraint_type == 'comparison':
            return 1  # For the indicator bit
        elif self.constraint_type == 'selection':
            return 1  # For the conditional selection
        else:
            return 1  # Default


@dataclass
class TropicalVariable:
    """A variable in the tropical circuit."""
    name: str
    value: Optional[int] = None
    is_input: bool = False
    is_output: bool = False


class TropicalCircuit:
    """
    A circuit using tropical arithmetic.
    
    This class tracks variables and constraints, and provides
    methods to compute the total constraint cost.
    """
    
    def __init__(self):
        self.variables: Dict[str, TropicalVariable] = {}
        self.constraints: List[TropicalConstraint] = []
        self.var_counter = 0
    
    def new_variable(self, prefix: str = "v", value: Optional[int] = None) -> str:
        """Create a new variable and return its name."""
        name = f"{prefix}_{self.var_counter}"
        self.var_counter += 1
        self.variables[name] = TropicalVariable(name=name, value=value)
        return name
    
    def add_constraint(self, constraint: TropicalConstraint):
        """Add a constraint to the circuit."""
        self.constraints.append(constraint)
    
    def total_constraints(self) -> int:
        """Return the total number of multiplicative constraints."""
        return sum(c.multiplicative_cost() for c in self.constraints)
    
    def constraint_breakdown(self) -> Dict[str, int]:
        """Return a breakdown of constraints by type."""
        breakdown = {}
        for c in self.constraints:
            breakdown[c.constraint_type] = breakdown.get(c.constraint_type, 0) + c.multiplicative_cost()
        return breakdown


class TropicalSemiring:
    """
    Tropical Semiring operations for ZK circuits.
    
    Uses the min-plus convention:
    - tropical_add(a, b) = min(a, b)
    - tropical_mul(a, b) = a + b
    """
    
    @staticmethod
    def tropical_mul(circuit: TropicalCircuit, a: str, b: str) -> str:
        """
        Tropical multiplication = standard addition.
        
        Cost: 0 multiplicative constraints (linear operation)
        
        Returns: name of result variable
        """
        result = circuit.new_variable("tmul")
        
        # Constraint: result = a + b (linear, no multiplicative cost)
        circuit.add_constraint(TropicalConstraint(
            constraint_type='linear',
            variables=[result, a, b],
            coefficients=[1, -1, -1],
            constant=0,
            metadata={'operation': 'tropical_mul'}
        ))
        
        return result
    
    @staticmethod
    def tropical_add(circuit: TropicalCircuit, a: str, b: str) -> Tuple[str, str]:
        """
        Tropical addition = minimum.
        
        Cost: 2 multiplicative constraints
        - 1 for comparison indicator
        - 1 for conditional selection
        
        Returns: (result_name, indicator_name)
        - indicator = 1 if a <= b, else 0
        - result = a if indicator else b
        """
        indicator = circuit.new_variable("ind")
        result = circuit.new_variable("tmin")
        
        # Constraint 1: indicator ∈ {0, 1} and encodes a <= b
        # This requires a comparison gadget
        circuit.add_constraint(TropicalConstraint(
            constraint_type='comparison',
            variables=[indicator, a, b],
            coefficients=[1, 1, -1],
            metadata={'operation': 'tropical_add_compare', 'meaning': 'a <= b'}
        ))
        
        # Constraint 2: result = indicator * a + (1 - indicator) * b
        # Simplified: result = b + indicator * (a - b)
        circuit.add_constraint(TropicalConstraint(
            constraint_type='selection',
            variables=[result, indicator, a, b],
            coefficients=[1, 1, 1, 1],
            metadata={'operation': 'tropical_add_select'}
        ))
        
        return result, indicator
    
    @staticmethod
    def tropical_max(circuit: TropicalCircuit, a: str, b: str) -> Tuple[str, str]:
        """
        Maximum of two values (dual of tropical addition).
        
        Cost: 2 multiplicative constraints (same as min)
        
        Returns: (result_name, indicator_name)
        """
        indicator = circuit.new_variable("ind")
        result = circuit.new_variable("tmax")
        
        # indicator = 1 if a >= b
        circuit.add_constraint(TropicalConstraint(
            constraint_type='comparison',
            variables=[indicator, a, b],
            coefficients=[1, 1, -1],
            metadata={'operation': 'tropical_max_compare', 'meaning': 'a >= b'}
        ))
        
        # result = a if indicator else b
        circuit.add_constraint(TropicalConstraint(
            constraint_type='selection',
            variables=[result, indicator, a, b],
            coefficients=[1, 1, 1, 1],
            metadata={'operation': 'tropical_max_select'}
        ))
        
        return result, indicator


class TropicalMaxPool:
    """
    Efficient Max-Pooling using tropical tournament tree.
    
    Standard approach: k elements → k-1 pairwise max → (k-1) × 2 constraints
    Tournament tree: k elements → log₂(k) rounds → same constraints but parallelizable
    
    The key optimization is that we track the maximum efficiently
    without redundant comparisons.
    """
    
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
    
    def compile(self, circuit: TropicalCircuit, inputs: List[str]) -> Tuple[str, List[str]]:
        """
        Compile max-pooling to tropical circuit.
        
        Args:
            circuit: The tropical circuit to add constraints to
            inputs: List of input variable names
        
        Returns:
            (max_result, all_indicators): The maximum and comparison indicators
        """
        if len(inputs) == 0:
            raise ValueError("Cannot max-pool empty list")
        
        if len(inputs) == 1:
            return inputs[0], []
        
        # Tournament tree approach
        current_level = inputs.copy()
        all_indicators = []
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Compare pair
                    result, indicator = TropicalSemiring.tropical_max(
                        circuit, current_level[i], current_level[i + 1]
                    )
                    next_level.append(result)
                    all_indicators.append(indicator)
                else:
                    # Odd element, pass through
                    next_level.append(current_level[i])
            
            current_level = next_level
        
        return current_level[0], all_indicators
    
    def constraint_count(self, num_inputs: int) -> int:
        """
        Calculate the number of constraints for max-pooling.
        
        For n inputs: n-1 comparisons, each costing 2 constraints
        Total: 2(n-1) constraints
        """
        return 2 * (num_inputs - 1)
    
    @staticmethod
    def standard_constraint_count(num_inputs: int, bits_per_comparison: int = 20) -> int:
        """
        Calculate constraints for standard (non-tropical) max-pooling.
        
        Standard approach uses bit decomposition for each comparison.
        For n inputs: n-1 comparisons × bits_per_comparison
        """
        return (num_inputs - 1) * bits_per_comparison


class TropicalArgmax:
    """
    Efficient Argmax using tropical tournament tree with index tracking.
    
    Returns both the maximum value and its index.
    """
    
    def compile(self, circuit: TropicalCircuit, inputs: List[str]) -> Tuple[str, str, List[str]]:
        """
        Compile argmax to tropical circuit.
        
        Args:
            circuit: The tropical circuit
            inputs: List of input variable names
        
        Returns:
            (max_value, max_index, indicators)
        """
        if len(inputs) == 0:
            raise ValueError("Cannot argmax empty list")
        
        if len(inputs) == 1:
            index = circuit.new_variable("idx")
            # index = 0 (constant)
            circuit.add_constraint(TropicalConstraint(
                constraint_type='linear',
                variables=[index],
                coefficients=[1],
                constant=0
            ))
            return inputs[0], index, []
        
        # Tournament with index tracking
        # Each element is (value_var, index_var)
        current_level = []
        for i, inp in enumerate(inputs):
            idx_var = circuit.new_variable(f"idx_{i}")
            # Set index to constant i (this is a linear constraint)
            circuit.add_constraint(TropicalConstraint(
                constraint_type='linear',
                variables=[idx_var],
                coefficients=[1],
                constant=-i,
                metadata={'meaning': f'index = {i}'}
            ))
            current_level.append((inp, idx_var))
        
        all_indicators = []
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    val_a, idx_a = current_level[i]
                    val_b, idx_b = current_level[i + 1]
                    
                    # Compare values
                    result_val, indicator = TropicalSemiring.tropical_max(
                        circuit, val_a, val_b
                    )
                    all_indicators.append(indicator)
                    
                    # Select index based on comparison
                    result_idx = circuit.new_variable("sel_idx")
                    circuit.add_constraint(TropicalConstraint(
                        constraint_type='selection',
                        variables=[result_idx, indicator, idx_a, idx_b],
                        coefficients=[1, 1, 1, 1],
                        metadata={'operation': 'argmax_index_select'}
                    ))
                    
                    next_level.append((result_val, result_idx))
                else:
                    next_level.append(current_level[i])
            
            current_level = next_level
        
        final_val, final_idx = current_level[0]
        return final_val, final_idx, all_indicators
    
    def constraint_count(self, num_inputs: int) -> int:
        """
        Constraints for tropical argmax.
        
        For n inputs:
        - n-1 value comparisons × 2 = 2(n-1)
        - n-1 index selections × 1 = n-1
        - n index initializations × 0 = 0 (linear)
        Total: 3(n-1) constraints
        """
        return 3 * (num_inputs - 1)
    
    @staticmethod
    def standard_constraint_count(num_inputs: int, bits_per_comparison: int = 20) -> int:
        """
        Standard argmax constraint count.
        
        Requires comparing each element with current max and tracking index.
        """
        return (num_inputs - 1) * (bits_per_comparison + 1)


class TropicalSoftmax:
    """
    Tropical approximation of Softmax using Log-Sum-Exp.
    
    Standard Softmax: softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
    
    Log-Softmax: log_softmax(x)_i = x_i - log(Σ_j exp(x_j))
                                  = x_i - logsumexp(x)
    
    Tropical Approximation:
        logsumexp(x) ≈ max(x) + log(k)  (where k = number of elements)
    
    This is exact when all x_i are equal, and a good approximation
    when one element dominates.
    
    For a tighter bound:
        logsumexp(x) = max(x) + log(Σ_j exp(x_j - max(x)))
                     ≈ max(x) + correction
    
    where correction ∈ [0, log(k)]
    """
    
    def __init__(self, num_classes: int, use_correction: bool = True):
        self.num_classes = num_classes
        self.use_correction = use_correction
    
    def compile(self, circuit: TropicalCircuit, inputs: List[str]) -> Tuple[List[str], str]:
        """
        Compile tropical softmax approximation.
        
        Args:
            circuit: The tropical circuit
            inputs: List of logit variable names
        
        Returns:
            (log_softmax_outputs, max_value)
        """
        n = len(inputs)
        
        # Step 1: Find max using tropical max-pool
        max_pool = TropicalMaxPool(n)
        max_val, _ = max_pool.compile(circuit, inputs)
        
        # Step 2: Compute log_softmax_i = x_i - max(x) - correction
        # For simplicity, we use correction = log(n) as a constant
        correction = math.log(n)
        
        outputs = []
        for i, inp in enumerate(inputs):
            # output_i = input_i - max_val - correction
            # This is a linear constraint (no multiplicative cost)
            out = circuit.new_variable(f"lsm_{i}")
            circuit.add_constraint(TropicalConstraint(
                constraint_type='linear',
                variables=[out, inp, max_val],
                coefficients=[1, -1, 1],
                constant=correction,
                metadata={'operation': 'log_softmax', 'index': i}
            ))
            outputs.append(out)
        
        return outputs, max_val
    
    def constraint_count(self, num_inputs: int) -> int:
        """
        Constraints for tropical softmax.
        
        - Max computation: 2(n-1)
        - Subtraction: 0 (linear)
        Total: 2(n-1)
        """
        return 2 * (num_inputs - 1)
    
    @staticmethod
    def standard_constraint_count(num_inputs: int) -> int:
        """
        Standard softmax constraint count.
        
        Requires:
        - n exponentiations (expensive in ZK)
        - n-1 additions
        - 1 division
        - n multiplications
        
        Rough estimate: ~50 constraints per element
        """
        return num_inputs * 50


# =============================================================================
# BENCHMARK AND TESTING
# =============================================================================

def benchmark_operations():
    """
    Benchmark tropical vs standard constraint counts.
    """
    print("=" * 70)
    print("TROPICAL OPERATIONS BENCHMARK")
    print("=" * 70)
    
    test_sizes = [2, 4, 8, 16, 32, 64, 128]
    
    print("\n1. MAX-POOLING")
    print("-" * 70)
    print(f"{'Size':<10} {'Standard':<15} {'Tropical':<15} {'Reduction':<15}")
    print("-" * 70)
    
    for size in test_sizes:
        standard = TropicalMaxPool.standard_constraint_count(size)
        tropical = TropicalMaxPool(size).constraint_count(size)
        reduction = (1 - tropical / standard) * 100
        print(f"{size:<10} {standard:<15} {tropical:<15} {reduction:.1f}%")
    
    print("\n2. ARGMAX")
    print("-" * 70)
    print(f"{'Size':<10} {'Standard':<15} {'Tropical':<15} {'Reduction':<15}")
    print("-" * 70)
    
    argmax = TropicalArgmax()
    for size in test_sizes:
        standard = TropicalArgmax.standard_constraint_count(size)
        tropical = argmax.constraint_count(size)
        reduction = (1 - tropical / standard) * 100
        print(f"{size:<10} {standard:<15} {tropical:<15} {reduction:.1f}%")
    
    print("\n3. SOFTMAX")
    print("-" * 70)
    print(f"{'Size':<10} {'Standard':<15} {'Tropical':<15} {'Reduction':<15}")
    print("-" * 70)
    
    for size in test_sizes:
        standard = TropicalSoftmax.standard_constraint_count(size)
        tropical = TropicalSoftmax(size).constraint_count(size)
        reduction = (1 - tropical / standard) * 100
        print(f"{size:<10} {standard:<15} {tropical:<15} {reduction:.1f}%")
    
    print("\n" + "=" * 70)
    print("CIRCUIT COMPILATION TEST")
    print("=" * 70)
    
    # Test actual circuit compilation
    circuit = TropicalCircuit()
    
    # Create input variables
    inputs = [circuit.new_variable(f"x_{i}") for i in range(8)]
    for i, inp in enumerate(inputs):
        circuit.variables[inp].is_input = True
    
    # Compile max-pool
    max_pool = TropicalMaxPool(8)
    max_result, indicators = max_pool.compile(circuit, inputs)
    
    print(f"\nMax-Pool (8 inputs):")
    print(f"  Variables created: {circuit.var_counter}")
    print(f"  Constraints: {circuit.total_constraints()}")
    print(f"  Breakdown: {circuit.constraint_breakdown()}")
    
    # Compile argmax
    circuit2 = TropicalCircuit()
    inputs2 = [circuit2.new_variable(f"x_{i}") for i in range(8)]
    
    argmax = TropicalArgmax()
    max_val, max_idx, _ = argmax.compile(circuit2, inputs2)
    
    print(f"\nArgmax (8 inputs):")
    print(f"  Variables created: {circuit2.var_counter}")
    print(f"  Constraints: {circuit2.total_constraints()}")
    print(f"  Breakdown: {circuit2.constraint_breakdown()}")
    
    # Compile softmax
    circuit3 = TropicalCircuit()
    inputs3 = [circuit3.new_variable(f"x_{i}") for i in range(8)]
    
    softmax = TropicalSoftmax(8)
    outputs, _ = softmax.compile(circuit3, inputs3)
    
    print(f"\nSoftmax (8 inputs):")
    print(f"  Variables created: {circuit3.var_counter}")
    print(f"  Constraints: {circuit3.total_constraints()}")
    print(f"  Breakdown: {circuit3.constraint_breakdown()}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Tropical arithmetic provides significant constraint reductions for:

1. MAX-POOLING: ~90% reduction
   - Standard: 20 constraints per comparison (bit decomposition)
   - Tropical: 2 constraints per comparison (indicator + selection)

2. ARGMAX: ~85% reduction
   - Standard: 21 constraints per comparison
   - Tropical: 3 constraints per comparison

3. SOFTMAX: ~96% reduction
   - Standard: ~50 constraints per element (exp, div, etc.)
   - Tropical: ~2 constraints per element (just max computation)

These optimizations are most impactful for:
- CNNs with max-pooling layers
- Transformers with attention (softmax)
- Classification networks (argmax output)
""")


if __name__ == "__main__":
    benchmark_operations()
