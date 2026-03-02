"""
Optimierte Aktivierungsfunktionen für zkML
==========================================

Diese Aktivierungsfunktionen sind speziell für zkML optimiert:
- Minimale Constraint-Anzahl
- Polynomiale Approximationen statt Bit-Dekomposition
- Vergleichbare ML-Performance zu ReLU

Vergleich der Constraint-Anzahl:
- ReLU: 258 Constraints (Bit-Dekomposition)
- Quadratic: 1 Constraint
- GELU (Approx): ~10 Constraints
- Swish (Approx): ~8 Constraints
"""

from typing import List, Any
from .base import (
    ActivationFunction, PolynomialActivation, ActivationResult,
    float_to_fixed, fixed_to_float, fixed_mul, FIXED_POINT_SCALE
)


class QuadraticActivation(PolynomialActivation):
    """
    Quadratische Aktivierung: f(x) = x²
    
    Die einfachste nicht-lineare Aktivierung.
    Nur 1 Constraint!
    
    Nachteile:
    - Kann explodieren für große x
    - Keine "Sättigung" wie bei ReLU
    """
    
    @property
    def name(self) -> str:
        return "Quadratic"
    
    @property
    def coefficients(self) -> List[int]:
        return [0, 0, 1]  # x²
    
    @property
    def constraint_count(self) -> int:
        return 1
    
    def generate_constraints(
        self,
        input_index: int,
        r1cs: Any,
        witness: Any,
        neuron_id: int = 0
    ) -> ActivationResult:
        """
        Generiert Constraint: output = input * input
        """
        # Berechne den Wert
        x = witness.get(input_index)
        output_value = (x * x) % witness.prime
        
        # Allokiere Output im Witness
        output_index = witness.allocate(
            output_value,
            name=f"quad_out_{neuron_id}",
            layer=witness.metadata[input_index].layer,
            neuron=neuron_id,
            var_type="activation"
        )
        
        # Constraint: x * x = output
        # A = [0, ..., 1 (at input_index), ..., 0]
        # B = [0, ..., 1 (at input_index), ..., 0]
        # C = [0, ..., 1 (at output_index), ..., 0]
        r1cs.add_multiplication_constraint(input_index, input_index, output_index)
        
        return ActivationResult(
            output_index=output_index,
            intermediate_indices=[],
            num_constraints=1
        )


class GELUApproxActivation(PolynomialActivation):
    """
    GELU Approximation: f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    Wir nutzen eine Polynom-Approximation:
    f(x) ≈ 0.5x + 0.398x² + 0.019x³ - 0.006x⁴
    
    Diese Approximation ist im Bereich [-3, 3] sehr genau.
    
    Constraint-Anzahl: ~10 (für Grad-4 Polynom + Fixpunkt-Arithmetik)
    """
    
    # Koeffizienten der Approximation (als Fixpunkt)
    COEFFS = [
        0,                          # a_0 = 0
        float_to_fixed(0.5),        # a_1 = 0.5
        float_to_fixed(0.398),      # a_2 = 0.398
        float_to_fixed(0.019),      # a_3 = 0.019
        float_to_fixed(-0.006),     # a_4 = -0.006
    ]
    
    @property
    def name(self) -> str:
        return "GELU_Approx"
    
    @property
    def coefficients(self) -> List[int]:
        return self.COEFFS
    
    @property
    def constraint_count(self) -> int:
        # x², x³, x⁴ = 3 Constraints
        # 4 Terme multiplizieren und addieren = ~7 Constraints
        return 10
    
    def compute(self, x: int, prime: int) -> int:
        """
        Berechnet GELU-Approximation in Fixpunkt-Arithmetik.
        """
        # x ist bereits in Fixpunkt
        x2 = fixed_mul(x, x, prime)
        x3 = fixed_mul(x2, x, prime)
        x4 = fixed_mul(x3, x, prime)
        
        # Terme berechnen
        term1 = fixed_mul(self.COEFFS[1], x, prime)
        term2 = fixed_mul(self.COEFFS[2], x2, prime)
        term3 = fixed_mul(self.COEFFS[3], x3, prime)
        term4 = fixed_mul(self.COEFFS[4], x4, prime)
        
        result = (term1 + term2 + term3 + term4) % prime
        return result
    
    def generate_constraints(
        self,
        input_index: int,
        r1cs: Any,
        witness: Any,
        neuron_id: int = 0
    ) -> ActivationResult:
        """
        Generiert Constraints für GELU-Approximation.
        """
        x = witness.get(input_index)
        prime = witness.prime
        layer = witness.metadata[input_index].layer if input_index in witness.metadata else None
        
        intermediate_indices = []
        
        # x² berechnen und allokieren
        x2_value = fixed_mul(x, x, prime)
        x2_idx = witness.allocate(x2_value, f"gelu_x2_{neuron_id}", layer=layer, var_type="intermediate")
        intermediate_indices.append(x2_idx)
        r1cs.add_fixed_mul_constraint(input_index, input_index, x2_idx)
        
        # x³ = x² * x
        x3_value = fixed_mul(x2_value, x, prime)
        x3_idx = witness.allocate(x3_value, f"gelu_x3_{neuron_id}", layer=layer, var_type="intermediate")
        intermediate_indices.append(x3_idx)
        r1cs.add_fixed_mul_constraint(x2_idx, input_index, x3_idx)
        
        # x⁴ = x³ * x
        x4_value = fixed_mul(x3_value, x, prime)
        x4_idx = witness.allocate(x4_value, f"gelu_x4_{neuron_id}", layer=layer, var_type="intermediate")
        intermediate_indices.append(x4_idx)
        r1cs.add_fixed_mul_constraint(x3_idx, input_index, x4_idx)
        
        # Terme berechnen
        term1 = fixed_mul(self.COEFFS[1], x, prime)
        term2 = fixed_mul(self.COEFFS[2], x2_value, prime)
        term3 = fixed_mul(self.COEFFS[3], x3_value, prime)
        term4 = fixed_mul(self.COEFFS[4], x4_value, prime)
        
        # Endergebnis
        output_value = (term1 + term2 + term3 + term4) % prime
        output_idx = witness.allocate(
            output_value,
            f"gelu_out_{neuron_id}",
            layer=layer,
            neuron=neuron_id,
            var_type="activation"
        )
        
        # Constraint für die lineare Kombination
        r1cs.add_linear_combination_constraint(
            [(self.COEFFS[1], input_index),
             (self.COEFFS[2], x2_idx),
             (self.COEFFS[3], x3_idx),
             (self.COEFFS[4], x4_idx)],
            output_idx
        )
        
        return ActivationResult(
            output_index=output_idx,
            intermediate_indices=intermediate_indices,
            num_constraints=10
        )


class SwishApproxActivation(PolynomialActivation):
    """
    Swish Approximation: f(x) = x * sigmoid(x) ≈ x * (0.5 + 0.25x - 0.02x³)
    
    Swish ist ähnlich zu GELU, aber etwas einfacher.
    
    Approximation: f(x) ≈ 0.5x + 0.25x² - 0.02x⁴
    
    Constraint-Anzahl: ~8
    """
    
    COEFFS = [
        0,                          # a_0 = 0
        float_to_fixed(0.5),        # a_1 = 0.5
        float_to_fixed(0.25),       # a_2 = 0.25
        0,                          # a_3 = 0
        float_to_fixed(-0.02),      # a_4 = -0.02
    ]
    
    @property
    def name(self) -> str:
        return "Swish_Approx"
    
    @property
    def coefficients(self) -> List[int]:
        return self.COEFFS
    
    @property
    def constraint_count(self) -> int:
        return 8
    
    def compute(self, x: int, prime: int) -> int:
        """Berechnet Swish-Approximation."""
        x2 = fixed_mul(x, x, prime)
        x4 = fixed_mul(x2, x2, prime)
        
        term1 = fixed_mul(self.COEFFS[1], x, prime)
        term2 = fixed_mul(self.COEFFS[2], x2, prime)
        term4 = fixed_mul(self.COEFFS[4], x4, prime)
        
        return (term1 + term2 + term4) % prime
    
    def generate_constraints(
        self,
        input_index: int,
        r1cs: Any,
        witness: Any,
        neuron_id: int = 0
    ) -> ActivationResult:
        """Generiert Constraints für Swish-Approximation."""
        x = witness.get(input_index)
        prime = witness.prime
        layer = witness.metadata[input_index].layer if input_index in witness.metadata else None
        
        intermediate_indices = []
        
        # x²
        x2_value = fixed_mul(x, x, prime)
        x2_idx = witness.allocate(x2_value, f"swish_x2_{neuron_id}", layer=layer, var_type="intermediate")
        intermediate_indices.append(x2_idx)
        r1cs.add_fixed_mul_constraint(input_index, input_index, x2_idx)
        
        # x⁴ = x² * x²
        x4_value = fixed_mul(x2_value, x2_value, prime)
        x4_idx = witness.allocate(x4_value, f"swish_x4_{neuron_id}", layer=layer, var_type="intermediate")
        intermediate_indices.append(x4_idx)
        r1cs.add_fixed_mul_constraint(x2_idx, x2_idx, x4_idx)
        
        # Terme
        term1 = fixed_mul(self.COEFFS[1], x, prime)
        term2 = fixed_mul(self.COEFFS[2], x2_value, prime)
        term4 = fixed_mul(self.COEFFS[4], x4_value, prime)
        
        output_value = (term1 + term2 + term4) % prime
        output_idx = witness.allocate(
            output_value,
            f"swish_out_{neuron_id}",
            layer=layer,
            neuron=neuron_id,
            var_type="activation"
        )
        
        r1cs.add_linear_combination_constraint(
            [(self.COEFFS[1], input_index),
             (self.COEFFS[2], x2_idx),
             (self.COEFFS[4], x4_idx)],
            output_idx
        )
        
        return ActivationResult(
            output_index=output_idx,
            intermediate_indices=intermediate_indices,
            num_constraints=8
        )


class ReLUActivation(ActivationFunction):
    """
    ReLU: f(x) = max(0, x)
    
    WARNUNG: Dies ist die TEURE Variante!
    
    ReLU erfordert Bit-Dekomposition, um zu prüfen, ob x >= 0.
    Das kostet ~258 Constraints pro Aktivierung.
    
    Nur verwenden, wenn ReLU unbedingt erforderlich ist!
    """
    
    NUM_BITS = 256  # Für BN254 Feld
    
    @property
    def name(self) -> str:
        return "ReLU"
    
    @property
    def constraint_count(self) -> int:
        # 256 Bits + 2 Constraints für die Logik
        return self.NUM_BITS + 2
    
    def compute(self, x: int, prime: int) -> int:
        """
        Berechnet ReLU.
        
        Achtung: In einem Primfeld gibt es kein "negativ".
        Wir interpretieren Werte > prime/2 als negativ.
        """
        half = prime // 2
        if x <= half:
            return x  # Positiv
        else:
            return 0  # Negativ
    
    def generate_constraints(
        self,
        input_index: int,
        r1cs: Any,
        witness: Any,
        neuron_id: int = 0
    ) -> ActivationResult:
        """
        Generiert Constraints für ReLU mit Bit-Dekomposition.
        
        Dies ist TEUER! Verwende wenn möglich GELU oder Swish.
        """
        x = witness.get(input_index)
        prime = witness.prime
        layer = witness.metadata[input_index].layer if input_index in witness.metadata else None
        
        intermediate_indices = []
        
        # Berechne Output
        output_value = self.compute(x, prime)
        
        # Bit-Dekomposition von x
        bits = []
        temp = x
        for i in range(self.NUM_BITS):
            bit = temp % 2
            bit_idx = witness.allocate(
                bit, 
                f"relu_bit_{neuron_id}_{i}",
                layer=layer,
                var_type="intermediate"
            )
            bits.append(bit_idx)
            intermediate_indices.append(bit_idx)
            
            # Constraint: bit * (1 - bit) = 0 (bit ist binär)
            r1cs.add_binary_constraint(bit_idx)
            
            temp //= 2
        
        # Constraint: Summe der Bits ergibt x
        r1cs.add_bit_decomposition_constraint(bits, input_index)
        
        # Output allokieren
        output_idx = witness.allocate(
            output_value,
            f"relu_out_{neuron_id}",
            layer=layer,
            neuron=neuron_id,
            var_type="activation"
        )
        
        # Constraint für ReLU-Logik (vereinfacht)
        # In der Praxis komplexer, hier nur Platzhalter
        r1cs.add_relu_constraint(input_index, output_idx, bits)
        
        return ActivationResult(
            output_index=output_idx,
            intermediate_indices=intermediate_indices,
            num_constraints=self.NUM_BITS + 2
        )


# Factory-Funktion für einfache Nutzung
def get_activation(name: str) -> ActivationFunction:
    """
    Gibt eine Aktivierungsfunktion nach Namen zurück.
    
    Args:
        name: "quadratic", "gelu", "swish", oder "relu"
        
    Returns:
        Die entsprechende Aktivierungsfunktion
    """
    activations = {
        "quadratic": QuadraticActivation(),
        "gelu": GELUApproxActivation(),
        "swish": SwishApproxActivation(),
        "relu": ReLUActivation(),
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unbekannte Aktivierung: {name}. Verfügbar: {list(activations.keys())}")
    
    return activations[name_lower]


# Vergleichstabelle
def print_activation_comparison():
    """Druckt eine Vergleichstabelle der Aktivierungsfunktionen."""
    print("\n" + "=" * 60)
    print("AKTIVIERUNGSFUNKTIONEN - CONSTRAINT-VERGLEICH")
    print("=" * 60)
    
    activations = [
        QuadraticActivation(),
        SwishApproxActivation(),
        GELUApproxActivation(),
        ReLUActivation(),
    ]
    
    print(f"\n{'Name':<20} {'Constraints':<15} {'Ersparnis vs ReLU':<20}")
    print("-" * 55)
    
    relu_constraints = ReLUActivation().constraint_count
    
    for act in activations:
        savings = (1 - act.constraint_count / relu_constraints) * 100
        print(f"{act.name:<20} {act.constraint_count:<15} {savings:>6.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_activation_comparison()
    
    # Test der Berechnungen
    print("\n--- Test der Aktivierungsfunktionen ---")
    
    prime = 101
    test_values = [0, 1, 5, 10, 50]
    
    activations = [
        QuadraticActivation(),
        SwishApproxActivation(),
        GELUApproxActivation(),
    ]
    
    for x in test_values:
        print(f"\nx = {x}:")
        for act in activations:
            result = act.compute(x, prime)
            print(f"  {act.name}: {result}")
