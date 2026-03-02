"""
Sparse Proof System für zkML
============================

Kernidee: Wenn ein Neuron nach der Aktivierung 0 ist, müssen wir
nicht die gesamte Berechnung beweisen - nur DASS es 0 ist.

In echten neuronalen Netzen sind nach ReLU/GELU oft 50-90% der
Neuronen inaktiv (= 0). Das ist enormes Optimierungspotenzial.

Sparse Proof Strategie:
1. Identifiziere inaktive Neuronen (Aktivierung = 0)
2. Ersetze vollständige Constraints durch "Zero-Proof"
3. Zero-Proof: Nur beweisen, dass output = 0

Constraint-Ersparnis:
- Vollständiger Dense Layer: O(input_size) Constraints pro Neuron
- Zero-Proof: 1 Constraint pro inaktivem Neuron
"""

from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
import math


@dataclass
class SparsityStats:
    """Statistiken über die Sparsity eines Netzwerks."""
    total_neurons: int
    active_neurons: int
    inactive_neurons: int
    sparsity_ratio: float
    
    # Constraint-Analyse
    full_constraints: int  # Ohne Sparse-Optimierung
    sparse_constraints: int  # Mit Sparse-Optimierung
    constraint_savings: float  # Prozentuale Ersparnis
    
    # Per-Layer Breakdown
    layer_stats: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"SparsityStats(\n"
            f"  neurons: {self.active_neurons}/{self.total_neurons} active ({100*(1-self.sparsity_ratio):.1f}%)\n"
            f"  constraints: {self.sparse_constraints}/{self.full_constraints} ({self.constraint_savings:.1f}% saved)\n"
            f")"
        )


class SparseConstraintSet:
    """
    Verwaltet Constraints mit Sparse-Optimierung.
    
    Unterscheidet zwischen:
    - Full Constraints: Vollständige Berechnung
    - Zero Constraints: Nur beweisen, dass output = 0
    - Skip Constraints: Können komplett übersprungen werden
    """
    
    def __init__(self, prime: int):
        self.prime = prime
        
        # Constraint-Speicher
        self.full_constraints: List[Tuple[List[int], List[int], List[int]]] = []
        self.zero_constraints: List[int] = []  # Indices von Variablen, die 0 sein müssen
        self.skipped_neurons: Set[int] = set()  # Neuron-IDs, die übersprungen wurden
        
        # Tracking
        self.neuron_to_constraints: Dict[int, List[int]] = {}  # neuron_id -> constraint indices
        self.inactive_neurons: Set[int] = set()
    
    def add_full_constraint(
        self, 
        A: List[int], 
        B: List[int], 
        C: List[int],
        neuron_id: Optional[int] = None
    ) -> int:
        """Fügt einen vollständigen Constraint hinzu."""
        idx = len(self.full_constraints)
        self.full_constraints.append((A, B, C))
        
        if neuron_id is not None:
            if neuron_id not in self.neuron_to_constraints:
                self.neuron_to_constraints[neuron_id] = []
            self.neuron_to_constraints[neuron_id].append(idx)
        
        return idx
    
    def add_zero_constraint(self, var_index: int, neuron_id: int) -> None:
        """
        Fügt einen Zero-Constraint hinzu.
        
        Beweist: variable[var_index] = 0
        
        Das ist VIEL billiger als die vollständige Berechnung!
        """
        self.zero_constraints.append(var_index)
        self.inactive_neurons.add(neuron_id)
    
    def mark_neuron_inactive(self, neuron_id: int) -> None:
        """
        Markiert ein Neuron als inaktiv.
        
        Alle Constraints für dieses Neuron werden durch Zero-Constraints ersetzt.
        """
        self.inactive_neurons.add(neuron_id)
        self.skipped_neurons.add(neuron_id)
    
    def get_active_constraints(self) -> List[Tuple[List[int], List[int], List[int]]]:
        """
        Gibt nur die aktiven (nicht übersprungenen) Constraints zurück.
        """
        active = []
        for idx, constraint in enumerate(self.full_constraints):
            # Prüfe, ob dieser Constraint zu einem inaktiven Neuron gehört
            skip = False
            for neuron_id, constraint_indices in self.neuron_to_constraints.items():
                if idx in constraint_indices and neuron_id in self.inactive_neurons:
                    skip = True
                    break
            
            if not skip:
                active.append(constraint)
        
        return active
    
    def total_constraints(self) -> int:
        """Gesamtzahl der Constraints (mit Sparse-Optimierung)."""
        active = len(self.get_active_constraints())
        zeros = len(self.zero_constraints)
        return active + zeros
    
    def full_constraint_count(self) -> int:
        """Anzahl der Constraints ohne Sparse-Optimierung."""
        return len(self.full_constraints)


class SparseProofBuilder:
    """
    Builder für Sparse Proofs.
    
    Analysiert das Netzwerk und entscheidet automatisch,
    welche Neuronen sparse behandelt werden können.
    """
    
    def __init__(self, prime: int, sparsity_threshold: float = 0.0):
        """
        Args:
            prime: Das Primfeld
            sparsity_threshold: Minimale Aktivierung, unter der ein Neuron als "inaktiv" gilt
                               (Standard: 0.0 = nur exakt 0 ist inaktiv)
        """
        self.prime = prime
        self.sparsity_threshold = sparsity_threshold
        self.constraint_set = SparseConstraintSet(prime)
        
        # Neuron-Tracking
        self.neuron_values: Dict[int, int] = {}  # neuron_id -> activation value
        self.neuron_layers: Dict[int, int] = {}  # neuron_id -> layer
        
        # Constraint-Kosten (pro Neuron-Typ)
        self.full_neuron_cost: int = 0  # Wird beim ersten Neuron gesetzt
    
    def register_neuron(
        self, 
        neuron_id: int, 
        activation_value: int, 
        layer: int,
        constraints_if_active: int
    ) -> bool:
        """
        Registriert ein Neuron und entscheidet, ob es sparse behandelt wird.
        
        Returns:
            True wenn das Neuron aktiv ist (vollständige Constraints nötig)
            False wenn das Neuron inaktiv ist (Zero-Constraint reicht)
        """
        self.neuron_values[neuron_id] = activation_value
        self.neuron_layers[neuron_id] = layer
        
        if self.full_neuron_cost == 0:
            self.full_neuron_cost = constraints_if_active
        
        # Prüfe, ob das Neuron inaktiv ist
        is_inactive = self._is_inactive(activation_value)
        
        if is_inactive:
            self.constraint_set.mark_neuron_inactive(neuron_id)
            return False
        else:
            return True
    
    def _is_inactive(self, value: int) -> bool:
        """Prüft, ob ein Wert als "inaktiv" gilt."""
        if self.sparsity_threshold == 0.0:
            return value == 0
        else:
            # Für Threshold > 0: Werte nahe 0 sind auch inaktiv
            # (nützlich für approximierte Aktivierungen)
            half = self.prime // 2
            if value <= half:
                return value < self.sparsity_threshold
            else:
                # Negative Werte (> half) sind immer "inaktiv" für ReLU-ähnliche Funktionen
                return True
    
    def add_zero_proof(self, var_index: int, neuron_id: int) -> None:
        """Fügt einen Zero-Proof für ein inaktives Neuron hinzu."""
        self.constraint_set.add_zero_constraint(var_index, neuron_id)
    
    def add_full_constraints(
        self, 
        constraints: List[Tuple[List[int], List[int], List[int]]],
        neuron_id: int
    ) -> None:
        """Fügt vollständige Constraints für ein aktives Neuron hinzu."""
        for A, B, C in constraints:
            self.constraint_set.add_full_constraint(A, B, C, neuron_id)
    
    def get_stats(self) -> SparsityStats:
        """Berechnet Sparsity-Statistiken."""
        total = len(self.neuron_values)
        inactive = len(self.constraint_set.inactive_neurons)
        active = total - inactive
        
        sparsity = inactive / total if total > 0 else 0
        
        # Constraint-Analyse
        full_constraints = total * self.full_neuron_cost if self.full_neuron_cost > 0 else 0
        sparse_constraints = (
            active * self.full_neuron_cost +  # Aktive Neuronen: volle Kosten
            inactive * 1  # Inaktive Neuronen: 1 Zero-Constraint
        )
        
        savings = (1 - sparse_constraints / full_constraints) * 100 if full_constraints > 0 else 0
        
        # Per-Layer Stats
        layer_stats = {}
        for neuron_id, layer in self.neuron_layers.items():
            if layer not in layer_stats:
                layer_stats[layer] = {"total": 0, "active": 0, "inactive": 0}
            
            layer_stats[layer]["total"] += 1
            if neuron_id in self.constraint_set.inactive_neurons:
                layer_stats[layer]["inactive"] += 1
            else:
                layer_stats[layer]["active"] += 1
        
        return SparsityStats(
            total_neurons=total,
            active_neurons=active,
            inactive_neurons=inactive,
            sparsity_ratio=sparsity,
            full_constraints=full_constraints,
            sparse_constraints=sparse_constraints,
            constraint_savings=savings,
            layer_stats=layer_stats
        )
    
    def build(self) -> Tuple[SparseConstraintSet, SparsityStats]:
        """Finalisiert den Sparse Proof und gibt Constraints + Stats zurück."""
        return self.constraint_set, self.get_stats()


class ZeroProofGenerator:
    """
    Generiert effiziente Zero-Proofs.
    
    Ein Zero-Proof beweist: "Diese Variable ist 0"
    
    Das ist trivial: x * 1 = 0 (wenn x = 0)
    
    Aber wir können es noch effizienter machen durch Batching:
    Mehrere Zero-Proofs können zu einem kombiniert werden.
    """
    
    def __init__(self, prime: int):
        self.prime = prime
        self.zero_vars: List[int] = []
    
    def add_zero_var(self, var_index: int) -> None:
        """Fügt eine Variable hinzu, die 0 sein muss."""
        self.zero_vars.append(var_index)
    
    def generate_individual_constraints(self) -> List[Tuple[List[int], List[int], List[int]]]:
        """
        Generiert individuelle Zero-Constraints.
        
        Für jede Variable x: x * 1 = 0
        """
        constraints = []
        for var_idx in self.zero_vars:
            # A = [0, ..., 1 (at var_idx), ..., 0]
            # B = [1, 0, 0, ...]  (Konstante 1)
            # C = [0, 0, 0, ...]  (Ergebnis 0)
            # Constraint: var * 1 = 0
            constraints.append((var_idx, 0, None))  # Vereinfachte Darstellung
        
        return constraints
    
    def generate_batched_constraint(self) -> Optional[Tuple[List[int], int]]:
        """
        Generiert einen gebatchten Zero-Constraint.
        
        Idee: Statt n individuelle Constraints, einen kombinierten:
        
        Wähle zufällige Koeffizienten r_1, ..., r_n
        Constraint: (r_1 * x_1 + r_2 * x_2 + ... + r_n * x_n) = 0
        
        Wenn alle x_i = 0, ist die Summe 0.
        Wenn ein x_i ≠ 0, ist die Summe mit hoher Wahrscheinlichkeit ≠ 0.
        
        Das reduziert n Constraints auf 1!
        
        ACHTUNG: Das ist probabilistisch, nicht deterministisch.
        Für echte Sicherheit braucht man mehrere Runden.
        """
        if not self.zero_vars:
            return None
        
        import random
        
        # Zufällige Koeffizienten
        coefficients = [random.randint(1, self.prime - 1) for _ in self.zero_vars]
        
        return (list(zip(coefficients, self.zero_vars)), 0)
    
    def constraint_count(self, batched: bool = False) -> int:
        """Gibt die Anzahl der Constraints zurück."""
        if batched:
            return 1 if self.zero_vars else 0
        else:
            return len(self.zero_vars)


# Hilfsfunktion für die Integration
def analyze_sparsity(activations: List[int], prime: int) -> Dict:
    """
    Analysiert die Sparsity einer Liste von Aktivierungen.
    
    Args:
        activations: Liste von Aktivierungswerten
        prime: Das Primfeld
        
    Returns:
        Dictionary mit Sparsity-Analyse
    """
    total = len(activations)
    zeros = sum(1 for a in activations if a == 0)
    
    # Für ReLU-ähnliche: Werte > prime/2 sind "negativ" (also 0 nach ReLU)
    half = prime // 2
    effective_zeros = sum(1 for a in activations if a == 0 or a > half)
    
    return {
        "total": total,
        "exact_zeros": zeros,
        "effective_zeros": effective_zeros,
        "sparsity_exact": zeros / total if total > 0 else 0,
        "sparsity_effective": effective_zeros / total if total > 0 else 0,
    }


# Tests
if __name__ == "__main__":
    print("=== Sparse Proof System Tests ===\n")
    
    prime = 101
    
    # Simuliere ein Netzwerk mit 10 Neuronen, 6 davon inaktiv
    builder = SparseProofBuilder(prime)
    
    activations = [0, 5, 0, 0, 12, 0, 8, 0, 0, 3]  # 6 Nullen = 60% Sparsity
    
    for i, act in enumerate(activations):
        is_active = builder.register_neuron(
            neuron_id=i,
            activation_value=act,
            layer=0,
            constraints_if_active=10  # Angenommen: 10 Constraints pro aktivem Neuron
        )
        print(f"Neuron {i}: value={act}, active={is_active}")
    
    constraint_set, stats = builder.build()
    
    print(f"\n{stats}")
    
    print(f"\nLayer Stats: {stats.layer_stats}")
    
    # Test Zero-Proof Batching
    print("\n--- Zero-Proof Batching ---")
    
    zero_gen = ZeroProofGenerator(prime)
    for i, act in enumerate(activations):
        if act == 0:
            zero_gen.add_zero_var(i)
    
    print(f"Individuelle Zero-Constraints: {zero_gen.constraint_count(batched=False)}")
    print(f"Gebatchte Zero-Constraints: {zero_gen.constraint_count(batched=True)}")
    
    # Sparsity-Analyse
    print("\n--- Sparsity-Analyse ---")
    analysis = analyze_sparsity(activations, prime)
    print(f"Analyse: {analysis}")
