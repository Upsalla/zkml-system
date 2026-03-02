"""
R1CS (Rank-1 Constraint System) für zkML
========================================

R1CS ist das Standard-Format für ZK-Proofs. Jede Berechnung wird als
eine Menge von Constraints der Form (A * B = C) dargestellt.

Mathematische Definition:
- A, B, C sind Linearkombinationen von Witness-Werten
- Eine Constraint ist erfüllt, wenn A(w) * B(w) = C(w) für den Witness w

Beispiel:
    Berechnung: z = x * y
    Constraint: x * y = z
    
    Mit Witness w = [1, x, y, z]:
    A = [0, 1, 0, 0]  (wählt x)
    B = [0, 0, 1, 0]  (wählt y)
    C = [0, 0, 0, 1]  (wählt z)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from .field import FieldElement, FieldConfig, FIELD_DEV, field_mul, field_add


@dataclass
class LinearCombination:
    """
    Eine Linearkombination von Witness-Werten.
    
    Repräsentiert: Σ (coefficient_i * witness[index_i])
    
    Beispiel:
        3*w[0] + 2*w[1] - w[3]
        = LinearCombination({0: 3, 1: 2, 3: -1})
    """
    terms: Dict[int, int] = field(default_factory=dict)
    
    def evaluate(self, witness: List[int], prime: int) -> int:
        """
        Wertet die Linearkombination für einen gegebenen Witness aus.
        
        Args:
            witness: Liste der Witness-Werte
            prime: Das Primfeld
            
        Returns:
            Der Wert der Linearkombination mod prime
        """
        result = 0
        for idx, coeff in self.terms.items():
            result = field_add(result, field_mul(coeff, witness[idx], prime), prime)
        return result
    
    def add_term(self, index: int, coefficient: int) -> 'LinearCombination':
        """Fügt einen Term hinzu oder aktualisiert ihn."""
        new_terms = self.terms.copy()
        if index in new_terms:
            new_terms[index] = (new_terms[index] + coefficient)
        else:
            new_terms[index] = coefficient
        return LinearCombination(new_terms)
    
    @classmethod
    def single(cls, index: int, coefficient: int = 1) -> 'LinearCombination':
        """Erstellt eine Linearkombination mit einem einzelnen Term."""
        return cls({index: coefficient})
    
    @classmethod
    def constant(cls, value: int) -> 'LinearCombination':
        """
        Erstellt eine Konstante (nutzt Index 0, der immer 1 ist).
        
        Konvention: witness[0] = 1 (Konstante)
        """
        return cls({0: value})
    
    @classmethod
    def zero(cls) -> 'LinearCombination':
        """Erstellt die Null-Linearkombination."""
        return cls({})
    
    def __add__(self, other: 'LinearCombination') -> 'LinearCombination':
        """Addition zweier Linearkombinationen."""
        new_terms = self.terms.copy()
        for idx, coeff in other.terms.items():
            if idx in new_terms:
                new_terms[idx] = new_terms[idx] + coeff
            else:
                new_terms[idx] = coeff
        return LinearCombination(new_terms)
    
    def __sub__(self, other: 'LinearCombination') -> 'LinearCombination':
        """Subtraktion zweier Linearkombinationen."""
        new_terms = self.terms.copy()
        for idx, coeff in other.terms.items():
            if idx in new_terms:
                new_terms[idx] = new_terms[idx] - coeff
            else:
                new_terms[idx] = -coeff
        return LinearCombination(new_terms)
    
    def __mul__(self, scalar: int) -> 'LinearCombination':
        """Skalarmultiplikation."""
        return LinearCombination({idx: coeff * scalar for idx, coeff in self.terms.items()})
    
    def __rmul__(self, scalar: int) -> 'LinearCombination':
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for idx, coeff in sorted(self.terms.items()):
            if coeff == 1:
                parts.append(f"w[{idx}]")
            elif coeff == -1:
                parts.append(f"-w[{idx}]")
            else:
                parts.append(f"{coeff}*w[{idx}]")
        return " + ".join(parts).replace(" + -", " - ")


@dataclass
class R1CSConstraint:
    """
    Eine einzelne R1CS-Constraint: A * B = C
    
    Attributes:
        a: Linke Seite der Multiplikation
        b: Rechte Seite der Multiplikation
        c: Ergebnis
        description: Menschenlesbare Beschreibung (für Debugging)
        neuron_id: Optional - welches Neuron diese Constraint betrifft (für Sparse)
    """
    a: LinearCombination
    b: LinearCombination
    c: LinearCombination
    description: str = ""
    neuron_id: Optional[int] = None
    
    def is_satisfied(self, witness: List[int], prime: int) -> bool:
        """
        Prüft, ob die Constraint für den gegebenen Witness erfüllt ist.
        
        Returns:
            True wenn A(w) * B(w) = C(w) mod prime
        """
        a_val = self.a.evaluate(witness, prime)
        b_val = self.b.evaluate(witness, prime)
        c_val = self.c.evaluate(witness, prime)
        
        return field_mul(a_val, b_val, prime) == c_val
    
    def get_violation(self, witness: List[int], prime: int) -> int:
        """
        Berechnet die "Verletzung" der Constraint.
        
        Returns:
            0 wenn erfüllt, sonst (A*B - C) mod prime
        """
        a_val = self.a.evaluate(witness, prime)
        b_val = self.b.evaluate(witness, prime)
        c_val = self.c.evaluate(witness, prime)
        
        return (field_mul(a_val, b_val, prime) - c_val) % prime


@dataclass
class R1CS:
    """
    Ein vollständiges R1CS-System.
    
    Attributes:
        constraints: Liste aller Constraints
        num_variables: Anzahl der Variablen im Witness
        num_public: Anzahl der öffentlichen Eingaben
        field: Das verwendete Primfeld
    """
    constraints: List[R1CSConstraint] = field(default_factory=list)
    num_variables: int = 0
    num_public: int = 0
    prime: int = FIELD_DEV.prime
    
    # Variablen-Tracking
    _variable_names: Dict[int, str] = field(default_factory=dict)
    _next_variable: int = 1  # 0 ist reserviert für die Konstante 1
    
    def __post_init__(self):
        # Variable 0 ist immer die Konstante 1
        self._variable_names[0] = "ONE"
    
    def allocate_variable(self, name: str = "") -> int:
        """
        Allokiert eine neue Variable im Witness.
        
        Returns:
            Der Index der neuen Variable
        """
        idx = self._next_variable
        self._next_variable += 1
        self.num_variables = self._next_variable
        if name:
            self._variable_names[idx] = name
        return idx
    
    def allocate_public_input(self, name: str = "") -> int:
        """
        Allokiert eine öffentliche Eingabe.
        
        Öffentliche Eingaben kommen direkt nach der Konstante 1.
        """
        idx = self.allocate_variable(name)
        self.num_public += 1
        return idx
    
    def add_constraint(
        self, 
        a: LinearCombination, 
        b: LinearCombination, 
        c: LinearCombination,
        description: str = "",
        neuron_id: Optional[int] = None
    ) -> None:
        """Fügt eine neue Constraint hinzu."""
        self.constraints.append(R1CSConstraint(a, b, c, description, neuron_id))
    
    def add_multiplication_constraint(
        self,
        left_idx: int,
        right_idx: int,
        result_idx: int,
        description: str = ""
    ) -> None:
        """
        Fügt eine einfache Multiplikations-Constraint hinzu.
        
        Constraint: witness[left_idx] * witness[right_idx] = witness[result_idx]
        """
        a = LinearCombination.single(left_idx)
        b = LinearCombination.single(right_idx)
        c = LinearCombination.single(result_idx)
        self.add_constraint(a, b, c, description)
    
    def verify(self, witness: List[int]) -> Tuple[bool, List[int]]:
        """
        Verifiziert, ob der Witness alle Constraints erfüllt.
        
        Returns:
            (all_satisfied, list_of_violated_constraint_indices)
        """
        violated = []
        for i, constraint in enumerate(self.constraints):
            if not constraint.is_satisfied(witness, self.prime):
                violated.append(i)
        return len(violated) == 0, violated
    
    def check_witness(self, witness: List[int]) -> bool:
        """Prüft, ob der Witness alle Constraints erfüllt."""
        is_valid, _ = self.verify(witness)
        return is_valid
    
    def num_constraints(self) -> int:
        """Gibt die Anzahl der Constraints zurück."""
        return len(self.constraints)
    
    def get_statistics(self) -> Dict[str, int]:
        """Gibt Statistiken über das R1CS-System zurück."""
        return {
            "num_constraints": len(self.constraints),
            "num_variables": self.num_variables,
            "num_public": self.num_public,
            "num_private": self.num_variables - self.num_public - 1,  # -1 für Konstante
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"R1CS(constraints={stats['num_constraints']}, "
            f"variables={stats['num_variables']}, "
            f"public={stats['num_public']})"
        )


class R1CSBuilder:
    """
    Builder-Klasse für die einfache Erstellung von R1CS-Systemen.
    
    Bietet High-Level-Methoden für häufige Operationen.
    """
    
    def __init__(self, prime: int = FIELD_DEV.prime):
        self.r1cs = R1CS(prime=prime)
        self.witness_values: Dict[int, int] = {0: 1}  # Konstante 1
    
    def input(self, name: str, value: int, public: bool = False) -> int:
        """Erstellt eine Eingabevariable."""
        if public:
            idx = self.r1cs.allocate_public_input(name)
        else:
            idx = self.r1cs.allocate_variable(name)
        self.witness_values[idx] = value % self.r1cs.prime
        return idx
    
    def constant(self, value: int) -> LinearCombination:
        """Erstellt eine Konstante als Linearkombination."""
        return LinearCombination.constant(value)
    
    def var(self, idx: int) -> LinearCombination:
        """Erstellt eine Linearkombination für eine einzelne Variable."""
        return LinearCombination.single(idx)
    
    def mul(self, left_idx: int, right_idx: int, name: str = "") -> int:
        """
        Multipliziert zwei Variablen und gibt den Index des Ergebnisses zurück.
        
        Fügt automatisch die entsprechende Constraint hinzu.
        """
        result_idx = self.r1cs.allocate_variable(name)
        
        # Berechne den Wert
        left_val = self.witness_values.get(left_idx, 0)
        right_val = self.witness_values.get(right_idx, 0)
        result_val = field_mul(left_val, right_val, self.r1cs.prime)
        self.witness_values[result_idx] = result_val
        
        # Füge Constraint hinzu
        self.r1cs.add_multiplication_constraint(left_idx, right_idx, result_idx, name)
        
        return result_idx
    
    def add(self, *indices: int, name: str = "") -> int:
        """
        Addiert mehrere Variablen.
        
        Addition braucht keine Constraint, nur eine neue Variable.
        """
        result_idx = self.r1cs.allocate_variable(name)
        
        # Berechne den Wert
        result_val = sum(self.witness_values.get(idx, 0) for idx in indices) % self.r1cs.prime
        self.witness_values[result_idx] = result_val
        
        return result_idx
    
    def square(self, idx: int, name: str = "") -> int:
        """Quadriert eine Variable (x * x)."""
        return self.mul(idx, idx, name)
    
    def get_witness(self) -> List[int]:
        """Gibt den Witness als Liste zurück."""
        max_idx = max(self.witness_values.keys())
        return [self.witness_values.get(i, 0) for i in range(max_idx + 1)]
    
    def add_constraint(
        self,
        a: LinearCombination,
        b: LinearCombination,
        c: LinearCombination,
        description: str = "",
        neuron_id: Optional[int] = None
    ) -> None:
        """Fügt eine Constraint direkt hinzu."""
        self.r1cs.add_constraint(a, b, c, description, neuron_id)
    
    def build(self, num_vars: Optional[int] = None) -> 'R1CS':
        """Gibt das fertige R1CS-System zurück."""
        if num_vars is not None:
            self.r1cs.num_variables = num_vars
        return self.r1cs
    
    def build_with_witness(self) -> Tuple[R1CS, List[int]]:
        """Gibt das fertige R1CS-System und den Witness zurück."""
        return self.r1cs, self.get_witness()


# Tests
if __name__ == "__main__":
    print("=== R1CS Tests ===\n")
    
    # Einfaches Beispiel: z = x * y
    builder = R1CSBuilder()
    
    x = builder.input("x", 3, public=True)
    y = builder.input("y", 4, public=True)
    z = builder.mul(x, y, "z = x * y")
    
    r1cs, witness = builder.build()
    
    print(f"R1CS: {r1cs}")
    print(f"Witness: {witness}")
    print(f"Statistics: {r1cs.get_statistics()}")
    
    # Verifizieren
    is_valid, violated = r1cs.verify(witness)
    print(f"Valid: {is_valid}")
    
    # Komplexeres Beispiel: z = x² + y²
    print("\n--- Komplexeres Beispiel: z = x² + y² ---")
    
    builder2 = R1CSBuilder()
    
    x2 = builder2.input("x", 3)
    y2 = builder2.input("y", 4)
    x_sq = builder2.square(x2, "x²")
    y_sq = builder2.square(y2, "y²")
    # Addition braucht keine Constraint, aber wir können das Ergebnis tracken
    
    r1cs2, witness2 = builder2.build()
    
    print(f"R1CS: {r1cs2}")
    print(f"Witness: {witness2}")
    print(f"x² = {witness2[x_sq]}, y² = {witness2[y_sq]}")
    
    is_valid2, violated2 = r1cs2.verify(witness2)
    print(f"Valid: {is_valid2}")
