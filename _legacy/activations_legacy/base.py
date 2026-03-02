"""
Basis-Interface für Aktivierungsfunktionen in zkML
==================================================

Jede Aktivierungsfunktion muss zwei Dinge können:
1. Den Wert berechnen (für den Witness)
2. Constraints generieren (für das R1CS)

Die Constraint-Anzahl ist der kritische Faktor für zkML-Performance.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ActivationResult:
    """Ergebnis einer Aktivierungsfunktion."""
    output_index: int  # Index des Ausgabewerts im Witness
    intermediate_indices: List[int]  # Indices der Zwischenwerte
    num_constraints: int  # Anzahl der generierten Constraints


class ActivationFunction(ABC):
    """
    Abstrakte Basisklasse für Aktivierungsfunktionen.
    
    Jede Aktivierungsfunktion muss implementieren:
    - compute(): Berechnet den Ausgabewert
    - generate_constraints(): Generiert R1CS-Constraints
    - get_constraint_count(): Gibt die Anzahl der Constraints zurück
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name der Aktivierungsfunktion."""
        pass
    
    @property
    @abstractmethod
    def constraint_count(self) -> int:
        """Anzahl der Constraints pro Aktivierung."""
        pass
    
    @abstractmethod
    def compute(self, x: int, prime: int) -> int:
        """
        Berechnet den Ausgabewert der Aktivierungsfunktion.
        
        Args:
            x: Eingabewert (im Feld)
            prime: Das Primfeld
            
        Returns:
            Der Ausgabewert (im Feld)
        """
        pass
    
    @abstractmethod
    def generate_constraints(
        self,
        input_index: int,
        r1cs: Any,  # R1CS Objekt
        witness: Any,  # Witness Objekt
        neuron_id: int = 0
    ) -> ActivationResult:
        """
        Generiert die R1CS-Constraints für diese Aktivierung.
        
        Args:
            input_index: Index des Eingabewerts im Witness
            r1cs: Das R1CS-System
            witness: Der Witness
            neuron_id: ID des Neurons (für Sparse-Tracking)
            
        Returns:
            ActivationResult mit Output-Index und Constraint-Info
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}(constraints={self.constraint_count})"


class PolynomialActivation(ActivationFunction):
    """
    Basisklasse für polynomiale Aktivierungsfunktionen.
    
    Polynome sind besonders effizient in R1CS, da sie nur
    Multiplikationen und Additionen benötigen.
    """
    
    @property
    @abstractmethod
    def coefficients(self) -> List[int]:
        """
        Koeffizienten des Polynoms.
        
        Für p(x) = a_0 + a_1*x + a_2*x² + ...
        Gibt [a_0, a_1, a_2, ...] zurück.
        """
        pass
    
    @property
    def degree(self) -> int:
        """Grad des Polynoms."""
        return len(self.coefficients) - 1
    
    def compute(self, x: int, prime: int) -> int:
        """Berechnet das Polynom an der Stelle x."""
        result = 0
        x_power = 1
        for coeff in self.coefficients:
            result = (result + coeff * x_power) % prime
            x_power = (x_power * x) % prime
        return result
    
    @property
    def constraint_count(self) -> int:
        """
        Anzahl der Constraints für ein Polynom vom Grad n.
        
        Wir brauchen:
        - (n-1) Constraints für x², x³, ..., x^n
        - 1 Constraint für das Endergebnis (wenn nötig)
        """
        return max(0, self.degree - 1) + 1


# Hilfsfunktionen für Fixpunkt-Arithmetik

FIXED_POINT_SCALE = 2**16  # 16 Bit Nachkommastellen


def float_to_fixed(x: float) -> int:
    """Konvertiert Float zu Fixpunkt."""
    return int(x * FIXED_POINT_SCALE)


def fixed_to_float(x: int) -> float:
    """Konvertiert Fixpunkt zu Float."""
    return x / FIXED_POINT_SCALE


def fixed_mul(a: int, b: int, prime: int) -> int:
    """Multiplikation in Fixpunkt-Arithmetik."""
    # a * b / SCALE
    product = (a * b) % prime
    # Division durch SCALE (Multiplikation mit Inversem)
    scale_inv = pow(FIXED_POINT_SCALE, prime - 2, prime)
    return (product * scale_inv) % prime
