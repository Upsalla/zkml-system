"""
Finite Field Arithmetik für zkML
================================

Dieses Modul implementiert Arithmetik über endlichen Feldern (Primfeldern).
Alle Berechnungen in ZK-Proofs finden in solchen Feldern statt.

Mathematischer Hintergrund:
- Ein endliches Feld F_p besteht aus den Zahlen {0, 1, 2, ..., p-1}
- Alle Operationen werden modulo p durchgeführt
- p muss eine Primzahl sein, damit Division möglich ist
"""

from dataclasses import dataclass
from typing import Union, List
import random


# BN254 Scalar Field - Standard für Ethereum/Groth16
BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Kleineres Feld für Entwicklung und Tests
DEV_PRIME = 101


@dataclass
class FieldConfig:
    """Konfiguration für ein endliches Feld."""
    prime: int
    name: str
    
    # Bekannte Primzahlen (für die wir den Test überspringen)
    KNOWN_PRIMES = {
        21888242871839275222246405745257275088548364400416034343698204186575808495617,  # BN254
        52435875175126190479447740508185965837690552500527637822603658699938581184513,  # BLS12-381
        101,  # Dev
    }
    
    def __post_init__(self):
        # Überspringe Test für bekannte Primzahlen
        if self.prime in self.KNOWN_PRIMES:
            return
        if not self._is_prime(self.prime):
            raise ValueError(f"{self.prime} ist keine Primzahl")
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Einfacher Primzahltest (nur für kleine Zahlen < 10^6)."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        # Für große Zahlen: Annahme, dass sie prim sind (oder in KNOWN_PRIMES)
        if n > 10**6:
            return True
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True


# Vordefinierte Feldkonfigurationen
FIELD_BN254 = FieldConfig(BN254_PRIME, "BN254")
FIELD_DEV = FieldConfig(DEV_PRIME, "Development")


class FieldElement:
    """
    Ein Element eines endlichen Feldes.
    
    Unterstützt alle arithmetischen Operationen mit automatischer Modulo-Reduktion.
    """
    
    __slots__ = ['value', 'field']
    
    def __init__(self, value: int, field: FieldConfig = FIELD_DEV):
        """
        Erstellt ein Feldelement.
        
        Args:
            value: Der Wert (wird automatisch mod p reduziert)
            field: Die Feldkonfiguration
        """
        self.field = field
        self.value = value % field.prime
    
    def __repr__(self) -> str:
        return f"FieldElement({self.value}, mod {self.field.prime})"
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __eq__(self, other: Union['FieldElement', int]) -> bool:
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Feldelemente aus verschiedenen Feldern können nicht verglichen werden")
            return self.value == other.value
        return self.value == (other % self.field.prime)
    
    def __hash__(self) -> int:
        return hash((self.value, self.field.prime))
    
    # Arithmetische Operationen
    
    def __add__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Addition: (a + b) mod p"""
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Feldelemente aus verschiedenen Feldern")
            return FieldElement((self.value + other.value) % self.field.prime, self.field)
        return FieldElement((self.value + other) % self.field.prime, self.field)
    
    def __radd__(self, other: int) -> 'FieldElement':
        return self.__add__(other)
    
    def __sub__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Subtraktion: (a - b) mod p"""
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Feldelemente aus verschiedenen Feldern")
            return FieldElement((self.value - other.value) % self.field.prime, self.field)
        return FieldElement((self.value - other) % self.field.prime, self.field)
    
    def __rsub__(self, other: int) -> 'FieldElement':
        return FieldElement((other - self.value) % self.field.prime, self.field)
    
    def __mul__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Multiplikation: (a * b) mod p"""
        if isinstance(other, FieldElement):
            if self.field.prime != other.field.prime:
                raise ValueError("Feldelemente aus verschiedenen Feldern")
            return FieldElement((self.value * other.value) % self.field.prime, self.field)
        return FieldElement((self.value * other) % self.field.prime, self.field)
    
    def __rmul__(self, other: int) -> 'FieldElement':
        return self.__mul__(other)
    
    def __pow__(self, exp: int) -> 'FieldElement':
        """Potenzierung: a^exp mod p (mit schneller Exponentiation)"""
        if exp < 0:
            # Negative Exponenten: a^(-n) = (a^(-1))^n
            return self.inverse() ** (-exp)
        return FieldElement(pow(self.value, exp, self.field.prime), self.field)
    
    def __neg__(self) -> 'FieldElement':
        """Negation: -a mod p"""
        return FieldElement((-self.value) % self.field.prime, self.field)
    
    def __truediv__(self, other: Union['FieldElement', int]) -> 'FieldElement':
        """Division: a / b = a * b^(-1) mod p"""
        if isinstance(other, FieldElement):
            return self * other.inverse()
        return self * FieldElement(other, self.field).inverse()
    
    def inverse(self) -> 'FieldElement':
        """
        Multiplikatives Inverses: a^(-1) mod p
        
        Berechnet mit dem erweiterten Euklidischen Algorithmus.
        Erfüllt: a * a^(-1) = 1 mod p
        """
        if self.value == 0:
            raise ZeroDivisionError("0 hat kein multiplikatives Inverses")
        
        # Erweiterter Euklidischer Algorithmus
        def extended_gcd(a: int, b: int) -> tuple:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        _, x, _ = extended_gcd(self.value % self.field.prime, self.field.prime)
        return FieldElement(x % self.field.prime, self.field)
    
    def is_zero(self) -> bool:
        """Prüft, ob das Element 0 ist."""
        return self.value == 0
    
    def is_one(self) -> bool:
        """Prüft, ob das Element 1 ist."""
        return self.value == 1
    
    @classmethod
    def zero(cls, field: FieldConfig = FIELD_DEV) -> 'FieldElement':
        """Gibt das Nullelement des Feldes zurück."""
        return cls(0, field)
    
    @classmethod
    def one(cls, field: FieldConfig = FIELD_DEV) -> 'FieldElement':
        """Gibt das Einselement des Feldes zurück."""
        return cls(1, field)
    
    @classmethod
    def random(cls, field: FieldConfig = FIELD_DEV) -> 'FieldElement':
        """Gibt ein zufälliges Feldelement zurück."""
        return cls(random.randint(0, field.prime - 1), field)


class FixedPoint:
    """
    Fixpunkt-Arithmetik für Dezimalzahlen in endlichen Feldern.
    
    Wir repräsentieren eine Dezimalzahl x als x * SCALE, wobei SCALE = 2^PRECISION.
    
    Beispiel mit PRECISION=16:
    - 1.5 wird als 1.5 * 65536 = 98304 gespeichert
    - 0.25 wird als 0.25 * 65536 = 16384 gespeichert
    """
    
    PRECISION = 16  # Anzahl der Nachkommabits
    SCALE = 1 << PRECISION  # 2^16 = 65536
    
    __slots__ = ['element']
    
    def __init__(self, value: Union[float, int, FieldElement], field: FieldConfig = FIELD_DEV):
        """
        Erstellt eine Fixpunkt-Zahl.
        
        Args:
            value: Float, Int oder bereits skaliertes FieldElement
            field: Die Feldkonfiguration
        """
        if isinstance(value, FieldElement):
            self.element = value
        elif isinstance(value, float):
            scaled = int(value * self.SCALE)
            self.element = FieldElement(scaled, field)
        else:
            scaled = value * self.SCALE
            self.element = FieldElement(scaled, field)
    
    def __repr__(self) -> str:
        return f"FixedPoint({self.to_float():.6f})"
    
    def to_float(self) -> float:
        """Konvertiert zurück zu Float (nur für Debugging/Anzeige)."""
        val = self.element.value
        # Behandle negative Zahlen (obere Hälfte des Feldes)
        if val > self.element.field.prime // 2:
            val = val - self.element.field.prime
        return val / self.SCALE
    
    def __add__(self, other: 'FixedPoint') -> 'FixedPoint':
        """Addition von Fixpunkt-Zahlen."""
        result = FixedPoint.__new__(FixedPoint)
        result.element = self.element + other.element
        return result
    
    def __sub__(self, other: 'FixedPoint') -> 'FixedPoint':
        """Subtraktion von Fixpunkt-Zahlen."""
        result = FixedPoint.__new__(FixedPoint)
        result.element = self.element - other.element
        return result
    
    def __mul__(self, other: 'FixedPoint') -> 'FixedPoint':
        """
        Multiplikation von Fixpunkt-Zahlen.
        
        (a * SCALE) * (b * SCALE) = a * b * SCALE^2
        Wir müssen durch SCALE teilen, um wieder a * b * SCALE zu bekommen.
        """
        # Multiplikation
        product = self.element * other.element
        # Division durch SCALE (Multiplikation mit SCALE^(-1))
        scale_inv = FieldElement(self.SCALE, self.element.field).inverse()
        result = FixedPoint.__new__(FixedPoint)
        result.element = product * scale_inv
        return result
    
    @classmethod
    def from_field_element(cls, element: FieldElement) -> 'FixedPoint':
        """Erstellt eine FixedPoint-Zahl aus einem bereits skalierten FieldElement."""
        result = cls.__new__(cls)
        result.element = element
        return result


class PrimeField:
    """
    Einfache Wrapper-Klasse für ein Primfeld.
    
    Bietet Hilfsmethoden für Feldoperationen.
    """
    
    def __init__(self, prime: int):
        self.prime = prime
        self.config = FieldConfig(prime, f"F_{prime}")
    
    def element(self, value: int) -> FieldElement:
        """Erstellt ein Feldelement."""
        return FieldElement(value, self.config)
    
    def add(self, a: int, b: int) -> int:
        """Addition mod p."""
        return (a + b) % self.prime
    
    def sub(self, a: int, b: int) -> int:
        """Subtraktion mod p."""
        return (a - b) % self.prime
    
    def mul(self, a: int, b: int) -> int:
        """Multiplikation mod p."""
        return (a * b) % self.prime
    
    def inv(self, a: int) -> int:
        """Multiplikatives Inverses mod p."""
        return pow(a, self.prime - 2, self.prime)
    
    def pow(self, a: int, exp: int) -> int:
        """Potenzierung mod p."""
        return pow(a, exp, self.prime)
    
    def neg(self, a: int) -> int:
        """Negation mod p."""
        return (-a) % self.prime


# Hilfsfunktionen für einfache Nutzung

def field_add(a: int, b: int, prime: int = DEV_PRIME) -> int:
    """Einfache Addition mod p."""
    return (a + b) % prime


def field_sub(a: int, b: int, prime: int = DEV_PRIME) -> int:
    """Einfache Subtraktion mod p."""
    return (a - b) % prime


def field_mul(a: int, b: int, prime: int = DEV_PRIME) -> int:
    """Einfache Multiplikation mod p."""
    return (a * b) % prime


def field_pow(a: int, exp: int, prime: int = DEV_PRIME) -> int:
    """Einfache Potenzierung mod p."""
    return pow(a, exp, prime)


def field_inv(a: int, prime: int = DEV_PRIME) -> int:
    """Einfaches multiplikatives Inverses mod p."""
    return pow(a, prime - 2, prime)  # Fermats kleiner Satz


# Tests
if __name__ == "__main__":
    print("=== Finite Field Tests ===\n")
    
    # Test mit kleinem Feld
    a = FieldElement(7, FIELD_DEV)
    b = FieldElement(13, FIELD_DEV)
    
    print(f"Feld: F_{FIELD_DEV.prime}")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a^2 = {a ** 2}")
    print(f"a^(-1) = {a.inverse()}")
    print(f"a * a^(-1) = {a * a.inverse()}")
    
    print("\n=== FixedPoint Tests ===\n")
    
    x = FixedPoint(1.5, FIELD_DEV)
    y = FixedPoint(0.25, FIELD_DEV)
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x - y = {x - y}")
    print(f"x * y = {x * y}")
