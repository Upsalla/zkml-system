# Technische Dokumentation: zkML System

## 1. Mathematische Grundlagen

### 1.1 Endliche Felder (Finite Fields)

Alle Berechnungen in ZK-Proofs finden in endlichen Feldern statt. Ein Primfeld F_p besteht aus den Zahlen {0, 1, 2, ..., p-1} mit Arithmetik modulo p.

**Warum Primfelder?**
- Division ist möglich (multiplikatives Inverses existiert)
- Keine Overflow-Probleme
- Kryptographische Sicherheit

**Implementierung** (`core/field.py`):
```python
class FieldElement:
    def __init__(self, value: int, field: FieldConfig):
        self.value = value % field.prime
    
    def inverse(self) -> 'FieldElement':
        # Erweiterter Euklidischer Algorithmus
        # a * a^(-1) = 1 mod p
```

### 1.2 R1CS (Rank-1 Constraint System)

R1CS ist das Standard-Format für ZK-Proofs. Jede Berechnung wird als Menge von Constraints dargestellt:

```
A(w) × B(w) = C(w)
```

**Beispiel**: z = x * y

```
Witness: w = [1, x, y, z]
A = [0, 1, 0, 0]  (wählt x)
B = [0, 0, 1, 0]  (wählt y)
C = [0, 0, 0, 1]  (wählt z)

Constraint: x * y = z ✓
```

**Implementierung** (`core/r1cs.py`):
```python
@dataclass
class R1CSConstraint:
    a: LinearCombination
    b: LinearCombination
    c: LinearCombination
    
    def is_satisfied(self, witness: List[int], prime: int) -> bool:
        return (a.evaluate(witness) * b.evaluate(witness)) % prime == c.evaluate(witness)
```

### 1.3 Witness

Der Witness enthält alle Zwischenwerte einer Berechnung. Er ist das "Geheimnis", das der Prover kennt, aber nicht offenlegt.

**Struktur**:
```
w[0] = 1          (Konstante)
w[1..n] = inputs  (öffentlich)
w[n+1..m] = intermediate (privat)
w[m+1..k] = outputs (öffentlich)
```

## 2. Aktivierungsfunktionen

### 2.1 Das ReLU-Problem

ReLU(x) = max(0, x) erfordert einen Vergleich, der in R1CS teuer ist:

**Bit-Dekomposition**:
```
x = Σ(b_i * 2^i)  für i = 0..255

Constraints pro Bit:
- b_i * (1 - b_i) = 0  (b_i ist 0 oder 1)
- Σ(b_i * 2^i) = x     (Rekonstruktion)

Total: 256+ Constraints pro ReLU
```

### 2.2 GELU-Lösung

GELU ist eine glatte Approximation von ReLU:

```
GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Polynomiale Approximation**:
```
GELU(x) ≈ ax³ + bx² + cx + d

Mit Koeffizienten:
a = 0.044715 * √(2/π) ≈ 0.0356
b = 0
c = 0.5 + 0.5 * √(2/π) ≈ 0.8989
d = 0
```

**Constraints**:
```
1. x² = x * x
2. x³ = x² * x
3. t1 = a * x³
4. t2 = c * x
5. output = t1 + t2

Total: 5-10 Constraints pro GELU (vs. 256+ für ReLU)
```

### 2.3 Swish und Quadratic

**Swish**: x * sigmoid(x) ≈ x * (0.5 + 0.25x)
```
Constraints: ~8 pro Aktivierung
```

**Quadratic**: x² (einfachste nicht-lineare Funktion)
```
Constraints: 1 pro Aktivierung
```

## 3. Sparse Proof Optimierung

### 3.1 Beobachtung

In trainierten neuronalen Netzen sind nach ReLU/GELU oft 50-90% der Neuronen inaktiv (Ausgabe = 0).

### 3.2 Zero-Proof

Statt die vollständige Berechnung zu beweisen, beweisen wir nur:
```
output = 0
```

**Constraint**:
```
output * 1 = 0
```

Das ist 1 Constraint statt Dutzenden!

### 3.3 Batched Zero-Proofs

Mehrere Zero-Proofs können kombiniert werden:

```
Wähle zufällige r_1, ..., r_n
Constraint: (r_1*x_1 + r_2*x_2 + ... + r_n*x_n) = 0

Wenn alle x_i = 0: Summe = 0 ✓
Wenn ein x_i ≠ 0: Summe ≠ 0 mit hoher Wahrscheinlichkeit
```

**Vorteil**: n Constraints → 1 Constraint

## 4. Proof-System

### 4.1 Prover

Der Prover generiert einen Beweis, dass er einen gültigen Witness kennt.

**Schritte**:
1. Forward Pass durchführen (Witness generieren)
2. R1CS-Constraints generieren
3. Witness-Commitment berechnen
4. Schnorr-artigen Proof generieren

**Commitment**:
```
C = Σ(w_i * g^i) mod p
```

**Schnorr-Proof**:
```
1. Wähle zufälliges k
2. R = g^k
3. c = Hash(R || C)  (Fiat-Shamir)
4. s = k + c * secret
```

### 4.2 Verifier

Der Verifier prüft den Proof ohne den Witness zu kennen.

**Checks**:
1. Network Hash stimmt überein
2. Primfeld stimmt überein
3. Schnorr-Proof ist gültig
4. R1CS-Struktur ist korrekt
5. Öffentliche Werte sind gültig

### 4.3 Sicherheitseigenschaften

**Completeness**: Ein ehrlicher Prover kann immer einen gültigen Proof erstellen.

**Soundness**: Ein unehrlicher Prover kann (fast) nie einen falschen Proof erstellen.

**Zero-Knowledge**: Der Verifier lernt nichts außer der Gültigkeit.

## 5. Constraint-Analyse

### 5.1 Dense Layer

```
Eingabe: n Neuronen
Ausgabe: m Neuronen

Matrixmultiplikation: n * m Constraints
Bias-Addition: m Constraints (in Matmul integriert)
Aktivierung: m * activation_cost Constraints

Total: m * (n + activation_cost)
```

### 5.2 Vergleich

| Aktivierung | Cost/Neuron | 128 Neuronen | 1024 Neuronen |
|-------------|-------------|--------------|---------------|
| ReLU        | 258         | 33,024       | 264,192       |
| GELU        | 10          | 1,280        | 10,240        |
| Swish       | 8           | 1,024        | 8,192         |
| Quadratic   | 1           | 128          | 1,024         |

### 5.3 Sparse Savings

Bei 60% Sparsity:
```
Aktive Neuronen: 40% * full_cost
Inaktive Neuronen: 60% * 1 (Zero-Proof)

Beispiel (128 GELU-Neuronen):
- Ohne Sparse: 128 * 10 = 1,280
- Mit Sparse: 51 * 10 + 77 * 1 = 587
- Ersparnis: 54%
```

## 6. Implementierungsdetails

### 6.1 Fixpunkt-Arithmetik

Dezimalzahlen werden als Ganzzahlen mit Skalierungsfaktor dargestellt:

```python
SCALE = 2^16 = 65536

1.5 → 1.5 * 65536 = 98304
0.25 → 0.25 * 65536 = 16384

Multiplikation:
(a * SCALE) * (b * SCALE) = a * b * SCALE²
→ Division durch SCALE nötig
```

### 6.2 Negative Zahlen

In Primfeldern gibt es keine negativen Zahlen. Wir interpretieren:
```
Werte 0 bis p/2: positiv
Werte p/2+1 bis p-1: negativ (als p - value)
```

### 6.3 Overflow-Handling

Bei großen Primfeldern (BN254) ist Overflow kein Problem. Bei kleinen Feldern (p=101) muss man vorsichtig sein:
```
50 + 60 = 110 → 110 % 101 = 9
```

## 7. Erweiterungsmöglichkeiten

### 7.1 Vollständiger SNARK

Aktuell: Vereinfachter Schnorr-artiger Proof
Ziel: Groth16 oder PLONK für echte Zero-Knowledge

### 7.2 Größeres Primfeld

BN254 für Ethereum-Kompatibilität:
```
p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
```

### 7.3 Weitere Layer-Typen

- Conv2D: Für Bildklassifikation
- BatchNorm: Für tiefere Netze
- Attention: Für Transformer

### 7.4 On-Chain Verifikation

Solidity-Verifier für Ethereum Smart Contracts.
