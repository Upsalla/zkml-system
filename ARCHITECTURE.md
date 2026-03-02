# zkML System Architecture

## Übersicht

```
zkml_system/
├── core/                    # Grundlegende Mathematik und Datenstrukturen
│   ├── field.py            # Finite Field Arithmetik
│   ├── r1cs.py             # R1CS Constraint System
│   └── witness.py          # Witness Management
│
├── activations/            # Optimierte Aktivierungsfunktionen
│   ├── base.py             # Basis-Interface
│   ├── relu.py             # ReLU (Referenz, teuer)
│   ├── gelu.py             # GELU Polynom-Approximation
│   └── swish.py            # Swish Polynom-Approximation
│
├── sparse/                 # Sparse Proof Optimierungen
│   ├── detector.py         # Sparsity-Erkennung
│   ├── pruner.py           # Constraint-Pruning
│   └── optimizer.py        # Sparse-Optimierung
│
├── network/                # Neural Network Komponenten
│   ├── layer.py            # Dense, Conv Layer
│   ├── model.py            # Netzwerk-Builder
│   └── compiler.py         # Netzwerk → R1CS Compiler
│
├── proof/                  # Proof-System
│   ├── prover.py           # Proof-Generierung
│   ├── verifier.py         # Proof-Verifikation
│   └── protocol.py         # ZK-Protokoll (vereinfacht)
│
├── tests/                  # Unit Tests
│   └── test_*.py
│
└── examples/               # Beispiele
    ├── mnist_simple.py     # MNIST Klassifikation
    └── benchmark.py        # Performance-Benchmarks
```

## Datenfluss

```
[Trainiertes Modell (PyTorch/ONNX)]
            ↓
    [Network Compiler]
            ↓
    [R1CS Constraints]
            ↓
    [Sparse Optimizer]  ← Entfernt inaktive Neuronen
            ↓
    [Optimierte Constraints]
            ↓
    [Prover] + [Witness]
            ↓
    [ZK Proof]
            ↓
    [Verifier] → True/False
```

## Design-Prinzipien

1. **Modularität:** Jede Komponente ist unabhängig testbar
2. **Optimierung First:** GELU und Sparse sind Standard, nicht optional
3. **Transparenz:** Jeder Schritt ist nachvollziehbar und dokumentiert
4. **Erweiterbarkeit:** Neue Aktivierungen/Layer können einfach hinzugefügt werden

## Finite Field

Wir nutzen ein großes Primfeld für kryptographische Sicherheit:

```python
# BN254 Scalar Field (Standard für Ethereum)
PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
```

Für Entwicklung/Tests nutzen wir ein kleineres Feld:

```python
# Entwicklungs-Feld
PRIME_DEV = 101
```

## Constraint-Typen

```python
# R1CS: A * B = C
# Wobei A, B, C lineare Kombinationen von Witness-Werten sind

class Constraint:
    a: LinearCombination  # Linke Seite
    b: LinearCombination  # Rechte Seite
    c: LinearCombination  # Ergebnis
```

## Aktivierungsfunktionen

| Funktion | Constraints | Implementierung |
|----------|-------------|-----------------|
| ReLU | 258 | Bit-Dekomposition |
| GELU | 10 | Taylor-Polynom (5 Terme) |
| Swish | 7 | x * σ(x) mit Polynom-σ |
| Square | 1 | x * x |

## Sparse Optimierung

```python
# Vor Optimierung
constraints = [c1, c2, c3, c4, c5, ...]  # Alle Constraints

# Nach Sparsity-Analyse
active_neurons = detect_sparsity(witness)

# Nach Optimierung
constraints_sparse = prune_inactive(constraints, active_neurons)
# Nur Constraints für aktive Neuronen bleiben
```
