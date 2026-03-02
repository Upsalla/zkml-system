# Architektur-Design: CS-Inspired Sparse Witness Commitment (CSWC)

## 1. Systemübersicht

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CSWC Architecture                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   Network   │───▶│   Witness   │───▶│   Sparse    │                 │
│  │  Inference  │    │  Generator  │    │  Extractor  │                 │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│                     ┌─────────────────────────┼─────────────────────┐  │
│                     │                         ▼                     │  │
│                     │  ┌─────────────┐  ┌─────────────┐            │  │
│                     │  │   Support   │  │   Sparse    │            │  │
│                     │  │   Set S     │  │  Values w_S │            │  │
│                     │  └──────┬──────┘  └──────┬──────┘            │  │
│                     │         │                │                    │  │
│                     │         ▼                ▼                    │  │
│                     │  ┌─────────────────────────────┐             │  │
│                     │  │      Sketch Computer        │             │  │
│                     │  │      y = A_S * w_S          │             │  │
│                     │  └──────────────┬──────────────┘             │  │
│                     │                 │                             │  │
│                     │                 ▼                             │  │
│                     │  ┌─────────────────────────────┐             │  │
│                     │  │    Commitment Generator     │             │  │
│                     │  │  C = (C_S, C_w, C_y)       │             │  │
│                     │  └──────────────┬──────────────┘             │  │
│                     │                 │                             │  │
│                     └─────────────────┼─────────────────────────────┘  │
│                                       │                                 │
│                                       ▼                                 │
│                          ┌─────────────────────────┐                   │
│                          │    CSWC Proof           │                   │
│                          │  (C_S, C_w, C_y, π)    │                   │
│                          └─────────────────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Komponenten-Spezifikation

### 2.1 Sensing Matrix Generator

**Zweck:** Generiert die Sensing-Matrix A für den Sketch.

**Interface:**
```python
class SensingMatrixGenerator:
    def __init__(self, n: int, m: int, field: FieldConfig):
        """
        n: Witness-Dimension
        m: Sketch-Dimension (Sicherheitsparameter)
        field: Endliches Feld für Arithmetik
        """
        
    def generate(self, seed: bytes) -> SensingMatrix:
        """
        Generiert deterministische Matrix aus Seed.
        Seed wird aus Public Parameters abgeleitet.
        """
        
    def get_submatrix(self, indices: List[int]) -> SensingMatrix:
        """
        Extrahiert Spalten für gegebene Indizes.
        Effizient für sparse Operationen.
        """
```

**Matrix-Typen:**

| Typ | Struktur | Speicher | Multiplikation | Empfehlung |
|-----|----------|----------|----------------|------------|
| Dense Random | Vollständig | O(mn) | O(mn) | Kleine n |
| Sparse Random | ~10 Einträge/Spalte | O(10n) | O(10k) | Mittlere n |
| Toeplitz | Zirkulant | O(n+m) | O(n log n) via FFT | Große n |

**Empfehlung:** Sparse Random für zkML (typisch n < 10^6).

### 2.2 Sparse Extractor

**Zweck:** Extrahiert Support und Werte aus dem vollen Witness.

**Interface:**
```python
class SparseExtractor:
    def __init__(self, threshold: float = 0.0):
        """
        threshold: Werte unter diesem Threshold werden als 0 behandelt.
        """
        
    def extract(self, witness: List[FieldElement]) -> SparseWitness:
        """
        Returns:
            SparseWitness mit:
            - support: List[int] (Indizes der nicht-null Einträge)
            - values: List[FieldElement] (Werte an diesen Indizes)
            - sparsity: float (Anteil der Nullen)
        """
```

### 2.3 Sketch Computer

**Zweck:** Berechnet den linearen Sketch y = A_S * w_S.

**Interface:**
```python
class SketchComputer:
    def __init__(self, sensing_matrix: SensingMatrix):
        """
        sensing_matrix: Die Sensing-Matrix A
        """
        
    def compute(self, sparse_witness: SparseWitness) -> List[FieldElement]:
        """
        Berechnet y = A_S * w_S effizient.
        Komplexität: O(m * k) für k nicht-null Einträge.
        """
        
    def verify(self, sparse_witness: SparseWitness, sketch: List[FieldElement]) -> bool:
        """
        Prüft, ob y == A_S * w_S.
        Verwendet vom Verifier.
        """
```

### 2.4 Commitment Generator

**Zweck:** Generiert kryptographische Commitments für S, w_S und y.

**Interface:**
```python
class CSWCCommitmentGenerator:
    def __init__(self, commitment_scheme: CommitmentScheme):
        """
        commitment_scheme: z.B. Pedersen, KZG
        """
        
    def commit_support(self, support: List[int]) -> Commitment:
        """
        Commitment auf die Support-Menge.
        Verwendet Merkle-Tree oder Polynom-Commitment.
        """
        
    def commit_values(self, values: List[FieldElement]) -> Commitment:
        """
        Commitment auf die Nicht-Null-Werte.
        """
        
    def commit_sketch(self, sketch: List[FieldElement]) -> Commitment:
        """
        Commitment auf den Sketch-Vektor.
        """
        
    def generate_proof(self) -> CSWCProof:
        """
        Generiert den vollständigen CSWC-Proof.
        """
```

## 3. Datenstrukturen

### 3.1 SparseWitness

```python
@dataclass
class SparseWitness:
    support: List[int]          # Indizes der nicht-null Einträge
    values: List[FieldElement]  # Werte an diesen Indizes
    full_size: int              # Ursprüngliche Witness-Größe n
    
    @property
    def sparsity(self) -> float:
        return 1.0 - len(self.support) / self.full_size
    
    def to_dense(self) -> List[FieldElement]:
        """Rekonstruiert den vollen Witness (für Tests)."""
        result = [FieldElement.zero()] * self.full_size
        for idx, val in zip(self.support, self.values):
            result[idx] = val
        return result
```

### 3.2 SensingMatrix

```python
@dataclass
class SensingMatrix:
    rows: int                   # m
    cols: int                   # n
    data: Dict[Tuple[int, int], FieldElement]  # Sparse Speicherung
    
    def multiply_sparse(self, sparse_witness: SparseWitness) -> List[FieldElement]:
        """
        Effiziente Multiplikation mit sparse Witness.
        Nur Spalten für Indizes in support werden verwendet.
        """
        result = [FieldElement.zero()] * self.rows
        for col_idx, val in zip(sparse_witness.support, sparse_witness.values):
            for row_idx in range(self.rows):
                if (row_idx, col_idx) in self.data:
                    result[row_idx] += self.data[(row_idx, col_idx)] * val
        return result
```

### 3.3 CSWCProof

```python
@dataclass
class CSWCProof:
    # Commitments
    support_commitment: Commitment
    values_commitment: Commitment
    sketch_commitment: Commitment
    
    # Öffnungen (für Verifikation)
    support: List[int]
    values: List[FieldElement]
    sketch: List[FieldElement]
    
    # Metadaten
    sparsity: float
    witness_size: int
    sketch_size: int
    
    def size_bytes(self) -> int:
        """Berechnet die Proof-Größe in Bytes."""
        return (
            len(self.support) * 4 +  # Indizes (32-bit)
            len(self.values) * 32 +  # Feldelemente (256-bit)
            len(self.sketch) * 32 +  # Sketch (256-bit)
            3 * 64  # Commitments (512-bit each)
        )
```

## 4. Protokoll-Ablauf

### 4.1 Setup (einmalig)

```
Input: Sicherheitsparameter λ, Witness-Größe n

1. Berechne Sketch-Größe: m = 2λ (z.B. m = 256 für λ = 128)
2. Generiere Sensing-Matrix: A = SensingMatrixGenerator(n, m).generate(seed)
3. Initialisiere Commitment-Schema: cs = CommitmentScheme.setup(λ)

Output: PublicParams = (A, cs, n, m)
```

### 4.2 Prove

```
Input: Witness w ∈ F^n, PublicParams

1. Extrahiere Sparse-Darstellung:
   sparse_w = SparseExtractor().extract(w)
   S = sparse_w.support
   w_S = sparse_w.values

2. Berechne Sketch:
   A_S = A.get_submatrix(S)
   y = A_S.multiply(w_S)

3. Generiere Commitments:
   C_S = commit_support(S)
   C_w = commit_values(w_S)
   C_y = commit_sketch(y)

4. Erstelle Proof:
   proof = CSWCProof(
       support_commitment=C_S,
       values_commitment=C_w,
       sketch_commitment=C_y,
       support=S,
       values=w_S,
       sketch=y,
       ...
   )

Output: proof
```

### 4.3 Verify

```
Input: proof, PublicParams

1. Verifiziere Commitments:
   assert verify_commitment(proof.support_commitment, proof.support)
   assert verify_commitment(proof.values_commitment, proof.values)
   assert verify_commitment(proof.sketch_commitment, proof.sketch)

2. Rekonstruiere Sketch:
   A_S = A.get_submatrix(proof.support)
   y_computed = A_S.multiply_sparse(SparseWitness(proof.support, proof.values, n))

3. Prüfe Sketch-Konsistenz:
   assert proof.sketch == y_computed

4. (Optional) Prüfe Circuit-Constraints mit sparse Witness

Output: Accept/Reject
```

## 5. Sicherheitsanalyse

### 5.1 Soundness

**Theorem:** Ein Angreifer kann mit Wahrscheinlichkeit höchstens k/|F| einen falschen Proof erstellen, wobei k die Sparsity und |F| die Feldgröße ist.

**Beweis-Skizze:**
- Angreifer committed auf (S', w_S', y')
- Er muss y' = A_S' * w_S' erfüllen
- Für falsches S' oder w_S': Pr[A_S * w_S = A_S' * w_S'] ≤ k/|F|
- Mit |F| = 2^256 und k < 10^6: Fehler < 10^(-70)

### 5.2 Zero-Knowledge

**Theorem:** Der Proof leakt keine Information über den Witness außer der Sparsity.

**Beweis-Skizze:**
- S ist durch C_S verborgen (Commitment-Hiding)
- w_S ist durch C_w verborgen (Commitment-Hiding)
- y = A_S * w_S ist eine deterministische Funktion von S und w_S
- Da A öffentlich ist, leakt y theoretisch Information
- **Lösung:** Randomisiere y mit Blinding: y' = y + r, wobei r zufällig

### 5.3 Komplexität

| Operation | Prover | Verifier |
|-----------|--------|----------|
| Sparse Extraction | O(n) | - |
| Sketch Computation | O(m * k) | O(m * k) |
| Commitment | O(k + m) | O(k + m) |
| **Gesamt** | **O(n + mk)** | **O(mk)** |

Für typische Werte (n = 10^6, k = 10^5, m = 256):
- Prover: O(10^6 + 256 * 10^5) = O(2.5 * 10^7)
- Verifier: O(256 * 10^5) = O(2.5 * 10^7)

**Vergleich mit Standard-Witness:**
- Standard: Prover O(n) = O(10^6), Verifier O(n) = O(10^6)
- CSWC: Prover O(n + mk), Verifier O(mk)

**Fazit:** CSWC ist für den Prover etwas teurer, aber der Verifier profitiert von der Sketch-basierten Konsistenzprüfung.

## 6. Implementierungsplan

### 6.1 Dateien

```
zkml_system/
├── compressed_sensing/
│   ├── __init__.py
│   ├── sensing_matrix.py      # SensingMatrixGenerator, SensingMatrix
│   ├── sparse_witness.py      # SparseExtractor, SparseWitness
│   ├── sketch.py              # SketchComputer
│   ├── commitment.py          # CSWCCommitmentGenerator, CSWCProof
│   └── test_cswc.py           # Tests
```

### 6.2 Abhängigkeiten

- Existierendes `core/field.py` für Feldarithmetik
- Existierendes `proof/` für Commitment-Integration
- NumPy für effiziente Matrixoperationen (optional)

### 6.3 Zeitschätzung

| Komponente | Aufwand |
|------------|---------|
| SensingMatrix | 2h |
| SparseExtractor | 1h |
| SketchComputer | 2h |
| Commitment Integration | 3h |
| Tests | 2h |
| **Gesamt** | **10h** |
