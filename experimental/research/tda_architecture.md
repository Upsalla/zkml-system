# TDA-Fingerprinting Architektur

## 1. System-Übersicht

### 1.1 Ziel

Ein **Modell-Fingerprinting-System**, das:
1. Einen kompakten, eindeutigen Fingerprint für jedes ML-Modell erzeugt
2. Effizient in ZK-Proofs verifizierbar ist
3. Robust gegen kleine Modifikationen ist (Stabilität)
4. Unterschiedliche Modelle zuverlässig unterscheidet (Eindeutigkeit)

### 1.2 Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TDA Model Fingerprinting System                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │    Model     │───▶│  Point Cloud │───▶│  Persistence │           │
│  │   Loader     │    │  Converter   │    │   Computer   │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                 │                    │
│                                                 ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │     ZK       │◀───│  Fingerprint │◀───│  Persistence │           │
│  │  Verifier    │    │  Extractor   │    │   Diagram    │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Komponenten-Design

### 2.1 Model Loader

**Funktion:** Lädt ein neuronales Netz und extrahiert die Gewichte.

**Input:** Modell-Datei (PyTorch, ONNX, oder eigenes Format)
**Output:** Liste von Gewichtsmatrizen `[W₁, W₂, ..., Wₗ]`

```python
class ModelLoader:
    def load(self, path: str) -> List[np.ndarray]:
        """Load model weights from file."""
        pass
    
    def from_network(self, network: Network) -> List[np.ndarray]:
        """Extract weights from in-memory network."""
        pass
```

### 2.2 Point Cloud Converter

**Funktion:** Konvertiert Gewichte in eine Punktwolke für TDA.

**Strategie: Layer-wise Neuron Embedding**

Für jeden Layer l mit Gewichtsmatrix Wₗ ∈ ℝ^(out × in):
- Jede Zeile (Neuron) wird als Punkt in ℝ^in behandelt
- Normalisiere auf Einheitskugel für Skalenunabhängigkeit

```python
class PointCloudConverter:
    def convert(self, weights: List[np.ndarray]) -> np.ndarray:
        """
        Convert model weights to point cloud.
        
        Returns:
            Point cloud as (N, D) array where N = total neurons,
            D = max input dimension (padded with zeros)
        """
        pass
```

**Normalisierung:**
- L2-Normalisierung jedes Punktes: p̂ = p / ||p||₂
- Optional: PCA-Reduktion auf feste Dimension

### 2.3 Persistence Computer

**Funktion:** Berechnet das Persistence Diagram der Punktwolke.

**Algorithmus: Vietoris-Rips mit Boundary Matrix Reduction**

```python
class PersistenceComputer:
    def compute(self, points: np.ndarray, max_dim: int = 1) -> PersistenceDiagram:
        """
        Compute persistence diagram up to dimension max_dim.
        
        Args:
            points: (N, D) point cloud
            max_dim: Maximum homology dimension (0 = components, 1 = loops)
            
        Returns:
            PersistenceDiagram with birth-death pairs
        """
        pass
```

**Komplexität:**
- Naive: O(n³) für n Punkte
- Optimiert (Ripser): O(n² log n) für sparse complexes

### 2.4 Fingerprint Extractor

**Funktion:** Extrahiert einen kompakten Fingerprint aus dem Persistence Diagram.

**Strategie: Top-k Persistent Features**

1. Sortiere Features nach Persistenz: `persistence = death - birth`
2. Wähle die k persistentesten Features
3. Quantisiere birth/death auf feste Präzision
4. Hash das Ergebnis

```python
@dataclass
class ModelFingerprint:
    """Compact model fingerprint."""
    features: List[Tuple[int, float, float]]  # (dimension, birth, death)
    hash: bytes  # 32-byte hash
    
class FingerprintExtractor:
    def extract(self, diagram: PersistenceDiagram, k: int = 20) -> ModelFingerprint:
        """Extract top-k persistent features as fingerprint."""
        pass
```

**Fingerprint-Größe:**
- k Features × (1 byte dim + 4 bytes birth + 4 bytes death) = 9k bytes
- Plus 32 bytes Hash
- Für k=20: **212 bytes** (unabhängig von Modellgröße!)

### 2.5 ZK Verifier

**Funktion:** Verifiziert, dass ein Proof zu einem bestimmten Fingerprint gehört.

**Protokoll: Commitment-basierte Verifikation**

1. **Setup:** Prover committet zu Modell M und Fingerprint F
2. **Challenge:** Verifier wählt zufällige Punkte aus dem Modell
3. **Response:** Prover zeigt, dass diese Punkte konsistent mit F sind
4. **Verify:** Verifier prüft Konsistenz

```python
class TDAVerifier:
    def verify_fingerprint(self, 
                          model_commitment: bytes,
                          fingerprint: ModelFingerprint,
                          proof: TDAProof) -> bool:
        """
        Verify that fingerprint matches committed model.
        """
        pass
```

---

## 3. Datenstrukturen

### 3.1 Persistence Diagram

```python
@dataclass
class PersistenceFeature:
    """A single topological feature."""
    dimension: int      # 0 = component, 1 = loop, 2 = void
    birth: float        # Filtration value at birth
    death: float        # Filtration value at death
    
    @property
    def persistence(self) -> float:
        return self.death - self.birth

@dataclass
class PersistenceDiagram:
    """Collection of persistence features."""
    features: List[PersistenceFeature]
    
    def top_k(self, k: int) -> List[PersistenceFeature]:
        """Return k most persistent features."""
        return sorted(self.features, key=lambda f: -f.persistence)[:k]
```

### 3.2 TDA Proof

```python
@dataclass
class TDAProof:
    """Proof that a model has a specific fingerprint."""
    model_commitment: bytes
    fingerprint: ModelFingerprint
    
    # Sampling-based verification data
    sampled_indices: List[int]
    sampled_points: List[List[float]]
    local_persistence_proofs: List[bytes]
    
    def size_bytes(self) -> int:
        """Calculate proof size."""
        pass
```

---

## 4. Sicherheitsanalyse

### 4.1 Eindeutigkeit

**Frage:** Können zwei verschiedene Modelle denselben Fingerprint haben?

**Analyse:**
- Persistent Homology ist eine **topologische Invariante**
- Verschiedene Punktwolken können theoretisch dasselbe Diagram haben
- Aber: Die Wahrscheinlichkeit ist extrem gering für reale Modelle
- Zusätzliche Sicherheit durch Sampling-Verifikation

**Sicherheitsniveau:** ~2^(-128) Kollisionswahrscheinlichkeit bei k=20 Features und 32-bit Präzision

### 4.2 Stabilität

**Frage:** Wie ändert sich der Fingerprint bei kleinen Modifikationen?

**Analyse:**
- Stabilitätstheorem: ||D(P) - D(Q)||_∞ ≤ d_H(P, Q)
- Kleine Gewichtsänderungen → kleine Fingerprint-Änderungen
- Quantisierung absorbiert Rauschen

**Praktische Implikation:** Modelle mit >1% Gewichtsunterschied haben unterschiedliche Fingerprints

### 4.3 Zero-Knowledge

**Frage:** Lernt der Verifier etwas über das Modell?

**Analyse:**
- Der Fingerprint selbst enthält topologische Information
- Diese Information ist jedoch **nicht invertierbar** (man kann das Modell nicht rekonstruieren)
- Sampling-basierte Verifikation offenbart nur zufällige Punkte

**Sicherheitsniveau:** Computational Zero-Knowledge (unter Random Oracle Model)

---

## 5. Performance-Schätzung

### 5.1 Fingerprint-Berechnung

| Modellgröße | Punkte | Persistence-Zeit | Fingerprint-Größe |
|-------------|--------|------------------|-------------------|
| 10K Parameter | ~100 | <1s | 212 bytes |
| 100K Parameter | ~1000 | ~10s | 212 bytes |
| 1M Parameter | ~10000 | ~100s | 212 bytes |

**Beobachtung:** Fingerprint-Größe ist **konstant**, Berechnungszeit skaliert mit O(n²).

### 5.2 ZK-Verifikation

| Metrik | Wert |
|--------|------|
| Verifier-Zeit | <10ms |
| Proof-Größe | ~1KB |
| On-Chain-Gas | ~50K |

---

## 6. Implementierungsplan

### Phase 1: Core TDA (2-3 Tage)
- Vietoris-Rips-Komplex
- Boundary-Matrix-Reduktion
- Persistence-Diagram-Berechnung

### Phase 2: Model Integration (1 Tag)
- Point Cloud Converter
- Fingerprint Extractor

### Phase 3: ZK Integration (2 Tage)
- Commitment-Schema
- Sampling-basierte Verifikation

### Phase 4: Benchmark (1 Tag)
- Eindeutigkeits-Tests
- Stabilitäts-Tests
- Performance-Messung
