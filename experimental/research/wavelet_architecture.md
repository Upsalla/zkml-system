# Wavelet-Architektur: Haar-Wavelet Witness Batching (HWWB)

## 1. Konzept-Übersicht

Nach der theoretischen Analyse ist klar, dass vollständige Wavelet-Kompression für R1CS zu komplex ist. Stattdessen implementieren wir einen **fokussierten Ansatz**: Haar-Wavelet Witness Batching (HWWB).

### 1.1 Kernidee

Benachbarte Witness-Werte in neuronalen Netzen sind oft korreliert (z.B. benachbarte Pixel, benachbarte Neuronen). HWWB nutzt dies aus:

1. **Transformation:** Zerlege Witness-Paare in Summe und Differenz
2. **Kompression:** Wenn Differenzen klein sind, können sie effizienter bewiesen werden
3. **Batching:** Mehrere kleine Differenzen können in einem Commitment zusammengefasst werden

### 1.2 Mathematische Basis

Für ein Witness-Paar `(w[2k], w[2k+1])`:

```
sum[k]  = w[2k] + w[2k+1]     (Approximation)
diff[k] = w[2k] - w[2k+1]     (Detail)
```

Rekonstruktion:
```
w[2k]   = (sum[k] + diff[k]) / 2
w[2k+1] = (sum[k] - diff[k]) / 2
```

**In Finite Fields:** Division durch 2 ist Multiplikation mit dem modularen Inversen von 2.

---

## 2. Architektur

### 2.1 Komponenten

```
┌─────────────────────────────────────────────────────────────┐
│                    HWWB System                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Analyzer   │───▶│ Transformer  │───▶│  Committer   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Correlation │    │   Haar DWT   │    │   Batched    │   │
│  │   Analysis   │    │  (sum/diff)  │    │  Commitment  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Workflow

**Prover:**
1. Empfange Witness-Vektor `w` der Länge `n`
2. Analysiere Korrelation benachbarter Werte
3. Wenn Korrelation hoch: Transformiere in `(sums, diffs)`
4. Identifiziere kleine Differenzen (unter Threshold)
5. Batche kleine Differenzen in ein einzelnes Commitment
6. Generiere Proof für `sums` + batched `diffs`

**Verifier:**
1. Empfange Proof mit `sums`, batched `diffs`, und Commitments
2. Verifiziere Commitments
3. Rekonstruiere `w` aus `sums` und `diffs`
4. Verifiziere, dass rekonstruierter `w` die Constraints erfüllt

### 2.3 Datenstrukturen

```python
@dataclass
class HaarCoefficients:
    """Haar-transformierte Witness-Darstellung."""
    sums: List[FieldElement]      # Approximationskoeffizienten
    diffs: List[FieldElement]     # Detailkoeffizienten
    small_diff_indices: List[int] # Indizes kleiner Differenzen
    
@dataclass
class HWWBProof:
    """Proof mit Haar-Wavelet-Batching."""
    sums_commitment: bytes
    diffs_commitment: bytes
    batched_small_diffs: bytes    # Einzelnes Commitment für alle kleinen Diffs
    small_diff_bound: int         # Obere Schranke für kleine Diffs
    revealed_sums: List[int]      # Für Verifikation benötigte Summen
```

---

## 3. Erwarteter Vorteil

### 3.1 Szenario-Analyse

**Best Case:** Alle benachbarten Werte sind identisch
- Alle Differenzen sind 0
- Nur `n/2` Summen müssen committed werden
- **50% Reduktion**

**Typical Case (CNN-Aktivierungen):** Benachbarte Werte sind ähnlich
- ~70% der Differenzen sind "klein" (unter Threshold)
- Diese können geballt werden
- **20-30% Reduktion erwartet**

**Worst Case:** Keine Korrelation
- Alle Differenzen sind signifikant
- Kein Vorteil, leichter Overhead durch Transformation
- **~5% Overhead**

### 3.2 Vergleich mit CSWC

| Aspekt | CSWC | HWWB |
|--------|------|------|
| Nutzt aus | Sparsity (Nullen) | Korrelation (Ähnlichkeit) |
| Best Case | 90%+ Reduktion | 50% Reduktion |
| Typical Case | 40-60% Reduktion | 20-30% Reduktion |
| Kombinierbar | Ja | Ja |
| Overhead | Hoch (Sketch-Berechnung) | Niedrig (einfache Addition) |

**Schlussfolgerung:** HWWB ist komplementär zu CSWC. Es adressiert einen anderen Aspekt (Korrelation statt Sparsity) und hat niedrigeren Overhead.

---

## 4. Implementierungsplan

### Phase 1: Core-Algorithmen
- `haar_transform()`: Witness → (sums, diffs)
- `haar_inverse()`: (sums, diffs) → Witness
- `analyze_correlation()`: Bestimme Korrelationsgrad

### Phase 2: Batching-Logik
- `identify_small_diffs()`: Finde Differenzen unter Threshold
- `batch_commitment()`: Erstelle einzelnes Commitment für kleine Diffs

### Phase 3: Integration
- In zkML-Pipeline einbinden
- Kombination mit CSWC testen

### Phase 4: Benchmark
- Vergleich: Standard vs. HWWB vs. CSWC vs. HWWB+CSWC
