# Machbarkeitsanalyse: Compressed Sensing für Witness-Kompression

## 1. Problemstellung

### 1.1 Aktueller Stand
In unserem zkML-System hat der Witness (alle Zwischenwerte einer NN-Inferenz) die Größe O(n), wobei n die Anzahl der Neuronen ist. Das führt zu:
- **Proof-Größe:** O(n) oder O(√n) je nach Schema
- **Prover-Zeit:** O(n) für Commitment
- **Verifier-Zeit:** O(1) bis O(log n) für Verifikation

### 1.2 Ziel
Reduziere die effektive Witness-Größe von O(n) auf O(k), wobei k die Anzahl der nicht-null Aktivierungen ist (typisch: k ≈ 0.1n bis 0.5n).

## 2. Drei Ansätze im Vergleich

### 2.1 Ansatz A: Direkte CS-Anwendung

**Idee:** Wende eine Sensing-Matrix A auf den Witness an.

```
Witness: w ∈ ℝⁿ (k-sparse)
Sensing: y = Aw ∈ ℝᵐ, wobei m = O(k log(n/k))
Commitment: Commit(y)
```

**Analyse:**

| Aspekt | Bewertung | Begründung |
|--------|-----------|------------|
| Proof-Größe | ✅ O(m) = O(k log n) | Signifikante Reduktion |
| Prover-Zeit | ⚠️ O(nm) | Matrix-Multiplikation ist teuer |
| Verifier-Zeit | ❌ O(n³) | L1-Minimierung zur Rekonstruktion |
| Soundness | ❌ Problematisch | Nicht-eindeutige Rekonstruktion |
| Zero-Knowledge | ⚠️ Unklar | y könnte Information leaken |

**Fazit:** Nicht praktikabel wegen Verifier-Komplexität und Soundness-Problemen.

### 2.2 Ansatz B: Sparse Commitment (unser aktueller Ansatz)

**Idee:** Committe nur die nicht-null Einträge.

```
Witness: w ∈ ℝⁿ (k-sparse)
Support: S = {i : w_i ≠ 0}, |S| = k
Sparse Witness: w_S ∈ ℝᵏ
Commitment: Commit(S, w_S)
```

**Analyse:**

| Aspekt | Bewertung | Begründung |
|--------|-----------|------------|
| Proof-Größe | ✅ O(k) | Nur nicht-null Einträge |
| Prover-Zeit | ✅ O(n + k) | Berechnung + Commitment |
| Verifier-Zeit | ✅ O(k + λ) | Commitment-Check + Stichproben |
| Soundness | ✅ Ja | Mit Stichproben-Verifikation |
| Zero-Knowledge | ✅ Ja | Commitments verbergen Werte |

**Fazit:** Praktikabel und bereits implementiert.

### 2.3 Ansatz C: CS-Inspired Structured Commitment (NEU)

**Idee:** Nutze CS-Strukturen für effizientere Stichproben.

```
Witness: w ∈ ℝⁿ (k-sparse)
Sensing Matrix: A ∈ ℝᵐˣⁿ (sparse, strukturiert)
Sketch: y = Aw ∈ ℝᵐ
Commitment: Commit(y) + Commit(S) + Commit(w_S)

Verifikation:
1. Prüfe Commit(y) gegen Commit(S, w_S) via A
2. Prüfe stichprobenartig, dass Nullen wirklich null sind
```

**Analyse:**

| Aspekt | Bewertung | Begründung |
|--------|-----------|------------|
| Proof-Größe | ✅ O(k + m) | Sketch + Sparse Commitment |
| Prover-Zeit | ✅ O(n) | Sparse Matrix-Multiplikation |
| Verifier-Zeit | ✅ O(k + m) | Keine Rekonstruktion nötig |
| Soundness | ✅ Ja | Sketch + Stichproben |
| Zero-Knowledge | ✅ Ja | Commitments verbergen alles |

**Vorteil gegenüber B:** Der Sketch y ermöglicht eine **globale Konsistenzprüfung** ohne alle Nullen einzeln zu prüfen.

## 3. Detaillierte Analyse von Ansatz C

### 3.1 Die Kernidee

Statt λ zufällige Positionen zu prüfen (wie in Ansatz B), nutzen wir eine **lineare Sketch-Funktion**:

```
y = Aw = A_S * w_S + A_{S̄} * w_{S̄}
        = A_S * w_S + 0  (wenn w wirklich k-sparse ist)
```

Der Verifier kann prüfen:
```
y ?= A_S * w_S
```

Wenn der Prover bei S oder w_S lügt, stimmt die Gleichung nicht.

### 3.2 Sicherheitsanalyse

**Soundness:**
- Angreifer will: y = A_S' * w_S' für falsches S' oder w_S'
- Wenn A zufällig ist: Pr[A_S * w_S = A_S' * w_S'] ≤ 1/|F| (vernachlässigbar)
- **Ergebnis:** Soundness ist garantiert.

**Zero-Knowledge:**
- y = A_S * w_S ist eine lineare Funktion von w_S
- Wenn A öffentlich ist, könnte y Information über w_S leaken
- **Lösung:** Randomisiere y mit einem Blinding-Faktor

### 3.3 Komplexitätsvergleich

| Metrik | Ansatz B (Sparse) | Ansatz C (CS-Inspired) | Verbesserung |
|--------|-------------------|------------------------|--------------|
| Proof-Größe | O(k + λ log n) | O(k + m) | ~ gleich |
| Prover-Zeit | O(n + k) | O(n + k) | gleich |
| Verifier-Zeit | O(k + λ) | O(k + m) | ~ gleich |
| Soundness-Fehler | 2^(-λ) | 2^(-m) | gleich |
| Stichproben nötig | Ja (λ) | Nein | ✅ Besser |

**Hauptvorteil:** Keine Stichproben nötig. Die Sketch-Prüfung ist **deterministisch**.

## 4. Mathematische Formalisierung

### 4.1 Protokoll: CS-Inspired Sparse Witness Commitment

**Setup:**
```
- Feld F mit |F| > 2^λ (Sicherheitsparameter)
- Sensing Matrix A ∈ F^(m×n), wobei m = O(λ)
- Commitment-Schema Commit: F^k → G (z.B. Pedersen)
```

**Prover:**
```
Input: Witness w ∈ F^n (k-sparse)

1. Berechne Support S = {i : w_i ≠ 0}
2. Extrahiere w_S = (w_i)_{i ∈ S}
3. Berechne Sketch y = A * w
4. Generiere Commitments:
   - C_S = Commit(S)
   - C_w = Commit(w_S)
   - C_y = Commit(y)
5. Sende (C_S, C_w, C_y) an Verifier
```

**Verifier:**
```
Input: Commitments (C_S, C_w, C_y), öffentliche Eingabe x

1. Fordere Öffnung von C_S → S
2. Fordere Öffnung von C_w → w_S
3. Fordere Öffnung von C_y → y
4. Berechne y' = A_S * w_S (nur Spalten von A für Indizes in S)
5. Prüfe: y == y'
6. Prüfe: Die Berechnung mit w ist korrekt (via Circuit-Check)
7. Akzeptiere wenn alle Prüfungen bestanden
```

### 4.2 Sicherheitsbeweis (Skizze)

**Theorem:** Das Protokoll ist sound mit Fehlerwahrscheinlichkeit ≤ |S|/|F|.

**Beweis:**
- Angenommen, der Prover sendet falsches S' oder w_S'
- Er muss y finden, sodass y = A_S' * w_S'
- Aber y wurde bereits committed (C_y)
- Für zufälliges A: Pr[A_S * w_S = A_S' * w_S'] ≤ |S|/|F|
- Mit |F| > 2^λ und |S| = k: Fehler ≤ k/2^λ ≈ 0

## 5. Implementierungsplan

### 5.1 Komponenten

1. **Sensing Matrix Generator:**
   - Generiere sparse, strukturierte Matrix A
   - Optionen: Toeplitz, Circulant, oder Random Sparse

2. **Sketch Computation:**
   - Effiziente Sparse-Matrix-Vektor-Multiplikation
   - Komplexität: O(nnz(A)) = O(m * sparsity)

3. **Commitment Integration:**
   - Erweitere existierendes Commitment-Schema
   - Füge Sketch-Commitment hinzu

4. **Verifier Logic:**
   - Implementiere Sketch-Verifikation
   - Entferne Stichproben-Logik

### 5.2 Erwartete Ergebnisse

| Metrik | Vorher (Sparse) | Nachher (CS-Inspired) |
|--------|-----------------|----------------------|
| Stichproben | λ = 128 | 0 |
| Verifier-Operationen | O(k + 128) | O(k + m) |
| Determinismus | Nein (probabilistisch) | Ja |
| Soundness | 2^(-128) | k/2^256 ≈ 0 |

## 6. Fazit

### 6.1 Machbarkeit
**Ja, der CS-inspirierte Ansatz ist machbar und bietet einen konkreten Vorteil:**
- Deterministische Verifikation statt probabilistischer Stichproben
- Gleiche asymptotische Komplexität
- Stärkere Soundness-Garantie

### 6.2 Innovationsgrad
**Mittel.** Die Idee kombiniert bekannte Techniken (Sparse Commitments, Linear Sketching) auf eine neue Weise. Es ist keine fundamentale Innovation, aber eine praktische Verbesserung.

### 6.3 Empfehlung
**Implementieren.** Der Aufwand ist gering (wenige Tage), und das Ergebnis ist eine messbare Verbesserung des Systems.
