# TDA-Fingerprinting: Theoretische Analyse

## 1. Persistent Homology Grundlagen

### 1.1 Was ist Persistent Homology?

Persistent Homology ist eine Technik aus der **Topological Data Analysis (TDA)**, die die "Form" von Daten erfasst. Sie identifiziert topologische Features (Löcher, Hohlräume, verbundene Komponenten), die über verschiedene Skalen persistent sind.

**Kernkonzepte:**

1. **Simplicial Complex:** Eine Menge von Punkten, Kanten, Dreiecken, etc., die einen Raum approximieren
2. **Filtration:** Eine Sequenz von wachsenden Komplexen, parametrisiert durch einen Schwellenwert ε
3. **Betti-Zahlen:** β₀ = Anzahl verbundener Komponenten, β₁ = Anzahl Löcher, β₂ = Anzahl Hohlräume
4. **Persistence Diagram:** Visualisierung der "Geburt" und "Tod" topologischer Features

### 1.2 Mathematische Formulierung

Für eine Punktwolke P = {p₁, ..., pₙ}:

1. **Vietoris-Rips-Komplex:** Für jeden Schwellenwert ε, verbinde Punkte mit Abstand ≤ ε
2. **Filtration:** R₀ ⊆ R₁ ⊆ ... ⊆ Rₘ (wachsende Komplexe)
3. **Homologie:** Berechne Hₖ(Rᵢ) für jede Dimension k und jeden Schritt i
4. **Persistence:** Verfolge, wann Features entstehen (birth) und verschwinden (death)

Das **Persistence Diagram** ist eine Menge von Punkten (birth, death) in ℝ².

### 1.3 Stabilität

**Schlüsseleigenschaft:** Persistent Homology ist **stabil** unter kleinen Perturbationen.

> Wenn zwei Punktwolken P und Q einen Hausdorff-Abstand von δ haben, dann unterscheiden sich ihre Persistence Diagrams um höchstens δ (im Bottleneck-Abstand).

Dies macht Persistent Homology ideal für **Fingerprinting**: Kleine Änderungen am Modell führen zu kleinen Änderungen am Fingerprint.

---

## 2. Anwendung auf ML-Modelle

### 2.1 Modell als Punktwolke

Ein neuronales Netz kann als Punktwolke interpretiert werden:

**Option A: Gewichte als Punkte**
- Jedes Gewicht wᵢⱼ wird als Punkt in ℝ¹ behandelt
- Problem: Verliert Strukturinformation (welche Gewichte verbunden sind)

**Option B: Neuronen als Punkte**
- Jedes Neuron wird als Punkt in ℝᵈ behandelt, wobei d = Anzahl eingehender Gewichte
- Die Koordinaten sind die Gewichte zu diesem Neuron
- Erhält Strukturinformation

**Option C: Layer-Matrizen als Punkte**
- Jede Gewichtsmatrix Wₗ wird als Punkt in ℝ^(m×n) behandelt
- Sehr hochdimensional, aber kompakt

### 2.2 Fingerprint-Berechnung

```
Modell M → Punktwolke P → Filtration → Persistence Diagram D → Fingerprint F
```

Der **Fingerprint F** kann sein:
1. Das vollständige Persistence Diagram (variabel groß)
2. Eine feste Anzahl der persistentesten Features (konstante Größe)
3. Ein Hash des Diagrams (32-64 bytes)

### 2.3 Erwartete Eigenschaften

| Eigenschaft | Beschreibung | Status |
|-------------|--------------|--------|
| **Eindeutigkeit** | Verschiedene Modelle → verschiedene Fingerprints | ✅ Erwartet |
| **Stabilität** | Ähnliche Modelle → ähnliche Fingerprints | ✅ Garantiert |
| **Kompaktheit** | Fingerprint-Größe unabhängig von Modellgröße | ⚠️ Abhängig von Implementierung |
| **Effizienz** | Schnelle Berechnung | ⚠️ O(n³) für naive Implementierung |

---

## 3. ZK-Integration

### 3.1 Das Protokoll

**Ziel:** Beweise, dass ein bestimmtes Modell M verwendet wurde, ohne M offenzulegen.

**Ansatz:**
1. **Setup:** Berechne Fingerprint F(M) offline
2. **Commitment:** Committe zu F(M) on-chain
3. **Inference:** Führe Inferenz aus
4. **Proof:** Beweise, dass die Inferenz mit einem Modell durchgeführt wurde, dessen Fingerprint F(M) ist

### 3.2 Herausforderung: Fingerprint-Verifikation

Das Problem: Der Verifier muss verifizieren können, dass der Fingerprint korrekt aus dem Modell berechnet wurde.

**Option A: Fingerprint im Proof**
- Der Prover berechnet den Fingerprint als Teil des Proofs
- Problem: Persistent Homology ist komplex und erzeugt viele Constraints

**Option B: Commitment-basiert**
- Der Prover committet sich zum Modell M und zum Fingerprint F
- Der Verifier prüft nur, dass F konsistent mit dem Commitment ist
- Der eigentliche Fingerprint-Beweis erfolgt off-chain oder durch einen separaten SNARK

**Option C: Sampling-basiert**
- Der Verifier wählt zufällige Punkte aus dem Modell
- Der Prover beweist, dass diese Punkte zum Fingerprint beitragen
- Probabilistischer Beweis, aber effizient

### 3.3 Machbarkeits-Einschätzung

| Aspekt | Bewertung | Begründung |
|--------|-----------|------------|
| Mathematische Basis | ✅ Solide | Persistent Homology ist gut verstanden |
| Eindeutigkeit | ✅ Hoch | Topologische Invarianten sind robust |
| ZK-Integration | ⚠️ Komplex | Fingerprint-Berechnung in Circuit ist teuer |
| Praktischer Nutzen | ✅ Hoch | Löst ein echtes Problem (Modell-Authentizität) |

---

## 4. Implementierungsplan

### Phase 1: Persistent Homology Berechnung
- Implementiere Vietoris-Rips-Komplex
- Implementiere Boundary-Matrix-Reduktion
- Berechne Persistence Diagrams

### Phase 2: Modell-zu-Punktwolke-Konversion
- Implementiere verschiedene Konversionsstrategien
- Evaluiere Eindeutigkeit und Stabilität

### Phase 3: Fingerprint-Extraktion
- Extrahiere feste Anzahl persistentester Features
- Implementiere Hash-basiertes Commitment

### Phase 4: ZK-Integration
- Implementiere Commitment-Schema
- Implementiere Sampling-basierte Verifikation

---

## 5. Erwartetes Ergebnis

**Best Case:**
- O(1) Fingerprint-Größe (z.B. 64 bytes)
- Eindeutige Identifikation jedes Modells
- Effiziente Verifikation (wenige Constraints)

**Realistic Case:**
- O(k) Fingerprint-Größe, wobei k = Anzahl persistenter Features (typisch 10-100)
- Hohe Eindeutigkeit für unterschiedliche Architekturen
- Moderate Verifikationskosten

**Worst Case:**
- Fingerprint nicht eindeutig genug für Sicherheitsanwendungen
- Verifikation zu teuer für praktischen Einsatz

---

## 6. Fazit

TDA-Fingerprinting ist **mathematisch fundiert und vielversprechend**. Die Hauptherausforderung liegt in der effizienten ZK-Integration. Der Sampling-basierte Ansatz scheint am praktikabelsten.

**Empfehlung:** Implementiere einen Prototyp und evaluiere die Eindeutigkeit empirisch.
