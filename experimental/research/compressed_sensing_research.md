# Compressed Sensing Recherche für zkML

## 1. Compressed Sensing Grundlagen

### 1.1 Kernkonzept
Compressed Sensing (CS) ermöglicht die Rekonstruktion eines Signals aus weniger Messungen als die Nyquist-Rate erfordert, **wenn das Signal sparse ist**.

**Mathematische Formulierung:**
```
Signal: x ∈ ℝⁿ (sparse, d.h. nur k << n Einträge sind nicht-null)
Messmatrix: A ∈ ℝᵐˣⁿ (m << n)
Messungen: y = Ax ∈ ℝᵐ

Rekonstruktion: Finde x aus y, obwohl m < n (unterbestimmt)
```

### 1.2 Restricted Isometry Property (RIP)
Eine Matrix A erfüllt RIP mit Konstante δ, wenn für alle k-sparse Vektoren x:
```
(1 - δ)||x||² ≤ ||Ax||² ≤ (1 + δ)||x||²
```

**Bedeutung:** Wenn A die RIP erfüllt, kann x aus y eindeutig rekonstruiert werden.

### 1.3 Rekonstruktionsalgorithmen
- **L1-Minimierung:** min ||x||₁ subject to Ax = y
- **Orthogonal Matching Pursuit (OMP):** Greedy-Algorithmus
- **Iterative Hard Thresholding (IHT):** Gradientenabstieg mit Thresholding

## 2. Existierende Arbeit: SpaGKR (2024)

### 2.1 Überblick
Paper: "Sparsity-Aware Protocol for ZK-friendly ML Models" (Li, Liang, Dong - 2024)

**Kernidee:** Nutze Sparsity in ML-Modellen (durch Pruning/Quantization) für effizientere ZK-Proofs.

### 2.2 Ansatz
- Basiert auf GKR-Protokoll (Goldwasser-Kalai-Rothblum)
- Speziell für Linear Layers mit sparsem Gewichtsmatrizen
- Erreicht **asymptotisch optimale Prover-Zeit** für sparse Matrizen

### 2.3 Unterschied zu unserer Idee
SpaGKR nutzt Sparsity in den **Gewichten** (Modellparameter).
Unsere Idee: Nutze Sparsity im **Witness** (Zwischenwerte der Berechnung).

## 3. Unsere Idee: CS für Witness-Kompression

### 3.1 Beobachtung
In neuronalen Netzen mit ReLU/GELU sind die Aktivierungen oft sparse:
- ReLU: ~50% der Aktivierungen sind 0
- Nach Pruning: ~90% können 0 sein

Der **Witness** (alle Zwischenwerte) ist also ein **sparse Signal**.

### 3.2 Hypothese
Statt den vollen Witness zu committen, könnten wir:
1. Eine Sensing-Matrix A anwenden: y = A * witness
2. Nur y committen (m << n)
3. Bei Verifikation: Rekonstruiere witness aus y

### 3.3 Potenzielle Vorteile
- **Proof-Größe:** O(m) statt O(n), wobei m ~ k log(n/k)
- **Prover-Zeit:** O(n) für Matrixmultiplikation (gleich)
- **Verifier-Zeit:** O(m) für Commitment-Check (besser)

### 3.4 Herausforderungen
1. **Soundness:** Kann ein Angreifer einen falschen Witness finden, der die gleichen Messungen y produziert?
2. **Zero-Knowledge:** Leaken die Messungen y Information über den Witness?
3. **Rekonstruktion:** Muss der Verifier den Witness rekonstruieren? (teuer!)

## 4. Kritische Analyse

### 4.1 Das Soundness-Problem
In CS ist die Rekonstruktion **nicht eindeutig** für nicht-sparse Signale. Ein Angreifer könnte:
1. Einen falschen (nicht-sparsem) Witness w' konstruieren
2. Der zufällig die gleichen Messungen y = Aw' produziert
3. Der Verifier würde akzeptieren, obwohl w' falsch ist

**Lösung:** Der Verifier muss prüfen, dass der rekonstruierte Witness tatsächlich sparse ist.

### 4.2 Das Zero-Knowledge-Problem
Die Messungen y = Aw enthalten Information über w. Selbst wenn A zufällig ist, könnte ein Angreifer:
1. Aus y und A Rückschlüsse auf w ziehen
2. Insbesondere die Position der Nicht-Null-Einträge könnte leaken

**Lösung:** Zusätzliche Randomisierung (z.B. y = Aw + r, wobei r ein Rauschvektor ist).

### 4.3 Das Rekonstruktions-Problem
L1-Minimierung ist O(n³) – viel zu teuer für den Verifier!

**Lösung:** Der Verifier rekonstruiert nicht. Stattdessen:
1. Prover sendet y und einen "Sparsity-Proof"
2. Verifier prüft nur, dass y konsistent mit dem Commitment ist

## 5. Revidierte Idee: CS-Inspired Witness Commitment

### 5.1 Neuer Ansatz
Statt CS direkt anzuwenden, nutzen wir die **Struktur** von CS:

1. **Prover:**
   - Berechnet Witness w (sparse)
   - Identifiziert Nicht-Null-Positionen: S = {i : w_i ≠ 0}
   - Committet nur w_S (die Nicht-Null-Einträge)
   - Committet S (die Positionen)

2. **Verifier:**
   - Prüft Commitment von w_S
   - Prüft, dass S konsistent mit der Berechnung ist
   - Prüft stichprobenartig, dass w_i = 0 für i ∉ S

### 5.2 Komplexität
- **Proof-Größe:** O(k) statt O(n), wobei k = |S| (Anzahl Nicht-Null-Einträge)
- **Prover-Zeit:** O(n) für Berechnung + O(k) für Commitment
- **Verifier-Zeit:** O(k) für Commitment-Check + O(λ) für Stichproben

### 5.3 Sicherheit
- **Soundness:** Wenn der Prover bei S lügt, wird er mit hoher Wahrscheinlichkeit bei den Stichproben erwischt.
- **Zero-Knowledge:** S und w_S werden durch Commitments verborgen.

## 6. Fazit der Recherche

### 6.1 Direkte CS-Anwendung: Nicht praktikabel
Die direkte Anwendung von Compressed Sensing auf Witness-Kompression scheitert an:
- Soundness-Problemen (nicht-eindeutige Rekonstruktion)
- Verifier-Komplexität (L1-Minimierung ist zu teuer)

### 6.2 CS-Inspirierte Optimierung: Vielversprechend
Die Kernidee von CS (Sparsity ausnutzen) kann indirekt angewendet werden:
- Nur Nicht-Null-Einträge committen
- Stichprobenartige Verifikation der Nullen
- Das ist im Wesentlichen, was wir bereits mit "Sparse Proofs" machen!

### 6.3 Erkenntnis
**Unsere existierende Sparse-Proof-Implementierung ist bereits die praktikable Version von "CS für Witness".**

Die Frage ist: Können wir sie weiter verbessern?

## 7. Nächste Schritte

1. **Formalisierung:** Die Sparse-Proof-Idee mathematisch rigoros formulieren
2. **Optimierung:** Effizientere Datenstrukturen für sparse Witnesses
3. **Benchmark:** Vergleich mit SpaGKR und anderen Ansätzen
4. **Publikation:** Falls die Ergebnisse signifikant sind
