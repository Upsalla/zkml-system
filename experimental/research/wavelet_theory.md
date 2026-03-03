# Wavelet-basierte Constraint-Kompression: Theoretische Analyse

## 1. Wavelet-Grundlagen

### 1.1 Was sind Wavelets?

Wavelets sind lokalisierte Wellenfunktionen, die ein Signal in Zeit-Frequenz-Komponenten zerlegen. Im Gegensatz zur Fourier-Transformation, die nur globale Frequenzinformationen liefert, erfassen Wavelets sowohl **Frequenz als auch Position**.

Die **Diskrete Wavelet-Transformation (DWT)** zerlegt ein Signal `x` der Länge `n` in:
- **Approximationskoeffizienten (cA):** Niedrigfrequente Komponenten (grobe Struktur)
- **Detailkoeffizienten (cD):** Hochfrequente Komponenten (feine Details)

### 1.2 Mathematische Formulierung

Für ein Signal `x[n]` mit Länge `N`:

```
DWT(x) = [cA, cD]

cA[k] = Σ x[n] * h[2k - n]  (Tiefpass-Filter)
cD[k] = Σ x[n] * g[2k - n]  (Hochpass-Filter)
```

Wobei `h` und `g` die Wavelet-Filterkoeffizienten sind (z.B. Haar, Daubechies).

### 1.3 Kompression durch Thresholding

Die Kernidee der Wavelet-Kompression:
1. Transformiere das Signal in den Wavelet-Bereich
2. Setze kleine Koeffizienten auf Null (Thresholding)
3. Speichere nur die signifikanten Koeffizienten

Dies funktioniert, weil natürliche Signale oft **spärlich im Wavelet-Bereich** sind.

---

## 2. Anwendbarkeit auf R1CS-Constraints

### 2.1 R1CS-Struktur

Ein R1CS-System besteht aus Constraints der Form:
```
(A · w) ⊙ (B · w) = C · w
```

Wobei `A`, `B`, `C` spärliche Matrizen und `w` der Witness-Vektor ist.

### 2.2 Wo können Wavelets ansetzen?

**Option A: Witness-Kompression**
- Der Witness `w` könnte wavelet-transformiert werden
- Problem: Der transformierte Witness muss in den Constraints verwendet werden
- Die Constraints müssten entsprechend angepasst werden

**Option B: Constraint-Matrix-Kompression**
- Die Matrizen `A`, `B`, `C` könnten komprimiert werden
- Problem: Die Matrizen sind bereits spärlich (wenige Nicht-Null-Einträge)
- Wavelets sind für dichte, strukturierte Daten optimiert

**Option C: Aktivierungs-Kompression (zkML-spezifisch)**
- Die Aktivierungsvektoren zwischen Schichten sind oft strukturiert
- Wavelets könnten diese komprimieren, bevor sie in Constraints eingehen
- **Dies ist der vielversprechendste Ansatz**

### 2.3 Kritische Analyse

**Problem 1: Linearität**
- Wavelets sind lineare Transformationen
- R1CS-Constraints sind bilinear (Multiplikation)
- Die Wavelet-Transformation kann nicht direkt in R1CS eingebettet werden, ohne zusätzliche Constraints

**Problem 2: Finite Fields**
- Wavelets arbeiten typischerweise über reellen Zahlen
- R1CS arbeitet über endlichen Feldern
- Die Wavelet-Koeffizienten müssen in Feldelemente konvertiert werden

**Problem 3: Rekonstruktion**
- Der Verifier muss die Rekonstruktion verifizieren können
- Dies erfordert zusätzliche Constraints für die inverse DWT

### 2.4 Machbarkeits-Einschätzung

| Aspekt | Bewertung | Begründung |
|--------|-----------|------------|
| Mathematische Kompatibilität | ⚠️ Mittel | Linearität ist kompatibel, aber Finite-Field-Arithmetik erfordert Anpassungen |
| Erwarteter Vorteil | ⚠️ Unsicher | Hängt stark von der Struktur der Aktivierungen ab |
| Implementierungsaufwand | ✅ Niedrig | DWT ist einfach zu implementieren |
| Zusätzliche Constraints | ❌ Hoch | Rekonstruktion erfordert viele zusätzliche Constraints |

---

## 3. Realistischer Ansatz: Wavelet-Preprocessing

Anstatt Wavelets direkt in das Proof-System zu integrieren, können wir sie als **Preprocessing-Schritt** verwenden:

### 3.1 Konzept

1. **Offline:** Analysiere die Aktivierungsmuster des Modells mit Wavelets
2. **Identifiziere:** Welche Wavelet-Koeffizienten sind typischerweise klein?
3. **Optimiere:** Trainiere das Modell so, dass es spärlich im Wavelet-Bereich ist
4. **Zur Laufzeit:** Nutze die bekannte Sparsity-Struktur für effizientere Proofs

### 3.2 Alternativer Ansatz: Haar-Wavelet für Witness-Batching

Die **Haar-Wavelet-Transformation** ist besonders einfach:
```
cA[k] = (x[2k] + x[2k+1]) / √2
cD[k] = (x[2k] - x[2k+1]) / √2
```

Dies kann als **Witness-Batching** interpretiert werden:
- Statt `n` einzelne Werte zu committen, committe `n/2` Summen und `n/2` Differenzen
- Wenn die Differenzen klein sind (benachbarte Werte ähnlich), können sie effizient bewiesen werden

---

## 4. Fazit und Empfehlung

**Wavelets sind für R1CS-Constraint-Kompression weniger geeignet als ursprünglich angenommen.**

Die Gründe:
1. R1CS-Matrizen sind bereits spärlich, nicht dicht-strukturiert
2. Die Integration in Finite-Field-Arithmetik erfordert signifikanten Overhead
3. Der Vorteil gegenüber direkter Sparsity-Ausnutzung (wie CSWC) ist unklar

**Empfehlung:** 

Anstatt Wavelets für Constraint-Kompression zu verwenden, sollten wir den Ansatz **pivotieren**:

1. **Haar-Wavelet für Witness-Batching:** Ein einfacher Ansatz, der benachbarte Witness-Werte zusammenfasst
2. **Wavelet-basierte Modell-Analyse:** Offline-Tool zur Identifikation von Kompressionspotenzialen
3. **Fokus auf TDA:** Die topologische Analyse ist mathematisch besser geeignet für das Fingerprinting-Problem

**Entscheidung:** Implementiere einen minimalen Haar-Wavelet-Prototyp für Witness-Batching, um die Idee zu testen. Wenn der Vorteil marginal ist, gehe direkt zu TDA über.
