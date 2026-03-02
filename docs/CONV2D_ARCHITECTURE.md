# Conv2D-Layer-Architektur für zkML

**Autor**: Manus AI
**Datum**: 26. Januar 2026
**Status**: Entwurf

## 1. Zusammenfassung

Dieses Dokument beschreibt die technische Architektur für die Erweiterung des zkML-Systems um **Convolutional Neural Network (CNN)**-Komponenten. Die derzeitige Implementierung ist auf Dense-Layer beschränkt, was die Anwendung auf einfache Vektor-Daten limitiert. Die Einführung von Convolutional- (Conv2D), Pooling- und Batch-Normalization-Layern ist entscheidend, um das System für Standard-Computer-Vision-Aufgaben wie die Bildklassifikation (z.B. mit MNIST oder CIFAR-10) zu ertüchtigen. Der Fokus liegt dabei auf der **Constraint-Optimierung**, um die für ZK-SNARKs untragbar hohen Kosten von naiven CNN-Implementierungen zu reduzieren.

## 2. Problemstellung und Ziele

Die direkte Übersetzung von CNN-Operationen in ein R1CS-System führt zu einer Explosion der Constraint-Anzahl, was die Proof-Generierung extrem langsam und teuer macht.

-   **Convolution**: Eine einzelne Faltungsoperation mit einem 3x3-Kernel über 64 Kanäle erfordert bereits `3 * 3 * 64 = 576` Multiplikations-Constraints pro Output-Pixel.
-   **ReLU-Aktivierung**: Wie bereits im bestehenden System analysiert, erfordert eine ReLU-Aktivierung hunderte von Constraints.
-   **Max-Pooling**: Eine `max(a, b)`-Operation ist ein Vergleich, der ebenfalls eine teure Bit-Dekomposition erfordert und hunderte Constraints kostet.

**Projektziele**:

1.  **Implementierung von CNN-Basisschichten**: Schaffung von `Conv2D`, `Pooling` und `BatchNorm` Layern, die nahtlos in den `NetworkBuilder` integriert werden können.
2.  **Constraint-Effizienz**: Systematische Anwendung von Optimierungstechniken, um die Anzahl der R1CS-Constraints auf ein praktikables Maß zu reduzieren.
3.  **Unterstützung von Standardmodellen**: Ermöglichung der Implementierung von klassischen CNN-Architekturen wie LeNet-5.
4.  **Modularität**: Die neuen Layer sollen als eigenständige Komponenten implementiert werden, die unabhängig getestet und konfiguriert werden können.

## 3. Architektur und Constraint-Optimierung

Die Architektur der CNN-Erweiterung konzentriert sich auf die Minimierung der erzeugten Constraints. Das folgende Diagramm zeigt den Datenfluss und die an jeder Stelle angewandten Optimierungen.

```
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│   Input Image     │──────▶│  Conv2D Layer     │──────▶│  Fused BatchNorm  │
│   (H x W x C)     │       │ (Optimized Kernel)│       │ (NEU)             │
└───────────────────┘       └───────────────────┘       └───────────────────┘
                                                                │
                                                                ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│   GELU Activation │◀──────│  Avg-Pooling      │◀──────│  Output Feature   │
│ (Constraint-arm)  │       │ (statt Max-Pool)  │       │  Map              │
└───────────────────┘       └───────────────────┘       └───────────────────┘
```

### 3.1 Conv2D-Layer in R1CS

Eine Faltungsoperation ist im Kern eine Summe von Produkten (ein inneres Produkt) zwischen dem Kernel und einem Ausschnitt des Inputs. Für jeden Output-Pixel `y[i,j,k]` gilt:

`y[i,j,k] = Σ_{di,dj,c} x[i+di, j+dj, c] * w[di, dj, c, k] + bias[k]`

Jede Multiplikation `x * w` erzeugt ein R1CS-Constraint. Die Summation hingegen erzeugt keine neuen Constraints, da sie als eine einzige `LinearCombination` dargestellt werden kann.

**Optimierung: Winograd-Faltung**
Für kleine Kernel (insbesondere 3x3) kann der **Winograd-Algorithmus** die Anzahl der Multiplikationen reduzieren [1]. Er transformiert die Input-Patches und den Kernel in eine Domäne, in der die Faltung mit weniger Multiplikationen durchgeführt werden kann, und transformiert das Ergebnis dann zurück. Dies kann die Anzahl der Multiplikationen um den Faktor 2-4 reduzieren, führt aber zu einer komplexeren Constraint-Struktur.

### 3.2 Pooling-Layer: Average vs. Max

Pooling-Layer reduzieren die räumliche Dimension der Feature-Maps. Die Wahl des Pooling-Typs hat massive Auswirkungen auf die Constraint-Anzahl.

| Pooling-Typ | Operation | R1CS-Kosten | Empfehlung |
| :--- | :--- | :--- | :--- |
| **Max-Pooling** | `y = max(x1, x2, x3, x4)` | **Extrem hoch** (~1000 Constraints) | **Vermeiden** |
| **Average-Pooling** | `y = (x1+x2+x3+x4) / 4` | **Sehr niedrig** (1 Constraint für Division) | **Standard** |

Die `max`-Operation erfordert bitweise Vergleiche, was in R1CS extrem teuer ist. `AvgPool` hingegen ist eine rein lineare Operation (Summe) gefolgt von einer einzigen Multiplikation mit dem Inversen der Fenstergröße. Daher wird **Average-Pooling als Standard für alle zkML-CNNs empfohlen**.

### 3.3 Fused Batch Normalization

Batch Normalization (`BatchNorm`) normalisiert die Aktivierungen eines Layers. Die Formel lautet:

`y = γ * (x - μ) / σ + β`

Eine naive Implementierung würde mehrere Constraints für Subtraktion, Division und Multiplikation erfordern. Da `BatchNorm` jedoch eine lineare Operation ist, können die Parameter `γ, μ, σ, β` während der Inferenz in die Gewichte und den Bias des vorhergehenden `Conv2D`-Layers **hineingefaltet** (gefused) werden [2].

-   `w_fused = w * (γ / σ)`
-   `b_fused = (b - μ) * (γ / σ) + β`

Dadurch entstehen **null zusätzliche Constraints** für den `BatchNorm`-Layer während der ZK-Inferenz.

## 4. Neue und modifizierte Module

Die Implementierung erfolgt durch Erweiterung des bestehenden `network`-Moduls.

```plaintext
zkml_system/
├── network/
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── dense.py        # Bestehend
│   │   ├── conv2d.py       # NEU: Convolutional Layer
│   │   ├── pooling.py      # NEU: AvgPool2D Layer
│   │   ├── batchnorm.py    # NEU: Fused BatchNorm Logik
│   │   └── flatten.py      # NEU: Layer zum Abflachen von Tensoren
│   │
│   ├── builder.py          # MODIFIZIERT: Fügt `add_conv2d`, `add_pooling` etc. hinzu
│   │
│   └── models/             # NEU: Vordefinierte, optimierte CNN-Architekturen
│       ├── __init__.py
│       ├── lenet.py        # zk-optimierte LeNet-5 für MNIST
│       └── resnet_tiny.py  # zk-optimierte Mini-ResNet für CIFAR-10
```

## 5. Beispiel: zk-optimierte LeNet-5 Architektur

Die klassische LeNet-5-Architektur [3] wird als erstes Zielmodell implementiert, wobei alle genannten Optimierungen zur Anwendung kommen.

| Layer | Original LeNet-5 | zk-optimierte LeNet-5 | Constraint-Kosten (geschätzt) |
| :--- | :--- | :--- | :--- |
| C1 | Conv (6 @ 5x5) | Conv (6 @ 5x5) | `24*24*6 * (25*1 + 10) ≈ 120k` |
| S2 | **Max-Pool** | **Avg-Pool** | `12*12*6 * 1 ≈ 0.9k` |
| C3 | Conv (16 @ 5x5) | Conv (16 @ 5x5) | `8*8*16 * (25*6 + 10) ≈ 164k` |
| S4 | **Max-Pool** | **Avg-Pool** | `4*4*16 * 1 ≈ 0.3k` |
| F5 | Dense (120) | Dense (120) | `256*120 + 120*10 ≈ 32k` |
| F6 | Dense (84) | Dense (84) | `120*84 + 84*10 ≈ 11k` |
| Out | Dense (10) | Dense (10) | `84*10 ≈ 0.8k` |
| **Total** | | | **~329k Constraints** |

Eine naive Implementierung mit `ReLU` und `MaxPool` würde mehrere Millionen Constraints erfordern. Durch die Kombination von `GELU`, `AvgPool` und `Fused BatchNorm` wird die Architektur für ZK-SNARKs praktikabel.

## 6. Implementierungsplan und Meilensteine

| Phase | Aufgabe | Dauer (Wochen) | Deliverable |
| :--- | :--- | :--- | :--- |
| 1 | **Conv2D Layer**: Implementierung der Faltungslogik und Constraint-Generierung. | 1 | Ein getesteter `Conv2D`-Layer. |
| 2 | **Pooling & Flatten**: Implementierung von `AvgPool2D` und `Flatten`. | 0.5 | Getestete `AvgPool2D`- und `Flatten`-Layer. |
| 3 | **Fused BatchNorm**: Implementierung der Logik zum Verschmelzen der Parameter. | 0.5 | Eine Hilfsfunktion, die `BatchNorm` in `Conv2D` faltet. |
| 4 | **Builder-Integration**: Erweiterung des `NetworkBuilder` um die neuen Layer. | 0.5 | `add_conv2d`, `add_pooling` sind im Builder verfügbar. |
| 5 | **LeNet-5 Modell**: Implementierung des zk-optimierten LeNet-5-Modells. | 0.5 | Ein `LeNet5`-Modell, das aus den neuen Layern aufgebaut ist. |
| 6 | **End-to-End-Test**: Training des Modells und Generierung eines ZK-Proofs für eine MNIST-Inferenz. | 1 | Ein Demo-Skript, das den gesamten Prozess zeigt. |

**Gesamtdauer**: 4 Wochen

## 7. Risiken

- **Tensor-Management**: Die Verwaltung von mehrdimensionalen Daten (Tensoren) und deren korrekte Indizierung (Padding, Strides) im flachen Witness-Vektor ist komplex und fehleranfällig. **Mitigation**: Umfangreiche Unit-Tests mit kleinen, nachvollziehbaren Tensor-Größen und Vergleich der Ergebnisse mit Standard-Frameworks wie PyTorch.
- **Constraint-Anzahl**: Trotz Optimierungen könnte die Anzahl der Constraints für tiefere Modelle immer noch zu hoch sein. **Mitigation**: Frühzeitige Benchmarks mit realistischen Modellgrößen, um die Skalierbarkeit zu bewerten und ggf. weitere Optimierungen (wie strukturierte Pruning-Techniken) zu evaluieren.

## 8. Referenzen

[1] Lavin, A., & Gray, S. (2016). *Fast Algorithms for Convolutional Neural Networks*. [arXiv:1509.09308](https://arxiv.org/abs/1509.09308)
[2] Jacob, B., et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
[3] LeCun, Y., et al. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE. [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
