# BN254 Integrationsarchitektur

**Autor**: Manus AI
**Datum**: 26. Januar 2026
**Status**: Entwurf

## 1. Zusammenfassung

Dieses Dokument beschreibt die technische Architektur für die Integration der elliptischen Kurve **BN254** in das zkML-System. Die Umstellung vom aktuellen Demonstrations-Primfeld (`p=101`) auf BN254 ist ein fundamentaler Schritt zur Erreichung von Produktionsreife. BN254 ist eine Pairing-freundliche Kurve, die für kryptographische Anwendungen mit einem Sicherheitsniveau von ca. 128 Bit ausgelegt ist und als De-facto-Standard im Ethereum-Ökosystem gilt. Die Integration umfasst die Implementierung der zugrundeliegenden endlichen Felder, der Kurvenarithmetik und der für ZK-SNARKs erforderlichen Pairing-Funktionen.

## 2. Problemstellung und Ziele

Das aktuelle System operiert über einem kleinen Primfeld (`p=101`), was für Demonstrations- und Testzwecke ausreichend ist, aber für einen produktiven Einsatz ungeeignet ist, da es keinerlei kryptographische Sicherheit bietet.

**Projektziele**:

1.  **Kryptographische Sicherheit**: Erreichen eines 128-Bit-Sicherheitsniveaus durch die Verwendung einer standardisierten, Pairing-freundlichen Kurve.
2.  **Ethereum-Kompatibilität**: Sicherstellen, dass die erzeugten Proofs mit den auf der Ethereum-Blockchain verfügbaren Precompiles für `ecAdd`, `ecMul` und `ecPairing` kompatibel sind.
3.  **Performance**: Implementierung hocheffizienter kryptographischer Primitive, die als Grundlage für das PLONK-SNARK-System dienen können, ohne einen signifikanten Performance-Overhead darzustellen.
4.  **Modularität**: Schaffung einer sauberen, in sich geschlossenen Kryptographie-Bibliothek, die unabhängig vom Rest des zkML-Systems getestet und gewartet werden kann.

## 3. BN254: Eine technische Einführung

BN254, auch bekannt als `alt_bn128`, ist eine Barreto-Naehrig-Kurve, die speziell für die effiziente Berechnung von Pairings konstruiert wurde [1]. Sie ist definiert über der Kurvengleichung `y² = x³ + 3`.

### 3.1 Hierarchie der mathematischen Strukturen

Die Implementierung von BN254 erfordert den Aufbau einer Hierarchie von mathematischen Strukturen, die aufeinander aufbauen:

```
        GT (Zielgruppe im Fp12-Feld)
        ▲
        │ (Pairing: e)
        │
┌───────┴───────┐
│               │
G1 (Punkte im Fp)   G2 (Punkte im Fp2)
▲               ▲
│               │
Fp (Basisfeld)    Fp2 (Erweiterungsfeld)
▲               ▲
│               │
└─── Fr (Skalarfeld) ───┘
```

-   **Fr**: Das Skalarfeld, in dem die Exponenten der Skalarmultiplikationen und die Koeffizienten der R1CS-Constraints leben.
-   **Fp**: Das Basisfeld, über dem die Koordinaten der G1-Punkte definiert sind.
-   **Fp2**: Ein Erweiterungsfeld `Fp[u]/(u² - β)`, das für die Koordinaten der G2-Punkte benötigt wird.
-   **Fp6** und **Fp12**: Weitere Turm-Erweiterungen, die für die Berechnung des Pairings notwendig sind.
-   **G1** und **G2**: Zwei unterschiedliche kryptographische Gruppen von Punkten auf der Kurve.
-   **GT**: Die Zielgruppe des Pairings, eine Untergruppe des multiplikativen Feldes Fp12.

### 3.2 Kurvenparameter

Die genauen Parameter sind entscheidend für die Kompatibilität.

| Parameter | Beschreibung | Wert (Ausschnitt) |
| :--- | :--- | :--- |
| `p` | Primmodul des Basisfeldes `Fp` | `21888242...8583` |
| `r` | Primmodul des Skalarfeldes `Fr` | `21888242...5617` |
| `b` | Kurvenparameter `y² = x³ + b` | `3` |
| `G1` | Generator der Gruppe G1 | `(1, 2)` |
| `G2` | Generator der Gruppe G2 | `((108570...2781, 11559...5634), (84956...1930, 40823...3531))` |

## 4. Architektur des Krypto-Moduls

Ein neues, dediziertes `crypto`-Modul wird eingeführt, das alle BN254-spezifischen Implementierungen kapselt.

```plaintext
zkml_system/
├── crypto/                 # NEU: Haupt-Kryptomodul
│   ├── __init__.py
│   ├── bn254/              # NEU: BN254-spezifische Implementierungen
│   │   ├── __init__.py
│   │   ├── field.py        # Implementierung von Fp, Fr, Fp2, Fp6, Fp12
│   │   ├── curve.py        # G1- und G2-Punktoperationen (add, mul, ...)
│   │   ├── pairing.py      # Implementierung des Optimal Ate Pairings
│   │   └── constants.py    # Alle Kurvenparameter und Generatoren
│   │
│   └── utils/              # NEU: Allgemeine kryptographische Hilfsfunktionen
│       ├── montgomery.py   # Montgomery-Reduktion für schnelle modulare Arithmetik
│       └── sqrt.py         # Tonelli-Shanks-Algorithmus für Quadratwurzeln
│
├── core/
│   └── field.py            # MODIFIZIERT: Wird die neue `Fr`-Implementierung nutzen
```

### 4.1 Kernkomponenten im Detail

#### 4.1.1 Modulare Arithmetik mit Montgomery-Reduktion

Standardmäßige modulare Multiplikation `(a * b) % p` ist langsam aufgrund der teuren Divisionsoperation. Die **Montgomery-Multiplikation** ersetzt diese Division durch schnellere Additionen und Bit-Shifts [2].

> Montgomery modular multiplication [...] is a method for performing fast modular multiplication. It replaces a division by the modulus with a multiplication and a few cheap bitwise operations. - Wikipedia

Alle Feldoperationen in `Fp` und `Fr` werden diese Technik verwenden, um maximale Performance zu erzielen.

```python
class FieldElement:
    # ...
    def __mul__(self, other: 'FieldElement') -> 'FieldElement':
        # Ruft die optimierte Montgomery-Multiplikation auf
        return montgomery_mul(self.value, other.value, self.field)
```

#### 4.1.2 Turm-Erweiterungsfelder (Tower Extension Fields)

Für das Pairing benötigen wir eine Kette von Erweiterungsfeldern. Diese werden als "Turm" implementiert, wobei jedes Feld auf dem vorherigen aufbaut:

-   `Fp2(u) = Fp[x] / (x² + 1)`
-   `Fp6(v) = Fp2[y] / (y³ - (u+1))`
-   `Fp12(w) = Fp6[z] / (z² - v)`

Jedes dieser Felder wird als Klasse implementiert, die die Arithmetik (Addition, Multiplikation, Inversion) für die jeweiligen Polynome kapselt.

#### 4.1.3 Optimal Ate Pairing

Das Pairing ist eine bilineare Abbildung `e: G1 × G2 → GT`. Wir werden das **Optimal Ate Pairing** implementieren, das als einer der effizientesten Pairing-Algorithmen gilt [3]. Der Algorithmus besteht aus zwei Hauptteilen:

1.  **Miller Loop**: Ein iterativer Algorithmus, der die Hauptberechnung durchführt und ein Element in `Fp12` erzeugt.
2.  **Final Exponentiation**: Eine abschließende Potenzierung, um das Ergebnis in die korrekte Untergruppe `GT` zu überführen und die Bilinearität sicherzustellen.

Die Komplexität und Korrektheit dieses Algorithmus sind von größter Bedeutung für das gesamte SNARK-System.

## 5. Performance-Optimierungen

Die Performance der kryptographischen Basisoperationen ist kritisch. Neben der Montgomery-Reduktion werden weitere Techniken eingesetzt:

| Technik | Ziel | Beschreibung |
| :--- | :--- | :--- |
| **Windowed NAF** | Skalarmultiplikation | Reduziert die Anzahl der Punktadditionen bei der Berechnung von `k * P` durch eine spezielle Darstellung des Skalars `k`. |
| **Jakobische Koordinaten** | Punktaddition/-verdopplung | Vermeidet teure modulare Inversionen bei jeder Addition/Verdopplung, indem die Inversion bis zum Ende aufgeschoben wird. |
| **Precomputation** | Skalarmultiplikation | Vorberechnung von Vielfachen des Generatorpunktes, um die Laufzeit von fixen Basis-Skalarmultiplikationen zu beschleunigen. |
| **Lazy Reduction** | Feldarithmetik | Aufschieben der modularen Reduktion über mehrere Additionen hinweg, um die Anzahl der Modulo-Operationen zu verringern. |

## 6. Implementierungsplan und Meilensteine

Die Implementierung ist in logische, aufeinander aufbauende Phasen unterteilt.

| Phase | Aufgabe | Dauer (Wochen) | Deliverable |
| :--- | :--- | :--- | :--- |
| 1 | **Feldarithmetik**: Implementierung von `Fp` und `Fr` mit Montgomery-Arithmetik. | 1 | Ein getestetes `field.py`-Modul für die Basis- und Skalarfelder. |
| 2 | **Erweiterungsfelder**: Implementierung des Turms `Fp2`, `Fp6`, `Fp12`. | 1 | Vollständige Arithmetik für alle Erweiterungsfelder. |
| 3 | **Kurvenarithmetik**: Implementierung von `G1`- und `G2`-Punktoperationen. | 1.5 | Ein `curve.py`-Modul mit Addition, Verdopplung und Skalarmultiplikation. |
| 4 | **Pairing**: Implementierung des Optimal Ate Pairings (Miller Loop & Final Exp.). | 1.5 | Eine `pairing()`-Funktion, die den Bilinearitätstest besteht. |
| 5 | **Integration & Refactoring**: Umstellung des bestehenden Systems auf `Fr`. | 1 | Das gesamte zkML-System nutzt die neue Krypto-Bibliothek. |

**Gesamtdauer**: 6 Wochen

## 7. Risiken und Mitigation

- **Implementierungsfehler**: Die Komplexität der Kryptographie ist extrem hoch. Ein kleiner Fehler kann die Sicherheit des gesamten Systems kompromittieren. **Mitigation**: 100% Testabdeckung mit Vektoren aus etablierten Quellen (z.B. Ethereum-Tests, `py_ecc`), Code-Reviews durch mehrere Entwickler.
- **Seitenkanalangriffe**: Eine nicht-konstante Laufzeitimplementierung von kryptographischen Operationen kann Informationen über die privaten Schlüssel leaken. **Mitigation**: Sicherstellen, dass alle Operationen (insbesondere Skalarmultiplikation) in konstanter Zeit ablaufen, unabhängig von den Eingabewerten.
- **Performance-Engpässe**: Eine suboptimale Implementierung könnte das System für größere neuronale Netze unbrauchbar machen. **Mitigation**: Kontinuierliches Benchmarking jeder Komponente und Vergleich mit Referenzimplementierungen.

## 8. Referenzen

[1] Beuchat, J. L., et al. (2010). *High-Speed Software Implementation of the Optimal Ate Pairing over a Barreto-Naehrig Curve*. [ePrint 2010/354](https://eprint.iacr.org/2010/354.pdf)
[2] Montgomery, P. L. (1985). *Modular Multiplication Without Trial Division*. Mathematics of Computation. [PDF](https://www.ams.org/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf)
[3] Vercauteren, F. (2009). *Optimal Pairings*. IEEE Transactions on Information Theory. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/5361495/)
[4])
[4] Buterin, V. (2017). *Exploring Elliptic Curve Pairings*. [vitalik.ca](https://vitalik.ca/general/2017/01/14/exploring_ec_pairings.html)
