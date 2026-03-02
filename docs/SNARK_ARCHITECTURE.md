# SNARK-Architekturplan: PLONK-Integration

**Autor**: Manus AI
**Datum**: 26. Januar 2026
**Status**: Entwurf

## 1. Zusammenfassung

Dieses Dokument beschreibt die technische Architektur für die Integration eines **PLONK** (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge) ZK-SNARK-Systems in das bestehende zkML-Framework. Ziel ist es, den aktuellen, nicht universellen und nicht vollständig Zero-Knowledge-fähigen Proof-Mechanismus durch ein produktionsreifes, universelles SNARK-System zu ersetzen. Die Entscheidung für PLONK basiert auf dessen universellem und aktualisierbarem Trusted Setup sowie seiner wachsenden Popularität und Flexibilität im Vergleich zu Alternativen wie Groth16.

## 2. Problemstellung und Ziele

Das derzeitige Proof-System weist mehrere Einschränkungen auf, die einen produktiven Einsatz verhindern:

- **Keine echte Zero-Knowledge-Eigenschaft**: Der vereinfachte Schnorr-artige Proof legt zwar nicht den gesamten Witness offen, bietet aber nicht die formalen Garantien eines echten ZK-SNARKs.
- **Keine On-Chain-Verifizierbarkeit**: Das Protokoll ist nicht für eine effiziente Verifikation in Smart Contracts ausgelegt.
- **Nicht-universelles Setup**: Ein neues Setup wäre für jede Änderung an der Netzwerkarchitektur erforderlich, was in der Praxis unhaltbar ist.

**Projektziele**:

1.  **Implementierung eines vollständigen PLONK-Systems**: Einschließlich Prover, Verifier und einem universellen Trusted Setup (SRS).
2.  **Gewährleistung der Zero-Knowledge-Eigenschaft**: Sicherstellen, dass keine Informationen über den privaten Witness aus dem Proof abgeleitet werden können.
3.  **Effiziente Verifikation**: Der Verifier muss so performant sein, dass eine On-Chain-Verifikation mit vertretbaren Gaskosten möglich ist.
4.  **Universalität**: Das System muss mit einem einzigen Trusted Setup für verschiedene neuronale Netze (innerhalb einer bestimmten Größenordnung) funktionieren.

## 3. Architekturentscheidung: PLONK

Die Wahl fiel auf PLONK gegenüber dem etablierteren Groth16 aus mehreren strategischen Gründen. Die folgende Tabelle fasst die wichtigsten Entscheidungskriterien zusammen:

| Kriterium | Groth16 | PLONK | Begründung für PLONK |
| :--- | :--- | :--- | :--- |
| **Trusted Setup** | Pro Circuit | **Universal & Aktualisierbar** | Ein einziges Setup für alle Circuits, was die operative Komplexität drastisch reduziert. |
| **Proof-Größe** | **~192 bytes** | ~400-600 bytes | Obwohl die Proofs größer sind, ist der Unterschied für die meisten Anwendungen akzeptabel. |
| **Verifikationszeit** | **~2-3 ms** | ~5-10 ms | Die etwas längere Verifikationszeit wird durch die immense Flexibilität aufgewogen. |
| **Flexibilität** | Gering | **Hoch** | PLONK unterstützt benutzerdefinierte Gates und ist besser für komplexe, wiederholte Strukturen geeignet. |
| **Entwickler-Ökosystem** | Etabliert | **Wachsend** | PLONK gewinnt schnell an Popularität und wird von modernen Frameworks wie Aztec und Zcash stark unterstützt. |

**Fazit**: Die Flexibilität und der universelle Charakter von PLONK sind für ein erweiterbares zkML-System, das verschiedene Modellarchitekturen unterstützen soll, von entscheidender Bedeutung und wiegen die Nachteile bei Proof-Größe und Verifikationszeit auf.

## 4. Systemarchitektur

Die Integration des PLONK-Systems erfordert eine Erweiterung der bestehenden Architektur um mehrere neue Kernkomponenten. Das folgende Diagramm zeigt den Datenfluss vom neuronalen Netz bis zum finalen, verifizierbaren Proof.

```
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│  Neural Network   │──────▶│ R1CS Representation ├──────▶│ PLONK Arithmetization │
│ (z.B. LeNet-5)    │       │ (Bestehend)       │       │ (NEU: Compiler)       │
└───────────────────┘       └───────────────────┘       └───────────────────┘
        │                                                         │
        │ (Forward Pass)                                          │
        ▼                                                         ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│   Witness Vektor  │       │  Universal SRS    │       │    PLONK Prover   │
│   (Privat)        │◀──────│ (Trusted Setup)   │──────▶│   (NEU)           │
└───────────────────┘       └───────────────────┘       └───────────────────┘
                                                                │
                                                                ▼
                                                        ┌───────────────────┐
                                                        │    PLONK Proof    │
                                                        └───────────────────┘
                                                                │
                                                                ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│  Universal SRS    │──────▶│   PLONK Verifier  │──────▶│  Accept / Reject  │
│ (Trusted Setup)   │       │   (NEU)           │       │                   │
└───────────────────┘       └───────────────────┘       └───────────────────┘
```

### 4.1 Neue und modifizierte Module

Die Implementierung erfordert die Schaffung eines neuen `snark`-Moduls sowie eines `compiler`-Moduls, das die Brücke zwischen R1CS und PLONK schlägt.

```plaintext
zkml_system/
├── snark/                  # NEU: SNARK-spezifische Logik
│   ├── __init__.py
│   ├── srs.py              # Laden und Verwalten des Structured Reference String (SRS)
│   ├── polynomial.py       # Arithmetik für Polynome in Lagrange- und Koeffizientenform
│   ├── commitment.py       # KZG Polynomial Commitment Scheme
│   ├── fft.py              # Fast Fourier Transform für Polynom-Konvertierungen
│   ├── plonk_prover.py     # Implementierung des PLONK Provers
│   ├── plonk_verifier.py   # Implementierung des PLONK Verifiers
│   └── transcript.py       # Fiat-Shamir-Transkript für Interaktivität
│
├── compiler/               # NEU: Circuit-Kompilierung
│   ├── __init__.py
│   ├── r1cs_to_plonk.py    # Konvertiert R1CS-Constraints in PLONK-Gates
│   └── gate_selector.py    # Definiert die PLONK-Gates (qL, qR, qO, qM, qC)
│
├── proof/                  # MODIFIZIERT
│   ├── prover.py           # Wird den plonk_prover aufrufen
│   └── verifier.py         # Wird den plonk_verifier aufrufen
```

## 5. Kernkomponenten im Detail

### 5.1 R1CS-zu-PLONK-Compiler

Dies ist eine entscheidende Komponente, die die bestehende R1CS-Darstellung in das für PLONK erforderliche arithmetische Circuit-Format umwandelt. Ein R1CS-Constraint `A(x) * B(x) - C(x) = 0` wird in ein oder mehrere PLONK-Gates `qL·a + qR·b + qO·c + qM·(a·b) + qC = 0` übersetzt.

**Beispiel: Multiplikations-Gate**

- **R1CS**: `w[i] * w[j] - w[k] = 0`
- **PLONK**: `a = w[i]`, `b = w[j]`, `c = w[k]`
  - `qL=0`, `qR=0`, `qO=-1`, `qM=1`, `qC=0`
  - `0·a + 0·b - 1·c + 1·(a·b) + 0 = 0`  => `a·b - c = 0`

### 5.2 KZG Polynomial Commitment

Das Herzstück von PLONK ist das KZG-Commitment-Schema, das es dem Prover ermöglicht, sich auf ein Polynom festzulegen und später zu beweisen, dass es an einer bestimmten Stelle einen bestimmten Wert hat, ohne das gesamte Polynom preiszugeben.

```python
class KZGCommitment:
    """
    Implementiert das Kate-Zaverucha-Goldberg Polynomial Commitment Scheme.
    Benötigt einen Structured Reference String (SRS) für die Operationen.
    """

    def __init__(self, srs: SRS):
        self.srs = srs

    def commit(self, polynomial: Polynomial) -> G1Point:
        """Erstellt ein Commitment zu einem Polynom p(x) als [p(τ)]₁.

        Dies geschieht durch eine Skalarmultiplikation der SRS-Punkte
        mit den Koeffizienten des Polynoms.
        """
        # C = Σ p_i * srs.g1_powers[i]
        pass

    def open(self, polynomial: Polynomial, z: FieldElement) -> G1Point:
        """Erstellt einen Öffnungs-Proof für den Wert p(z).

        Berechnet das Quotientenpolynom q(x) = (p(x) - p(z)) / (x - z)
        und committet zu diesem: π = [q(τ)]₁.
        """
        pass

    def verify(self, commitment: G1Point, z: FieldElement, value: FieldElement, proof: G1Point) -> bool:
        """Verifiziert einen Öffnungs-Proof.

        Prüft die Pairing-Gleichung: e(C - [value]₁, [1]₂) = e(proof, [τ - z]₂).
        """
        # lhs = pairing(commitment - G1.generator() * value, srs.g2_powers[0])
        # rhs = pairing(proof, srs.g2_powers[1] - G2.generator() * z)
        # return lhs == rhs
        pass
```

### 5.3 PLONK Prover-Protokoll (vereinfacht)

1.  **Arithmetization**: Konvertiere R1CS in PLONK-Gates und Witness-Werte in Polynome `a(x)`, `b(x)`, `c(x)`.
2.  **Round 1 (Commitment)**: Committe zu den Witness-Polynomen `a(x)`, `b(x)`, `c(x)` mittels KZG. Sende die Commitments an den Verifier (via Transkript).
3.  **Round 2 (Permutation)**: Erzeuge das Permutationspolynom `z(x)`, das die Korrektheit der "Kopier-Constraints" (d.h. die Verbindungen zwischen den Gates) sicherstellt. Committe zu `z(x)`.
4.  **Round 3 (Quotient)**: Kombiniere alle Gate-, Permutations- und Public-Input-Constraints in einem einzigen großen Polynom `t(x)`. Dieses `t(x)` ist nur dann null an den relevanten Punkten, wenn alle Constraints erfüllt sind. Berechne das Quotientenpolynom `q(x) = t(x) / Z_H(x)`, wobei `Z_H(x)` das Vanishing-Polynom ist.
5.  **Round 4 (Linearization)**: Linearsiere die Gleichungen, um die Anzahl der teuren `open`-Operationen zu reduzieren.
6.  **Round 5 (Opening)**: Erzeuge die KZG-Öffnungs-Proofs für alle Polynome an den vom Verifier (via Transkript) gewählten Zufallspunkten.

## 6. Implementierungsplan und Meilensteine

Die Implementierung wird in mehreren Phasen durchgeführt, die auf den im Haupt-Roadmap-Dokument definierten Abhängigkeiten aufbauen.

| Phase | Aufgabe | Dauer (Wochen) | Deliverable |
| :--- | :--- | :--- | :--- |
| 1 | **Grundlagen**: Polynom-Arithmetik, FFT | 1 | Ein getestetes `polynomial.py`-Modul. |
| 2 | **Commitment**: KZG-Schema | 1 | Ein `commitment.py`-Modul mit `commit`, `open`, `verify`. |
| 3 | **Compiler**: R1CS-zu-PLONK-Konverter | 2 | Ein `compiler`-Modul, das R1CS in PLONK-Gates umwandelt. |
| 4 | **Prover**: Implementierung des PLONK-Provers | 2 | Ein `plonk_prover.py`, der einen gültigen Proof erzeugen kann. |
| 5 | **Verifier**: Implementierung des PLONK-Verifiers | 1 | Ein `plonk_verifier.py`, der einen gültigen Proof akzeptiert. |
| 6 | **Integration**: Anbindung an das zkML-System | 1 | Die `prove()`- und `verify()`-Funktionen nutzen das neue SNARK-System. |

**Gesamtdauer**: 8 Wochen

## 7. Risiken

- **Komplexität der Kryptographie**: Die Implementierung von elliptischen Kurven, Pairings und dem PLONK-Protokoll ist fehleranfällig. **Mitigation**: Strenge Test-Driven-Entwicklung (TDD), Cross-Validierung gegen etablierte Bibliotheken (z.B. `py_ecc`) und Einholung von externem Review.
- **Performance**: Eine naive Implementierung der Polynom-Arithmetik oder der Skalarmultiplikationen kann das System unbrauchbar langsam machen. **Mitigation**: Einsatz von optimierten Algorithmen wie der Number-Theoretic Transform (NTT) für Polynom-Multiplikation und Benchmarking jeder Komponente.
- **Sicherheit des Trusted Setups**: Obwohl PLONK ein universelles Setup hat, muss der Prozess zur Erzeugung des SRS (Structured Reference String) sicher sein. **Mitigation**: Verwendung eines öffentlich verifizierbaren Multi-Party Computation (MPC) Zeremonie-Outputs, wie dem von der Zcash- oder Aztec-Community erzeugten.

## 8. Referenzen

[1] Gabizon, A., Williamson, Z. J., & Ciobotaru, O. (2019). *PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge*. [ePrint 2019/953](https://eprint.iacr.org/2019/953)
[2] Kate, A., Zaverucha, G., & Goldberg, I. (2010). *Constant-Size Commitments to Polynomials and Their Applications*. [ASIACRYPT 2010](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)
[3] Vitalik Buterin (2019). *Understanding PLONK*. [blog.ethereum.org](https://blog.ethereum.org/2019/09/22/plonk-by-hand-part-1)
