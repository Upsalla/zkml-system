# zkML System: Architekturplan zur Produktionsreife

**Autor**: David Weyhe  
**Version**: 1.0  
**Datum**: 27. Januar 2026  
**Status**: Aktiv

---

## Executive Summary

Nach kritischer Analyse des bestehenden Systems identifiziere ich folgende Architektur-Defizite, die eine Produktionsreife verhindern:

| Defizit | Schweregrad | Aufwand | Priorität |
|---------|-------------|---------|-----------|
| KZG-Verifikation ohne Pairing | **KRITISCH** | 2-3 Tage | 1 |
| Insecure SRS Generation | **KRITISCH** | 1-2 Tage | 2 |
| Solidity Verifier unvollständig | **HOCH** | 2-3 Tage | 3 |
| Keine Folding Schemes | **MITTEL** | 5-7 Tage | 4 |

**Kritische Erkenntnis**: Der aktuelle KZG-Verifier in `plonk/core.py` Zeile 486 gibt `return True` zurück, ohne jegliche Pairing-Prüfung. Das System ist in diesem Zustand **kryptographisch wertlos**.

---

## Phase 1: Sicherheitsfundament

### 1.1 KZG-Verifikation mit echtem Pairing

**Problem**: Die aktuelle `KZG.verify()` Methode ist ein Stub:
```python
def verify(self, proof: KZGProof) -> bool:
    # ...
    return True  # Placeholder - real verification needs pairing
```

**Lösung**: Implementierung der vollständigen Pairing-basierten Verifikation.

**Mathematische Grundlage**:
Die KZG-Verifikation prüft die Gleichung:
```
e(C - v·G₁, G₂) = e(π, τ·G₂ - z·G₂)
```
wobei:
- `C` = Commitment auf Polynom p(X)
- `v` = p(z), der behauptete Wert
- `π` = Quotient-Commitment (der Beweis)
- `z` = Evaluationspunkt
- `τ` = Geheimnis aus dem Trusted Setup (nur als τ·G₂ bekannt)

**Äquivalente Formulierung** (effizienter für Batch-Verifikation):
```
e(C - v·G₁, G₂) · e(-π, τ·G₂ - z·G₂) = 1
```

**Architektur-Entscheidung**: 
- Python-Implementierung nutzt die existierende `pairing.py` Bibliothek
- Solidity-Implementierung nutzt den EIP-197 Precompile (bn256Pairing)

### 1.2 Trusted Setup Infrastruktur

**Problem**: `SRS.generate_insecure()` verwendet einen bekannten Tau-Wert.

**Lösung**: Zwei Optionen:
1. **Perpetual Powers of Tau**: Nutzung existierender Zeremonien (z.B. Hermez, Zcash)
2. **Universal Setup**: Migration zu einem System wie MARLIN/PLONK mit universeller SRS

**Architektur-Entscheidung für v1**:
- Implementierung eines `SRS.load_from_ptau()` Loaders für existierende Zeremonien
- Beibehaltung von `generate_insecure()` nur für Tests (mit Warnung)

---

## Phase 2: Vollständige Verifikation

### 2.1 Python KZG-Verifier

**Datei**: `plonk/core.py`

**Neue Methode**:
```python
def verify_with_pairing(self, proof: KZGProof) -> Tuple[bool, str]:
    """
    Vollständige KZG-Verifikation mit Pairing-Check.
    
    Returns:
        (is_valid, reason)
    """
```

**Abhängigkeiten**:
- `crypto/bn254/pairing.py`: `pairing()`, `verify_pairing_equation()`
- `crypto/bn254/curve.py`: `G1Point`, `G2Point`

### 2.2 Solidity Verifier Korrektur

**Problem**: Der aktuelle `_verifyPairing()` in `PlonkVerifier.sol` hat einen Fehler in der G2-Punkt-Konstruktion (Zeilen 437-449). Die x2-Koordinate wird falsch verwendet.

**Korrektur erforderlich**:
- Korrekte Serialisierung der G2-Punkte für den Precompile
- Vollständige Linearisierung statt Placeholder (Zeile 335)

---

## Implementierungsplan

### Schritt 1: KZG-Verifikation (Python)
1. Erweitere `KZG.verify()` um echte Pairing-Prüfung
2. Füge `verify_batch()` für Multi-Proof-Verifikation hinzu
3. Implementiere Fehlerbehandlung mit aussagekräftigen Fehlermeldungen

### Schritt 2: SRS-Loader
1. Implementiere `SRS.load_from_ptau(path)` für Hermez/Zcash PTAU-Dateien
2. Füge Validierung der geladenen SRS hinzu
3. Deprecate `generate_insecure()` mit Warnung

### Schritt 3: Solidity-Verifier
1. Korrigiere G2-Punkt-Serialisierung
2. Implementiere vollständige Linearisierung
3. Füge Gas-Optimierungen hinzu

### Schritt 4: Integration & Tests
1. End-to-End-Tests mit echten Pairings
2. Cross-Validierung Python ↔ Solidity
3. Performance-Benchmarks

---

## Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Pairing-Performance zu langsam | Mittel | Hoch | Batch-Verifikation, Caching |
| PTAU-Inkompatibilität | Niedrig | Mittel | Mehrere Formate unterstützen |
| Solidity Gas-Limit | Mittel | Hoch | Proof-Aggregation |

---

## Metriken für Erfolg

1. **Sicherheit**: Alle Proofs werden kryptographisch verifiziert
2. **Performance**: Verifikation < 500ms (Python), < 400k Gas (Solidity)
3. **Kompatibilität**: Interoperabilität mit Standard-PTAU-Dateien

---

## Nächste Schritte

1. ✅ Architekturplan erstellt
2. ⏳ KZG-Verifikation implementieren
3. ⏳ SRS-Loader implementieren
4. ⏳ Solidity-Verifier korrigieren
5. ⏳ Tests und Validierung
