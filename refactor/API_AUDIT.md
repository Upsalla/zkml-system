# API Audit: zkML System

## Identifizierte Inkonsistenzen

### 1. FieldConfig

**Problem:** `FieldConfig` erfordert `name` als Pflichtparameter, aber viele Module übergeben nur `prime`.

**Aktuelle Signatur:**
```python
@dataclass
class FieldConfig:
    prime: int
    name: str  # PFLICHT
```

**Erwartete Nutzung in Tests:**
```python
config = FieldConfig(prime=101)  # FEHLT: name
```

**Lösung:** `name` optional machen mit Default-Wert.

---

### 2. Wavelet-Modul

**Problem:** Klasse heißt `HaarTransformer`, nicht `HaarWaveletTransform`.

| Erwartet | Tatsächlich |
|----------|-------------|
| `HaarWaveletTransform` | `HaarTransformer` |
| `HWWBProver.prove(witness)` | `HWWBProver.prove(witness, field_config)` |

---

### 3. Tropical-Modul

**Problem:** `TropicalElement` existiert nicht.

| Erwartet | Tatsächlich |
|----------|-------------|
| `TropicalElement` | `TropicalVariable` |
| `TropicalSemiring.from_standard()` | Methode existiert nicht |

---

### 4. Network/CNN-Modul

**Problem:** `MaxPoolLayer` existiert nicht.

| Erwartet | Tatsächlich |
|----------|-------------|
| `MaxPoolLayer` | `MaxPool2D` |
| `Conv2DLayer(in_channels, out_channels, kernel_size, field_config)` | Andere Signatur |

---

### 5. PLONK-Modul

**Problem:** `NetworkConfig` existiert nicht.

| Erwartet | Tatsächlich |
|----------|-------------|
| `NetworkConfig` | Existiert nicht |
| `CircuitCompiler.compile(network_config)` | Andere Signatur |

---

### 6. BN254 Curve

**Problem:** `scalar_mul` heißt anders.

| Erwartet | Tatsächlich |
|----------|-------------|
| `G1Point.scalar_mul(n)` | `G1Point.__mul__(n)` oder `G1Point.mul(n)` |

---

## Refaktorierungs-Plan

### Phase 1: Core-Modul
1. `FieldConfig.name` optional machen
2. Konsistente Factory-Methoden hinzufügen

### Phase 2: Crypto-Modul
1. `G1Point.scalar_mul()` Alias hinzufügen
2. Konsistente Methoden-Namen

### Phase 3: Network-Modul
1. `MaxPoolLayer` Alias für `MaxPool2D`
2. Konsistente Konstruktor-Signaturen

### Phase 4: Optimierungs-Module
1. `HaarWaveletTransform` Alias für `HaarTransformer`
2. `TropicalElement` Alias für `TropicalVariable`
3. `TropicalSemiring.from_standard()` hinzufügen

### Phase 5: PLONK-Modul
1. `NetworkConfig` Klasse hinzufügen
2. Konsistente Circuit-Compiler-API
