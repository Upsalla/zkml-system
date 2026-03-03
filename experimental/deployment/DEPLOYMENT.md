# zkML System Deployment Guide

**Version:** 0.1.0
**Datum:** 26. Januar 2026

## Übersicht

Dieses Dokument beschreibt die Deployment-Optionen für das zkML PLONK Pipeline System. Das System kann auf drei Arten betrieben werden:

1. **Lokale Installation** – Direkte Python-Installation für Entwicklung und Tests
2. **REST API** – FastAPI-basierter Server für Integration in andere Systeme
3. **Docker** – Containerisiertes Deployment für Produktion

## Schnellstart

### Option 1: Lokale Installation

```bash
# Repository klonen
git clone <repository-url>
cd zkml_system

# Dependencies installieren
pip install -r requirements.txt

# Package installieren
pip install -e .

# CLI testen
python -m zkml_system.deployment.cli.main --help
```

### Option 2: Docker

```bash
# Image bauen
docker build -t zkml-system:latest -f deployment/docker/Dockerfile .

# API-Server starten
docker run -p 8000:8000 zkml-system serve --host 0.0.0.0 --port 8000

# CLI im Container nutzen
docker run zkml-system --help
```

### Option 3: Docker Compose

```bash
cd deployment/docker
docker-compose up -d
```

## CLI-Befehle

Das CLI-Tool bietet folgende Befehle:

| Befehl | Beschreibung |
|--------|--------------|
| `init` | Erstellt eine Beispiel-Netzwerk-Konfiguration |
| `compile` | Kompiliert ein Netzwerk in einen PLONK-Circuit |
| `prove` | Generiert einen Zero-Knowledge Proof |
| `verify` | Verifiziert einen bestehenden Proof |
| `benchmark` | Führt einen Performance-Benchmark durch |
| `serve` | Startet den REST API Server |

### Beispiel-Workflow

```bash
# 1. Beispiel-Netzwerk erstellen
zkml init --output my_network.json

# 2. Circuit analysieren
zkml compile -n my_network.json -i my_input.json

# 3. Proof generieren
zkml prove -n my_network.json -i my_input.json -o proof.json --verbose

# 4. Proof verifizieren
zkml verify -p proof.json -n my_network.json
```

## REST API

### Endpunkte

| Endpunkt | Methode | Beschreibung |
|----------|---------|--------------|
| `/health` | GET | Health-Check |
| `/compile` | POST | Kompiliert ein Netzwerk |
| `/prove` | POST | Generiert einen Proof |
| `/verify` | POST | Verifiziert einen Proof |
| `/docs` | GET | OpenAPI Dokumentation |

### Beispiel: Proof generieren

```bash
curl -X POST http://localhost:8000/prove \
  -H "Content-Type: application/json" \
  -d '{
    "network": {
      "name": "test_network",
      "input_size": 4,
      "layers": [
        {
          "type": "dense",
          "weights": [[1,2,3,4], [5,6,7,8]],
          "biases": [1, 2],
          "activation": "gelu"
        }
      ]
    },
    "inputs": [10, 20, 30, 40],
    "use_sparse": true,
    "use_gelu": true
  }'
```

### Beispiel-Antwort

```json
{
  "success": true,
  "proof": {
    "circuit_hash": "a1b2c3d4e5f6g7h8",
    "num_gates": 47,
    "num_sparse_gates": 0,
    "num_gelu_gates": 9,
    "public_inputs": ["10", "20", "30", "40"],
    "public_outputs": ["1234", "5678"],
    "proof_size_bytes": 416
  },
  "outputs": [1234.0, 5678.0],
  "prover_time_ms": 2500.5
}
```

## Netzwerk-Konfiguration

Das Netzwerk wird als JSON-Datei definiert:

```json
{
  "name": "my_network",
  "input_size": 4,
  "layers": [
    {
      "type": "dense",
      "weights": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
      "biases": [1, 2, 3],
      "activation": "gelu"
    },
    {
      "type": "dense",
      "weights": [[1, 2, 3], [4, 5, 6]],
      "biases": [1, 2],
      "activation": "linear"
    }
  ]
}
```

### Unterstützte Aktivierungsfunktionen

- `gelu` – Gaussian Error Linear Unit (empfohlen, 3 Gates pro Neuron)
- `relu` – Rectified Linear Unit (teurer, ~255 Gates pro Neuron)
- `swish` – Swish/SiLU Aktivierung
- `linear` – Keine Aktivierung

## Optimierungen

### GELU-Optimierung

Die GELU-Aktivierung wird durch eine Polynom-Approximation implementiert, die nur 3 Gates pro Neuron benötigt, im Vergleich zu ~255 Gates für eine sichere ReLU-Implementierung.

```bash
# GELU aktivieren (Standard)
zkml prove -n network.json -i input.json --gelu

# GELU deaktivieren
zkml prove -n network.json -i input.json --no-gelu
```

### Sparse-Optimierung

Inaktive Neuronen (Aktivierungswert = 0) werden automatisch erkannt und durch ein einziges "Zero-Gate" ersetzt.

```bash
# Sparse aktivieren (Standard)
zkml prove -n network.json -i input.json --sparse

# Sparse deaktivieren
zkml prove -n network.json -i input.json --no-sparse
```

## Performance-Hinweise

1. **Python-Implementierung**: Die aktuelle Implementierung ist eine Referenzimplementierung in Python. Für Produktion sollten kritische Komponenten in Rust portiert werden.

2. **SRS-Größe**: Die Structured Reference String (SRS) Größe beeinflusst die maximale Circuit-Größe. Standard ist 2048.

3. **Proof-Generierung**: Die Proof-Generierung ist CPU-intensiv. Für große Netzwerke kann dies mehrere Sekunden dauern.

## Troubleshooting

### Häufige Probleme

**Problem**: `ModuleNotFoundError: No module named 'zkml_system'`

**Lösung**: Stellen Sie sicher, dass das Package installiert ist:
```bash
pip install -e .
```

**Problem**: API-Server startet nicht

**Lösung**: Prüfen Sie, ob Port 8000 frei ist:
```bash
lsof -i :8000
```

**Problem**: Proof-Generierung ist langsam

**Lösung**: Reduzieren Sie die Netzwerk-Größe oder aktivieren Sie Sparse-Optimierung.

## Support

Bei Fragen oder Problemen öffnen Sie ein Issue im Repository oder kontaktieren Sie das Entwicklungsteam.
