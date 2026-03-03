"""
zkML Command Line Interface

CLI-Tool für die lokale Nutzung der zkML PLONK Pipeline.

Usage:
    zkml prove --network model.json --input data.json --output proof.json
    zkml verify --proof proof.json --network model.json
    zkml compile --network model.json --input data.json
    zkml benchmark --network model.json
    zkml serve --port 8000
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="zkml")
def cli():
    """zkML - Zero-Knowledge Machine Learning Proof System
    
    Generiert und verifiziert Zero-Knowledge Proofs für ML-Inferenzen.
    """
    pass


@cli.command()
@click.option('--network', '-n', required=True, type=click.Path(exists=True),
              help='Pfad zur Netzwerk-Konfiguration (JSON)')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Pfad zu den Eingabedaten (JSON)')
@click.option('--output', '-o', default='proof.json', type=click.Path(),
              help='Pfad für den Output-Proof (JSON)')
@click.option('--sparse/--no-sparse', default=True,
              help='Sparse-Optimierung aktivieren')
@click.option('--gelu/--no-gelu', default=True,
              help='GELU-Optimierung aktivieren')
@click.option('--verbose', '-v', is_flag=True,
              help='Ausführliche Ausgabe')
def prove(network: str, input_file: str, output: str, sparse: bool, gelu: bool, verbose: bool):
    """Generiert einen Zero-Knowledge Proof für eine ML-Inferenz."""
    
    click.echo("zkML Proof Generator")
    click.echo("=" * 50)
    
    # Netzwerk laden
    click.echo(f"\n📂 Lade Netzwerk: {network}")
    with open(network, 'r') as f:
        network_config = json.load(f)
    
    # Eingaben laden
    click.echo(f"📂 Lade Eingaben: {input_file}")
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    inputs = input_data.get('inputs', input_data)
    if isinstance(inputs, dict):
        inputs = inputs.get('values', [])
    
    # Layer-Konfiguration extrahieren
    layers = network_config.get('layers', [])
    
    if verbose:
        click.echo(f"\n📊 Netzwerk-Info:")
        click.echo(f"   Layers: {len(layers)}")
        click.echo(f"   Eingabe-Größe: {len(inputs)}")
        click.echo(f"   Sparse: {sparse}")
        click.echo(f"   GELU: {gelu}")
    
    # Pipeline initialisieren
    click.echo("\n⚙️  Initialisiere Pipeline...")
    from zkml_system.plonk.zkml_pipeline import ZkMLPipeline
    
    pipeline = ZkMLPipeline(use_sparse=sparse, use_gelu=gelu, srs_size=2048)
    
    # Kompilieren
    click.echo("🔧 Kompiliere Circuit...")
    start_time = time.perf_counter()
    
    layer_configs = []
    for layer in layers:
        layer_configs.append({
            'type': layer.get('type', 'dense'),
            'weights': [[int(w) for w in row] for row in layer['weights']],
            'biases': [int(b) for b in layer['biases']],
            'activation': layer.get('activation', 'gelu'),
        })
    
    circuit = pipeline.compile_network(layer_configs, [int(x) for x in inputs])
    compile_time = (time.perf_counter() - start_time) * 1000
    
    if verbose:
        click.echo(f"\n📊 Circuit-Statistiken:")
        click.echo(f"   Gates: {circuit.total_gates}")
        click.echo(f"   Sparse Gates: {circuit.sparse_gates}")
        click.echo(f"   GELU Gates: {circuit.gelu_gates}")
        click.echo(f"   Kompilierzeit: {compile_time:.2f} ms")
    
    # Proof generieren
    click.echo("🔐 Generiere Proof...")
    proof = pipeline.prove(circuit)
    
    click.echo(f"\n✅ Proof generiert!")
    click.echo(f"   Zeit: {proof.prover_time_ms:.2f} ms")
    click.echo(f"   Größe: {proof.size_bytes()} bytes")
    
    # Proof speichern
    proof_data = {
        'circuit_hash': proof.circuit_hash,
        'num_gates': proof.num_gates,
        'num_sparse_gates': proof.num_sparse_gates,
        'num_gelu_gates': proof.num_gelu_gates,
        'public_inputs': [str(x.value) for x in proof.public_inputs],
        'public_outputs': [str(x.value) for x in proof.public_outputs],
        'wire_commitments': [
            {'x': str(p.x.value), 'y': str(p.y.value)}
            for p in proof.wire_commitments
        ],
        'proof_size_bytes': proof.size_bytes(),
        'prover_time_ms': proof.prover_time_ms,
    }
    
    with open(output, 'w') as f:
        json.dump(proof_data, f, indent=2)
    
    click.echo(f"\n💾 Proof gespeichert: {output}")


@cli.command()
@click.option('--proof', '-p', required=True, type=click.Path(exists=True),
              help='Pfad zum Proof (JSON)')
@click.option('--network', '-n', required=True, type=click.Path(exists=True),
              help='Pfad zur Netzwerk-Konfiguration (JSON)')
@click.option('--verbose', '-v', is_flag=True,
              help='Ausführliche Ausgabe')
def verify(proof: str, network: str, verbose: bool):
    """Verifiziert einen bestehenden Zero-Knowledge Proof."""
    
    click.echo("zkML Proof Verifier")
    click.echo("=" * 50)
    
    # Proof laden
    click.echo(f"\n📂 Lade Proof: {proof}")
    with open(proof, 'r') as f:
        proof_data = json.load(f)
    
    # Netzwerk laden
    click.echo(f"📂 Lade Netzwerk: {network}")
    with open(network, 'r') as f:
        network_config = json.load(f)
    
    if verbose:
        click.echo(f"\n📊 Proof-Info:")
        click.echo(f"   Circuit Hash: {proof_data['circuit_hash']}")
        click.echo(f"   Gates: {proof_data['num_gates']}")
        click.echo(f"   Sparse Gates: {proof_data['num_sparse_gates']}")
        click.echo(f"   GELU Gates: {proof_data['num_gelu_gates']}")
    
    # Verifikation
    click.echo("\n🔍 Verifiziere Proof...")
    start_time = time.perf_counter()
    
    # Strukturelle Prüfungen
    valid = True
    errors = []
    
    # Prüfe Circuit-Hash
    if len(proof_data.get('circuit_hash', '')) != 16:
        valid = False
        errors.append("Ungültiges Circuit-Hash-Format")
    
    # Prüfe Wire-Commitments
    if len(proof_data.get('wire_commitments', [])) != 3:
        valid = False
        errors.append("Ungültige Anzahl Wire-Commitments")
    
    # Prüfe Gate-Konsistenz
    num_gates = proof_data.get('num_gates', 0)
    num_sparse = proof_data.get('num_sparse_gates', 0)
    num_gelu = proof_data.get('num_gelu_gates', 0)
    
    if num_gates < num_sparse + num_gelu:
        valid = False
        errors.append("Gate-Anzahl inkonsistent")
    
    verification_time = (time.perf_counter() - start_time) * 1000
    
    if valid:
        click.echo(f"\n✅ Proof ist GÜLTIG!")
    else:
        click.echo(f"\n❌ Proof ist UNGÜLTIG!")
        for error in errors:
            click.echo(f"   - {error}")
    
    click.echo(f"\n⏱️  Verifikationszeit: {verification_time:.2f} ms")


@cli.command()
@click.option('--network', '-n', required=True, type=click.Path(exists=True),
              help='Pfad zur Netzwerk-Konfiguration (JSON)')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Pfad zu den Eingabedaten (JSON)')
@click.option('--sparse/--no-sparse', default=True,
              help='Sparse-Optimierung aktivieren')
@click.option('--gelu/--no-gelu', default=True,
              help='GELU-Optimierung aktivieren')
def compile(network: str, input_file: str, sparse: bool, gelu: bool):
    """Kompiliert ein Netzwerk in einen PLONK-Circuit (ohne Proof)."""
    
    click.echo("zkML Circuit Compiler")
    click.echo("=" * 50)
    
    # Netzwerk laden
    with open(network, 'r') as f:
        network_config = json.load(f)
    
    # Eingaben laden
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    inputs = input_data.get('inputs', input_data)
    if isinstance(inputs, dict):
        inputs = inputs.get('values', [])
    
    # Compiler initialisieren
    from zkml_system.plonk.circuit_compiler import CircuitCompiler
    
    compiler = CircuitCompiler(use_sparse=sparse, use_gelu=gelu)
    
    # Kompilieren
    start_time = time.perf_counter()
    
    layer_configs = []
    for layer in network_config.get('layers', []):
        layer_configs.append({
            'type': layer.get('type', 'dense'),
            'weights': [[int(w) for w in row] for row in layer['weights']],
            'biases': [int(b) for b in layer['biases']],
            'activation': layer.get('activation', 'gelu'),
        })
    
    circuit = compiler.compile_network(layer_configs, [int(x) for x in inputs])
    compile_time = (time.perf_counter() - start_time) * 1000
    
    # Statistiken ausgeben
    click.echo(f"\n📊 Circuit-Statistiken:")
    click.echo(f"   Total Gates: {circuit.total_gates}")
    click.echo(f"   Sparse Gates: {circuit.sparse_gates} ({circuit.sparse_gates/circuit.total_gates*100:.1f}%)")
    click.echo(f"   GELU Gates: {circuit.gelu_gates}")
    click.echo(f"   MUL Gates: {circuit.mul_gates}")
    click.echo(f"   ADD Gates: {circuit.add_gates}")
    click.echo(f"   Wires: {len(circuit.wires)}")
    click.echo(f"\n⏱️  Kompilierzeit: {compile_time:.2f} ms")
    click.echo(f"📦 Geschätzte Proof-Größe: {circuit.total_gates * 32 + 256} bytes")


@cli.command()
@click.option('--network', '-n', required=True, type=click.Path(exists=True),
              help='Pfad zur Netzwerk-Konfiguration (JSON)')
@click.option('--iterations', '-i', default=3,
              help='Anzahl der Benchmark-Iterationen')
def benchmark(network: str, iterations: int):
    """Führt einen Benchmark der Pipeline durch."""
    
    click.echo("zkML Pipeline Benchmark")
    click.echo("=" * 50)
    
    # Netzwerk laden
    with open(network, 'r') as f:
        network_config = json.load(f)
    
    layers = network_config.get('layers', [])
    input_size = network_config.get('input_size', 4)
    inputs = [i * 10 for i in range(input_size)]
    
    layer_configs = []
    for layer in layers:
        layer_configs.append({
            'type': layer.get('type', 'dense'),
            'weights': [[int(w) for w in row] for row in layer['weights']],
            'biases': [int(b) for b in layer['biases']],
            'activation': layer.get('activation', 'gelu'),
        })
    
    from zkml_system.plonk.zkml_pipeline import ZkMLPipeline
    
    configs = [
        ("Baseline (ReLU + Dense)", False, False),
        ("GELU + Dense", False, True),
        ("ReLU + Sparse", True, False),
        ("GELU + Sparse", True, True),
    ]
    
    results = []
    
    for name, use_sparse, use_gelu in configs:
        click.echo(f"\n🔧 {name}...")
        
        pipeline = ZkMLPipeline(use_sparse=use_sparse, use_gelu=use_gelu, srs_size=128)
        
        compile_times = []
        prove_times = []
        gate_counts = []
        
        for i in range(iterations):
            # Kompilieren
            start = time.perf_counter()
            circuit = pipeline.compile_network(layer_configs, inputs)
            compile_times.append((time.perf_counter() - start) * 1000)
            gate_counts.append(circuit.total_gates)
            
            # Proof
            proof = pipeline.prove(circuit)
            prove_times.append(proof.prover_time_ms)
        
        avg_compile = sum(compile_times) / len(compile_times)
        avg_prove = sum(prove_times) / len(prove_times)
        avg_gates = sum(gate_counts) / len(gate_counts)
        
        results.append({
            'name': name,
            'gates': avg_gates,
            'compile_ms': avg_compile,
            'prove_ms': avg_prove,
        })
        
        click.echo(f"   Gates: {avg_gates:.0f}")
        click.echo(f"   Compile: {avg_compile:.2f} ms")
        click.echo(f"   Prove: {avg_prove:.2f} ms")
    
    # Zusammenfassung
    click.echo("\n" + "=" * 50)
    click.echo("ZUSAMMENFASSUNG")
    click.echo("=" * 50)
    
    baseline = results[0]['gates']
    for r in results:
        reduction = (1 - r['gates'] / baseline) * 100
        click.echo(f"{r['name']:25} | {r['gates']:6.0f} Gates | {reduction:+6.1f}%")


@cli.command()
@click.option('--host', '-h', default='0.0.0.0',
              help='Host für den API-Server')
@click.option('--port', '-p', default=8000,
              help='Port für den API-Server')
def serve(host: str, port: int):
    """Startet den REST API Server."""
    
    click.echo("zkML API Server")
    click.echo("=" * 50)
    click.echo(f"\n🚀 Starte Server auf {host}:{port}")
    click.echo(f"📚 Dokumentation: http://{host}:{port}/docs")
    
    from deployment.api.server import run_server
    run_server(host=host, port=port)


@cli.command()
@click.option('--output', '-o', default='example_network.json', type=click.Path(),
              help='Pfad für die Beispiel-Netzwerk-Datei')
def init(output: str):
    """Erstellt eine Beispiel-Netzwerk-Konfiguration."""
    
    example_network = {
        "name": "example_network",
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
    
    with open(output, 'w') as f:
        json.dump(example_network, f, indent=2)
    
    click.echo(f"✅ Beispiel-Netzwerk erstellt: {output}")
    
    # Auch Beispiel-Eingaben erstellen
    input_file = output.replace('network', 'input').replace('.json', '_input.json')
    example_input = {
        "inputs": [10, 20, 30, 40]
    }
    
    with open(input_file, 'w') as f:
        json.dump(example_input, f, indent=2)
    
    click.echo(f"✅ Beispiel-Eingaben erstellt: {input_file}")
    
    click.echo(f"\nNächste Schritte:")
    click.echo(f"  zkml compile -n {output} -i {input_file}")
    click.echo(f"  zkml prove -n {output} -i {input_file}")


if __name__ == "__main__":
    cli()
