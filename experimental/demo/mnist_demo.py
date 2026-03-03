"""
MNIST zkML Demo
===============

Demonstriert das vollständige zkML-System mit einem MNIST-ähnlichen
Klassifikationsproblem.

Da wir kein echtes MNIST laden, simulieren wir:
- Eingabe: 784 Pixel (28x28 Bild)
- Hidden Layer: 128 Neuronen mit GELU
- Hidden Layer: 64 Neuronen mit GELU
- Ausgabe: 10 Klassen (Ziffern 0-9)

Zeigt:
1. Netzwerk-Erstellung mit optimierten Aktivierungen
2. Forward Pass und Witness-Generierung
3. Proof-Generierung
4. Verifikation
5. Constraint-Vergleich: GELU vs ReLU
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.builder import NetworkBuilder, Network, NetworkStats
from proof.prover import Prover, Proof, ProofStats
from proof.verifier import Verifier, VerificationResult


def create_mnist_network(prime: int, seed: int = 42) -> Network:
    """
    Erstellt ein MNIST-ähnliches Netzwerk.
    
    Architektur:
    - Input: 784 (28x28 Pixel)
    - Hidden 1: 128 Neuronen, GELU
    - Hidden 2: 64 Neuronen, GELU
    - Output: 10 Klassen
    """
    builder = NetworkBuilder(prime)
    builder.add_input(784)
    builder.add_dense(128, activation="gelu", name="hidden1")
    builder.add_dense(64, activation="gelu", name="hidden2")
    builder.add_dense(10, activation="none", name="output")
    builder.add_output()
    
    return builder.build(random_seed=seed)


def create_small_network(prime: int, seed: int = 42) -> Network:
    """
    Erstellt ein kleineres Netzwerk für schnellere Tests.
    
    Architektur:
    - Input: 64 (8x8 Pixel)
    - Hidden 1: 32 Neuronen, GELU
    - Hidden 2: 16 Neuronen, GELU
    - Output: 10 Klassen
    """
    builder = NetworkBuilder(prime)
    builder.add_input(64)
    builder.add_dense(32, activation="gelu", name="hidden1")
    builder.add_dense(16, activation="gelu", name="hidden2")
    builder.add_dense(10, activation="none", name="output")
    builder.add_output()
    
    return builder.build(random_seed=seed)


def generate_random_image(size: int, prime: int, seed: int = None) -> list:
    """Generiert ein zufälliges 'Bild' (Liste von Pixelwerten)."""
    if seed is not None:
        random.seed(seed)
    # Pixelwerte zwischen 0 und 255, normalisiert auf Primfeld
    return [random.randint(0, 255) % prime for _ in range(size)]


def run_demo(use_small: bool = True):
    """
    Führt die vollständige Demo durch.
    """
    print("=" * 70)
    print("zkML MNIST Demo")
    print("=" * 70)
    print()
    
    # Konfiguration
    prime = 101  # Kleines Primfeld für Demo
    
    # Netzwerk erstellen
    print("1. Netzwerk erstellen...")
    print("-" * 40)
    
    if use_small:
        network = create_small_network(prime)
        input_size = 64
        print("   Verwende kleines Netzwerk (64 -> 32 -> 16 -> 10)")
    else:
        network = create_mnist_network(prime)
        input_size = 784
        print("   Verwende MNIST-Netzwerk (784 -> 128 -> 64 -> 10)")
    
    print(f"\n{network}")
    print(f"\nNetzwerk-Statistiken:")
    print(f"   Parameter: {network.stats.total_parameters}")
    print(f"   Constraints (GELU): {network.stats.total_constraints}")
    print(f"   Constraints (ReLU): {network.stats.relu_constraints}")
    print(f"   Ersparnis: {network.stats.constraint_savings:.1%}")
    
    # Eingabe generieren
    print("\n2. Eingabe generieren...")
    print("-" * 40)
    
    image = generate_random_image(input_size, prime, seed=123)
    print(f"   Eingabegröße: {len(image)} Pixel")
    print(f"   Erste 10 Werte: {image[:10]}")
    
    # Forward Pass
    print("\n3. Forward Pass (Inferenz)...")
    print("-" * 40)
    
    start_time = time.time()
    outputs, witness, updated_stats = network.forward(image)
    forward_time = time.time() - start_time
    
    print(f"   Ausgabe (Logits): {outputs}")
    print(f"   Vorhergesagte Klasse: {outputs.index(max(outputs))}")
    print(f"   Witness-Größe: {witness.size()} Variablen")
    print(f"   Durchschnittliche Sparsity: {updated_stats.avg_sparsity:.1%}")
    print(f"   Zeit: {forward_time*1000:.2f} ms")
    
    # Proof generieren
    print("\n4. Zero-Knowledge Proof generieren...")
    print("-" * 40)
    
    prover = Prover(network, use_sparse=True)
    
    start_time = time.time()
    proof, proof_outputs = prover.prove(image)
    proof_time = time.time() - start_time
    
    print(f"   Network Hash: {proof.network_hash}")
    print(f"   Witness Commitment: {proof.components.witness_commitment}")
    print(f"   Challenge: {proof.components.challenge}")
    print(f"   Response: {proof.components.response}")
    print(f"   Constraints im Proof: {proof.components.num_constraints}")
    print(f"   Zeit: {proof_time*1000:.2f} ms")
    
    # Proof serialisieren
    serialized = proof.serialize()
    print(f"   Proof-Größe: {len(serialized)} bytes")
    
    # Verifikation
    print("\n5. Proof verifizieren...")
    print("-" * 40)
    
    verifier = Verifier(prover.network_hash, prime)
    
    start_time = time.time()
    result = verifier.verify_computation(proof, image, outputs)
    verify_time = time.time() - start_time
    
    print(f"   Ergebnis: {'GÜLTIG' if result.valid else 'UNGÜLTIG'}")
    print(f"   Checks: {result.checks_passed}")
    print(f"   Zeit: {verify_time*1000:.2f} ms")
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    
    print(f"""
Netzwerk-Architektur:
   Eingabe:    {input_size} Neuronen
   Hidden 1:   {network.hidden_layers[0].config.output_size} Neuronen (GELU)
   Hidden 2:   {network.hidden_layers[1].config.output_size} Neuronen (GELU)
   Ausgabe:    {network.hidden_layers[2].config.output_size} Neuronen

Constraint-Analyse:
   Mit GELU:   {network.stats.total_constraints:,} Constraints
   Mit ReLU:   {network.stats.relu_constraints:,} Constraints
   Ersparnis:  {network.stats.constraint_savings:.1%}

Performance:
   Forward Pass:  {forward_time*1000:.2f} ms
   Proof-Gen:     {proof_time*1000:.2f} ms
   Verifikation:  {verify_time*1000:.2f} ms

Proof-Details:
   Größe:      {len(serialized)} bytes
   Gültig:     {result.valid}
""")
    
    return proof, result


def compare_activations():
    """
    Vergleicht verschiedene Aktivierungsfunktionen.
    """
    print("\n" + "=" * 70)
    print("AKTIVIERUNGSFUNKTIONEN-VERGLEICH")
    print("=" * 70)
    
    prime = 101
    
    activations = ["gelu", "swish", "quadratic"]
    results = {}
    
    for act in activations:
        builder = NetworkBuilder(prime)
        builder.add_input(64)
        builder.add_dense(32, activation=act)
        builder.add_dense(16, activation=act)
        builder.add_dense(10, activation="none")
        builder.add_output()
        
        network = builder.build(random_seed=42)
        results[act] = {
            "constraints": network.stats.total_constraints,
            "relu_equivalent": network.stats.relu_constraints,
            "savings": network.stats.constraint_savings
        }
    
    print("\nAktivierung | Constraints | ReLU-Äquivalent | Ersparnis")
    print("-" * 60)
    for act, data in results.items():
        print(f"{act:11} | {data['constraints']:11,} | {data['relu_equivalent']:15,} | {data['savings']:8.1%}")
    
    return results


def benchmark_scaling():
    """
    Benchmarkt das Skalierungsverhalten.
    """
    print("\n" + "=" * 70)
    print("SKALIERUNGS-BENCHMARK")
    print("=" * 70)
    
    prime = 101
    
    # Verschiedene Netzwerkgrößen
    configs = [
        (16, 8, 4),
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
        (256, 128, 64),
    ]
    
    print("\nEingabe | Hidden1 | Hidden2 | Constraints | ReLU | Ersparnis | Zeit (ms)")
    print("-" * 80)
    
    for input_size, h1, h2 in configs:
        builder = NetworkBuilder(prime)
        builder.add_input(input_size)
        builder.add_dense(h1, activation="gelu")
        builder.add_dense(h2, activation="gelu")
        builder.add_dense(10, activation="none")
        builder.add_output()
        
        network = builder.build(random_seed=42)
        
        # Forward Pass timen
        image = generate_random_image(input_size, prime)
        
        start = time.time()
        outputs, witness, stats = network.forward(image)
        elapsed = (time.time() - start) * 1000
        
        print(f"{input_size:7} | {h1:7} | {h2:7} | {stats.total_constraints:11,} | {stats.relu_constraints:4,} | {stats.constraint_savings:8.1%} | {elapsed:9.2f}")


if __name__ == "__main__":
    # Hauptdemo
    proof, result = run_demo(use_small=True)
    
    # Aktivierungsvergleich
    compare_activations()
    
    # Skalierungsbenchmark
    benchmark_scaling()
    
    print("\n" + "=" * 70)
    print("Demo abgeschlossen!")
    print("=" * 70)
