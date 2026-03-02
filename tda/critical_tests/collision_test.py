"""
TDA-Fingerprinting: Kritischer Kollisionstest

Ziel: Prüfen, ob verschiedene Modelle unterschiedliche Fingerprints erzeugen.
Methode: Generiere 10,000 zufällige Modelle und prüfe auf Kollisionen.

Eine Kollision liegt vor, wenn zwei verschiedene Modelle einen Fingerprint-Abstand
von < 0.01 haben (praktisch identisch).
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tda.fingerprint import TDAFingerprintSystem, TDAProver


def generate_random_model(num_params: int, seed: int) -> np.ndarray:
    """Generiere ein zufälliges Modell mit gegebener Parameterzahl."""
    np.random.seed(seed)
    # Verschiedene Verteilungen für Realismus
    distribution = seed % 4
    if distribution == 0:
        # Normal-Verteilung (typisch für initialisierte Modelle)
        weights = np.random.randn(num_params) * 0.1
    elif distribution == 1:
        # Uniform-Verteilung
        weights = np.random.uniform(-0.5, 0.5, num_params)
    elif distribution == 2:
        # Xavier-ähnliche Initialisierung
        fan_in = int(np.sqrt(num_params))
        weights = np.random.randn(num_params) * np.sqrt(2.0 / fan_in)
    else:
        # Sparse Modell (viele Nullen)
        weights = np.random.randn(num_params) * 0.1
        mask = np.random.random(num_params) < 0.5
        weights[mask] = 0
    
    return weights


def run_collision_test(
    num_models: int = 10000,
    model_size: int = 1000,
    collision_threshold: float = 0.01,
    near_collision_threshold: float = 0.05
) -> Dict:
    """
    Führe den Kollisionstest durch.
    
    Args:
        num_models: Anzahl der zu testenden Modelle
        model_size: Anzahl der Parameter pro Modell
        collision_threshold: Distanz unter der eine Kollision vorliegt
        near_collision_threshold: Distanz unter der eine "Near-Collision" vorliegt
    
    Returns:
        Dictionary mit Testergebnissen
    """
    print(f"=" * 60)
    print(f"TDA-FINGERPRINTING: KOLLISIONSTEST")
    print(f"=" * 60)
    print(f"Anzahl Modelle: {num_models}")
    print(f"Modellgröße: {model_size} Parameter")
    print(f"Kollisions-Schwelle: {collision_threshold}")
    print(f"Near-Collision-Schwelle: {near_collision_threshold}")
    print(f"=" * 60)
    
    # Initialisiere das TDA-System
    tda_system = TDAFingerprintSystem(
        n_features=20,
        n_samples=10
    )
    
    # Generiere Fingerprints
    fingerprints = []
    generation_times = []
    
    print(f"\nGeneriere {num_models} Fingerprints...")
    start_total = time.time()
    
    for i in range(num_models):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_total
            rate = i / elapsed
            eta = (num_models - i) / rate
            print(f"  Progress: {i}/{num_models} ({100*i/num_models:.1f}%) - ETA: {eta:.1f}s")
        
        # Generiere zufälliges Modell
        weights = generate_random_model(model_size, seed=i)
        
        # Generiere Fingerprint
        start = time.time()
        fingerprint = tda_system.fingerprint([weights])
        generation_times.append(time.time() - start)
        
        # Konvertiere Features zu Vektor für Distanzberechnung
        feature_vector = np.array([f[1] + f[2] for f in fingerprint.features])  # birth + death
        if len(feature_vector) < 20:
            feature_vector = np.pad(feature_vector, (0, 20 - len(feature_vector)))
        
        fingerprints.append({
            'id': i,
            'fingerprint': fingerprint,
            'vector': feature_vector,
            'hash': fingerprint.hash
        })
    
    total_time = time.time() - start_total
    print(f"\nFingerprint-Generierung abgeschlossen in {total_time:.2f}s")
    print(f"Durchschnittliche Zeit pro Fingerprint: {np.mean(generation_times)*1000:.2f}ms")
    
    # Prüfe auf Kollisionen
    print(f"\nPrüfe auf Kollisionen (O(n²) Vergleiche)...")
    collisions = []
    near_collisions = []
    distances = []
    
    num_comparisons = num_models * (num_models - 1) // 2
    comparison_count = 0
    
    start_comparison = time.time()
    
    for i in range(num_models):
        for j in range(i + 1, num_models):
            comparison_count += 1
            
            if comparison_count % 10000000 == 0:
                elapsed = time.time() - start_comparison
                rate = comparison_count / elapsed
                eta = (num_comparisons - comparison_count) / rate
                print(f"  Vergleiche: {comparison_count}/{num_comparisons} - ETA: {eta:.1f}s")
            
            # Berechne Distanz
            vec_i = fingerprints[i]['vector']
            vec_j = fingerprints[j]['vector']
            
            # Euclidean distance
            dist = np.linalg.norm(vec_i - vec_j)
            distances.append(dist)
            
            if dist < collision_threshold:
                collisions.append({
                    'model_i': i,
                    'model_j': j,
                    'distance': dist
                })
            elif dist < near_collision_threshold:
                near_collisions.append({
                    'model_i': i,
                    'model_j': j,
                    'distance': dist
                })
    
    comparison_time = time.time() - start_comparison
    
    # Statistiken
    distances = np.array(distances)
    
    results = {
        'num_models': num_models,
        'model_size': model_size,
        'num_comparisons': num_comparisons,
        'num_collisions': len(collisions),
        'num_near_collisions': len(near_collisions),
        'collision_rate': len(collisions) / num_comparisons,
        'near_collision_rate': len(near_collisions) / num_comparisons,
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'percentile_1': float(np.percentile(distances, 1)),
        'percentile_5': float(np.percentile(distances, 5)),
        'percentile_10': float(np.percentile(distances, 10)),
        'total_time': total_time,
        'comparison_time': comparison_time,
        'collisions': collisions[:10],  # Nur erste 10
        'near_collisions': near_collisions[:10]  # Nur erste 10
    }
    
    # Ausgabe
    print(f"\n" + "=" * 60)
    print(f"ERGEBNISSE")
    print(f"=" * 60)
    print(f"Anzahl Vergleiche: {num_comparisons:,}")
    print(f"Kollisionen (dist < {collision_threshold}): {len(collisions)}")
    print(f"Near-Collisions (dist < {near_collision_threshold}): {len(near_collisions)}")
    print(f"Kollisionsrate: {results['collision_rate']:.2e}")
    print(f"\nDistanz-Statistiken:")
    print(f"  Min: {results['min_distance']:.6f}")
    print(f"  Max: {results['max_distance']:.6f}")
    print(f"  Mean: {results['mean_distance']:.6f}")
    print(f"  Median: {results['median_distance']:.6f}")
    print(f"  Std: {results['std_distance']:.6f}")
    print(f"  1. Perzentil: {results['percentile_1']:.6f}")
    print(f"  5. Perzentil: {results['percentile_5']:.6f}")
    print(f"  10. Perzentil: {results['percentile_10']:.6f}")
    
    if collisions:
        print(f"\nGefundene Kollisionen:")
        for c in collisions[:5]:
            print(f"  Modell {c['model_i']} vs {c['model_j']}: dist = {c['distance']:.6f}")
    
    # Bewertung
    print(f"\n" + "=" * 60)
    print(f"BEWERTUNG")
    print(f"=" * 60)
    
    if len(collisions) == 0:
        print("✅ BESTANDEN: Keine Kollisionen gefunden.")
        print(f"   Bei {num_comparisons:,} Vergleichen ist das ein starkes Ergebnis.")
    else:
        collision_rate = len(collisions) / num_comparisons
        if collision_rate < 1e-6:
            print(f"⚠️ WARNUNG: {len(collisions)} Kollisionen gefunden.")
            print(f"   Kollisionsrate: {collision_rate:.2e}")
            print(f"   Das ist niedrig, aber nicht null. Weitere Analyse nötig.")
        else:
            print(f"❌ KRITISCH: {len(collisions)} Kollisionen gefunden!")
            print(f"   Kollisionsrate: {collision_rate:.2e}")
            print(f"   Das ist zu hoch für ein produktionsreifes System.")
    
    if results['min_distance'] < 0.1:
        print(f"\n⚠️ WARNUNG: Minimale Distanz ist {results['min_distance']:.6f}")
        print(f"   Das deutet auf geringe Trennschärfe hin.")
    
    return results


if __name__ == "__main__":
    # Führe Test mit verschiedenen Konfigurationen durch
    
    # Test 1: Kleine Modelle, viele Samples
    print("\n" + "=" * 80)
    print("TEST 1: Kleine Modelle (1000 Params), 5000 Samples")
    print("=" * 80)
    results_small = run_collision_test(
        num_models=5000,
        model_size=1000,
        collision_threshold=0.01,
        near_collision_threshold=0.05
    )
    
    # Test 2: Größere Modelle, weniger Samples
    print("\n" + "=" * 80)
    print("TEST 2: Größere Modelle (10000 Params), 1000 Samples")
    print("=" * 80)
    results_large = run_collision_test(
        num_models=1000,
        model_size=10000,
        collision_threshold=0.01,
        near_collision_threshold=0.05
    )
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"Test 1 (klein): {results_small['num_collisions']} Kollisionen, min_dist = {results_small['min_distance']:.6f}")
    print(f"Test 2 (groß): {results_large['num_collisions']} Kollisionen, min_dist = {results_large['min_distance']:.6f}")
