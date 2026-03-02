"""
TDA-Fingerprinting: Schneller Kollisionstest

Reduzierte Version für schnelle Validierung.
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tda.fingerprint import TDAFingerprintSystem


def generate_random_model(num_params: int, seed: int) -> np.ndarray:
    """Generiere ein zufälliges Modell."""
    np.random.seed(seed)
    distribution = seed % 4
    if distribution == 0:
        weights = np.random.randn(num_params) * 0.1
    elif distribution == 1:
        weights = np.random.uniform(-0.5, 0.5, num_params)
    elif distribution == 2:
        fan_in = int(np.sqrt(num_params))
        weights = np.random.randn(num_params) * np.sqrt(2.0 / fan_in)
    else:
        weights = np.random.randn(num_params) * 0.1
        mask = np.random.random(num_params) < 0.5
        weights[mask] = 0
    return weights


def run_fast_collision_test(num_models: int = 500, model_size: int = 500):
    """Schneller Kollisionstest."""
    print(f"=" * 60)
    print(f"TDA-FINGERPRINTING: SCHNELLER KOLLISIONSTEST")
    print(f"=" * 60)
    print(f"Anzahl Modelle: {num_models}")
    print(f"Modellgröße: {model_size} Parameter")
    print(f"=" * 60)
    
    tda_system = TDAFingerprintSystem(n_features=20, n_samples=5)
    
    fingerprints = []
    hashes = []
    generation_times = []
    
    print(f"\nGeneriere {num_models} Fingerprints...")
    start_total = time.time()
    
    for i in range(num_models):
        if i % 100 == 0:
            print(f"  Progress: {i}/{num_models}")
        
        weights = generate_random_model(model_size, seed=i)
        
        start = time.time()
        fingerprint = tda_system.fingerprint([weights])
        generation_times.append(time.time() - start)
        
        # Extrahiere Vektor aus Features
        if fingerprint.features:
            feature_vector = np.array([f[1] + f[2] for f in fingerprint.features])
        else:
            feature_vector = np.zeros(20)
        
        if len(feature_vector) < 20:
            feature_vector = np.pad(feature_vector, (0, 20 - len(feature_vector)))
        elif len(feature_vector) > 20:
            feature_vector = feature_vector[:20]
        
        fingerprints.append({
            'id': i,
            'vector': feature_vector,
            'hash': fingerprint.hash
        })
        hashes.append(fingerprint.hash)
    
    total_time = time.time() - start_total
    print(f"\nGenerierung abgeschlossen in {total_time:.2f}s")
    print(f"Durchschnitt: {np.mean(generation_times)*1000:.2f}ms pro Fingerprint")
    
    # Test 1: Hash-Kollisionen
    print(f"\n--- TEST 1: Hash-Kollisionen ---")
    unique_hashes = len(set([h.hex() for h in hashes]))
    hash_collisions = num_models - unique_hashes
    print(f"Einzigartige Hashes: {unique_hashes}/{num_models}")
    print(f"Hash-Kollisionen: {hash_collisions}")
    
    if hash_collisions == 0:
        print("✅ Keine Hash-Kollisionen")
    else:
        print(f"❌ {hash_collisions} Hash-Kollisionen gefunden!")
    
    # Test 2: Vektor-Distanzen
    print(f"\n--- TEST 2: Vektor-Distanzen ---")
    distances = []
    collisions = []
    near_collisions = []
    
    collision_threshold = 0.01
    near_threshold = 0.05
    
    for i in range(num_models):
        for j in range(i + 1, num_models):
            vec_i = fingerprints[i]['vector']
            vec_j = fingerprints[j]['vector']
            
            # Normalisierte Distanz
            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)
            
            if norm_i > 0 and norm_j > 0:
                dist = np.linalg.norm(vec_i/norm_i - vec_j/norm_j)
            else:
                dist = 0 if norm_i == norm_j else 1
            
            distances.append(dist)
            
            if dist < collision_threshold:
                collisions.append((i, j, dist))
            elif dist < near_threshold:
                near_collisions.append((i, j, dist))
    
    distances = np.array(distances)
    
    print(f"Anzahl Vergleiche: {len(distances):,}")
    print(f"Kollisionen (dist < {collision_threshold}): {len(collisions)}")
    print(f"Near-Collisions (dist < {near_threshold}): {len(near_collisions)}")
    print(f"\nDistanz-Statistiken:")
    print(f"  Min: {np.min(distances):.6f}")
    print(f"  Max: {np.max(distances):.6f}")
    print(f"  Mean: {np.mean(distances):.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  1. Perzentil: {np.percentile(distances, 1):.6f}")
    print(f"  5. Perzentil: {np.percentile(distances, 5):.6f}")
    
    if collisions:
        print(f"\nGefundene Kollisionen:")
        for i, j, d in collisions[:5]:
            print(f"  Modell {i} vs {j}: dist = {d:.6f}")
    
    # Test 3: Ähnliche Modelle sollten ähnliche Fingerprints haben
    print(f"\n--- TEST 3: Perturbations-Stabilität ---")
    base_weights = generate_random_model(model_size, seed=999)
    base_fp = tda_system.fingerprint([base_weights])
    base_vec = np.array([f[1] + f[2] for f in base_fp.features])
    if len(base_vec) < 20:
        base_vec = np.pad(base_vec, (0, 20 - len(base_vec)))
    
    perturbation_results = []
    for noise_level in [0.001, 0.01, 0.05, 0.1, 0.2]:
        perturbed = base_weights + np.random.randn(model_size) * noise_level
        perturbed_fp = tda_system.fingerprint([perturbed])
        perturbed_vec = np.array([f[1] + f[2] for f in perturbed_fp.features])
        if len(perturbed_vec) < 20:
            perturbed_vec = np.pad(perturbed_vec, (0, 20 - len(perturbed_vec)))
        
        norm_base = np.linalg.norm(base_vec)
        norm_pert = np.linalg.norm(perturbed_vec)
        if norm_base > 0 and norm_pert > 0:
            dist = np.linalg.norm(base_vec/norm_base - perturbed_vec/norm_pert)
        else:
            dist = 0
        
        perturbation_results.append((noise_level, dist))
        print(f"  Noise {noise_level}: Fingerprint-Distanz = {dist:.6f}")
    
    # Bewertung
    print(f"\n" + "=" * 60)
    print(f"GESAMTBEWERTUNG")
    print(f"=" * 60)
    
    issues = []
    
    if hash_collisions > 0:
        issues.append(f"Hash-Kollisionen: {hash_collisions}")
    
    if len(collisions) > 0:
        issues.append(f"Vektor-Kollisionen: {len(collisions)}")
    
    if np.min(distances) < 0.01:
        issues.append(f"Minimale Distanz zu klein: {np.min(distances):.6f}")
    
    # Prüfe ob Perturbation monoton ist
    dists = [d for _, d in perturbation_results]
    if not all(dists[i] <= dists[i+1] for i in range(len(dists)-1)):
        issues.append("Perturbations-Distanz nicht monoton steigend")
    
    if not issues:
        print("✅ ALLE TESTS BESTANDEN")
        print("   Das System zeigt gute Kollisionsresistenz und Stabilität.")
    else:
        print("⚠️ PROBLEME GEFUNDEN:")
        for issue in issues:
            print(f"   - {issue}")
    
    return {
        'hash_collisions': hash_collisions,
        'vector_collisions': len(collisions),
        'near_collisions': len(near_collisions),
        'min_distance': float(np.min(distances)),
        'mean_distance': float(np.mean(distances)),
        'perturbation_results': perturbation_results,
        'issues': issues
    }


if __name__ == "__main__":
    results = run_fast_collision_test(num_models=500, model_size=500)
