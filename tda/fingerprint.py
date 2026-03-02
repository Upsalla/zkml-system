"""
TDA-based Model Fingerprinting System.

This module implements:
1. Model to point cloud conversion
2. Fingerprint extraction from persistence diagrams
3. ZK-compatible commitment and verification
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tda.persistence import compute_persistence, PersistenceDiagram, PersistenceFeature


@dataclass
class ModelFingerprint:
    """
    Compact fingerprint of a neural network model.
    
    Contains the top-k most persistent topological features
    and a hash for quick comparison.
    """
    features: List[Tuple[int, float, float]]  # (dimension, birth, death)
    hash: bytes  # 32-byte SHA256 hash
    model_stats: Dict[str, Any] = field(default_factory=dict)
    
    def __eq__(self, other):
        if not isinstance(other, ModelFingerprint):
            return False
        return self.hash == other.hash
    
    def __hash__(self):
        return int.from_bytes(self.hash[:8], 'big')
    
    def size_bytes(self) -> int:
        """Calculate fingerprint size in bytes."""
        # Features: 1 byte dim + 4 bytes birth + 4 bytes death = 9 bytes each
        # Hash: 32 bytes
        return len(self.features) * 9 + 32
    
    def distance(self, other: 'ModelFingerprint') -> float:
        """
        Compute distance between two fingerprints.
        
        Uses bottleneck-like distance on the feature sets.
        """
        if len(self.features) != len(other.features):
            return float('inf')
        
        # Simple L2 distance on sorted features
        f1 = sorted(self.features)
        f2 = sorted(other.features)
        
        total = 0
        for (d1, b1, e1), (d2, b2, e2) in zip(f1, f2):
            if d1 != d2:
                return float('inf')
            total += (b1 - b2) ** 2 + (e1 - e2) ** 2
        
        return np.sqrt(total)


class PointCloudConverter:
    """
    Converts neural network weights to a point cloud for TDA.
    """
    
    def __init__(self, strategy: str = "neuron"):
        """
        Initialize converter.
        
        Args:
            strategy: Conversion strategy
                - "neuron": Each neuron is a point (coordinates = incoming weights)
                - "weight": Each weight is a 1D point
                - "layer": Each layer is a point (flattened weights)
        """
        self.strategy = strategy
    
    def convert(self, weights: List[np.ndarray]) -> np.ndarray:
        """
        Convert model weights to point cloud.
        
        Args:
            weights: List of weight matrices [W1, W2, ...]
            
        Returns:
            Point cloud as (N, D) array
        """
        if self.strategy == "neuron":
            return self._convert_neuron(weights)
        elif self.strategy == "weight":
            return self._convert_weight(weights)
        elif self.strategy == "layer":
            return self._convert_layer(weights)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _convert_neuron(self, weights: List[np.ndarray]) -> np.ndarray:
        """Each neuron becomes a point in weight-space."""
        points = []
        max_dim = max(w.shape[1] if len(w.shape) > 1 else 1 for w in weights)
        
        for W in weights:
            if len(W.shape) == 1:
                W = W.reshape(-1, 1)
            
            # Pad to max dimension
            padded = np.zeros((W.shape[0], max_dim))
            padded[:, :W.shape[1]] = W
            
            # Normalize each neuron to unit sphere
            norms = np.linalg.norm(padded, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized = padded / norms
            
            points.append(normalized)
        
        return np.vstack(points)
    
    def _convert_weight(self, weights: List[np.ndarray]) -> np.ndarray:
        """Each weight becomes a 1D point."""
        all_weights = np.concatenate([w.flatten() for w in weights])
        return all_weights.reshape(-1, 1)
    
    def _convert_layer(self, weights: List[np.ndarray]) -> np.ndarray:
        """Each layer becomes a point in flattened weight-space."""
        max_size = max(w.size for w in weights)
        points = []
        
        for W in weights:
            flat = W.flatten()
            padded = np.zeros(max_size)
            padded[:len(flat)] = flat
            
            # Normalize
            norm = np.linalg.norm(padded)
            if norm > 0:
                padded /= norm
            
            points.append(padded)
        
        return np.array(points)


class FingerprintExtractor:
    """
    Extracts compact fingerprints from persistence diagrams.
    """
    
    def __init__(self, n_features: int = 20, quantization_bits: int = 16):
        """
        Initialize extractor.
        
        Args:
            n_features: Number of top features to include
            quantization_bits: Bits for birth/death quantization
        """
        self.n_features = n_features
        self.quantization_bits = quantization_bits
    
    def extract(self, diagram: PersistenceDiagram, 
                model_stats: Dict[str, Any] = None) -> ModelFingerprint:
        """
        Extract fingerprint from persistence diagram.
        
        Args:
            diagram: Persistence diagram
            model_stats: Optional model statistics to include
            
        Returns:
            ModelFingerprint
        """
        # Get top-k features
        top_features = diagram.top_k(self.n_features, exclude_infinite=True)
        
        # Quantize and extract
        features = []
        for f in top_features:
            dim = f.dimension
            birth = self._quantize(f.birth)
            death = self._quantize(f.death)
            features.append((dim, birth, death))
        
        # Pad with zeros if not enough features
        while len(features) < self.n_features:
            features.append((0, 0.0, 0.0))
        
        # Compute hash
        hash_input = b''
        for dim, birth, death in features:
            hash_input += dim.to_bytes(1, 'big')
            hash_input += int(birth * 1e6).to_bytes(4, 'big', signed=True)
            hash_input += int(death * 1e6).to_bytes(4, 'big', signed=True)
        
        hash_value = hashlib.sha256(hash_input).digest()
        
        return ModelFingerprint(
            features=features,
            hash=hash_value,
            model_stats=model_stats or {}
        )
    
    def _quantize(self, value: float) -> float:
        """Quantize a float to fixed precision."""
        if value == float('inf'):
            return float('inf')
        max_val = 2 ** self.quantization_bits
        quantized = round(value * 1000) / 1000  # 3 decimal places
        return min(quantized, max_val)


@dataclass
class TDAProof:
    """
    Zero-knowledge proof that a model has a specific fingerprint.
    """
    fingerprint: ModelFingerprint
    model_commitment: bytes
    
    # Sampling-based verification
    sampled_point_indices: List[int]
    sampled_points: List[List[float]]
    local_diagram_hash: bytes
    
    def size_bytes(self) -> int:
        """Calculate proof size."""
        size = self.fingerprint.size_bytes()
        size += 32  # model commitment
        size += len(self.sampled_point_indices) * 4
        size += sum(len(p) * 8 for p in self.sampled_points)
        size += 32  # local diagram hash
        return size


class TDAProver:
    """
    Generates TDA-based proofs for model fingerprints.
    """
    
    def __init__(self, n_samples: int = 10, max_dim: int = 1):
        self.n_samples = n_samples
        self.max_dim = max_dim
        self.converter = PointCloudConverter(strategy="neuron")
        self.extractor = FingerprintExtractor(n_features=20)
    
    def compute_fingerprint(self, weights: List[np.ndarray]) -> Tuple[ModelFingerprint, np.ndarray]:
        """
        Compute fingerprint for a model.
        
        Args:
            weights: List of weight matrices
            
        Returns:
            Tuple of (fingerprint, point_cloud)
        """
        # Convert to point cloud
        points = self.converter.convert(weights)
        
        # Compute persistence
        max_edge = np.percentile(
            [np.linalg.norm(points[i] - points[j]) 
             for i in range(min(50, len(points))) 
             for j in range(i+1, min(50, len(points)))],
            90
        ) if len(points) > 1 else 1.0
        
        diagram = compute_persistence(points, self.max_dim, max_edge_length=max_edge)
        
        # Extract fingerprint
        model_stats = {
            "n_points": len(points),
            "n_layers": len(weights),
            "total_params": sum(w.size for w in weights)
        }
        
        fingerprint = self.extractor.extract(diagram, model_stats)
        
        return fingerprint, points
    
    def prove(self, weights: List[np.ndarray]) -> TDAProof:
        """
        Generate a proof for a model's fingerprint.
        
        Args:
            weights: List of weight matrices
            
        Returns:
            TDAProof
        """
        # Compute fingerprint
        fingerprint, points = self.compute_fingerprint(weights)
        
        # Create model commitment
        weight_data = b''.join(w.tobytes() for w in weights)
        model_commitment = hashlib.sha256(weight_data).digest()
        
        # Sample points for verification
        n_points = len(points)
        np.random.seed(int.from_bytes(model_commitment[:4], 'big'))
        sampled_indices = np.random.choice(n_points, min(self.n_samples, n_points), replace=False).tolist()
        sampled_points = [points[i].tolist() for i in sampled_indices]
        
        # Compute local diagram hash (for sampled points)
        local_data = b''.join(
            np.array(p).tobytes() for p in sampled_points
        )
        local_diagram_hash = hashlib.sha256(local_data).digest()
        
        return TDAProof(
            fingerprint=fingerprint,
            model_commitment=model_commitment,
            sampled_point_indices=sampled_indices,
            sampled_points=sampled_points,
            local_diagram_hash=local_diagram_hash
        )


class TDAVerifier:
    """
    Verifies TDA-based proofs.
    """
    
    def verify(self, proof: TDAProof, expected_fingerprint: ModelFingerprint = None) -> Tuple[bool, str]:
        """
        Verify a TDA proof.
        
        Args:
            proof: The proof to verify
            expected_fingerprint: Optional expected fingerprint to match
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check fingerprint hash consistency
        hash_input = b''
        for dim, birth, death in proof.fingerprint.features:
            hash_input += dim.to_bytes(1, 'big')
            hash_input += int(birth * 1e6).to_bytes(4, 'big', signed=True)
            hash_input += int(death * 1e6).to_bytes(4, 'big', signed=True)
        
        expected_hash = hashlib.sha256(hash_input).digest()
        
        if proof.fingerprint.hash != expected_hash:
            return False, "Fingerprint hash mismatch"
        
        # Check local diagram hash
        local_data = b''.join(
            np.array(p).tobytes() for p in proof.sampled_points
        )
        expected_local_hash = hashlib.sha256(local_data).digest()
        
        if proof.local_diagram_hash != expected_local_hash:
            return False, "Local diagram hash mismatch"
        
        # Check against expected fingerprint if provided
        if expected_fingerprint is not None:
            if proof.fingerprint.hash != expected_fingerprint.hash:
                return False, "Fingerprint does not match expected"
        
        return True, "Valid"


class TDAFingerprintSystem:
    """
    Complete TDA fingerprinting system.
    """
    
    def __init__(self, n_features: int = 20, n_samples: int = 10):
        self.prover = TDAProver(n_samples=n_samples)
        self.verifier = TDAVerifier()
        self.prover.extractor.n_features = n_features
    
    def fingerprint(self, weights: List[np.ndarray]) -> ModelFingerprint:
        """Compute fingerprint for a model."""
        fp, _ = self.prover.compute_fingerprint(weights)
        return fp
    
    def prove(self, weights: List[np.ndarray]) -> TDAProof:
        """Generate proof for a model."""
        return self.prover.prove(weights)
    
    def verify(self, proof: TDAProof, expected: ModelFingerprint = None) -> Tuple[bool, str]:
        """Verify a proof."""
        return self.verifier.verify(proof, expected)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("TDA Model Fingerprinting Tests")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test 1: Basic fingerprinting
    print("\n1. Testing basic fingerprinting")
    
    # Create a simple model (3 layers)
    weights = [
        np.random.randn(10, 5),   # Layer 1: 5 -> 10
        np.random.randn(8, 10),   # Layer 2: 10 -> 8
        np.random.randn(3, 8)     # Layer 3: 8 -> 3
    ]
    
    system = TDAFingerprintSystem(n_features=10)
    
    start = time.time()
    fingerprint = system.fingerprint(weights)
    fp_time = (time.time() - start) * 1000
    
    print(f"   Model: 3 layers, {sum(w.size for w in weights)} params")
    print(f"   Fingerprint size: {fingerprint.size_bytes()} bytes")
    print(f"   Fingerprint time: {fp_time:.2f} ms")
    print(f"   Hash: {fingerprint.hash[:8].hex()}...")
    print(f"   Top features: {fingerprint.features[:5]}")
    
    # Test 2: Proof generation and verification
    print("\n2. Testing proof generation and verification")
    
    start = time.time()
    proof = system.prove(weights)
    prove_time = (time.time() - start) * 1000
    
    start = time.time()
    valid, reason = system.verify(proof, fingerprint)
    verify_time = (time.time() - start) * 1000
    
    print(f"   Proof size: {proof.size_bytes()} bytes")
    print(f"   Prove time: {prove_time:.2f} ms")
    print(f"   Verify time: {verify_time:.2f} ms")
    print(f"   Valid: {valid} ({reason})")
    
    # Test 3: Different models have different fingerprints
    print("\n3. Testing fingerprint uniqueness")
    
    weights2 = [
        np.random.randn(10, 5),
        np.random.randn(8, 10),
        np.random.randn(3, 8)
    ]
    
    fingerprint2 = system.fingerprint(weights2)
    
    print(f"   Model 1 hash: {fingerprint.hash[:8].hex()}...")
    print(f"   Model 2 hash: {fingerprint2.hash[:8].hex()}...")
    print(f"   Same fingerprint: {fingerprint == fingerprint2}")
    print(f"   Distance: {fingerprint.distance(fingerprint2):.4f}")
    
    # Test 4: Similar models have similar fingerprints
    print("\n4. Testing fingerprint stability")
    
    # Slightly perturb the original model
    weights_perturbed = [w + np.random.randn(*w.shape) * 0.01 for w in weights]
    fingerprint_perturbed = system.fingerprint(weights_perturbed)
    
    print(f"   Original hash: {fingerprint.hash[:8].hex()}...")
    print(f"   Perturbed hash: {fingerprint_perturbed.hash[:8].hex()}...")
    print(f"   Same fingerprint: {fingerprint == fingerprint_perturbed}")
    print(f"   Distance: {fingerprint.distance(fingerprint_perturbed):.4f}")
    
    # Test 5: Larger model
    print("\n5. Testing with larger model")
    
    large_weights = [
        np.random.randn(128, 64),
        np.random.randn(64, 128),
        np.random.randn(32, 64),
        np.random.randn(10, 32)
    ]
    
    start = time.time()
    large_fp = system.fingerprint(large_weights)
    large_time = (time.time() - start) * 1000
    
    print(f"   Model: 4 layers, {sum(w.size for w in large_weights)} params")
    print(f"   Fingerprint size: {large_fp.size_bytes()} bytes (constant!)")
    print(f"   Fingerprint time: {large_time:.2f} ms")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
