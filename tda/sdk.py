"""
TDA Model Fingerprinting SDK.

A Python SDK for interacting with the TDA Fingerprinting API
and performing local fingerprint operations.
"""

import numpy as np
import requests
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Fingerprint:
    """Represents a model fingerprint."""
    hash: str
    features: List[Dict[str, float]]
    n_features: int
    size_bytes: int
    model_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "features": self.features,
            "n_features": self.n_features,
            "size_bytes": self.size_bytes,
            "model_stats": self.model_stats
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, path: str):
        """Save fingerprint to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> 'Fingerprint':
        """Load fingerprint from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            hash=data["hash"],
            features=data["features"],
            n_features=data["n_features"],
            size_bytes=data["size_bytes"],
            model_stats=data["model_stats"]
        )


@dataclass
class VerificationResult:
    """Result of fingerprint verification."""
    is_registered: bool
    owner: Optional[str]
    model_name: Optional[str]
    registration_time: Optional[int]


@dataclass
class RegistrationResult:
    """Result of model registration."""
    success: bool
    fingerprint_hash: str
    transaction_hash: Optional[str]
    block_number: Optional[int]
    error: Optional[str]


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    fingerprint_a: str
    fingerprint_b: str
    distance: float
    are_same: bool
    similarity_percent: float


class TDAFingerprintSDK:
    """
    SDK for TDA Model Fingerprinting.
    
    Can operate in two modes:
    1. API mode: Connects to a remote API server
    2. Local mode: Performs computations locally
    """
    
    def __init__(self, api_url: Optional[str] = None, local_mode: bool = True):
        """
        Initialize the SDK.
        
        Args:
            api_url: URL of the API server (e.g., "http://localhost:8000")
            local_mode: If True, perform computations locally
        """
        self.api_url = api_url
        self.local_mode = local_mode or (api_url is None)
        
        if self.local_mode:
            # Import local components
            from tda.fingerprint import TDAFingerprintSystem
            from tda.scaled_persistence import compute_scaled_persistence
            from tda.chain_integration import (
                ChainConfig, OnChainFingerprintManager
            )
            
            self._fingerprinter = TDAFingerprintSystem()
            self._compute_persistence = compute_scaled_persistence
            self._chain_manager = OnChainFingerprintManager(ChainConfig.local())
    
    def fingerprint(
        self,
        weights: Union[List[np.ndarray], Dict[str, np.ndarray]],
        n_landmarks: int = 100,
        top_k: int = 20
    ) -> Fingerprint:
        """
        Generate a fingerprint for a model.
        
        Args:
            weights: Model weights as list of arrays or dict
            n_landmarks: Number of landmarks for scaling
            top_k: Number of top features to include
            
        Returns:
            Fingerprint object
        """
        if isinstance(weights, dict):
            weights = list(weights.values())
        
        if self.local_mode:
            return self._fingerprint_local(weights, n_landmarks, top_k)
        else:
            return self._fingerprint_api(weights, n_landmarks, top_k)
    
    def _fingerprint_local(
        self,
        weights: List[np.ndarray],
        n_landmarks: int,
        top_k: int
    ) -> Fingerprint:
        """Generate fingerprint locally."""
        # Flatten weights to point cloud
        all_weights = []
        for w in weights:
            all_weights.extend(w.flatten().tolist())
        
        n_points = len(all_weights)
        dim = min(50, n_points // 10) if n_points > 100 else max(2, n_points // 5)
        
        # Pad
        padded_length = ((n_points + dim - 1) // dim) * dim
        padded_weights = all_weights + [0.0] * (padded_length - n_points)
        points = np.array(padded_weights).reshape(-1, dim)
        
        # Compute persistence
        diagram, meta = self._compute_persistence(points, n_landmarks=n_landmarks)
        
        # Extract features
        features = diagram.features
        sorted_features = sorted(features, key=lambda f: f.death - f.birth, reverse=True)
        top_features = sorted_features[:top_k]
        
        feature_list = [
            {"dimension": f.dimension, "birth": f.birth, "death": f.death}
            for f in top_features
        ]
        
        # Create fingerprint
        fingerprint_data = {
            "features": feature_list,
            "n_features": len(features),
            "n_params": n_points
        }
        
        fingerprint_bytes = json.dumps(fingerprint_data, sort_keys=True).encode()
        fingerprint_hash = "0x" + hashlib.sha256(fingerprint_bytes).hexdigest()
        
        return Fingerprint(
            hash=fingerprint_hash,
            features=feature_list,
            n_features=len(features),
            size_bytes=len(fingerprint_bytes),
            model_stats={
                "n_layers": len(weights),
                "n_params": n_points,
                "n_landmarks_used": meta["n_landmarks"]
            }
        )
    
    def _fingerprint_api(
        self,
        weights: List[np.ndarray],
        n_landmarks: int,
        top_k: int
    ) -> Fingerprint:
        """Generate fingerprint via API."""
        payload = {
            "weights": {
                "layers": [w.tolist() for w in weights]
            },
            "n_landmarks": n_landmarks,
            "top_k_features": top_k
        }
        
        response = requests.post(f"{self.api_url}/fingerprint", json=payload)
        response.raise_for_status()
        data = response.json()
        
        return Fingerprint(
            hash=data["fingerprint_hash"],
            features=data["top_features"],
            n_features=data["n_features"],
            size_bytes=data["fingerprint_size_bytes"],
            model_stats=data["model_stats"]
        )
    
    def verify(self, fingerprint: Union[Fingerprint, str]) -> VerificationResult:
        """
        Verify if a fingerprint is registered.
        
        Args:
            fingerprint: Fingerprint object or hash string
            
        Returns:
            VerificationResult
        """
        fp_hash = fingerprint.hash if isinstance(fingerprint, Fingerprint) else fingerprint
        
        if self.local_mode:
            result = self._chain_manager.client.verify_by_hash(fp_hash)
            return VerificationResult(
                is_registered=result.is_registered,
                owner=result.owner,
                model_name=result.model_name,
                registration_time=result.registration_time
            )
        else:
            response = requests.post(
                f"{self.api_url}/verify",
                json={"fingerprint_hash": fp_hash}
            )
            response.raise_for_status()
            data = response.json()
            
            return VerificationResult(
                is_registered=data["is_registered"],
                owner=data.get("owner"),
                model_name=data.get("model_name"),
                registration_time=data.get("registration_time")
            )
    
    def register(
        self,
        weights: Union[List[np.ndarray], Dict[str, np.ndarray]],
        model_name: str,
        model_version: str = "1.0.0"
    ) -> RegistrationResult:
        """
        Register a model on-chain.
        
        Args:
            weights: Model weights
            model_name: Name for the model
            model_version: Version string
            
        Returns:
            RegistrationResult
        """
        if isinstance(weights, dict):
            weights = list(weights.values())
        
        # First compute fingerprint
        fp = self.fingerprint(weights)
        
        if self.local_mode:
            result = self._chain_manager.register_from_fingerprint(
                fingerprint_data={
                    "features": fp.features,
                    "n_features": fp.n_features,
                    "n_params": fp.model_stats.get("n_params", 0)
                },
                model_name=model_name,
                model_version=model_version
            )
            
            return RegistrationResult(
                success=result.success,
                fingerprint_hash=result.fingerprint_hash,
                transaction_hash=result.transaction_hash,
                block_number=result.block_number,
                error=result.error
            )
        else:
            payload = {
                "weights": {"layers": [w.tolist() for w in weights]},
                "model_name": model_name,
                "model_version": model_version
            }
            
            response = requests.post(f"{self.api_url}/register", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return RegistrationResult(
                success=data["success"],
                fingerprint_hash=data["fingerprint_hash"],
                transaction_hash=data.get("transaction_hash"),
                block_number=data.get("block_number"),
                error=data.get("error")
            )
    
    def compare(
        self,
        weights_a: Union[List[np.ndarray], Dict[str, np.ndarray]],
        weights_b: Union[List[np.ndarray], Dict[str, np.ndarray]]
    ) -> ComparisonResult:
        """
        Compare two models.
        
        Args:
            weights_a: First model weights
            weights_b: Second model weights
            
        Returns:
            ComparisonResult
        """
        if isinstance(weights_a, dict):
            weights_a = list(weights_a.values())
        if isinstance(weights_b, dict):
            weights_b = list(weights_b.values())
        
        fp_a = self.fingerprint(weights_a)
        fp_b = self.fingerprint(weights_b)
        
        # Compute distance
        def feature_vector(features):
            return [f["death"] - f["birth"] for f in features]
        
        vec_a = np.array(feature_vector(fp_a.features))
        vec_b = np.array(feature_vector(fp_b.features))
        
        max_len = max(len(vec_a), len(vec_b))
        vec_a = np.pad(vec_a, (0, max_len - len(vec_a)))
        vec_b = np.pad(vec_b, (0, max_len - len(vec_b)))
        
        # Handle potential NaN values
        vec_a = np.nan_to_num(vec_a, nan=0.0)
        vec_b = np.nan_to_num(vec_b, nan=0.0)
        
        distance = np.linalg.norm(vec_a - vec_b)
        max_norm = max(np.linalg.norm(vec_a), np.linalg.norm(vec_b), 1e-10)
        normalized_distance = distance / max_norm if max_norm > 1e-10 else 0.0
        
        are_same = (fp_a.hash == fp_b.hash) or (normalized_distance < 0.01)
        similarity = max(0, 1 - normalized_distance) * 100
        
        return ComparisonResult(
            fingerprint_a=fp_a.hash,
            fingerprint_b=fp_b.hash,
            distance=float(normalized_distance),
            are_same=are_same,
            similarity_percent=float(similarity)
        )


# Convenience functions
def fingerprint_model(weights: Union[List[np.ndarray], Dict[str, np.ndarray]]) -> Fingerprint:
    """Quick fingerprint generation."""
    sdk = TDAFingerprintSDK(local_mode=True)
    return sdk.fingerprint(weights)


def compare_models(
    weights_a: Union[List[np.ndarray], Dict[str, np.ndarray]],
    weights_b: Union[List[np.ndarray], Dict[str, np.ndarray]]
) -> ComparisonResult:
    """Quick model comparison."""
    sdk = TDAFingerprintSDK(local_mode=True)
    return sdk.compare(weights_a, weights_b)


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("TDA Fingerprinting SDK Tests")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Initialize SDK
    sdk = TDAFingerprintSDK(local_mode=True)
    
    # Test 1: Generate fingerprint
    print("\n1. Generating fingerprint")
    weights = [
        np.random.randn(10, 5),
        np.random.randn(5, 10),
        np.random.randn(3, 5)
    ]
    
    fp = sdk.fingerprint(weights)
    print(f"   Hash: {fp.hash[:20]}...")
    print(f"   Size: {fp.size_bytes} bytes")
    print(f"   Features: {fp.n_features}")
    print(f"   Model stats: {fp.model_stats}")
    
    # Test 2: Register model
    print("\n2. Registering model")
    result = sdk.register(weights, model_name="TestSDKModel", model_version="1.0.0")
    print(f"   Success: {result.success}")
    print(f"   Hash: {result.fingerprint_hash[:20]}...")
    print(f"   TX: {result.transaction_hash[:20]}..." if result.transaction_hash else "   TX: None")
    
    # Test 3: Verify registration
    print("\n3. Verifying registration")
    verification = sdk.verify(fp)
    print(f"   Registered: {verification.is_registered}")
    print(f"   Owner: {verification.owner}")
    print(f"   Name: {verification.model_name}")
    
    # Test 4: Compare models
    print("\n4. Comparing models")
    weights_similar = [w + np.random.randn(*w.shape) * 0.01 for w in weights]
    weights_different = [np.random.randn(*w.shape) for w in weights]
    
    comp_similar = sdk.compare(weights, weights_similar)
    print(f"   Similar models:")
    print(f"     Distance: {comp_similar.distance:.4f}")
    print(f"     Same: {comp_similar.are_same}")
    print(f"     Similarity: {comp_similar.similarity_percent:.1f}%")
    
    comp_different = sdk.compare(weights, weights_different)
    print(f"   Different models:")
    print(f"     Distance: {comp_different.distance:.4f}")
    print(f"     Same: {comp_different.are_same}")
    print(f"     Similarity: {comp_different.similarity_percent:.1f}%")
    
    # Test 5: Save and load fingerprint
    print("\n5. Save and load fingerprint")
    fp.save("/tmp/test_fingerprint.json")
    loaded_fp = Fingerprint.load("/tmp/test_fingerprint.json")
    print(f"   Original hash: {fp.hash[:20]}...")
    print(f"   Loaded hash: {loaded_fp.hash[:20]}...")
    print(f"   Match: {fp.hash == loaded_fp.hash}")
    
    # Test 6: Convenience functions
    print("\n6. Convenience functions")
    quick_fp = fingerprint_model(weights)
    print(f"   Quick fingerprint: {quick_fp.hash[:20]}...")
    
    quick_comp = compare_models(weights, weights_similar)
    print(f"   Quick comparison: {quick_comp.similarity_percent:.1f}% similar")
    
    print("\n" + "=" * 70)
    print("SDK Tests Complete!")
    print("=" * 70)
