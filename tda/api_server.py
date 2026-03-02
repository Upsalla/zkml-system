"""
TDA Model Fingerprinting API Server.

A standalone REST API for generating, verifying, and registering
TDA-based model fingerprints.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import time
import hashlib
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TDA components
from tda.fingerprint import ModelFingerprinter, TDAProofSystem
from tda.scaled_persistence import compute_scaled_persistence
from tda.chain_integration import (
    ChainConfig, ChainType, OnChainFingerprintManager,
    RegistrationResult, VerificationResult
)


# ============================================================
# API Models
# ============================================================

class ModelWeights(BaseModel):
    """Input model for neural network weights."""
    layers: List[List[List[float]]] = Field(
        ..., 
        description="List of weight matrices, each as a 2D list"
    )
    model_name: Optional[str] = Field(None, description="Optional model name")
    model_version: Optional[str] = Field("1.0.0", description="Model version")


class FingerprintRequest(BaseModel):
    """Request for fingerprint generation."""
    weights: ModelWeights
    n_landmarks: Optional[int] = Field(100, description="Number of landmarks for scaling")
    top_k_features: Optional[int] = Field(20, description="Number of top features to include")


class FingerprintResponse(BaseModel):
    """Response containing the fingerprint."""
    fingerprint_hash: str
    fingerprint_size_bytes: int
    n_features: int
    top_features: List[Dict[str, float]]
    computation_time_ms: float
    model_stats: Dict[str, Any]


class VerifyRequest(BaseModel):
    """Request for fingerprint verification."""
    fingerprint_hash: str
    expected_owner: Optional[str] = None


class VerifyResponse(BaseModel):
    """Response for verification."""
    is_registered: bool
    owner: Optional[str]
    model_name: Optional[str]
    registration_time: Optional[int]
    matches_expected_owner: Optional[bool]


class RegisterRequest(BaseModel):
    """Request for on-chain registration."""
    weights: ModelWeights
    model_name: str
    model_version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None


class RegisterResponse(BaseModel):
    """Response for registration."""
    success: bool
    fingerprint_hash: str
    transaction_hash: Optional[str]
    block_number: Optional[int]
    gas_used: Optional[int]
    error: Optional[str]


class CompareRequest(BaseModel):
    """Request for comparing two models."""
    weights_a: ModelWeights
    weights_b: ModelWeights


class CompareResponse(BaseModel):
    """Response for model comparison."""
    fingerprint_hash_a: str
    fingerprint_hash_b: str
    distance: float
    are_same_model: bool
    similarity_percentage: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    total_fingerprints_generated: int


# ============================================================
# API Server
# ============================================================

# Global state
START_TIME = time.time()
FINGERPRINT_COUNT = 0

app = FastAPI(
    title="TDA Model Fingerprinting API",
    description="Generate, verify, and register topological fingerprints for ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chain manager (mock for now)
chain_config = ChainConfig.local()
chain_manager = OnChainFingerprintManager(chain_config)


def weights_to_numpy(weights: ModelWeights) -> List[np.ndarray]:
    """Convert ModelWeights to list of numpy arrays."""
    return [np.array(layer) for layer in weights.layers]


def compute_fingerprint(weights: List[np.ndarray], 
                        n_landmarks: int = 100,
                        top_k: int = 20) -> Dict[str, Any]:
    """Compute TDA fingerprint for weights."""
    global FINGERPRINT_COUNT
    
    start_time = time.time()
    
    # Flatten weights to point cloud
    all_weights = []
    for w in weights:
        all_weights.extend(w.flatten().tolist())
    
    # Create point cloud (reshape to 2D)
    n_points = len(all_weights)
    dim = min(50, n_points // 10) if n_points > 100 else max(2, n_points // 5)
    
    # Pad to make divisible
    padded_length = ((n_points + dim - 1) // dim) * dim
    padded_weights = all_weights + [0.0] * (padded_length - n_points)
    
    points = np.array(padded_weights).reshape(-1, dim)
    
    # Compute scaled persistence
    diagram, meta = compute_scaled_persistence(points, n_landmarks=n_landmarks)
    
    # Extract top features
    features = diagram.features
    sorted_features = sorted(features, key=lambda f: f.death - f.birth, reverse=True)
    top_features = sorted_features[:top_k]
    
    # Create fingerprint data
    feature_list = [
        {"dimension": f.dimension, "birth": f.birth, "death": f.death}
        for f in top_features
    ]
    
    fingerprint_data = {
        "features": feature_list,
        "n_features": len(features),
        "n_params": n_points
    }
    
    # Hash
    fingerprint_bytes = json.dumps(fingerprint_data, sort_keys=True).encode()
    fingerprint_hash = "0x" + hashlib.sha256(fingerprint_bytes).hexdigest()
    
    computation_time = (time.time() - start_time) * 1000
    FINGERPRINT_COUNT += 1
    
    return {
        "fingerprint_hash": fingerprint_hash,
        "fingerprint_data": fingerprint_data,
        "fingerprint_bytes": fingerprint_bytes,
        "fingerprint_size_bytes": len(fingerprint_bytes),
        "n_features": len(features),
        "top_features": feature_list,
        "computation_time_ms": computation_time,
        "model_stats": {
            "n_layers": len(weights),
            "n_params": n_points,
            "n_landmarks_used": meta["n_landmarks"]
        }
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - START_TIME,
        total_fingerprints_generated=FINGERPRINT_COUNT
    )


@app.post("/fingerprint", response_model=FingerprintResponse)
async def generate_fingerprint(request: FingerprintRequest):
    """
    Generate a TDA fingerprint for a model.
    
    The fingerprint is a topological signature that uniquely identifies
    the model's weight structure.
    """
    try:
        weights = weights_to_numpy(request.weights)
        result = compute_fingerprint(
            weights, 
            n_landmarks=request.n_landmarks,
            top_k=request.top_k_features
        )
        
        return FingerprintResponse(
            fingerprint_hash=result["fingerprint_hash"],
            fingerprint_size_bytes=result["fingerprint_size_bytes"],
            n_features=result["n_features"],
            top_features=result["top_features"],
            computation_time_ms=result["computation_time_ms"],
            model_stats=result["model_stats"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerifyResponse)
async def verify_fingerprint(request: VerifyRequest):
    """
    Verify if a fingerprint is registered on-chain.
    """
    try:
        result = chain_manager.client.verify_by_hash(request.fingerprint_hash)
        
        matches_expected = None
        if request.expected_owner and result.is_registered:
            matches_expected = result.owner.lower() == request.expected_owner.lower()
        
        return VerifyResponse(
            is_registered=result.is_registered,
            owner=result.owner,
            model_name=result.model_name,
            registration_time=result.registration_time,
            matches_expected_owner=matches_expected
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register", response_model=RegisterResponse)
async def register_model(request: RegisterRequest):
    """
    Register a model fingerprint on-chain.
    
    This computes the fingerprint and submits it to the blockchain.
    """
    try:
        weights = weights_to_numpy(request.weights)
        fingerprint_result = compute_fingerprint(weights)
        
        # Register on chain
        result = chain_manager.register_from_fingerprint(
            fingerprint_data=fingerprint_result["fingerprint_data"],
            model_name=request.model_name,
            model_version=request.model_version
        )
        
        return RegisterResponse(
            success=result.success,
            fingerprint_hash=result.fingerprint_hash,
            transaction_hash=result.transaction_hash,
            block_number=result.block_number,
            gas_used=result.gas_used,
            error=result.error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """
    Compare two models by their fingerprints.
    
    Returns the distance between fingerprints and whether they
    represent the same model.
    """
    try:
        weights_a = weights_to_numpy(request.weights_a)
        weights_b = weights_to_numpy(request.weights_b)
        
        fp_a = compute_fingerprint(weights_a)
        fp_b = compute_fingerprint(weights_b)
        
        # Compute distance between feature sets
        features_a = fp_a["top_features"]
        features_b = fp_b["top_features"]
        
        # Simple distance: compare persistence values
        def feature_vector(features):
            return [f["death"] - f["birth"] for f in features]
        
        vec_a = np.array(feature_vector(features_a))
        vec_b = np.array(feature_vector(features_b))
        
        # Pad to same length
        max_len = max(len(vec_a), len(vec_b))
        vec_a = np.pad(vec_a, (0, max_len - len(vec_a)))
        vec_b = np.pad(vec_b, (0, max_len - len(vec_b)))
        
        distance = np.linalg.norm(vec_a - vec_b)
        
        # Normalize distance
        max_norm = max(np.linalg.norm(vec_a), np.linalg.norm(vec_b), 1e-10)
        normalized_distance = distance / max_norm
        
        # Same model if hashes match or distance is very small
        are_same = (fp_a["fingerprint_hash"] == fp_b["fingerprint_hash"]) or (normalized_distance < 0.01)
        similarity = max(0, 1 - normalized_distance) * 100
        
        return CompareResponse(
            fingerprint_hash_a=fp_a["fingerprint_hash"],
            fingerprint_hash_b=fp_b["fingerprint_hash"],
            distance=float(normalized_distance),
            are_same_model=are_same,
            similarity_percentage=float(similarity)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "total_fingerprints_generated": FINGERPRINT_COUNT,
        "total_registered_models": chain_manager.client.get_registration_count(),
        "uptime_seconds": time.time() - START_TIME,
        "gas_estimates": chain_manager.client.estimate_registration_gas()
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("TDA Model Fingerprinting API Server")
    print("=" * 70)
    print("\nStarting server on http://0.0.0.0:8000")
    print("\nEndpoints:")
    print("  GET  /health     - Health check")
    print("  POST /fingerprint - Generate fingerprint")
    print("  POST /verify     - Verify registration")
    print("  POST /register   - Register on-chain")
    print("  POST /compare    - Compare two models")
    print("  GET  /stats      - API statistics")
    print("\nAPI docs: http://0.0.0.0:8000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
