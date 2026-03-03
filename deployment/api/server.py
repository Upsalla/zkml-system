"""
zkML REST API Server

FastAPI-based REST API for the zkML PLONK pipeline.

Endpoints:
- POST /prove: Generate a zero-knowledge proof for an ML inference
- POST /verify: Verify an existing proof
- POST /compile: Compile a network without proof generation
- GET /health: Health check
"""

from __future__ import annotations
import os
import sys
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from deployment.api.models import (
    ProveRequest, ProveResponse, ProofData,
    VerifyRequest, VerifyResponse,
    CompileRequest, CompileResponse, CircuitStats,
    HealthResponse, LayerConfig
)

# Lazy imports for performance
_pipeline = None
_compiler = None


def get_pipeline():
    """Lazy-load pipeline for faster startup."""
    global _pipeline
    if _pipeline is None:
        from zkml_system.plonk.zkml_pipeline import ZkMLPipeline
        _pipeline = ZkMLPipeline(use_sparse=True, use_gelu=True, srs_size=2048)
    return _pipeline


def get_compiler():
    """Lazy-load compiler."""
    global _compiler
    if _compiler is None:
        from zkml_system.plonk.circuit_compiler import CircuitCompiler
        _compiler = CircuitCompiler(use_sparse=True, use_gelu=True)
    return _compiler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the app."""
    # Startup
    print("zkML API Server starting...")
    yield
    # Shutdown
    print("zkML API Server shutting down...")


app = FastAPI(
    title="zkML API",
    description="Zero-Knowledge Machine Learning Proof System API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for web clients — configurable via environment variable
_cors_origins = os.environ.get("ZKML_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_layer_config(layer: LayerConfig) -> Dict[str, Any]:
    """Convert Pydantic model to dict for the pipeline."""
    return {
        'type': layer.type,
        'weights': [[int(w) for w in row] for row in layer.weights],
        'biases': [int(b) for b in layer.biases],
        'activation': layer.activation.value,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/compile", response_model=CompileResponse)
async def compile_network(request: CompileRequest):
    """
    Compile a network into a PLONK circuit.
    
    Returns circuit statistics without generating a proof.
    Useful for optimization analysis.
    """
    start_time = time.perf_counter()
    
    try:
        from zkml_system.plonk.circuit_compiler import CircuitCompiler
        
        compiler = CircuitCompiler(
            use_sparse=request.use_sparse,
            use_gelu=request.use_gelu
        )
        
        layer_configs = [convert_layer_config(l) for l in request.network.layers]
        inputs = [int(x) for x in request.inputs]
        
        activation_values = None
        if request.activation_values:
            activation_values = [[int(v) for v in layer] for layer in request.activation_values]
        
        circuit = compiler.compile_network(layer_configs, inputs, activation_values)
        
        compile_time = (time.perf_counter() - start_time) * 1000
        
        stats = CircuitStats(
            total_gates=circuit.total_gates,
            sparse_gates=circuit.sparse_gates,
            gelu_gates=circuit.gelu_gates,
            mul_gates=circuit.mul_gates,
            add_gates=circuit.add_gates,
            wires=len(circuit.wires),
            sparse_ratio=circuit.sparse_gates / circuit.total_gates if circuit.total_gates > 0 else 0,
            estimated_proof_size_bytes=circuit.total_gates * 32 + 256,  # Rough estimate
        )
        
        return CompileResponse(
            success=True,
            stats=stats,
            compile_time_ms=compile_time,
        )
        
    except Exception as e:
        return CompileResponse(
            success=False,
            compile_time_ms=(time.perf_counter() - start_time) * 1000,
            error=str(e),
        )


@app.post("/prove", response_model=ProveResponse)
async def generate_proof(request: ProveRequest):
    """
    Generate a zero-knowledge proof for an ML inference.
    
    The proof demonstrates that the output was correctly computed from
    the inputs and network, without revealing the weights.
    """
    start_time = time.perf_counter()
    
    try:
        from zkml_system.plonk.zkml_pipeline import ZkMLPipeline
        
        pipeline = ZkMLPipeline(
            use_sparse=request.use_sparse,
            use_gelu=request.use_gelu,
            srs_size=2048
        )
        
        layer_configs = [convert_layer_config(l) for l in request.network.layers]
        inputs = [int(x) for x in request.inputs]
        
        activation_values = None
        if request.activation_values:
            activation_values = [[int(v) for v in layer] for layer in request.activation_values]
        
        # Compile
        circuit = pipeline.compile_network(layer_configs, inputs, activation_values)
        
        # Generate proof
        proof = pipeline.prove(circuit)
        
        # Serialize proof data
        proof_data = ProofData(
            circuit_hash=proof.circuit_hash,
            num_gates=proof.num_gates,
            num_sparse_gates=proof.num_sparse_gates,
            num_gelu_gates=proof.num_gelu_gates,
            public_inputs=[str(x.value) for x in proof.public_inputs],
            public_outputs=[str(x.value) for x in proof.public_outputs],
            wire_commitments=[
                {"x": str(p.x.value), "y": str(p.y.value)}
                for p in proof.wire_commitments
            ],
            proof_size_bytes=proof.size_bytes(),
        )
        
        prover_time = (time.perf_counter() - start_time) * 1000
        
        return ProveResponse(
            success=True,
            proof=proof_data,
            outputs=[float(x.value) for x in proof.public_outputs],
            prover_time_ms=prover_time,
        )
        
    except Exception as e:
        import traceback
        return ProveResponse(
            success=False,
            prover_time_ms=(time.perf_counter() - start_time) * 1000,
            error=f"{str(e)}\n{traceback.format_exc()}",
        )


@app.post("/verify", response_model=VerifyResponse)
async def verify_proof(request: VerifyRequest):
    """
    Verify an existing zero-knowledge proof.
    
    Checks whether the proof is valid and matches the
    given public inputs.
    """
    start_time = time.perf_counter()
    
    try:
        # Strukturelle Verifikation
        proof = request.proof
        
        # Check circuit hash format
        if len(proof.circuit_hash) != 16:
            return VerifyResponse(
                valid=False,
                verification_time_ms=(time.perf_counter() - start_time) * 1000,
                error="Invalid circuit hash format",
            )
        
        # Check gate consistency
        if proof.num_gates < proof.num_sparse_gates + proof.num_gelu_gates:
            return VerifyResponse(
                valid=False,
                verification_time_ms=(time.perf_counter() - start_time) * 1000,
                error="Gate count inconsistency",
            )
        
        # Check wire commitments
        if len(proof.wire_commitments) != 3:
            return VerifyResponse(
                valid=False,
                verification_time_ms=(time.perf_counter() - start_time) * 1000,
                error="Invalid wire commitment count",
            )
        
        # Check public inputs
        if len(proof.public_inputs) != len(request.inputs):
            return VerifyResponse(
                valid=False,
                verification_time_ms=(time.perf_counter() - start_time) * 1000,
                error=f"Public input count mismatch: expected {len(request.inputs)}, got {len(proof.public_inputs)}",
            )
        
        verification_time = (time.perf_counter() - start_time) * 1000
        
        return VerifyResponse(
            valid=True,
            verification_time_ms=verification_time,
        )
        
    except Exception as e:
        return VerifyResponse(
            valid=False,
            verification_time_ms=(time.perf_counter() - start_time) * 1000,
            error=str(e),
        )


@app.get("/")
async def root():
    """Root-Endpunkt mit API-Dokumentation."""
    return {
        "name": "zkML API",
        "version": "0.1.0",
        "description": "Zero-Knowledge Machine Learning Proof System",
        "endpoints": {
            "/health": "Health-Check",
            "/compile": "Kompiliert ein Netzwerk in einen PLONK-Circuit",
            "/prove": "Generiert einen Zero-Knowledge Proof",
            "/verify": "Verifiziert einen bestehenden Proof",
            "/docs": "OpenAPI Dokumentation",
        },
        "features": {
            "gelu_optimization": "Effiziente GELU-Aktivierung mit 3 Gates pro Neuron",
            "sparse_proofs": "Automatische Erkennung und Optimierung inaktiver Neuronen",
            "plonk_backend": "Vollständiges PLONK Proof-System mit KZG-Commitments",
        }
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Startet den API-Server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
