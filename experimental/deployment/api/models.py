"""
Pydantic Models für die zkML API

Diese Modelle definieren die Eingabe- und Ausgabeformate für die REST API.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ActivationType(str, Enum):
    """Unterstützte Aktivierungsfunktionen."""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    LINEAR = "linear"


class LayerConfig(BaseModel):
    """Konfiguration für einen einzelnen Layer."""
    type: Literal["dense", "conv2d"] = Field(
        description="Typ des Layers"
    )
    weights: List[List[float]] = Field(
        description="Gewichtsmatrix [output_dim x input_dim]"
    )
    biases: List[float] = Field(
        description="Bias-Vektor [output_dim]"
    )
    activation: ActivationType = Field(
        default=ActivationType.GELU,
        description="Aktivierungsfunktion"
    )


class NetworkConfig(BaseModel):
    """Vollständige Netzwerk-Konfiguration."""
    name: str = Field(
        default="unnamed_network",
        description="Name des Netzwerks"
    )
    layers: List[LayerConfig] = Field(
        description="Liste der Layer-Konfigurationen"
    )
    input_size: int = Field(
        description="Größe der Eingabe"
    )


class ProveRequest(BaseModel):
    """Anfrage für Proof-Generierung."""
    network: NetworkConfig = Field(
        description="Netzwerk-Konfiguration"
    )
    inputs: List[float] = Field(
        description="Eingabewerte für die Inferenz"
    )
    use_sparse: bool = Field(
        default=True,
        description="Sparse-Optimierung aktivieren"
    )
    use_gelu: bool = Field(
        default=True,
        description="GELU-Optimierung aktivieren"
    )
    activation_values: Optional[List[List[float]]] = Field(
        default=None,
        description="Aktivierungswerte für Sparse-Optimierung (optional)"
    )


class ProofData(BaseModel):
    """Serialisierte Proof-Daten."""
    circuit_hash: str = Field(
        description="Hash des kompilierten Circuits"
    )
    num_gates: int = Field(
        description="Anzahl der Gates im Circuit"
    )
    num_sparse_gates: int = Field(
        description="Anzahl der Sparse-Gates"
    )
    num_gelu_gates: int = Field(
        description="Anzahl der GELU-Gates"
    )
    public_inputs: List[str] = Field(
        description="Öffentliche Eingaben (als Strings)"
    )
    public_outputs: List[str] = Field(
        description="Öffentliche Ausgaben (als Strings)"
    )
    wire_commitments: List[Dict[str, str]] = Field(
        description="KZG-Commitments für die Wires"
    )
    proof_size_bytes: int = Field(
        description="Größe des Proofs in Bytes"
    )


class ProveResponse(BaseModel):
    """Antwort der Proof-Generierung."""
    success: bool = Field(
        description="Ob die Proof-Generierung erfolgreich war"
    )
    proof: Optional[ProofData] = Field(
        default=None,
        description="Der generierte Proof"
    )
    outputs: List[float] = Field(
        default=[],
        description="Inferenz-Ausgaben"
    )
    prover_time_ms: float = Field(
        description="Zeit für Proof-Generierung in Millisekunden"
    )
    error: Optional[str] = Field(
        default=None,
        description="Fehlermeldung bei Misserfolg"
    )


class VerifyRequest(BaseModel):
    """Anfrage für Proof-Verifikation."""
    proof: ProofData = Field(
        description="Der zu verifizierende Proof"
    )
    network: NetworkConfig = Field(
        description="Netzwerk-Konfiguration für Verifikation"
    )
    inputs: List[float] = Field(
        description="Öffentliche Eingaben"
    )


class VerifyResponse(BaseModel):
    """Antwort der Proof-Verifikation."""
    valid: bool = Field(
        description="Ob der Proof gültig ist"
    )
    verification_time_ms: float = Field(
        description="Zeit für Verifikation in Millisekunden"
    )
    error: Optional[str] = Field(
        default=None,
        description="Fehlermeldung bei ungültigem Proof"
    )


class CircuitStats(BaseModel):
    """Statistiken über einen kompilierten Circuit."""
    total_gates: int
    sparse_gates: int
    gelu_gates: int
    mul_gates: int
    add_gates: int
    wires: int
    sparse_ratio: float
    estimated_proof_size_bytes: int


class CompileRequest(BaseModel):
    """Anfrage für Circuit-Kompilierung (ohne Proof)."""
    network: NetworkConfig
    inputs: List[float]
    use_sparse: bool = True
    use_gelu: bool = True
    activation_values: Optional[List[List[float]]] = None


class CompileResponse(BaseModel):
    """Antwort der Circuit-Kompilierung."""
    success: bool
    stats: Optional[CircuitStats] = None
    compile_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health-Check Antwort."""
    status: str = "healthy"
    version: str = "0.1.0"
    features: Dict[str, bool] = Field(
        default_factory=lambda: {
            "gelu_optimization": True,
            "sparse_proofs": True,
            "plonk_backend": True,
        }
    )
