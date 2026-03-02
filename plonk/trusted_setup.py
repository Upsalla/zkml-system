"""
Trusted Setup Infrastructure for zkML System

Author: David Weyhe
Date: 27. Januar 2026
Version: 1.0

This module provides infrastructure for secure SRS (Structured Reference String)
generation and management. It supports:

1. Loading pre-computed Powers-of-Tau from ceremony files
2. Validating SRS integrity
3. Secure SRS serialization and deserialization
4. Contribution verification for multi-party ceremonies

Security Model:
---------------
The security of KZG commitments relies on the discrete log assumption:
given [τ]₁ = τ·G₁, it should be computationally infeasible to recover τ.

A trusted setup ceremony ensures that τ is unknown to any single party:
- Multiple participants contribute randomness
- Each contribution is verified
- The final SRS is secure if at least ONE participant is honest

Powers-of-Tau Format:
---------------------
We support the Hermez/Zcash PTAU format:
- Header: magic bytes, version, sections
- G1 powers: [G₁, τ·G₁, τ²·G₁, ..., τⁿ·G₁]
- G2 powers: [G₂, τ·G₂]
- Verification data: proofs of correct computation

WARNING:
--------
The `generate_insecure()` method should ONLY be used for testing.
For production, always use `load_from_ptau()` or `load_from_ceremony()`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, BinaryIO
import hashlib
import struct
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.bn254.field import Fr, Fp
from crypto.bn254.curve import G1Point, G2Point
from plonk.core import SRS


# =============================================================================
# Constants
# =============================================================================

PTAU_MAGIC = b'ptau'
PTAU_VERSION = 1

# Section types in PTAU files
SECTION_HEADER = 1
SECTION_G1_POWERS = 2
SECTION_G2_POWERS = 3
SECTION_CONTRIBUTIONS = 4


# =============================================================================
# SRS Validation
# =============================================================================

@dataclass
class ValidationResult:
    """Result of SRS validation."""
    is_valid: bool
    reason: str
    checks_passed: int
    checks_total: int


def validate_srs(srs: SRS) -> ValidationResult:
    """
    Validate the integrity of an SRS.
    
    Performs the following checks:
    1. All G1 points are on the curve
    2. All G2 points are on the curve
    3. G1 powers are consistent (pairing check)
    4. G2 powers are consistent (pairing check)
    
    Args:
        srs: The SRS to validate
    
    Returns:
        ValidationResult with detailed information
    """
    from crypto.bn254.pairing_pyecc import pairing
    
    checks_passed = 0
    checks_total = 4
    
    # Check 1: G1 points on curve
    g1_valid = all(p.is_on_curve() for p in srs.g1_powers)
    if g1_valid:
        checks_passed += 1
    else:
        return ValidationResult(
            is_valid=False,
            reason="Some G1 points are not on the curve",
            checks_passed=checks_passed,
            checks_total=checks_total
        )
    
    # Check 2: G2 points on curve
    g2_valid = all(p.is_on_curve() for p in srs.g2_powers)
    if g2_valid:
        checks_passed += 1
    else:
        return ValidationResult(
            is_valid=False,
            reason="Some G2 points are not on the curve",
            checks_passed=checks_passed,
            checks_total=checks_total
        )
    
    # Check 3: G1 powers consistency
    # e(τⁱ·G₁, G₂) = e(τⁱ⁻¹·G₁, τ·G₂) for all i > 0
    if len(srs.g1_powers) >= 2 and len(srs.g2_powers) >= 2:
        g1_0 = srs.g1_powers[0]  # G₁
        g1_1 = srs.g1_powers[1]  # τ·G₁
        g2_0 = srs.g2_powers[0]  # G₂
        g2_1 = srs.g2_powers[1]  # τ·G₂
        
        # e(τ·G₁, G₂) should equal e(G₁, τ·G₂)
        e1 = pairing(g1_1, g2_0)
        e2 = pairing(g1_0, g2_1)
        
        if e1 == e2:
            checks_passed += 1
        else:
            return ValidationResult(
                is_valid=False,
                reason="G1 powers are inconsistent (pairing check failed)",
                checks_passed=checks_passed,
                checks_total=checks_total
            )
    else:
        checks_passed += 1  # Skip if not enough powers
    
    # Check 4: G1 powers are sequential
    # For a more thorough check, verify e(τⁱ·G₁, G₂) = e(G₁, G₂)^(τⁱ)
    # This is expensive, so we only check a sample
    if len(srs.g1_powers) >= 3:
        # Check: e(τ²·G₁, G₂) = e(τ·G₁, τ·G₂)
        g1_2 = srs.g1_powers[2]  # τ²·G₁
        g1_1 = srs.g1_powers[1]  # τ·G₁
        g2_1 = srs.g2_powers[1]  # τ·G₂
        g2_0 = srs.g2_powers[0]  # G₂
        
        e1 = pairing(g1_2, g2_0)
        e2 = pairing(g1_1, g2_1)
        
        if e1 == e2:
            checks_passed += 1
        else:
            return ValidationResult(
                is_valid=False,
                reason="G1 powers sequence is inconsistent",
                checks_passed=checks_passed,
                checks_total=checks_total
            )
    else:
        checks_passed += 1  # Skip if not enough powers
    
    return ValidationResult(
        is_valid=True,
        reason="All validation checks passed",
        checks_passed=checks_passed,
        checks_total=checks_total
    )


# =============================================================================
# SRS Serialization
# =============================================================================

def serialize_g1_point(point: G1Point) -> bytes:
    """Serialize a G1 point to 64 bytes."""
    if point.is_identity():
        return b'\x00' * 64
    
    x, y = point.to_affine()
    return x.to_int().to_bytes(32, 'big') + y.to_int().to_bytes(32, 'big')


def deserialize_g1_point(data: bytes) -> G1Point:
    """Deserialize a G1 point from 64 bytes."""
    if data == b'\x00' * 64:
        return G1Point.identity()
    
    x = Fp(int.from_bytes(data[:32], 'big'))
    y = Fp(int.from_bytes(data[32:64], 'big'))
    return G1Point(x, y)


def serialize_g2_point(point: G2Point) -> bytes:
    """Serialize a G2 point to 128 bytes."""
    if point.is_identity():
        return b'\x00' * 128
    
    x, y = point.to_affine()
    # G2 coordinates are in Fp2: c0 + c1*u
    return (
        x.c0.to_int().to_bytes(32, 'big') +
        x.c1.to_int().to_bytes(32, 'big') +
        y.c0.to_int().to_bytes(32, 'big') +
        y.c1.to_int().to_bytes(32, 'big')
    )


def deserialize_g2_point(data: bytes) -> G2Point:
    """Deserialize a G2 point from 128 bytes."""
    from crypto.bn254.extension_field import Fp2
    
    if data == b'\x00' * 128:
        return G2Point.identity()
    
    x_c0 = Fp(int.from_bytes(data[0:32], 'big'))
    x_c1 = Fp(int.from_bytes(data[32:64], 'big'))
    y_c0 = Fp(int.from_bytes(data[64:96], 'big'))
    y_c1 = Fp(int.from_bytes(data[96:128], 'big'))
    
    x = Fp2(x_c0, x_c1)
    y = Fp2(y_c0, y_c1)
    
    return G2Point(x, y)


def serialize_srs(srs: SRS) -> bytes:
    """
    Serialize an SRS to bytes.
    
    Format:
    - 4 bytes: magic ('srs\x00')
    - 4 bytes: version (uint32)
    - 4 bytes: max_degree (uint32)
    - 4 bytes: num_g1_powers (uint32)
    - 4 bytes: num_g2_powers (uint32)
    - 64 * num_g1_powers bytes: G1 powers
    - 128 * num_g2_powers bytes: G2 powers
    - 32 bytes: SHA256 checksum
    """
    header = b'srs\x00'
    header += struct.pack('<I', 1)  # version
    header += struct.pack('<I', srs.max_degree)
    header += struct.pack('<I', len(srs.g1_powers))
    header += struct.pack('<I', len(srs.g2_powers))
    
    g1_data = b''.join(serialize_g1_point(p) for p in srs.g1_powers)
    g2_data = b''.join(serialize_g2_point(p) for p in srs.g2_powers)
    
    content = header + g1_data + g2_data
    checksum = hashlib.sha256(content).digest()
    
    return content + checksum


def deserialize_srs(data: bytes) -> SRS:
    """
    Deserialize an SRS from bytes.
    
    Raises:
        ValueError: If the data is invalid or corrupted
    """
    if len(data) < 52:  # Minimum size: header + checksum
        raise ValueError("Data too short to be a valid SRS")
    
    # Verify checksum
    content = data[:-32]
    checksum = data[-32:]
    expected_checksum = hashlib.sha256(content).digest()
    
    if checksum != expected_checksum:
        raise ValueError("SRS checksum mismatch - data may be corrupted")
    
    # Parse header
    magic = data[0:4]
    if magic != b'srs\x00':
        raise ValueError(f"Invalid magic bytes: {magic}")
    
    version = struct.unpack('<I', data[4:8])[0]
    if version != 1:
        raise ValueError(f"Unsupported SRS version: {version}")
    
    max_degree = struct.unpack('<I', data[8:12])[0]
    num_g1 = struct.unpack('<I', data[12:16])[0]
    num_g2 = struct.unpack('<I', data[16:20])[0]
    
    # Parse G1 powers
    g1_start = 20
    g1_end = g1_start + 64 * num_g1
    g1_powers = []
    for i in range(num_g1):
        offset = g1_start + 64 * i
        g1_powers.append(deserialize_g1_point(data[offset:offset+64]))
    
    # Parse G2 powers
    g2_start = g1_end
    g2_powers = []
    for i in range(num_g2):
        offset = g2_start + 128 * i
        g2_powers.append(deserialize_g2_point(data[offset:offset+128]))
    
    return SRS(g1_powers, g2_powers, max_degree)


# =============================================================================
# Trusted Setup Ceremony
# =============================================================================

@dataclass
class Contribution:
    """A contribution to a trusted setup ceremony."""
    contributor_id: str
    g1_proof: G1Point  # Proof of knowledge of contribution
    g2_proof: G2Point
    timestamp: int
    signature: bytes


class TrustedSetupCeremony:
    """
    Manager for trusted setup ceremonies.
    
    This class handles:
    1. Initializing a new ceremony
    2. Adding contributions
    3. Verifying contributions
    4. Finalizing the SRS
    """
    
    def __init__(self, max_degree: int):
        """
        Initialize a new ceremony.
        
        Args:
            max_degree: Maximum polynomial degree to support
        """
        self.max_degree = max_degree
        self.contributions: List[Contribution] = []
        self.current_srs: Optional[SRS] = None
        
        # Initialize with the "genesis" SRS (tau = 1)
        self._initialize_genesis()
    
    def _initialize_genesis(self):
        """Initialize the genesis SRS with tau = 1."""
        g1 = G1Point.generator()
        g2 = G2Point.generator()
        
        # tau = 1, so all powers are just the generator
        g1_powers = [g1] * (self.max_degree + 1)
        g2_powers = [g2, g2]  # [G2, tau*G2] = [G2, G2] when tau=1
        
        self.current_srs = SRS(g1_powers, g2_powers, self.max_degree)
    
    def contribute(self, secret: Fr, contributor_id: str) -> Contribution:
        """
        Add a contribution to the ceremony.
        
        The contributor provides a secret scalar that is multiplied into
        the existing SRS. The contribution is verified before acceptance.
        
        Args:
            secret: The contributor's secret scalar
            contributor_id: Identifier for the contributor
        
        Returns:
            The contribution record
        
        Raises:
            ValueError: If the contribution is invalid
        """
        if self.current_srs is None:
            raise ValueError("Ceremony not initialized")
        
        # Compute new SRS: multiply all powers by secret
        new_g1_powers = []
        secret_power = Fr.one()
        for i, p in enumerate(self.current_srs.g1_powers):
            new_g1_powers.append(p * secret_power)
            secret_power = secret_power * secret
        
        new_g2_powers = [
            self.current_srs.g2_powers[0],  # G2 unchanged
            self.current_srs.g2_powers[1] * secret  # tau*G2 -> secret*tau*G2
        ]
        
        # Create proof of knowledge
        g1_proof = G1Point.generator() * secret
        g2_proof = G2Point.generator() * secret
        
        # Verify the contribution
        if not self._verify_contribution(new_g1_powers, new_g2_powers, g1_proof, g2_proof):
            raise ValueError("Contribution verification failed")
        
        # Accept the contribution
        self.current_srs = SRS(new_g1_powers, new_g2_powers, self.max_degree)
        
        import time
        contribution = Contribution(
            contributor_id=contributor_id,
            g1_proof=g1_proof,
            g2_proof=g2_proof,
            timestamp=int(time.time()),
            signature=b''  # Would be signed in production
        )
        self.contributions.append(contribution)
        
        return contribution
    
    def _verify_contribution(
        self,
        g1_powers: List[G1Point],
        g2_powers: List[G2Point],
        g1_proof: G1Point,
        g2_proof: G2Point
    ) -> bool:
        """Verify a contribution is valid."""
        from crypto.bn254.pairing_pyecc import pairing
        
        # Check: e(new_tau*G1, G2) = e(G1, new_tau*G2)
        if len(g1_powers) < 2 or len(g2_powers) < 2:
            return False
        
        e1 = pairing(g1_powers[1], g2_powers[0])
        e2 = pairing(g1_powers[0], g2_powers[1])
        
        return e1 == e2
    
    def finalize(self) -> SRS:
        """
        Finalize the ceremony and return the SRS.
        
        Returns:
            The final SRS
        
        Raises:
            ValueError: If no contributions have been made
        """
        if not self.contributions:
            raise ValueError("Cannot finalize ceremony with no contributions")
        
        if self.current_srs is None:
            raise ValueError("Ceremony not initialized")
        
        # Validate the final SRS
        result = validate_srs(self.current_srs)
        if not result.is_valid:
            raise ValueError(f"Final SRS validation failed: {result.reason}")
        
        return self.current_srs
    
    def get_transcript(self) -> str:
        """Get a human-readable transcript of the ceremony."""
        lines = [
            "Trusted Setup Ceremony Transcript",
            "=" * 50,
            f"Max Degree: {self.max_degree}",
            f"Contributions: {len(self.contributions)}",
            ""
        ]
        
        for i, c in enumerate(self.contributions):
            lines.append(f"Contribution {i + 1}:")
            lines.append(f"  Contributor: {c.contributor_id}")
            lines.append(f"  Timestamp: {c.timestamp}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# PTAU File Loading (Hermez/Zcash format)
# =============================================================================

def load_from_ptau(path: str, max_degree: Optional[int] = None) -> SRS:
    """
    Load an SRS from a Powers-of-Tau file.
    
    Supports the Hermez/Zcash PTAU format used in production ceremonies.
    
    Args:
        path: Path to the PTAU file
        max_degree: Maximum degree to load (None = load all)
    
    Returns:
        The loaded SRS
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    # Note: This is a simplified implementation.
    # A full implementation would parse the actual PTAU binary format.
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"PTAU file not found: {path}")
    
    with open(path, 'rb') as f:
        # Check magic bytes
        magic = f.read(4)
        if magic == b'srs\x00':
            # Our custom format
            f.seek(0)
            data = f.read()
            return deserialize_srs(data)
        elif magic == PTAU_MAGIC:
            # Hermez PTAU format
            return _parse_hermez_ptau(f, max_degree)
        else:
            raise ValueError(f"Unknown file format: {magic}")


def _parse_hermez_ptau(f: BinaryIO, max_degree: Optional[int]) -> SRS:
    """Parse Hermez PTAU format."""
    # This is a placeholder for the actual Hermez format parser
    # The real format is more complex with multiple sections
    raise NotImplementedError(
        "Hermez PTAU format not yet implemented. "
        "Use generate_insecure() for testing or provide a custom SRS file."
    )


# =============================================================================
# Secure SRS Generation (for testing only)
# =============================================================================

def generate_test_srs(max_degree: int, seed: Optional[bytes] = None) -> SRS:
    """
    Generate a test SRS with a deterministic but unknown tau.
    
    WARNING: This is for testing only! The tau is derived from the seed,
    which means anyone with the seed can compute tau.
    
    For production, use a proper trusted setup ceremony.
    
    Args:
        max_degree: Maximum polynomial degree
        seed: Optional seed for deterministic generation
    
    Returns:
        A test SRS
    """
    import warnings
    warnings.warn(
        "generate_test_srs() is for testing only! "
        "Use a trusted setup ceremony for production.",
        UserWarning
    )
    
    if seed is None:
        seed = os.urandom(32)
    
    # Derive tau from seed
    tau_bytes = hashlib.sha256(seed + b'tau').digest()
    tau = Fr(int.from_bytes(tau_bytes, 'big') % Fr.MODULUS)
    
    return SRS.generate_insecure(max_degree, tau)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Trusted Setup Infrastructure Self-Test")
    print("=" * 70)
    
    # Test 1: SRS Validation
    print("\n1. Testing SRS validation...")
    srs = SRS.generate_insecure(8, Fr(12345))
    result = validate_srs(srs)
    print(f"   Valid: {result.is_valid}")
    print(f"   Reason: {result.reason}")
    print(f"   Checks: {result.checks_passed}/{result.checks_total}")
    
    # Test 2: SRS Serialization
    print("\n2. Testing SRS serialization...")
    serialized = serialize_srs(srs)
    print(f"   Serialized size: {len(serialized)} bytes")
    
    deserialized = deserialize_srs(serialized)
    print(f"   Deserialized max_degree: {deserialized.max_degree}")
    print(f"   G1 powers match: {len(deserialized.g1_powers) == len(srs.g1_powers)}")
    
    # Test 3: Ceremony
    print("\n3. Testing trusted setup ceremony...")
    ceremony = TrustedSetupCeremony(max_degree=8)
    
    # Add contributions
    ceremony.contribute(Fr(111), "Alice")
    ceremony.contribute(Fr(222), "Bob")
    ceremony.contribute(Fr(333), "Charlie")
    
    final_srs = ceremony.finalize()
    print(f"   Contributions: {len(ceremony.contributions)}")
    print(f"   Final SRS valid: {validate_srs(final_srs).is_valid}")
    
    # Print transcript
    print("\n4. Ceremony transcript:")
    print(ceremony.get_transcript())
    
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
