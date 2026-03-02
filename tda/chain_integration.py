"""
On-Chain Integration for TDA Model Fingerprinting.

This module provides Python bindings for interacting with the ModelRegistry
smart contract on Ethereum-compatible blockchains.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import time


class ChainType(Enum):
    """Supported blockchain networks."""
    ETHEREUM_MAINNET = "ethereum"
    ETHEREUM_SEPOLIA = "sepolia"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    BASE = "base"
    LOCAL = "local"


@dataclass
class ChainConfig:
    """Configuration for blockchain connection."""
    chain_type: ChainType
    rpc_url: str
    contract_address: str
    private_key: Optional[str] = None  # For signing transactions
    
    @classmethod
    def local(cls, contract_address: str = "0x0000000000000000000000000000000000000000"):
        return cls(
            chain_type=ChainType.LOCAL,
            rpc_url="http://localhost:8545",
            contract_address=contract_address
        )


@dataclass
class RegistrationResult:
    """Result of a model registration."""
    success: bool
    transaction_hash: Optional[str]
    fingerprint_hash: str
    block_number: Optional[int]
    gas_used: Optional[int]
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of a model verification."""
    is_registered: bool
    owner: Optional[str]
    registration_time: Optional[int]
    model_name: Optional[str]
    model_version: Optional[str]


class FingerprintHasher:
    """
    Converts TDA fingerprints to on-chain compatible hashes.
    """
    
    @staticmethod
    def hash_fingerprint(fingerprint_bytes: bytes) -> bytes:
        """
        Hash a fingerprint to a 32-byte value suitable for on-chain storage.
        
        Args:
            fingerprint_bytes: Raw fingerprint bytes
            
        Returns:
            32-byte hash
        """
        return hashlib.sha256(fingerprint_bytes).digest()
    
    @staticmethod
    def hash_to_hex(hash_bytes: bytes) -> str:
        """Convert hash bytes to hex string with 0x prefix."""
        return "0x" + hash_bytes.hex()
    
    @staticmethod
    def hex_to_bytes32(hex_str: str) -> bytes:
        """Convert hex string to bytes32."""
        if hex_str.startswith("0x"):
            hex_str = hex_str[2:]
        return bytes.fromhex(hex_str.zfill(64))


class ModelRegistryClient:
    """
    Client for interacting with the ModelRegistry smart contract.
    
    This is a simulation/mock implementation. In production, this would
    use web3.py to interact with the actual blockchain.
    """
    
    def __init__(self, config: ChainConfig):
        self.config = config
        self.hasher = FingerprintHasher()
        
        # Mock storage for local testing
        self._mock_registry: Dict[str, Dict] = {}
        self._mock_block_number = 1000000
        self._mock_gas_price = 20_000_000_000  # 20 gwei
    
    def register_model(
        self,
        fingerprint: bytes,
        model_name: str,
        model_version: str,
        metadata: bytes = b""
    ) -> RegistrationResult:
        """
        Register a model fingerprint on-chain.
        
        Args:
            fingerprint: Raw fingerprint bytes
            model_name: Human-readable model name
            model_version: Version string
            metadata: Additional metadata
            
        Returns:
            RegistrationResult with transaction details
        """
        # Hash the fingerprint
        fingerprint_hash = self.hasher.hash_fingerprint(fingerprint)
        hash_hex = self.hasher.hash_to_hex(fingerprint_hash)
        
        # Check if already registered
        if hash_hex in self._mock_registry:
            return RegistrationResult(
                success=False,
                transaction_hash=None,
                fingerprint_hash=hash_hex,
                block_number=None,
                gas_used=None,
                error="Model already registered"
            )
        
        # Simulate transaction
        self._mock_block_number += 1
        tx_hash = hashlib.sha256(
            fingerprint_hash + str(time.time()).encode()
        ).hexdigest()
        
        # Store in mock registry
        self._mock_registry[hash_hex] = {
            "owner": "0x" + "1" * 40,  # Mock owner address
            "registration_time": int(time.time()),
            "model_name": model_name,
            "model_version": model_version,
            "metadata": metadata.hex(),
            "is_active": True
        }
        
        return RegistrationResult(
            success=True,
            transaction_hash="0x" + tx_hash,
            fingerprint_hash=hash_hex,
            block_number=self._mock_block_number,
            gas_used=150000  # Estimated gas
        )
    
    def verify_model(self, fingerprint: bytes) -> VerificationResult:
        """
        Verify if a model fingerprint is registered.
        
        Args:
            fingerprint: Raw fingerprint bytes
            
        Returns:
            VerificationResult with registration details
        """
        fingerprint_hash = self.hasher.hash_fingerprint(fingerprint)
        hash_hex = self.hasher.hash_to_hex(fingerprint_hash)
        
        if hash_hex not in self._mock_registry:
            return VerificationResult(
                is_registered=False,
                owner=None,
                registration_time=None,
                model_name=None,
                model_version=None
            )
        
        entry = self._mock_registry[hash_hex]
        if not entry["is_active"]:
            return VerificationResult(
                is_registered=False,
                owner=entry["owner"],
                registration_time=entry["registration_time"],
                model_name=entry["model_name"],
                model_version=entry["model_version"]
            )
        
        return VerificationResult(
            is_registered=True,
            owner=entry["owner"],
            registration_time=entry["registration_time"],
            model_name=entry["model_name"],
            model_version=entry["model_version"]
        )
    
    def verify_by_hash(self, fingerprint_hash: str) -> VerificationResult:
        """
        Verify using a pre-computed hash.
        
        Args:
            fingerprint_hash: Hex string of the fingerprint hash
            
        Returns:
            VerificationResult
        """
        if not fingerprint_hash.startswith("0x"):
            fingerprint_hash = "0x" + fingerprint_hash
        
        if fingerprint_hash not in self._mock_registry:
            return VerificationResult(
                is_registered=False,
                owner=None,
                registration_time=None,
                model_name=None,
                model_version=None
            )
        
        entry = self._mock_registry[fingerprint_hash]
        return VerificationResult(
            is_registered=entry["is_active"],
            owner=entry["owner"],
            registration_time=entry["registration_time"],
            model_name=entry["model_name"],
            model_version=entry["model_version"]
        )
    
    def get_registration_count(self) -> int:
        """Get total number of registered models."""
        return sum(1 for e in self._mock_registry.values() if e["is_active"])
    
    def estimate_registration_gas(self) -> Dict[str, int]:
        """
        Estimate gas costs for registration.
        
        Returns:
            Dictionary with gas estimates
        """
        return {
            "gas_limit": 200000,
            "gas_price_gwei": self._mock_gas_price // 10**9,
            "estimated_cost_wei": 200000 * self._mock_gas_price,
            "estimated_cost_eth": (200000 * self._mock_gas_price) / 10**18
        }


class OnChainFingerprintManager:
    """
    High-level manager for on-chain fingerprint operations.
    
    Combines TDA fingerprinting with blockchain registration.
    """
    
    def __init__(self, chain_config: ChainConfig):
        self.client = ModelRegistryClient(chain_config)
        self.hasher = FingerprintHasher()
    
    def register_from_fingerprint(
        self,
        fingerprint_data: Dict[str, Any],
        model_name: str,
        model_version: str = "1.0.0"
    ) -> RegistrationResult:
        """
        Register a model using its TDA fingerprint data.
        
        Args:
            fingerprint_data: Dictionary containing fingerprint info
            model_name: Name for the model
            model_version: Version string
            
        Returns:
            RegistrationResult
        """
        # Serialize fingerprint to bytes
        fingerprint_bytes = json.dumps(fingerprint_data, sort_keys=True).encode()
        
        # Create metadata
        metadata = json.dumps({
            "n_features": fingerprint_data.get("n_features", 0),
            "fingerprint_version": "1.0",
            "algorithm": "TDA-PersistentHomology"
        }).encode()
        
        return self.client.register_model(
            fingerprint=fingerprint_bytes,
            model_name=model_name,
            model_version=model_version,
            metadata=metadata
        )
    
    def verify_from_fingerprint(
        self,
        fingerprint_data: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify a model using its TDA fingerprint data.
        
        Args:
            fingerprint_data: Dictionary containing fingerprint info
            
        Returns:
            VerificationResult
        """
        fingerprint_bytes = json.dumps(fingerprint_data, sort_keys=True).encode()
        return self.client.verify_model(fingerprint_bytes)
    
    def compute_fingerprint_hash(self, fingerprint_data: Dict[str, Any]) -> str:
        """
        Compute the on-chain hash for a fingerprint.
        
        Args:
            fingerprint_data: Dictionary containing fingerprint info
            
        Returns:
            Hex string of the hash
        """
        fingerprint_bytes = json.dumps(fingerprint_data, sort_keys=True).encode()
        hash_bytes = self.hasher.hash_fingerprint(fingerprint_bytes)
        return self.hasher.hash_to_hex(hash_bytes)


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("On-Chain Integration Tests")
    print("=" * 70)
    
    # Create local config
    config = ChainConfig.local()
    manager = OnChainFingerprintManager(config)
    
    # Test 1: Register a model
    print("\n1. Registering a model")
    fingerprint_data = {
        "hash": "abc123",
        "features": [(0, 0.0, 1.5), (0, 0.1, 0.8)],
        "n_features": 2
    }
    
    result = manager.register_from_fingerprint(
        fingerprint_data,
        model_name="TestModel",
        model_version="1.0.0"
    )
    
    print(f"   Success: {result.success}")
    print(f"   Transaction hash: {result.transaction_hash}")
    print(f"   Fingerprint hash: {result.fingerprint_hash}")
    print(f"   Block number: {result.block_number}")
    print(f"   Gas used: {result.gas_used}")
    
    # Test 2: Verify the model
    print("\n2. Verifying the registered model")
    verification = manager.verify_from_fingerprint(fingerprint_data)
    
    print(f"   Is registered: {verification.is_registered}")
    print(f"   Owner: {verification.owner}")
    print(f"   Model name: {verification.model_name}")
    print(f"   Registration time: {verification.registration_time}")
    
    # Test 3: Try to register same model again
    print("\n3. Attempting duplicate registration")
    result2 = manager.register_from_fingerprint(
        fingerprint_data,
        model_name="TestModel",
        model_version="1.0.1"
    )
    
    print(f"   Success: {result2.success}")
    print(f"   Error: {result2.error}")
    
    # Test 4: Verify unregistered model
    print("\n4. Verifying unregistered model")
    other_fingerprint = {"hash": "xyz789", "features": [], "n_features": 0}
    verification2 = manager.verify_from_fingerprint(other_fingerprint)
    
    print(f"   Is registered: {verification2.is_registered}")
    
    # Test 5: Gas estimation
    print("\n5. Gas estimation")
    gas_estimate = manager.client.estimate_registration_gas()
    
    print(f"   Gas limit: {gas_estimate['gas_limit']}")
    print(f"   Gas price: {gas_estimate['gas_price_gwei']} gwei")
    print(f"   Estimated cost: {gas_estimate['estimated_cost_eth']:.6f} ETH")
    
    # Test 6: Registration count
    print("\n6. Registry statistics")
    count = manager.client.get_registration_count()
    print(f"   Total registered models: {count}")
    
    # Test 7: Compute hash without registration
    print("\n7. Computing fingerprint hash")
    hash_hex = manager.compute_fingerprint_hash(fingerprint_data)
    print(f"   Fingerprint hash: {hash_hex}")
    
    print("\n" + "=" * 70)
    print("On-Chain Integration Tests Complete!")
    print("=" * 70)
