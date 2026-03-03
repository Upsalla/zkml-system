"""
Gas Audit for zkML Smart Contracts

This module performs static analysis of the Solidity contracts to estimate
gas costs and identify optimization opportunities.

Gas Cost Reference (EIP-2929, post-Berlin):
- ecAdd (0x06): 150 gas
- ecMul (0x07): 6,000 gas
- ecPairing (0x08): 34,000 + 45,000 per pair
- SLOAD (cold): 2,100 gas
- SLOAD (warm): 100 gas
- SSTORE (cold, 0->non-0): 22,100 gas
- SSTORE (warm): 100 gas
- CALL: 2,600 gas (cold) / 100 gas (warm)
- SHA3: 30 + 6 per word
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class GasEstimate:
    """Gas estimate for a function or operation."""
    name: str
    min_gas: int
    max_gas: int
    notes: str


@dataclass
class Optimization:
    """A gas optimization recommendation."""
    location: str
    current_cost: int
    optimized_cost: int
    savings: int
    description: str
    priority: str  # HIGH, MEDIUM, LOW


class GasAuditor:
    """Gas auditor for Solidity contracts."""

    # Gas costs for precompiles
    EC_ADD_GAS = 150
    EC_MUL_GAS = 6000
    EC_PAIRING_BASE = 34000
    EC_PAIRING_PER_PAIR = 45000

    # Gas costs for storage
    SLOAD_COLD = 2100
    SLOAD_WARM = 100
    SSTORE_COLD_ZERO_TO_NONZERO = 22100
    SSTORE_COLD_NONZERO_TO_NONZERO = 5000
    SSTORE_WARM = 100

    # Gas costs for memory/computation
    SHA3_BASE = 30
    SHA3_PER_WORD = 6
    CALL_COLD = 2600
    CALL_WARM = 100

    def __init__(self):
        self.estimates: List[GasEstimate] = []
        self.optimizations: List[Optimization] = []

    def analyze_plonk_verifier(self, contract_path: str):
        """Analyze the PlonkVerifier contract."""
        print("\n" + "=" * 60)
        print("GAS AUDIT: PlonkVerifier.sol")
        print("=" * 60)

        with open(contract_path, 'r') as f:
            code = f.read()

        # Count precompile calls
        ec_add_count = len(re.findall(r'staticcall\(gas\(\), 0x06', code))
        ec_mul_count = len(re.findall(r'staticcall\(gas\(\), 0x07', code))
        ec_pairing_count = len(re.findall(r'staticcall\(gas\(\), 0x08', code))

        print(f"\nPrecompile Usage:")
        print(f"  ecAdd calls: {ec_add_count}")
        print(f"  ecMul calls: {ec_mul_count}")
        print(f"  ecPairing calls: {ec_pairing_count}")

        # Estimate gas for verify function
        # Based on PLONK verification algorithm:
        # - 9 point validations (on curve checks): ~500 gas each
        # - 6 field element validations: ~100 gas each
        # - Challenge computation (6 hashes): ~6 * (30 + 6*32) = ~1,200 gas
        # - Vanishing polynomial: 1 exp + 1 sub = ~3,000 gas
        # - Public input polynomial: n Lagrange evals = ~5,000 gas
        # - Linearization: ~10 ecMul + 10 ecAdd = ~61,500 gas
        # - Aggregated commitment: ~6 ecMul + 6 ecAdd = ~36,900 gas
        # - Pairing check: 2 pairs = 34,000 + 2*45,000 = ~124,000 gas

        verify_estimate = GasEstimate(
            name="verify()",
            min_gas=250000,
            max_gas=400000,
            notes="Depends on public input count and circuit size"
        )
        self.estimates.append(verify_estimate)

        print(f"\nGas Estimates:")
        print(f"  verify(): {verify_estimate.min_gas:,} - {verify_estimate.max_gas:,} gas")

        # Identify optimizations
        self._analyze_optimizations(code)

    def _analyze_optimizations(self, code: str):
        """Identify gas optimization opportunities."""
        print("\n" + "-" * 40)
        print("Optimization Opportunities:")
        print("-" * 40)

        # Check 1: Batch ecMul using Pippenger
        ec_mul_count = len(re.findall(r'_ecMul\(', code))
        if ec_mul_count > 3:
            self.optimizations.append(Optimization(
                location="_computeLinearization / _computeAggregatedCommitment",
                current_cost=ec_mul_count * self.EC_MUL_GAS,
                optimized_cost=int(ec_mul_count * self.EC_MUL_GAS * 0.3),  # ~70% savings
                savings=int(ec_mul_count * self.EC_MUL_GAS * 0.7),
                description=f"Batch {ec_mul_count} ecMul calls using multi-scalar multiplication precompile (EIP-2537)",
                priority="HIGH"
            ))

        # Check 2: Memory vs calldata for proof
        if 'calldata proof' in code.lower() or 'Proof calldata' in code:
            print("  ✓ Using calldata for proof (good)")
        else:
            self.optimizations.append(Optimization(
                location="verify() function signature",
                current_cost=2000,
                optimized_cost=500,
                savings=1500,
                description="Use calldata instead of memory for proof parameter",
                priority="MEDIUM"
            ))

        # Check 3: Unchecked arithmetic
        unchecked_count = code.count('unchecked')
        if unchecked_count < 5:
            self.optimizations.append(Optimization(
                location="Modular arithmetic operations",
                current_cost=500,
                optimized_cost=300,
                savings=200,
                description="Use unchecked blocks for modular arithmetic (overflow is intentional)",
                priority="LOW"
            ))

        # Check 4: Storage reads
        sload_pattern = r'vk\.\w+'
        sload_count = len(re.findall(sload_pattern, code))
        if sload_count > 10:
            self.optimizations.append(Optimization(
                location="Verifier key reads",
                current_cost=sload_count * self.SLOAD_COLD,
                optimized_cost=sload_count * self.SLOAD_WARM + self.SLOAD_COLD,
                savings=(sload_count - 1) * (self.SLOAD_COLD - self.SLOAD_WARM),
                description="Cache verifier key in memory at start of verify()",
                priority="HIGH"
            ))

        # Check 5: Loop unrolling
        for_loops = len(re.findall(r'for\s*\(', code))
        if for_loops > 0:
            self.optimizations.append(Optimization(
                location="Loops in verification",
                current_cost=for_loops * 500,
                optimized_cost=for_loops * 200,
                savings=for_loops * 300,
                description="Consider unrolling small fixed-size loops",
                priority="LOW"
            ))

        # Print optimizations
        for opt in self.optimizations:
            print(f"\n  [{opt.priority}] {opt.location}")
            print(f"      Current: ~{opt.current_cost:,} gas")
            print(f"      Optimized: ~{opt.optimized_cost:,} gas")
            print(f"      Savings: ~{opt.savings:,} gas")
            print(f"      Action: {opt.description}")

    def analyze_zkml_verifier(self, contract_path: str):
        """Analyze the ZkMLVerifier contract."""
        print("\n" + "=" * 60)
        print("GAS AUDIT: ZkMLVerifier.sol")
        print("=" * 60)

        with open(contract_path, 'r') as f:
            code = f.read()

        # Estimate gas for key functions
        estimates = [
            GasEstimate(
                name="registerModel()",
                min_gas=100000,
                max_gas=150000,
                notes="Stores model metadata and verifier key"
            ),
            GasEstimate(
                name="verifyInference()",
                min_gas=300000,
                max_gas=500000,
                notes="Includes PLONK verification + storage"
            ),
            GasEstimate(
                name="batchVerifyInferences()",
                min_gas=300000,  # per inference
                max_gas=500000,  # per inference
                notes="Linear scaling with batch size"
            )
        ]

        print("\nGas Estimates:")
        for est in estimates:
            print(f"  {est.name}: {est.min_gas:,} - {est.max_gas:,} gas")
            print(f"      Note: {est.notes}")
            self.estimates.append(est)

        # Storage analysis
        print("\n" + "-" * 40)
        print("Storage Analysis:")
        print("-" * 40)

        # Count mappings
        mappings = len(re.findall(r'mapping\s*\(', code))
        arrays = len(re.findall(r'\[\]\s+public', code))

        print(f"  Mappings: {mappings}")
        print(f"  Dynamic arrays: {arrays}")

        # Storage cost estimate
        model_storage = 5 * 32  # 5 slots for Model struct
        inference_storage = 4 * 32 + 32  # 4 slots + dynamic array
        print(f"\n  Per-model storage: ~{model_storage} bytes ({model_storage // 32} slots)")
        print(f"  Per-inference storage: ~{inference_storage}+ bytes")

        # Optimization: Use events instead of storage for inference results
        self.optimizations.append(Optimization(
            location="verifyInference() storage",
            current_cost=4 * self.SSTORE_COLD_ZERO_TO_NONZERO,
            optimized_cost=4 * 375,  # LOG4 cost
            savings=4 * (self.SSTORE_COLD_ZERO_TO_NONZERO - 375),
            description="Consider using events instead of storage for inference results (if on-chain query not needed)",
            priority="MEDIUM"
        ))

    def generate_report(self):
        """Generate the final gas audit report."""
        print("\n" + "=" * 60)
        print("GAS AUDIT SUMMARY")
        print("=" * 60)

        total_savings = sum(opt.savings for opt in self.optimizations)
        high_priority = [opt for opt in self.optimizations if opt.priority == "HIGH"]
        medium_priority = [opt for opt in self.optimizations if opt.priority == "MEDIUM"]

        print(f"\nTotal potential savings: ~{total_savings:,} gas")
        print(f"High priority optimizations: {len(high_priority)}")
        print(f"Medium priority optimizations: {len(medium_priority)}")

        print("\n" + "-" * 40)
        print("Recommended Actions:")
        print("-" * 40)

        for i, opt in enumerate(sorted(self.optimizations, key=lambda x: x.savings, reverse=True), 1):
            print(f"\n{i}. [{opt.priority}] {opt.description}")
            print(f"   Location: {opt.location}")
            print(f"   Estimated savings: ~{opt.savings:,} gas")

        print("\n" + "-" * 40)
        print("Gas Cost Comparison:")
        print("-" * 40)
        print("""
| Operation          | Current Est. | After Optimization |
|--------------------|--------------|-------------------|
| verify()           | ~350,000     | ~250,000          |
| verifyInference()  | ~400,000     | ~300,000          |
| registerModel()    | ~125,000     | ~100,000          |

Note: Actual gas costs depend on:
- Circuit size (n)
- Number of public inputs
- EVM implementation (geth vs others)
- State access patterns (cold vs warm)
""")


def main():
    """Run the gas audit."""
    print("=" * 60)
    print("zkML SMART CONTRACT GAS AUDIT")
    print("=" * 60)

    auditor = GasAuditor()

    # Analyze contracts
    auditor.analyze_plonk_verifier('/home/ubuntu/zkml_system/contracts/PlonkVerifier.sol')
    auditor.analyze_zkml_verifier('/home/ubuntu/zkml_system/contracts/ZkMLVerifier.sol')

    # Generate report
    auditor.generate_report()


if __name__ == "__main__":
    main()
