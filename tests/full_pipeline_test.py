"""
Full Pipeline Test for zkML System v2.0.0

This script tests all major components of the zkML system to ensure
they work correctly after integration.

Tests:
1. Core: Field arithmetic, R1CS, Witness
2. Crypto: BN254 field and curve operations
3. Network: Layer building, CNN support
4. Optimizations: CSWC, HWWB, Tropical
5. PLONK: Polynomial, KZG, Circuit Compiler
6. Proof: Prover and Verifier
"""

import sys
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None


class PipelineTester:
    """Comprehensive tester for the zkML pipeline."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def run_test(self, name: str, test_func):
        """Run a single test and record the result."""
        print(f"  Testing {name}...", end=" ", flush=True)
        start = time.time()
        
        try:
            test_func()
            duration = (time.time() - start) * 1000
            self.results.append(TestResult(name, True, duration))
            print(f"PASS ({duration:.1f}ms)")
            return True
        except Exception as e:
            duration = (time.time() - start) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.results.append(TestResult(name, False, duration, error_msg))
            print(f"FAIL ({error_msg})")
            return False
    
    def test_core_field(self):
        """Test core field arithmetic."""
        from core.field import FieldElement, FieldConfig
        
        config = FieldConfig(prime=101)
        a = FieldElement(50, config)
        b = FieldElement(60, config)
        
        # Addition
        c = a + b
        assert c.value == 9, f"Expected 9, got {c.value}"
        
        # Multiplication
        d = a * b
        assert d.value == (50 * 60) % 101, f"Multiplication failed"
        
        # Inverse
        inv_a = a.inverse()
        assert (a * inv_a).value == 1, "Inverse failed"
    
    def test_core_r1cs(self):
        """Test R1CS constraint system."""
        from core.r1cs import R1CSBuilder, LinearCombination
        from core.field import FieldConfig
        
        config = FieldConfig(prime=101)
        builder = R1CSBuilder(config)
        
        # Create variables
        x = builder.new_variable("x")
        y = builder.new_variable("y")
        z = builder.new_variable("z")
        
        # Add constraint: x * y = z
        a = LinearCombination()
        a.add_term(x, 1)
        
        b = LinearCombination()
        b.add_term(y, 1)
        
        c = LinearCombination()
        c.add_term(z, 1)
        
        builder.add_constraint(a, b, c)
        
        r1cs = builder.build()
        assert r1cs.num_constraints() == 1
    
    def test_crypto_bn254_field(self):
        """Test BN254 field arithmetic."""
        from crypto.bn254.field import Fp, Fr
        
        # Test Fp
        a = Fp(12345)
        b = Fp(67890)
        c = a + b
        d = a * b
        
        # Test Fr
        x = Fr(11111)
        y = Fr(22222)
        z = x + y
        w = x * y
        
        # Test inverse
        inv_a = a.inverse()
        assert (a * inv_a) == Fp.one(), "Fp inverse failed"
    
    def test_crypto_bn254_curve(self):
        """Test BN254 curve operations."""
        from crypto.bn254.curve import G1Point
        
        # Generator point
        g = G1Point.generator()
        
        # Scalar multiplication
        p = g.scalar_mul(5)
        q = g.scalar_mul(3)
        
        # Point addition
        r = p + q
        
        # Verify: 5G + 3G = 8G
        expected = g.scalar_mul(8)
        assert r == expected, "Curve arithmetic failed"
    
    def test_network_builder(self):
        """Test network builder."""
        from network.builder import NetworkBuilder
        from core.field import FieldConfig
        
        config = FieldConfig(prime=101)
        
        network = (NetworkBuilder(config)
            .input(4)
            .dense(8, activation='relu')
            .dense(2, activation='softmax')
            .build())
        
        assert network is not None
        assert len(network.layers) == 3
    
    def test_cnn_layers(self):
        """Test CNN layers."""
        from network.cnn.conv2d import Conv2DLayer
        from network.cnn.pooling import MaxPoolLayer
        from core.field import FieldConfig
        
        config = FieldConfig(prime=101)
        
        # Conv2D
        conv = Conv2DLayer(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            field_config=config
        )
        
        # MaxPool
        pool = MaxPoolLayer(pool_size=2, field_config=config)
        
        assert conv is not None
        assert pool is not None
    
    def test_compressed_sensing(self):
        """Test CSWC (Compressed Sensing Witness Compression)."""
        from compressed_sensing.sensing_matrix import SensingMatrix
        from compressed_sensing.sparse_witness import SparseWitness
        from core.field import FieldConfig
        
        config = FieldConfig(prime=101)
        
        # Create sensing matrix
        matrix = SensingMatrix(
            n_measurements=10,
            n_witness=50,
            field_config=config
        )
        
        # Create sparse witness (50% sparsity)
        witness_values = [i if i % 2 == 0 else 0 for i in range(50)]
        witness = SparseWitness(witness_values, config)
        
        assert witness.sparsity > 0.4
    
    def test_wavelet_hwwb(self):
        """Test HWWB (Haar Wavelet Witness Batching)."""
        from wavelet.haar_transform import HaarWaveletTransform, HWWBProver
        from core.field import FieldElement, FieldConfig
        
        config = FieldConfig(prime=101)
        
        # Create transform
        transform = HaarWaveletTransform(config)
        
        # Test data (power of 2)
        data = [FieldElement(i, config) for i in range(8)]
        
        # Forward transform
        coeffs = transform.forward(data)
        
        # Inverse transform
        recovered = transform.inverse(coeffs)
        
        # Verify reconstruction
        for i, (orig, rec) in enumerate(zip(data, recovered)):
            assert orig.value == rec.value, f"HWWB reconstruction failed at {i}"
    
    def test_tropical_operations(self):
        """Test Tropical Geometry operations."""
        from tropical.tropical_ops import (
            TropicalSemiring, TropicalCircuit, 
            TropicalMaxPool, TropicalArgmax
        )
        
        # Test semiring
        a = TropicalSemiring.from_standard(10)
        b = TropicalSemiring.from_standard(20)
        
        # Tropical addition (min)
        c = TropicalSemiring.add(a, b)
        assert c == a, "Tropical add (min) failed"
        
        # Tropical multiplication (standard add)
        d = TropicalSemiring.mul(a, b)
        assert d == a + b, "Tropical mul failed"
        
        # Test circuit
        circuit = TropicalCircuit()
        max_pool = TropicalMaxPool(4)
        
        inputs = [circuit.new_variable(f"x{i}") for i in range(4)]
        result, constraints = max_pool.compile(circuit, inputs)
        
        assert result is not None
        assert constraints > 0
    
    def test_tropical_optimizer(self):
        """Test Tropical optimizer integration."""
        from tropical.pipeline_integration import (
            TropicalOptimizer, LayerInfo, LayerType, OptimizationConfig
        )
        
        config = OptimizationConfig(
            enable_tropical_maxpool=True,
            enable_tropical_softmax=True
        )
        
        optimizer = TropicalOptimizer(config)
        
        layers = [
            LayerInfo(LayerType.CONV2D, 784, 1024, {'kernel_size': 3, 'in_channels': 1}),
            LayerInfo(LayerType.RELU, 1024, 1024),
            LayerInfo(LayerType.MAX_POOL, 1024, 256, {'pool_size': 2}),
            LayerInfo(LayerType.DENSE, 256, 10),
            LayerInfo(LayerType.SOFTMAX, 10, 10),
        ]
        
        result = optimizer.optimize_network(layers)
        
        assert result.reduction_percent > 0, "No optimization achieved"
        assert result.layers_optimized > 0, "No layers optimized"
    
    def test_plonk_polynomial(self):
        """Test PLONK polynomial arithmetic."""
        from plonk.polynomial import Polynomial
        from crypto.bn254.field import Fr
        
        # Create polynomials
        p1 = Polynomial.from_ints([1, 2, 3])  # 1 + 2x + 3x^2
        p2 = Polynomial.from_ints([4, 5])     # 4 + 5x
        
        # Addition
        p3 = p1 + p2
        
        # Multiplication
        p4 = p1 * p2
        
        # Evaluation
        val = p1.evaluate(Fr(2))
        expected = Fr(1 + 2*2 + 3*4)  # 1 + 4 + 12 = 17
        
        assert val == expected, f"Polynomial evaluation failed: {val} != {expected}"
    
    def test_plonk_circuit_compiler(self):
        """Test PLONK circuit compiler."""
        from plonk.circuit_compiler import CircuitCompiler, NetworkConfig
        from network.builder import NetworkBuilder
        from core.field import FieldConfig
        
        field_config = FieldConfig(prime=101)
        
        # Build a simple network
        network = (NetworkBuilder(field_config)
            .input(4)
            .dense(4, activation='gelu')
            .dense(2)
            .build())
        
        # Compile to circuit
        compiler = CircuitCompiler()
        net_config = NetworkConfig(
            input_size=4,
            layers=[
                {'type': 'dense', 'size': 4, 'activation': 'gelu'},
                {'type': 'dense', 'size': 2, 'activation': 'none'}
            ]
        )
        
        circuit = compiler.compile(net_config)
        
        assert circuit is not None
        assert circuit.num_constraints > 0
    
    def test_proof_system(self):
        """Test the proof generation and verification."""
        from proof.prover import Prover
        from proof.verifier import Verifier
        from network.builder import NetworkBuilder
        from core.field import FieldConfig
        
        config = FieldConfig(prime=101)
        
        # Build network
        network = (NetworkBuilder(config)
            .input(4)
            .dense(4, activation='gelu')
            .dense(2)
            .build())
        
        # Create prover and verifier
        prover = Prover(network)
        verifier = Verifier(network)
        
        # Generate proof
        input_data = [1, 2, 3, 4]
        proof, output = prover.prove(input_data)
        
        # Verify proof
        is_valid = verifier.verify(proof, input_data, output)
        
        assert is_valid, "Proof verification failed"
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("=" * 70)
        print("zkML System v2.0.0 - Full Pipeline Test")
        print("=" * 70)
        
        # Core tests
        print("\n[1/7] Core Module Tests:")
        self.run_test("Field Arithmetic", self.test_core_field)
        self.run_test("R1CS Constraints", self.test_core_r1cs)
        
        # Crypto tests
        print("\n[2/7] Crypto Module Tests:")
        self.run_test("BN254 Field", self.test_crypto_bn254_field)
        self.run_test("BN254 Curve", self.test_crypto_bn254_curve)
        
        # Network tests
        print("\n[3/7] Network Module Tests:")
        self.run_test("Network Builder", self.test_network_builder)
        self.run_test("CNN Layers", self.test_cnn_layers)
        
        # Optimization tests
        print("\n[4/7] Optimization Module Tests:")
        self.run_test("Compressed Sensing (CSWC)", self.test_compressed_sensing)
        self.run_test("Wavelet (HWWB)", self.test_wavelet_hwwb)
        self.run_test("Tropical Operations", self.test_tropical_operations)
        self.run_test("Tropical Optimizer", self.test_tropical_optimizer)
        
        # PLONK tests
        print("\n[5/7] PLONK Module Tests:")
        self.run_test("Polynomial Arithmetic", self.test_plonk_polynomial)
        self.run_test("Circuit Compiler", self.test_plonk_circuit_compiler)
        
        # Proof system tests
        print("\n[6/7] Proof System Tests:")
        self.run_test("Prover & Verifier", self.test_proof_system)
        
        # Generate report
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        total_time = sum(r.duration_ms for r in self.results)
        
        print(f"\nTotal: {total} tests")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total:.1f}%)")
        print(f"Total time: {total_time:.1f}ms")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
        
        print("\n" + "=" * 70)
        if failed == 0:
            print("ALL TESTS PASSED")
        else:
            print(f"TESTS FAILED: {failed}/{total}")
        print("=" * 70)
        
        return failed == 0


if __name__ == "__main__":
    tester = PipelineTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
