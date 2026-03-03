"""
Comprehensive Tests for the Refactored PLONK zkML System

Tests cover:
1. Core PLONK components (Field, Polynomial, KZG, Circuit)
2. BN254 cryptographic primitives
3. Optimizations (CSWC, HWWB, Tropical)
4. Complete zkML pipeline
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestField(unittest.TestCase):
    """Tests for field arithmetic."""
    
    def test_field_element_creation(self):
        from plonk.core import Field, Fr
        
        a = Field.element(42)
        self.assertEqual(a.to_int(), 42)
        
        b = Field.element(-1)
        self.assertEqual(b.to_int(), Fr.MODULUS - 1)
    
    def test_field_arithmetic(self):
        from plonk.core import Field
        
        a = Field.element(100)
        b = Field.element(200)
        
        # Addition
        c = a + b
        self.assertEqual(c.to_int(), 300)
        
        # Multiplication
        d = a * b
        self.assertEqual(d.to_int(), 20000)
        
        # Subtraction
        e = b - a
        self.assertEqual(e.to_int(), 100)
    
    def test_field_inverse(self):
        from plonk.core import Field
        
        a = Field.element(42)
        a_inv = a.inverse()
        
        # a * a^(-1) = 1
        product = a * a_inv
        self.assertEqual(product.to_int(), 1)


class TestPolynomial(unittest.TestCase):
    """Tests for polynomial operations."""
    
    def test_polynomial_evaluation(self):
        from plonk.core import Polynomial, Field
        
        # p(x) = 1 + 2x + 3x^2
        p = Polynomial.from_ints([1, 2, 3])
        
        # p(2) = 1 + 4 + 12 = 17
        result = p.evaluate(Field.element(2))
        self.assertEqual(result.to_int(), 17)
    
    def test_polynomial_addition(self):
        from plonk.core import Polynomial
        
        p1 = Polynomial.from_ints([1, 2, 3])
        p2 = Polynomial.from_ints([4, 5, 6])
        
        p3 = p1 + p2
        self.assertEqual([c.to_int() for c in p3.coeffs], [5, 7, 9])
    
    def test_polynomial_multiplication(self):
        from plonk.core import Polynomial
        
        # (1 + x) * (1 + x) = 1 + 2x + x^2
        p = Polynomial.from_ints([1, 1])
        p_squared = p * p
        
        self.assertEqual([c.to_int() for c in p_squared.coeffs], [1, 2, 1])
    
    def test_lagrange_interpolation(self):
        from plonk.core import Polynomial, Field
        
        # Points: (0, 1), (1, 2), (2, 5)
        # Should give p(x) = 1 + 0.5x + 0.5x^2 (but in field arithmetic)
        points = [
            (Field.element(0), Field.element(1)),
            (Field.element(1), Field.element(2)),
            (Field.element(2), Field.element(5))
        ]
        
        p = Polynomial.lagrange_interpolate(points)
        
        # Verify interpolation
        for x, y in points:
            self.assertEqual(p.evaluate(x).to_int(), y.to_int())


class TestKZG(unittest.TestCase):
    """Tests for KZG commitment scheme."""
    
    def test_kzg_commitment(self):
        from plonk.core import Polynomial, SRS, KZG
        
        srs = SRS.generate_insecure(16)
        kzg = KZG(srs)
        
        p = Polynomial.from_ints([1, 2, 3])
        commitment = kzg.commit(p)
        
        # Commitment should be a valid curve point
        self.assertTrue(commitment.point.is_on_curve())
    
    def test_kzg_opening_proof(self):
        from plonk.core import Polynomial, SRS, KZG, Field
        
        srs = SRS.generate_insecure(16)
        kzg = KZG(srs)
        
        p = Polynomial.from_ints([1, 2, 3])
        point = Field.element(5)
        
        proof = kzg.create_proof(p, point)
        
        # Verify the evaluation is correct
        expected = p.evaluate(point)
        self.assertEqual(proof.value.to_int(), expected.to_int())


class TestCircuit(unittest.TestCase):
    """Tests for circuit representation."""
    
    def test_circuit_creation(self):
        from plonk.core import Circuit, Field
        
        circuit = Circuit()
        
        x = circuit.add_public_input('x')
        y = circuit.add_public_input('y')
        z = circuit.mul(x, y, 'z')
        
        self.assertEqual(circuit.num_wires(), 3)
        self.assertEqual(circuit.num_constraints(), 3)
    
    def test_witness_assignment(self):
        from plonk.core import Circuit, Witness, Field
        
        circuit = Circuit()
        x = circuit.add_public_input('x')
        y = circuit.add_public_input('y')
        z = circuit.mul(x, y, 'z')
        
        witness = Witness()
        witness.set(x, Field.element(3))
        witness.set(y, Field.element(4))
        witness.set(z, Field.element(12))
        
        self.assertEqual(witness.get(x).to_int(), 3)
        self.assertEqual(witness.get(y).to_int(), 4)
        self.assertEqual(witness.get(z).to_int(), 12)


class TestBN254(unittest.TestCase):
    """Tests for BN254 cryptographic primitives."""
    
    def test_g1_generator(self):
        from crypto.bn254 import G1Point
        
        g1 = G1Point.generator()
        self.assertTrue(g1.is_on_curve())
    
    def test_g1_scalar_multiplication(self):
        from crypto.bn254 import G1Point, Fr
        
        g1 = G1Point.generator()
        
        # 2 * G1
        g1_2 = g1 * Fr(2)
        self.assertTrue(g1_2.is_on_curve())
        
        # G1 + G1 should equal 2 * G1
        g1_sum = g1 + g1
        self.assertEqual(g1_2, g1_sum)
    
    def test_g2_generator(self):
        from crypto.bn254 import G2Point
        
        g2 = G2Point.generator()
        # G2 point should be valid
        self.assertFalse(g2.is_identity())


class TestCSWC(unittest.TestCase):
    """Tests for Compressed Sensing Witness Compression."""
    
    def test_sparse_witness_creation(self):
        from plonk.optimizations import SparseWitness
        from plonk.core import Field
        
        # Create sparse data (67% zeros)
        values = [Field.element(i if i % 3 == 0 else 0) for i in range(100)]
        sparse = SparseWitness.from_dense(values)
        
        self.assertAlmostEqual(sparse.sparsity, 0.67, places=2)
        self.assertEqual(sparse.nnz, 33)
    
    def test_cswc_compression(self):
        from plonk.optimizations import SparseWitness, CSWCCompressor
        from plonk.core import Field
        
        # Create highly sparse data
        values = [Field.element(i if i % 10 == 0 else 0) for i in range(100)]
        sparse = SparseWitness.from_dense(values)
        
        compressor = CSWCCompressor()
        _, meta = compressor.compress(sparse)
        
        self.assertTrue(meta.get('compressed', False))


class TestHWWB(unittest.TestCase):
    """Tests for Haar Wavelet Witness Batching."""
    
    def test_haar_transform(self):
        from plonk.optimizations import HaarTransform
        from plonk.core import Field
        
        # Test with power-of-2 length
        values = [Field.element(i) for i in range(8)]
        
        transform = HaarTransform()
        coeffs = transform.transform_1d(values)
        
        # Transform should preserve length
        self.assertEqual(len(coeffs), 8)
    
    def test_haar_roundtrip(self):
        from plonk.optimizations import HaarTransform
        from plonk.core import Field
        
        values = [Field.element(i * 10) for i in range(8)]
        
        transform = HaarTransform()
        coeffs = transform.transform_1d(values)
        recovered = transform.inverse_1d(coeffs)
        
        # Should recover original values
        for i in range(8):
            self.assertEqual(values[i].to_int(), recovered[i].to_int())


class TestTropical(unittest.TestCase):
    """Tests for Tropical Geometry optimizations."""
    
    def test_tropical_max(self):
        from plonk.optimizations import TropicalSemiring
        from plonk.core import Field
        
        values = [Field.element(x) for x in [5, 2, 8, 1, 9, 3]]
        
        max_val, max_idx = TropicalSemiring.tropical_max(values)
        
        self.assertEqual(max_val.to_int(), 9)
        self.assertEqual(max_idx, 4)
    
    def test_tropical_reduction_estimate(self):
        from plonk.optimizations import TropicalOptimizer
        
        optimizer = TropicalOptimizer()
        
        # Max operation on 16 elements
        reduction = optimizer.calculate_reduction('max', 16)
        
        # Should achieve >90% reduction
        self.assertGreater(reduction, 90.0)


class TestZkMLPipeline(unittest.TestCase):
    """Tests for the complete zkML pipeline."""
    
    def test_network_configuration(self):
        from plonk.zkml import NetworkConfig, LayerType
        
        config = NetworkConfig(
            name="test_net",
            layers=[
                ('dense', {'input_size': 10, 'output_size': 5}),
                ('relu', {'input_size': 5}),
                ('argmax', {'input_size': 5})
            ]
        )
        
        self.assertEqual(config.input_size, 10)
        self.assertEqual(len(config.layers), 3)
    
    def test_circuit_compilation(self):
        from plonk.zkml import ZkML, NetworkConfig
        
        config = NetworkConfig(
            name="test_net",
            layers=[
                ('dense', {'input_size': 8, 'output_size': 4}),
                ('relu', {'input_size': 4})
            ]
        )
        
        zkml = ZkML(config, srs_max_degree=256)
        stats = zkml.get_circuit_stats()
        
        self.assertGreater(stats['num_wires'], 0)
        self.assertGreater(stats['num_gates'], 0)
    
    def test_proof_generation_raises_not_implemented(self):
        """ZkMLProver.prove() must raise NotImplementedError (quarantined hollow shell)."""
        from plonk.zkml import ZkML, NetworkConfig
        
        config = NetworkConfig(
            name="test_classifier",
            layers=[
                ('dense', {'input_size': 8, 'output_size': 4}),
                ('relu', {'input_size': 4}),
                ('argmax', {'input_size': 4})
            ]
        )
        
        zkml = ZkML(config, srs_max_degree=512)
        
        input_data = [i + 1 for i in range(8)]
        with self.assertRaises(NotImplementedError):
            zkml.prove(input_data)
    
    def test_optimization_estimates(self):
        from plonk.zkml import ZkML, NetworkConfig
        
        config = NetworkConfig(
            name="optimized_net",
            layers=[
                ('dense', {'input_size': 16, 'output_size': 8}),
                ('max_pool', {'input_size': 8, 'output_size': 4, 'pool_size': 2}),
                ('argmax', {'input_size': 4})
            ]
        )
        
        zkml = ZkML(config, srs_max_degree=512)
        stats = zkml.estimate_optimizations()
        
        # Should show constraint reduction
        self.assertGreater(stats.constraint_reduction, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_raises_not_implemented(self):
        """E2E proof must raise NotImplementedError (quarantined hollow shell)."""
        from plonk.zkml import ZkML, NetworkConfig
        
        config = NetworkConfig(
            name="mnist_small",
            layers=[
                ('dense', {'input_size': 16, 'output_size': 8}),
                ('relu', {'input_size': 8}),
                ('dense', {'input_size': 8, 'output_size': 4}),
                ('argmax', {'input_size': 4})
            ]
        )
        
        zkml = ZkML(config, srs_max_degree=1024)
        
        input_data = [j % 256 for j in range(16)]
        with self.assertRaises(NotImplementedError):
            zkml.prove(input_data)
    
    def test_proof_size_raises_not_implemented(self):
        """Proof size test must raise NotImplementedError (quarantined)."""
        from plonk.zkml import ZkML, NetworkConfig
        
        config = NetworkConfig(
            name="size_test",
            layers=[
                ('dense', {'input_size': 8, 'output_size': 4}),
                ('argmax', {'input_size': 4})
            ]
        )
        
        zkml = ZkML(config, srs_max_degree=256)
        
        input_data = [i for i in range(8)]
        with self.assertRaises(NotImplementedError):
            zkml.prove(input_data)


def run_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestField))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomial))
    suite.addTests(loader.loadTestsFromTestCase(TestKZG))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuit))
    suite.addTests(loader.loadTestsFromTestCase(TestBN254))
    suite.addTests(loader.loadTestsFromTestCase(TestCSWC))
    suite.addTests(loader.loadTestsFromTestCase(TestHWWB))
    suite.addTests(loader.loadTestsFromTestCase(TestTropical))
    suite.addTests(loader.loadTestsFromTestCase(TestZkMLPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
