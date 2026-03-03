"""
End-to-End Integration Tests for the zkML System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time


def test_core_modules():
    """Test core modules."""
    print("Testing core modules...")
    from zkml_system.core.field import FieldElement, FIELD_DEV
    a = FieldElement(7, FIELD_DEV)
    b = FieldElement(13, FIELD_DEV)
    assert (a * b).value == 91
    print("  Core modules: OK")


def test_bn254_crypto():
    """Test BN254 crypto."""
    print("Testing BN254 crypto...")
    from zkml_system.crypto.bn254.field import Fp, Fr
    from zkml_system.crypto.bn254.curve import G1Point, G2Point

    a = Fp(123456789)
    b = Fp(987654321)
    c = a * b
    assert c.to_int() == (123456789 * 987654321) % Fp.MODULUS

    g1 = G1Point.generator()
    assert g1.is_on_curve()
    g1_3 = g1 * 3
    assert g1_3 == g1 + g1 + g1

    g2 = G2Point.generator()
    assert g2.is_on_curve()
    print("  BN254 crypto: OK")


def test_cnn_layers():
    """Test CNN layers."""
    print("Testing CNN layers...")
    from zkml_system.network.cnn.conv2d import Conv2DConfig, Conv2DLayer
    from zkml_system.network.cnn.pooling import PoolConfig, AvgPool2D

    config = Conv2DConfig(in_channels=1, out_channels=1, kernel_size=3)
    weights = [[[[1/9] * 3 for _ in range(3)]]]
    layer = Conv2DLayer(config, weights)
    input_tensor = [[[1.0] * 5 for _ in range(5)]]
    output = layer.forward(input_tensor)
    assert len(output) == 1

    pool_config = PoolConfig(kernel_size=2, stride=2)
    avg_pool = AvgPool2D(pool_config)
    pool_input = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                   [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]
    pool_output = avg_pool.forward(pool_input)
    assert pool_output[0][0][0] == 3.5
    print("  CNN layers: OK")


def test_polynomial():
    """Test polynomial arithmetic."""
    print("Testing polynomial arithmetic...")
    from zkml_system.crypto.bn254.fr_adapter import Fr
    from zkml_system.plonk.polynomial import Polynomial, FFT

    p = Polynomial.from_ints([1, 2, 3])
    q = Polynomial.from_ints([4, 5])
    r = p * q
    assert r.coeffs[0].to_int() == 4

    fft = FFT(4)
    coeffs = [Fr(1), Fr(2), Fr(3), Fr(4)]
    evals = fft.fft(coeffs)
    recovered = fft.ifft(evals)
    for i in range(4):
        assert recovered[i].to_int() == coeffs[i].to_int()
    print("  Polynomial: OK")


def test_kzg():
    """Test KZG commitments."""
    print("Testing KZG commitments...")
    from zkml_system.crypto.bn254.fr_adapter import Fr
    from zkml_system.plonk.polynomial import Polynomial
    from zkml_system.plonk.kzg import SRS, KZG

    srs = SRS.generate(16)
    kzg = KZG(srs)
    poly = Polynomial.from_ints([1, 2, 3])
    commitment = kzg.commit(poly)
    assert commitment.point.is_on_curve()

    z = Fr(5)
    proof, y = kzg.create_proof(poly, z)
    assert y == poly.evaluate(z)
    assert kzg.verify(commitment, z, y, proof)
    print("  KZG: OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("zkML System Integration Tests")
    print("=" * 60)

    test_core_modules()
    test_bn254_crypto()
    test_cnn_layers()
    test_polynomial()
    test_kzg()

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
