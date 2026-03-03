"""
zkml-system — Zero-Knowledge Proofs for Neural Network Inference.

Research prototype: PLONK prover/verifier on BN254, with a hybrid
TDA+ZK bridge for privacy-preserving model similarity proofs.
"""

from setuptools import setup, find_packages

VERSION = "3.1.0"

setup(
    name="zkml-system",
    version=VERSION,
    description="Zero-Knowledge Machine Learning — PLONK proofs for neural network inference and model similarity",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Upsalla",
    author_email="",
    url="https://github.com/Upsalla/zkml-system",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*", "experimental", "experimental.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "py_ecc>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    keywords="zero-knowledge, machine-learning, zkml, plonk, cryptography, tda",
)
