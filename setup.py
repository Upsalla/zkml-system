"""
zkML System - Zero-Knowledge Machine Learning Framework

A comprehensive framework for generating zero-knowledge proofs of
machine learning inference with interdisciplinary optimizations.

Features:
- PLONK-based proof system
- BN254 elliptic curve cryptography
- Compressed Sensing witness compression (CSWC)
- Haar-Wavelet witness batching (HWWB)
- Tropical Geometry optimizations for max-pooling/softmax
- CNN support with Conv2D and Pooling layers
- Smart contract integration for on-chain verification

Installation:
    pip install -e .
    pip install zkml-system-2.0.0.tar.gz

Usage:
    zkml --help
    zkml prove --network model.json --input data.json
    zkml verify --proof proof.json
    zkml benchmark --network model.json
"""

from setuptools import setup, find_packages

VERSION = "2.0.0"

setup(
    name="zkml-system",
    version=VERSION,
    description="Zero-Knowledge Machine Learning Framework with Interdisciplinary Optimizations",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="zkML Research Team",
    author_email="research@zkml.dev",
    url="https://github.com/zkml/zkml-system",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "web3": [
            "web3>=6.0.0",
            "eth-account>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zkml=deployment.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.sol", "*.md"],
        "contracts": ["*.sol"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
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
    keywords="zero-knowledge, machine-learning, zkml, plonk, cryptography, neural-networks",
)
