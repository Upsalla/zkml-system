# Technical Documentation: zkML System

> [!WARNING]
> **Status: PARTIALLY STALE** — Sections 1-3 (field arithmetic, R1CS, activations) are
> conceptually valid. Section 4 (Prover/Verifier) describes a legacy Schnorr-based
> system that has been replaced by PLONK. See `SNARK_ARCHITECTURE.md` for the current
> proof system and `plonk/plonk_prover.py` for the implementation.

## 1. Mathematical Foundations

### 1.1 Finite Fields

All computations in ZK-proofs take place in finite fields. A prime field F_p consists of the numbers {0, 1, 2, ..., p-1} with arithmetic modulo p.

**Why prime fields?**
- Division is possible (multiplicative inverse exists)
- No overflow problems
- Cryptographic security

**Implementation** (`crypto/bn254/fr_adapter.py`):
```python
class FieldElement:
    def __init__(self, value: int, field: FieldConfig):
        self.value = value % field.prime
    
    def inverse(self) -> 'FieldElement':
        # Extended Euclidean Algorithm
        # a * a^(-1) = 1 mod p
```

### 1.2 R1CS (Rank-1 Constraint System)

R1CS is the standard format for ZK-proofs. Each computation is represented as a set of constraints:

```
A(w) × B(w) = C(w)
```

**Example**: z = x * y

```
Witness: w = [1, x, y, z]
A = [0, 1, 0, 0]  (selects x)
B = [0, 0, 1, 0]  (selects y)
C = [0, 0, 0, 1]  (selects z)

Constraint: x * y = z ✓
```

**Implementation** (`plonk/core.py`):
```python
@dataclass
class R1CSConstraint:
    a: LinearCombination
    b: LinearCombination
    c: LinearCombination
    
    def is_satisfied(self, witness: List[int], prime: int) -> bool:
        return (a.evaluate(witness) * b.evaluate(witness)) % prime == c.evaluate(witness)
```

### 1.3 Witness

The witness contains all intermediate values of a computation. It is the "secret" that the prover knows but does not reveal.

**Structure**:
```
w[0] = 1          (constant)
w[1..n] = inputs  (public)
w[n+1..m] = intermediate (private)
w[m+1..k] = outputs (public)
```

## 2. Activation Functions

### 2.1 The ReLU Problem

ReLU(x) = max(0, x) requires a comparison, which is expensive in R1CS:

**Bit decomposition**:
```
x = Σ(b_i * 2^i)  for i = 0..255

Constraints per bit:
- b_i * (1 - b_i) = 0  (b_i is 0 or 1)
- Σ(b_i * 2^i) = x     (reconstruction)

Total: 256+ constraints per ReLU
```

### 2.2 GELU Solution [IMPLEMENTED]

GELU is a smooth approximation of ReLU:

```
GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Polynomial approximation**:
```
GELU(x) ≈ ax³ + bx² + cx + d

With coefficients:
a = 0.044715 * √(2/π) ≈ 0.0356
b = 0
c = 0.5 + 0.5 * √(2/π) ≈ 0.8989
d = 0
```

**Constraints**:
```
1. x² = x * x
2. x³ = x² * x
3. t1 = a * x³
4. t2 = c * x
5. output = t1 + t2

Total: 5-10 constraints per GELU (vs. 256+ for ReLU)
```

### 2.3 Swish and Quadratic

**Swish**: x * sigmoid(x) ≈ x * (0.5 + 0.25x)
```
Constraints: ~8 per activation
```

**Quadratic**: x² (simplest non-linear function)
```
Constraints: 1 per activation
```

## 3. Sparse Proof Optimization [IMPLEMENTED]

### 3.1 Observation

In trained neural networks, 50-90% of neurons are inactive (output = 0) after ReLU/GELU.

### 3.2 Zero-Proof

Instead of proving the full computation, we prove only:
```
output = 0
```

**Constraint**:
```
output * 1 = 0
```

That is 1 constraint instead of dozens!

### 3.3 Batched Zero-Proofs

Multiple zero-proofs can be combined:

```
Choose random r_1, ..., r_n
Constraint: (r_1*x_1 + r_2*x_2 + ... + r_n*x_n) = 0

If all x_i = 0: sum = 0 ✓
If any x_i ≠ 0: sum ≠ 0 with high probability
```

**Advantage**: n constraints → 1 constraint

## 4. Proof System [DEPRECATED — replaced by PLONK]

> [!CAUTION]
> This section describes the legacy Schnorr-based proof system that has been
> replaced by PLONK. The implementation is in `plonk/plonk_prover.py` (5-round
> Fiat-Shamir protocol with KZG commitments). The old Schnorr system is no
> longer used.

### 4.1 Prover (Legacy)

The prover generates a proof that it knows a valid witness.

**Steps**:
1. Perform forward pass (generate witness)
2. Generate R1CS constraints
3. Compute witness commitment
4. Generate Schnorr-style proof

### 4.2 Verifier (Legacy)

The verifier checks the proof without knowing the witness.

**Checks**:
1. Network hash matches
2. Prime field matches
3. Schnorr proof is valid
4. R1CS structure is correct
5. Public values are valid

### 4.3 Security Properties

**Completeness**: An honest prover can always create a valid proof.

**Soundness**: A dishonest prover can (almost) never create a false proof.

**Zero-Knowledge**: The verifier learns nothing beyond validity.

## 5. Constraint Analysis [IMPLEMENTED]

### 5.1 Dense Layer

```
Input: n neurons
Output: m neurons

Matrix multiplication: n * m constraints
Bias addition: m constraints (integrated into matmul)
Activation: m * activation_cost constraints

Total: m * (n + activation_cost)
```

### 5.2 Comparison

| Activation | Cost/Neuron | 128 Neurons | 1024 Neurons |
|------------|-------------|-------------|--------------|
| ReLU       | 258         | 33,024      | 264,192      |
| GELU       | 10          | 1,280       | 10,240       |
| Swish      | 8           | 1,024       | 8,192        |
| Quadratic  | 1           | 128         | 1,024        |

### 5.3 Sparse Savings

At 60% sparsity:
```
Active neurons: 40% * full_cost
Inactive neurons: 60% * 1 (zero-proof)

Example (128 GELU neurons):
- Without sparse: 128 * 10 = 1,280
- With sparse: 51 * 10 + 77 * 1 = 587
- Savings: 54%
```

## 6. Implementation Details [IMPLEMENTED]

### 6.1 Fixed-Point Arithmetic

Decimal numbers are represented as integers with a scaling factor:

```python
SCALE = 2^16 = 65536

1.5 → 1.5 * 65536 = 98304
0.25 → 0.25 * 65536 = 16384

Multiplication:
(a * SCALE) * (b * SCALE) = a * b * SCALE²
→ Division by SCALE required
```

### 6.2 Negative Numbers

In prime fields there are no negative numbers. We interpret:
```
Values 0 to p/2: positive
Values p/2+1 to p-1: negative (as p - value)
```

### 6.3 Overflow Handling

With large prime fields (BN254) overflow is not a problem. With small fields (p=101) one must be careful:
```
50 + 60 = 110 → 110 % 101 = 9
```

## 7. Extension Possibilities

### 7.1 Full SNARK [IMPLEMENTED]

~~Current: Simplified Schnorr-style proof~~
~~Goal: Groth16 or PLONK for real zero-knowledge~~

**Status**: PLONK is implemented in `plonk/plonk_prover.py` with 5-round Fiat-Shamir, KZG commitments over BN254.

### 7.2 Larger Prime Field [IMPLEMENTED]

BN254 for Ethereum compatibility:
```
p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
```

**Status**: Fully implemented in `crypto/bn254/`.

### 7.3 Additional Layer Types [PARTIALLY IMPLEMENTED]

- Conv2D: For image classification → **Implemented** in `network/cnn/conv2d.py`
- BatchNorm: For deeper networks → **Not yet implemented**
- Attention: For transformers → **Not yet implemented**

### 7.4 On-Chain Verification [PARTIALLY IMPLEMENTED]

Solidity verifier for Ethereum smart contracts.

**Status**: Contracts exist in `contracts/` (`PlonkVerifier.sol`, `ZkMLVerifier.sol`, `ModelRegistry.sol`). Not yet deployed or end-to-end tested.
