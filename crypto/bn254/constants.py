"""
BN254 Curve Constants (alt_bn128)

This module defines all cryptographic parameters for the BN254 elliptic curve,
which is used by Ethereum's precompiled contracts for pairing operations.

The curve is defined by y² = x³ + 3 over the prime field Fp.
"""

# --- Base Field Fp ---
# The prime modulus for the base field Fp.
# All G1 point coordinates are elements of this field.
FIELD_MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583

# --- Scalar Field Fr ---
# The prime modulus for the scalar field Fr.
# This is the order of the G1 and G2 groups.
# All R1CS witness values and SNARK scalars are elements of this field.
CURVE_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# --- Curve Parameters ---
# Curve equation: y² = x³ + b
CURVE_B = 3

# --- G1 Generator ---
# The generator point for the G1 group on the curve.
G1_X = 1
G1_Y = 2

# --- G2 Generator ---
# The generator point for the G2 group, which lives in the Fp2 extension field.
# Coordinates are represented as (c0, c1) for the element c0 + c1 * u.
G2_X = (
    10857046999023057135944570762232829481370756359578518086990519993285655852781,
    11559732032986387107991004021392285783925812861821192530917403151452391805634,
)
G2_Y = (
    8495653923123431417604973247489272438418190587263600148770280649306958101930,
    4082367875863433681332203403145435568316851327593401208105741076214120093531,
)

# --- Montgomery Parameters for Fp ---
# R = 2^256 mod p
# R_SQUARED = R^2 mod p
# N_PRIME = -p^(-1) mod 2^64
# These are pre-computed for efficient Montgomery multiplication.
FP_R = 6350874878119819312338956282401532409788428879151445726012394534686998597021
FP_R_SQUARED = 3096616502983703923843567936837374451735540968419076528771170197431451843209
FP_N_PRIME = 9786893198990664585  # This is -p^(-1) mod 2^64

# --- Montgomery Parameters for Fr ---
FR_R = 6350874878119819312338956282401532410528162663560392320966563075034087161851
FR_R_SQUARED = 944936681149208446651664254269745548490766851729442924617792859073125903783
FR_N_PRIME = 14042775128853446655  # This is -r^(-1) mod 2^64

# --- Frobenius Coefficients for Fp2 ---
# Used for the Frobenius endomorphism x^p in Fp2.
# FP2_FROBENIUS_COEFF_C1[i] = (u+1)^((p^i - 1) / 2)
# For BN254, u^2 = -1, so u^p = -u.
# FP2_FROBENIUS_COEFF_C1[1] = u^((p-1)/2) = -1 (since (p-1)/2 is odd)
FP2_NON_RESIDUE = FIELD_MODULUS - 1  # This is -1 mod p, used as u^2 = -1

# --- Twist Parameters for G2 ---
# The twist curve is y² = x³ + b/ξ where ξ = u + 1 (a non-residue in Fp2).
# b_twist = b / ξ = 3 / (1 + u)
# We need to compute 3 * (1 + u)^(-1) in Fp2.
# (1+u)^(-1) = (1-u) / (1 - u^2) = (1-u) / 2
# So b_twist = 3 * (1-u) / 2
# The components are:
TWIST_B_C0 = 19485874751759354771024239261021720505790618469301721065564631296452457478373
TWIST_B_C1 = 266929791119991161246907387137283842545076965332900288569378510910307636690

# --- Ate Pairing Loop Count ---
# The loop count for the Miller loop in the Ate pairing.
# This is 6 * x + 2 where x is the BN parameter.
# x = 4965661367192848881
ATE_LOOP_COUNT = 29793968203157093288
ATE_LOOP_COUNT_IS_NEGATIVE = False

# --- Final Exponentiation Parameters ---
# (p^12 - 1) / r = (p^6 - 1) * (p^2 + 1) * (p^4 - p^2 + 1) / r
# The hard part is (p^4 - p^2 + 1) / r.
# We use the formula from "Faster Squaring in the Cyclotomic Subgroup of Sixth Degree Extensions".
