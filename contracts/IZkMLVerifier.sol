// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IZkMLVerifier
 * @notice Interface for zkML verification contracts
 * @dev Allows other contracts to integrate with zkML verification
 */

interface IZkMLVerifier {
    // ============================================================
    // Structs
    // ============================================================

    struct Proof {
        uint256[2] a_commit;
        uint256[2] b_commit;
        uint256[2] c_commit;
        uint256[2] z_commit;
        uint256[2] t_lo_commit;
        uint256[2] t_mid_commit;
        uint256[2] t_hi_commit;
        uint256 a_eval;
        uint256 b_eval;
        uint256 c_eval;
        uint256 sigma_1_eval;
        uint256 sigma_2_eval;
        uint256 z_omega_eval;
        uint256[2] w_zeta;
        uint256[2] w_zeta_omega;
    }

    // ============================================================
    // Events
    // ============================================================

    event ModelRegistered(
        bytes32 indexed modelId,
        address indexed owner,
        bytes32 modelHash,
        uint256 inputSize,
        uint256 outputSize
    );

    event InferenceVerified(
        bytes32 indexed inferenceId,
        bytes32 indexed modelId,
        address indexed prover,
        bytes32 inputHash,
        uint256[] output
    );

    // ============================================================
    // Model Management
    // ============================================================

    /**
     * @notice Check if a model is registered and active
     * @param modelId The model to check
     * @return True if the model exists and is active
     */
    function isModelActive(bytes32 modelId) external view returns (bool);

    /**
     * @notice Get model information
     * @param modelId The model to query
     * @return modelHash Hash of model weights
     * @return inputSize Expected input dimension
     * @return outputSize Expected output dimension
     * @return owner Model owner address
     */
    function getModelInfo(bytes32 modelId) external view returns (
        bytes32 modelHash,
        uint256 inputSize,
        uint256 outputSize,
        address owner
    );

    // ============================================================
    // Verification
    // ============================================================

    /**
     * @notice Verify an ML inference proof
     * @param modelId The model used for inference
     * @param inputHash Hash of the input data
     * @param output The claimed inference output
     * @param proof The PLONK proof
     * @return inferenceId The unique identifier for this verified inference
     */
    function verifyInference(
        bytes32 modelId,
        bytes32 inputHash,
        uint256[] calldata output,
        Proof calldata proof
    ) external returns (bytes32 inferenceId);

    /**
     * @notice Check if an inference has been verified
     * @param inferenceId The inference to check
     * @return True if the inference is verified
     */
    function isInferenceVerified(bytes32 inferenceId) external view returns (bool);

    /**
     * @notice Get inference output
     * @param inferenceId The inference to query
     * @return output The inference output array
     */
    function getInferenceOutput(bytes32 inferenceId) external view returns (uint256[] memory);
}
