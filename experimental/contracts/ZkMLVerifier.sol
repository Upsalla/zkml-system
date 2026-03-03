// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./PlonkVerifier.sol";

/**
 * @title ZkMLVerifier
 * @notice Verifies zero-knowledge proofs of ML model inference
 * @dev Extends PlonkVerifier with ML-specific functionality
 *
 * This contract allows:
 * 1. Registration of ML models with their verifier keys
 * 2. Verification of inference proofs
 * 3. Storage of verified inference results
 *
 * Use cases:
 * - Verifiable AI: Prove that a prediction came from a specific model
 * - Private ML: Verify inference without revealing inputs
 * - Decentralized AI: On-chain verification of off-chain computation
 */

contract ZkMLVerifier is PlonkVerifier {
    // ============================================================
    // Data Structures
    // ============================================================

    struct Model {
        bytes32 modelHash;          // Hash of model weights
        uint256 inputSize;          // Expected input dimension
        uint256 outputSize;         // Expected output dimension
        address owner;              // Model owner
        bool isActive;              // Whether model is active
        uint256 verificationCount;  // Number of verified inferences
    }

    struct InferenceResult {
        bytes32 modelId;            // Model used for inference
        bytes32 inputHash;          // Hash of input data
        uint256[] output;           // Inference output
        uint256 timestamp;          // Verification timestamp
        address prover;             // Who submitted the proof
    }

    // ============================================================
    // State Variables
    // ============================================================

    // Model registry
    mapping(bytes32 => Model) public models;
    bytes32[] public modelIds;

    // Verified inference results
    mapping(bytes32 => InferenceResult) public inferences;
    bytes32[] public inferenceIds;

    // Model-specific verifier keys
    mapping(bytes32 => VerifierKey) public modelVerifierKeys;

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

    event ModelDeactivated(bytes32 indexed modelId);

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
     * @notice Register a new ML model
     * @param modelHash Hash of the model weights
     * @param inputSize Expected input dimension
     * @param outputSize Expected output dimension
     * @param verifierKey The verifier key for this model's circuit
     * @return modelId The unique identifier for this model
     */
    function registerModel(
        bytes32 modelHash,
        uint256 inputSize,
        uint256 outputSize,
        VerifierKey calldata verifierKey
    ) external returns (bytes32 modelId) {
        modelId = keccak256(abi.encodePacked(
            modelHash,
            inputSize,
            outputSize,
            msg.sender,
            block.timestamp
        ));

        require(models[modelId].owner == address(0), "Model already exists");

        models[modelId] = Model({
            modelHash: modelHash,
            inputSize: inputSize,
            outputSize: outputSize,
            owner: msg.sender,
            isActive: true,
            verificationCount: 0
        });

        modelVerifierKeys[modelId] = verifierKey;
        modelIds.push(modelId);

        emit ModelRegistered(modelId, msg.sender, modelHash, inputSize, outputSize);
    }

    /**
     * @notice Deactivate a model
     * @param modelId The model to deactivate
     */
    function deactivateModel(bytes32 modelId) external {
        require(models[modelId].owner == msg.sender, "Not model owner");
        models[modelId].isActive = false;
        emit ModelDeactivated(modelId);
    }

    // ============================================================
    // Inference Verification
    // ============================================================

    /**
     * @notice Verify an ML inference proof
     * @param modelId The model used for inference
     * @param inputHash Hash of the input data (for privacy)
     * @param output The claimed inference output
     * @param proof The PLONK proof
     * @return inferenceId The unique identifier for this verified inference
     */
    function verifyInference(
        bytes32 modelId,
        bytes32 inputHash,
        uint256[] calldata output,
        Proof calldata proof
    ) external returns (bytes32 inferenceId) {
        Model storage model = models[modelId];
        require(model.isActive, "Model not active");
        require(output.length == model.outputSize, "Output size mismatch");

        // Set the verifier key for this model
        vk = modelVerifierKeys[modelId];

        // Prepare public inputs: [inputHash, output...]
        uint256[] memory publicInputs = new uint256[](1 + output.length);
        publicInputs[0] = uint256(inputHash);
        for (uint256 i = 0; i < output.length; i++) {
            publicInputs[i + 1] = output[i];
        }

        // Verify the proof
        require(this.verify(proof, publicInputs), "Proof verification failed");

        // Store the verified inference
        inferenceId = keccak256(abi.encodePacked(
            modelId,
            inputHash,
            output,
            msg.sender,
            block.timestamp
        ));

        inferences[inferenceId] = InferenceResult({
            modelId: modelId,
            inputHash: inputHash,
            output: output,
            timestamp: block.timestamp,
            prover: msg.sender
        });

        inferenceIds.push(inferenceId);
        model.verificationCount++;

        emit InferenceVerified(inferenceId, modelId, msg.sender, inputHash, output);
    }

    /**
     * @notice Batch verify multiple inferences
     * @param modelId The model used for all inferences
     * @param inputHashes Hashes of input data
     * @param outputs The claimed inference outputs
     * @param proofs The PLONK proofs
     * @return inferenceIdList The unique identifiers for verified inferences
     */
    function batchVerifyInferences(
        bytes32 modelId,
        bytes32[] calldata inputHashes,
        uint256[][] calldata outputs,
        Proof[] calldata proofs
    ) external returns (bytes32[] memory inferenceIdList) {
        require(
            inputHashes.length == outputs.length &&
            outputs.length == proofs.length,
            "Array length mismatch"
        );

        inferenceIdList = new bytes32[](inputHashes.length);

        for (uint256 i = 0; i < inputHashes.length; i++) {
            inferenceIdList[i] = this.verifyInference(
                modelId,
                inputHashes[i],
                outputs[i],
                proofs[i]
            );
        }
    }

    // ============================================================
    // Query Functions
    // ============================================================

    /**
     * @notice Get the number of registered models
     */
    function getModelCount() external view returns (uint256) {
        return modelIds.length;
    }

    /**
     * @notice Get the number of verified inferences
     */
    function getInferenceCount() external view returns (uint256) {
        return inferenceIds.length;
    }

    /**
     * @notice Get inference output by ID
     * @param inferenceId The inference to query
     * @return output The inference output array
     */
    function getInferenceOutput(bytes32 inferenceId) external view returns (uint256[] memory) {
        return inferences[inferenceId].output;
    }

    /**
     * @notice Check if an inference has been verified
     * @param inferenceId The inference to check
     * @return True if the inference exists and is verified
     */
    function isInferenceVerified(bytes32 inferenceId) external view returns (bool) {
        return inferences[inferenceId].timestamp > 0;
    }

    /**
     * @notice Get model verification statistics
     * @param modelId The model to query
     * @return verificationCount Number of verified inferences
     * @return isActive Whether the model is active
     */
    function getModelStats(bytes32 modelId) external view returns (
        uint256 verificationCount,
        bool isActive
    ) {
        Model storage model = models[modelId];
        return (model.verificationCount, model.isActive);
    }
}
