// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ModelRegistry
 * @author David Weyhe
 * @notice On-chain registry for TDA-based model fingerprints
 * @dev Enables registration, verification, and ownership tracking of ML model fingerprints
 */
contract ModelRegistry {
    
    // ============================================================
    // STRUCTS
    // ============================================================
    
    struct ModelFingerprint {
        bytes32 fingerprintHash;      // SHA256 of the TDA fingerprint
        address owner;                 // Owner of the model
        uint256 registrationTime;      // Block timestamp of registration
        string modelName;              // Human-readable model name
        string modelVersion;           // Version string
        bytes metadata;                // Additional metadata (IPFS hash, description, etc.)
        bool isActive;                 // Whether the registration is active
    }
    
    struct VerificationResult {
        bool isRegistered;
        address owner;
        uint256 registrationTime;
        string modelName;
    }
    
    // ============================================================
    // STATE VARIABLES
    // ============================================================
    
    /// @notice Mapping from fingerprint hash to model registration
    mapping(bytes32 => ModelFingerprint) public models;
    
    /// @notice Mapping from owner to list of their registered fingerprints
    mapping(address => bytes32[]) public ownerModels;
    
    /// @notice Total number of registered models
    uint256 public totalModels;
    
    /// @notice Registration fee (can be 0)
    uint256 public registrationFee;
    
    /// @notice Contract owner for admin functions
    address public admin;
    
    /// @notice Pause state for emergency
    bool public paused;
    
    // ============================================================
    // EVENTS
    // ============================================================
    
    event ModelRegistered(
        bytes32 indexed fingerprintHash,
        address indexed owner,
        string modelName,
        string modelVersion,
        uint256 timestamp
    );
    
    event ModelTransferred(
        bytes32 indexed fingerprintHash,
        address indexed previousOwner,
        address indexed newOwner,
        uint256 timestamp
    );
    
    event ModelDeactivated(
        bytes32 indexed fingerprintHash,
        address indexed owner,
        uint256 timestamp
    );
    
    event ModelReactivated(
        bytes32 indexed fingerprintHash,
        address indexed owner,
        uint256 timestamp
    );
    
    event RegistrationFeeUpdated(
        uint256 oldFee,
        uint256 newFee
    );
    
    // ============================================================
    // ERRORS
    // ============================================================
    
    error ModelAlreadyRegistered(bytes32 fingerprintHash);
    error ModelNotRegistered(bytes32 fingerprintHash);
    error NotModelOwner(bytes32 fingerprintHash, address caller);
    error InsufficientFee(uint256 required, uint256 provided);
    error InvalidFingerprint();
    error ContractPaused();
    error NotAdmin();
    error TransferToZeroAddress();
    
    // ============================================================
    // MODIFIERS
    // ============================================================
    
    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAdmin();
        _;
    }
    
    modifier whenNotPaused() {
        if (paused) revert ContractPaused();
        _;
    }
    
    modifier onlyModelOwner(bytes32 fingerprintHash) {
        if (models[fingerprintHash].owner != msg.sender) {
            revert NotModelOwner(fingerprintHash, msg.sender);
        }
        _;
    }
    
    // ============================================================
    // CONSTRUCTOR
    // ============================================================
    
    constructor(uint256 _registrationFee) {
        admin = msg.sender;
        registrationFee = _registrationFee;
        paused = false;
        totalModels = 0;
    }
    
    // ============================================================
    // REGISTRATION FUNCTIONS
    // ============================================================
    
    /**
     * @notice Register a new model fingerprint
     * @param fingerprintHash The SHA256 hash of the TDA fingerprint
     * @param modelName Human-readable name for the model
     * @param modelVersion Version string (e.g., "1.0.0")
     * @param metadata Additional metadata (e.g., IPFS hash of full fingerprint)
     */
    function registerModel(
        bytes32 fingerprintHash,
        string calldata modelName,
        string calldata modelVersion,
        bytes calldata metadata
    ) external payable whenNotPaused {
        // Validate fingerprint
        if (fingerprintHash == bytes32(0)) revert InvalidFingerprint();
        
        // Check if already registered
        if (models[fingerprintHash].isActive) {
            revert ModelAlreadyRegistered(fingerprintHash);
        }
        
        // Check fee
        if (msg.value < registrationFee) {
            revert InsufficientFee(registrationFee, msg.value);
        }
        
        // Create registration
        models[fingerprintHash] = ModelFingerprint({
            fingerprintHash: fingerprintHash,
            owner: msg.sender,
            registrationTime: block.timestamp,
            modelName: modelName,
            modelVersion: modelVersion,
            metadata: metadata,
            isActive: true
        });
        
        // Track owner's models
        ownerModels[msg.sender].push(fingerprintHash);
        
        // Increment counter
        totalModels++;
        
        // Emit event
        emit ModelRegistered(
            fingerprintHash,
            msg.sender,
            modelName,
            modelVersion,
            block.timestamp
        );
        
        // Refund excess payment
        if (msg.value > registrationFee) {
            payable(msg.sender).transfer(msg.value - registrationFee);
        }
    }
    
    /**
     * @notice Batch register multiple models
     * @param fingerprintHashes Array of fingerprint hashes
     * @param modelNames Array of model names
     * @param modelVersions Array of version strings
     * @param metadataArray Array of metadata
     */
    function batchRegisterModels(
        bytes32[] calldata fingerprintHashes,
        string[] calldata modelNames,
        string[] calldata modelVersions,
        bytes[] calldata metadataArray
    ) external payable whenNotPaused {
        uint256 count = fingerprintHashes.length;
        require(
            count == modelNames.length && 
            count == modelVersions.length && 
            count == metadataArray.length,
            "Array length mismatch"
        );
        
        uint256 totalFee = registrationFee * count;
        if (msg.value < totalFee) {
            revert InsufficientFee(totalFee, msg.value);
        }
        
        for (uint256 i = 0; i < count; i++) {
            bytes32 hash = fingerprintHashes[i];
            
            if (hash == bytes32(0)) continue;
            if (models[hash].isActive) continue;
            
            models[hash] = ModelFingerprint({
                fingerprintHash: hash,
                owner: msg.sender,
                registrationTime: block.timestamp,
                modelName: modelNames[i],
                modelVersion: modelVersions[i],
                metadata: metadataArray[i],
                isActive: true
            });
            
            ownerModels[msg.sender].push(hash);
            totalModels++;
            
            emit ModelRegistered(
                hash,
                msg.sender,
                modelNames[i],
                modelVersions[i],
                block.timestamp
            );
        }
        
        // Refund excess
        if (msg.value > totalFee) {
            payable(msg.sender).transfer(msg.value - totalFee);
        }
    }
    
    // ============================================================
    // VERIFICATION FUNCTIONS
    // ============================================================
    
    /**
     * @notice Verify if a fingerprint is registered
     * @param fingerprintHash The fingerprint hash to verify
     * @return result Verification result struct
     */
    function verifyModel(bytes32 fingerprintHash) 
        external 
        view 
        returns (VerificationResult memory result) 
    {
        ModelFingerprint storage model = models[fingerprintHash];
        
        result.isRegistered = model.isActive;
        result.owner = model.owner;
        result.registrationTime = model.registrationTime;
        result.modelName = model.modelName;
    }
    
    /**
     * @notice Check if a fingerprint is registered (simple boolean)
     * @param fingerprintHash The fingerprint hash to check
     * @return True if registered and active
     */
    function isRegistered(bytes32 fingerprintHash) external view returns (bool) {
        return models[fingerprintHash].isActive;
    }
    
    /**
     * @notice Get the owner of a registered model
     * @param fingerprintHash The fingerprint hash
     * @return Owner address (zero if not registered)
     */
    function getOwner(bytes32 fingerprintHash) external view returns (address) {
        return models[fingerprintHash].owner;
    }
    
    /**
     * @notice Get full model details
     * @param fingerprintHash The fingerprint hash
     * @return model The full ModelFingerprint struct
     */
    function getModel(bytes32 fingerprintHash) 
        external 
        view 
        returns (ModelFingerprint memory model) 
    {
        return models[fingerprintHash];
    }
    
    // ============================================================
    // OWNERSHIP FUNCTIONS
    // ============================================================
    
    /**
     * @notice Transfer model ownership to a new address
     * @param fingerprintHash The fingerprint hash of the model
     * @param newOwner The new owner address
     */
    function transferModelOwnership(bytes32 fingerprintHash, address newOwner) 
        external 
        whenNotPaused 
        onlyModelOwner(fingerprintHash) 
    {
        if (newOwner == address(0)) revert TransferToZeroAddress();
        
        address previousOwner = models[fingerprintHash].owner;
        models[fingerprintHash].owner = newOwner;
        
        // Update owner tracking
        ownerModels[newOwner].push(fingerprintHash);
        
        emit ModelTransferred(
            fingerprintHash,
            previousOwner,
            newOwner,
            block.timestamp
        );
    }
    
    /**
     * @notice Deactivate a model registration
     * @param fingerprintHash The fingerprint hash of the model
     */
    function deactivateModel(bytes32 fingerprintHash) 
        external 
        whenNotPaused 
        onlyModelOwner(fingerprintHash) 
    {
        models[fingerprintHash].isActive = false;
        totalModels--;
        
        emit ModelDeactivated(fingerprintHash, msg.sender, block.timestamp);
    }
    
    /**
     * @notice Reactivate a previously deactivated model
     * @param fingerprintHash The fingerprint hash of the model
     */
    function reactivateModel(bytes32 fingerprintHash) 
        external 
        whenNotPaused 
        onlyModelOwner(fingerprintHash) 
    {
        require(!models[fingerprintHash].isActive, "Model already active");
        
        models[fingerprintHash].isActive = true;
        totalModels++;
        
        emit ModelReactivated(fingerprintHash, msg.sender, block.timestamp);
    }
    
    // ============================================================
    // QUERY FUNCTIONS
    // ============================================================
    
    /**
     * @notice Get all models owned by an address
     * @param owner The owner address
     * @return Array of fingerprint hashes
     */
    function getModelsByOwner(address owner) 
        external 
        view 
        returns (bytes32[] memory) 
    {
        return ownerModels[owner];
    }
    
    /**
     * @notice Get the count of models owned by an address
     * @param owner The owner address
     * @return Count of models
     */
    function getModelCountByOwner(address owner) external view returns (uint256) {
        return ownerModels[owner].length;
    }
    
    // ============================================================
    // ADMIN FUNCTIONS
    // ============================================================
    
    /**
     * @notice Update the registration fee
     * @param newFee The new fee amount in wei
     */
    function setRegistrationFee(uint256 newFee) external onlyAdmin {
        uint256 oldFee = registrationFee;
        registrationFee = newFee;
        emit RegistrationFeeUpdated(oldFee, newFee);
    }
    
    /**
     * @notice Pause the contract
     */
    function pause() external onlyAdmin {
        paused = true;
    }
    
    /**
     * @notice Unpause the contract
     */
    function unpause() external onlyAdmin {
        paused = false;
    }
    
    /**
     * @notice Transfer admin role
     * @param newAdmin The new admin address
     */
    function transferAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "Invalid admin address");
        admin = newAdmin;
    }
    
    /**
     * @notice Withdraw accumulated fees
     */
    function withdrawFees() external onlyAdmin {
        uint256 balance = address(this).balance;
        require(balance > 0, "No fees to withdraw");
        payable(admin).transfer(balance);
    }
    
    // ============================================================
    // RECEIVE FUNCTION
    // ============================================================
    
    receive() external payable {}
}
