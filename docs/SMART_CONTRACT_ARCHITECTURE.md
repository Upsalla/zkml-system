# Smart-Contract-Architektur für On-Chain zkML-Verifikation

**Autor**: Manus AI
**Datum**: 26. Januar 2026
**Status**: Entwurf

## 1. Zusammenfassung

Dieses Dokument beschreibt die Architektur der Smart Contracts für die On-Chain-Verifikation von zkML-Proofs. Ziel ist es, ein dezentrales, sicheres und gas-effizientes System auf der Ethereum-Blockchain zu schaffen, das es ermöglicht, die Integrität von Machine-Learning-Inferenzen trustless zu überprüfen. Die Architektur basiert auf der Nutzung der BN254-Pairing-Precompiles von Ethereum und ist für die Verifikation von PLONK-Proofs ausgelegt. Sie umfasst einen zentralen Verifier-Contract, eine Modell-Registry und Hilfsbibliotheken.

## 2. Problemstellung und Ziele

Eine Off-Chain-Verifikation, selbst wenn sie kryptographisch sicher ist, erfordert Vertrauen in die Partei, die die Verifikation durchführt. Für eine echte Dezentralisierung und Trustlessness muss die Verifikation auf einer öffentlichen Blockchain stattfinden.

**Projektziele**:

1.  **Trustless Verifikation**: Jeder Teilnehmer im Netzwerk muss in der Lage sein, einen Proof zu verifizieren, indem er eine Transaktion an einen Smart Contract sendet.
2.  **Gas-Effizienz**: Die Kosten für eine On-Chain-Verifikation müssen so gering wie möglich gehalten werden, um die praktische Anwendbarkeit zu gewährleisten. Das Ziel liegt bei unter 500.000 Gas pro Verifikation.
3.  **Ethereum-Kompatibilität**: Die Smart Contracts müssen in Solidity geschrieben sein und die existierenden Precompiles für BN254-Operationen nutzen.
4.  **Modularität und Erweiterbarkeit**: Die Architektur muss es ermöglichen, neue Modelle (und deren Verification Keys) einfach hinzuzufügen und potenziell auch zukünftige ZK-SNARK-Systeme zu unterstützen.

## 3. On-Chain-Architektur

Die On-Chain-Architektur besteht aus mehreren interagierenden Smart Contracts, die jeweils eine spezifische Aufgabe erfüllen. Dies fördert die Modularität und Sicherheit.

```
┌───────────────────────────┐
│      Application Layer    │
│ (z.B. dApp, Skript)       │
└─────────────┬─────────────┘
              │ 1. submitProof(modelHash, proof, inputs)
              ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│      ZKMLRegistry.sol     │──────▶│      ZKMLVerifier.sol     │
│ (Verwaltet Modelle & VKs) │ 2. getVK(modelHash) │ (Haupt-Verifier-Logik)    │
└─────────────┬─────────────┘       └─────────────┬─────────────┘
              │                                   │ 3. verify(vk, proof, inputs)
              │                                   ▼
              │                         ┌───────────────────────────┐
              │                         │       PairingLib.sol      │
              │                         │ (Wrapper für Precompiles) │
              │                         └─────────────┬─────────────┘
              │                                       │ 4. ecPairing(proofData)
              │                                       ▼
              │                             ┌───────────────────────────┐
              │                             │ Ethereum Precompiles    │
              │                             │ (0x06, 0x07, 0x08)        │
              │                             └───────────────────────────┘
              ▼
┌───────────────────────────┐
│       Event Logs          │
│ (ProofVerified, etc.)     │
└───────────────────────────┘
```

### 3.1 Komponenten

-   **ZKMLVerifier.sol**: Der Kern-Contract, der die `verify`-Funktion enthält. Er nimmt einen Proof und öffentliche Eingaben entgegen und gibt `true` oder `false` zurück. Er ist zustandslos bezüglich der Modelle und delegiert die Speicherung der Verification Keys (VKs) an die Registry.
-   **ZKMLRegistry.sol**: Ein separater Contract, der eine Zuordnung von einem Modell-Hash (z.B. `keccak256` der Modellarchitektur und Gewichte) zu seinem spezifischen PLONK Verification Key speichert. Dies ermöglicht das Hinzufügen neuer Modelle ohne den Verifier-Contract neu deployen zu müssen.
-   **PairingLib.sol**: Eine statische Bibliothek, die die komplexen Aufrufe der Ethereum-Precompiles für `ecAdd`, `ecMul` und `ecPairing` kapselt. Sie stellt eine saubere und lesbare Schnittstelle für den `ZKMLVerifier` bereit.

## 4. Smart Contract Design im Detail

### 4.1 ZKMLVerifier.sol

Dieser Contract enthält die Hauptlogik für die Verifikation. Er ist so konzipiert, dass er möglichst wenig Speicher (Storage) verwendet, um die Gaskosten zu minimieren.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./PairingLib.sol";
import "./ZKMLRegistry.sol";

contract ZKMLVerifier {
    ZKMLRegistry public immutable registry;

    event ProofVerified(bytes32 indexed modelHash, bool indexed isValid);

    constructor(address _registryAddress) {
        registry = ZKMLRegistry(_registryAddress);
    }

    function verify(
        bytes32 modelHash,
        bytes calldata proofData,
        uint256[] calldata publicInputs
    ) external returns (bool) {
        // 1. Lade den Verification Key aus der Registry
        ZKMLRegistry.VerificationKey memory vk = registry.getVerificationKey(modelHash);
        require(vk.alpha[0] != 0, "Model not registered");

        // 2. Dekodiere den Proof
        (PairingLib.G1Point memory a, PairingLib.G2Point memory b, PairingLib.G1Point memory c) = abi.decode(proofData, (PairingLib.G1Point, PairingLib.G2Point, PairingLib.G1Point));

        // 3. Berechne das Commitment zu den öffentlichen Eingaben
        PairingLib.G1Point memory vk_x = computeInputCommitment(vk.ic, publicInputs);

        // 4. Führe den finalen Pairing-Check durch
        // e(A, B) == e(α, β) * e(vk_x, γ) * e(C, δ)
        bool isValid = PairingLib.pairingCheck(
            a, b, vk.alpha, vk.beta, vk_x, vk.gamma, c, vk.delta
        );

        emit ProofVerified(modelHash, isValid);
        return isValid;
    }

    function computeInputCommitment(
        PairingLib.G1Point[] memory ic,
        uint256[] calldata inputs
    ) internal view returns (PairingLib.G1Point memory) {
        // ... Logik zur Berechnung des Input-Commitments via ecAdd und ecMul
    }
}
```

### 4.2 ZKMLRegistry.sol

Dieser Contract dient als upgrade-fähige Datenbank für Verification Keys.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./PairingLib.sol";

contract ZKMLRegistry {
    struct VerificationKey {
        PairingLib.G1Point alpha;
        PairingLib.G2Point beta;
        PairingLib.G2Point gamma;
        PairingLib.G2Point delta;
        PairingLib.G1Point[] ic;
    }

    mapping(bytes32 => VerificationKey) public verificationKeys;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function registerModel(bytes32 modelHash, VerificationKey calldata vk) external {
        require(msg.sender == owner, "Only owner");
        verificationKeys[modelHash] = vk;
    }

    function getVerificationKey(bytes32 modelHash) external view returns (VerificationKey memory) {
        return verificationKeys[modelHash];
    }
}
```

### 4.3 Gas-Analyse und Optimierung

Die Gaskosten sind der kritischste Faktor. Die Hauptkosten entstehen durch den `ecPairing`-Precompile-Aufruf.

| Operation | Gas-Kosten (ca.) | Anzahl | Gesamt-Gas (ca.) |
| :--- | :--- | :--- | :--- |
| `ecPairing` (4 Paare) | `45,000 + 34,000 * 4` | 1 | 181,000 |
| `ecMul` (pro öffentl. Input) | `6,000` | `n` | `6,000 * n` |
| `ecAdd` (pro öffentl. Input) | `150` | `n` | `150 * n` |
| `SLOAD` (für VK) | `2,100` | `~5` | 10,500 |
| Sonstige (Memory, etc.) | - | - | 20,000 |
| **Total (für n=2 Inputs)** | | | **~225,000** |

**Optimierungsstrategien**:

1.  **Minimierung der öffentlichen Eingaben**: Jede öffentliche Eingabe erfordert ein `ecMul` und ein `ecAdd`. Daher sollten nur absolut notwendige Werte (z.B. ein Hash des vollständigen Inputs und der Output) als öffentlich deklariert werden.
2.  **Effiziente Datenstrukturen**: Die `VerificationKey`-Struktur sollte so gepackt wie möglich sein, um die `SLOAD`-Kosten zu minimieren.
3.  **Assembly für Precompiles**: Die Verwendung von Inline-Assembly für die Precompile-Aufrufe ist notwendig und ermöglicht eine präzise Kontrolle über Speicherlayout und Gaskosten.

## 5. Implementierungsplan

| Phase | Aufgabe | Dauer (Wochen) | Deliverable |
| :--- | :--- | :--- | :--- |
| 1 | **Pairing-Bibliothek**: Entwicklung und Test von `PairingLib.sol`. | 1 | Eine getestete Bibliothek, die die BN254-Precompiles kapselt. |
| 2 | **Registry-Contract**: Implementierung und Test von `ZKMLRegistry.sol`. | 0.5 | Ein deploy-fähiger Registry-Contract. |
| 3 | **Verifier-Contract**: Implementierung von `ZKMLVerifier.sol`. | 1.5 | Ein Verifier, der gegen Testvektoren validiert wurde. |
| 4 | **Deployment & Test-Skripte**: Erstellung von Skripten für Deployment, VK-Export und Proof-Submission. | 0.5 | Ein Satz von Hardhat/Foundry-Skripten. |
| 5 | **Gas-Optimierung & Audit-Vorbereitung**: Systematisches Review und Optimierung des Codes. | 0.5 | Ein finaler, optimierter Satz von Smart Contracts. |

**Gesamtdauer**: 4 Wochen

## 6. Risiken

- **Gas-Limit-Überschreitung**: Bei sehr vielen öffentlichen Eingaben oder einem komplexeren Verifikationsprozess könnten die Gaskosten das Block-Gas-Limit überschreiten. **Mitigation**: Strikte Begrenzung der Anzahl öffentlicher Eingaben; Prüfung von L2-Lösungen (Rollups), die höhere Gas-Limits haben.
- **Fehler bei der Precompile-Nutzung**: Die Schnittstelle zu den Precompiles ist auf niedriger Ebene und fehleranfällig. **Mitigation**: Verwendung von etablierten und geprüften Bibliotheken (z.B. von OpenZeppelin oder anderen ZK-Projekten) als Referenz.
- **Upgradeability**: Ein Fehler im `ZKMLVerifier` könnte ein erneutes Deployment und die Migration aller Daten erfordern. **Mitigation**: Verwendung eines Proxy-Patterns (z.B. UUPS), um den Verifier-Contract upgrade-fähig zu machen, während die Registry als persistenter Speicher dient.

## 7. Referenzen

[1] Ethereum Foundation. *EIP-197: Precompiled contracts for optimal ate pairing check on the elliptic curve alt_bn128*. [eips.ethereum.org](https://eips.ethereum.org/EIPS/eip-197)
[2] Solidity Documentation. *Precompiled Contracts*. [docs.soliditylang.org](https://docs.soliditylang.org/en/latest/contracts.html#precompiled-contracts)
