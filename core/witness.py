"""
Witness Management for zkML
===========================

The witness is the prover's "secret input". It contains all values
that occur in the computation — both public and private.

Witness structure:
- Index 0: Constant 1 (always)
- Index 1 to num_public: Public inputs
- Remainder: Private values (intermediate results, weights, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class WitnessVariable:
    """Metadata for a witness variable."""
    index: int
    name: str
    value: int
    is_public: bool = False
    layer: Optional[int] = None   # For neural networks: which layer
    neuron: Optional[int] = None  # For neural networks: which neuron
    var_type: str = "generic"     # "input", "weight", "bias", "activation", "output"


class Witness:
    """
    Manages the witness for a ZK proof system.

    Provides methods for adding, retrieving, and analyzing witness values.
    """

    def __init__(self, prime: int):
        """
        Initialize a new witness.

        Args:
            prime: The prime field for arithmetic
        """
        self.prime = prime
        self.values: List[int] = [1]  # Index 0 is always 1
        self.metadata: Dict[int, WitnessVariable] = {
            0: WitnessVariable(0, "ONE", 1, is_public=True, var_type="constant")
        }
        self.num_public = 1  # The constant 1 counts as public
        self.activation_indices: Dict[int, List[int]] = {}
        self.zero_activations: List[int] = []

    def allocate(
        self,
        value: int,
        name: str = "",
        is_public: bool = False,
        layer: Optional[int] = None,
        neuron: Optional[int] = None,
        var_type: str = "generic",
    ) -> int:
        """
        Allocate a new witness variable.

        Args:
            value: The value for the variable
            name: Optional name
            is_public: Whether the variable is public
            layer: Optional layer number
            neuron: Optional neuron number
            var_type: Variable type

        Returns:
            Index of the new variable
        """
        normalized_value = value % self.prime
        idx = len(self.values)
        self.values.append(normalized_value)

        if not name:
            name = f"w_{idx}"

        self.metadata[idx] = WitnessVariable(
            index=idx,
            name=name,
            value=normalized_value,
            is_public=is_public,
            layer=layer,
            neuron=neuron,
            var_type=var_type,
        )

        if is_public:
            self.num_public += 1

        # Track activations for sparsity analysis
        if var_type == "activation" and layer is not None:
            if layer not in self.activation_indices:
                self.activation_indices[layer] = []
            self.activation_indices[layer].append(idx)

            if normalized_value == 0:
                self.zero_activations.append(idx)

        return idx

    def get(self, idx: int) -> int:
        """Return the value at a given index."""
        return self.values[idx]

    def set(self, idx: int, value: int) -> None:
        """Set the value at a given index."""
        self.values[idx] = value % self.prime
        if idx in self.metadata:
            self.metadata[idx].value = self.values[idx]

    def get_public(self) -> List[int]:
        """Return all public values."""
        return [
            self.values[idx]
            for idx, meta in self.metadata.items()
            if meta.is_public
        ]

    def get_private(self) -> List[int]:
        """Return all private values."""
        return [
            self.values[idx]
            for idx, meta in self.metadata.items()
            if not meta.is_public
        ]

    def as_list(self) -> List[int]:
        """Return the entire witness as a list."""
        return self.values.copy()

    def size(self) -> int:
        """Return the size of the witness."""
        return len(self.values)

    def get_sparsity_info(self) -> Dict:
        """
        Analyze the sparsity of the witness.

        Returns:
            Dictionary with sparsity statistics
        """
        total_activations = sum(
            len(indices) for indices in self.activation_indices.values()
        )
        zero_count = len(self.zero_activations)

        return {
            "total_activations": total_activations,
            "zero_activations": zero_count,
            "sparsity": zero_count / total_activations if total_activations > 0 else 0,
            "zero_indices": self.zero_activations.copy(),
            "by_layer": {
                layer: {
                    "total": len(indices),
                    "zeros": sum(1 for idx in indices if self.values[idx] == 0),
                }
                for layer, indices in self.activation_indices.items()
            },
        }

    def get_variables_by_type(self, var_type: str) -> List[WitnessVariable]:
        """Return all variables of a given type."""
        return [meta for meta in self.metadata.values() if meta.var_type == var_type]

    def get_variables_by_layer(self, layer: int) -> List[WitnessVariable]:
        """Return all variables of a given layer."""
        return [meta for meta in self.metadata.values() if meta.layer == layer]

    def to_json(self) -> str:
        """Serialize the witness to JSON."""
        return json.dumps(
            {
                "prime": self.prime,
                "values": self.values,
                "num_public": self.num_public,
                "metadata": {
                    str(idx): {
                        "name": meta.name,
                        "is_public": meta.is_public,
                        "layer": meta.layer,
                        "neuron": meta.neuron,
                        "var_type": meta.var_type,
                    }
                    for idx, meta in self.metadata.items()
                },
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Witness":
        """Deserialize a witness from JSON."""
        data = json.loads(json_str)
        witness = cls(data["prime"])
        witness.values = data["values"]
        witness.num_public = data["num_public"]

        for idx_str, meta_data in data["metadata"].items():
            idx = int(idx_str)
            witness.metadata[idx] = WitnessVariable(
                index=idx,
                name=meta_data["name"],
                value=witness.values[idx],
                is_public=meta_data["is_public"],
                layer=meta_data["layer"],
                neuron=meta_data["neuron"],
                var_type=meta_data["var_type"],
            )

        return witness

    def __repr__(self) -> str:
        return (
            f"Witness(size={self.size()}, public={self.num_public}, "
            f"private={self.size() - self.num_public})"
        )


class WitnessBuilder:
    """
    Builder for structured witness creation.

    Particularly useful for neural networks where variables need to be
    organized by layer and type.
    """

    def __init__(self, prime: int):
        self.witness = Witness(prime)
        self.current_layer = 0

    def set_layer(self, layer: int) -> "WitnessBuilder":
        """Set the current layer for subsequent allocations."""
        self.current_layer = layer
        return self

    def add_input(self, value: int, name: str = "", public: bool = True) -> int:
        """Add an input."""
        return self.witness.allocate(
            value, name, is_public=public, layer=0, var_type="input"
        )

    def add_weight(self, value: int, name: str = "") -> int:
        """Add a weight."""
        return self.witness.allocate(
            value, name, is_public=False, layer=self.current_layer, var_type="weight"
        )

    def add_bias(self, value: int, name: str = "") -> int:
        """Add a bias."""
        return self.witness.allocate(
            value, name, is_public=False, layer=self.current_layer, var_type="bias"
        )

    def add_activation(self, value: int, neuron: int, name: str = "") -> int:
        """Add an activation."""
        return self.witness.allocate(
            value,
            name,
            is_public=False,
            layer=self.current_layer,
            neuron=neuron,
            var_type="activation",
        )

    def add_intermediate(self, value: int, name: str = "") -> int:
        """Add an intermediate value."""
        return self.witness.allocate(
            value,
            name,
            is_public=False,
            layer=self.current_layer,
            var_type="intermediate",
        )

    def add_output(self, value: int, name: str = "", public: bool = True) -> int:
        """Add an output."""
        return self.witness.allocate(
            value,
            name,
            is_public=public,
            layer=self.current_layer,
            var_type="output",
        )

    def build(self) -> Witness:
        """Return the completed witness."""
        return self.witness
