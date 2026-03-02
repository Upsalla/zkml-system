"""
Witness Management für zkML
===========================

Der Witness ist die "geheime Eingabe" des Provers. Er enthält alle Werte,
die in der Berechnung vorkommen – sowohl öffentliche als auch private.

Struktur des Witness:
- Index 0: Konstante 1 (immer)
- Index 1 bis num_public: Öffentliche Eingaben
- Rest: Private Werte (Zwischenergebnisse, Gewichte, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class WitnessVariable:
    """Metadaten für eine Witness-Variable."""
    index: int
    name: str
    value: int
    is_public: bool = False
    layer: Optional[int] = None  # Für Neural Network: welcher Layer
    neuron: Optional[int] = None  # Für Neural Network: welches Neuron
    var_type: str = "generic"  # "input", "weight", "bias", "activation", "output"


class Witness:
    """
    Verwaltet den Witness für ein ZK-Proof-System.
    
    Bietet Methoden zum Hinzufügen, Abrufen und Analysieren von Witness-Werten.
    """
    
    def __init__(self, prime: int):
        """
        Initialisiert einen neuen Witness.
        
        Args:
            prime: Das Primfeld für die Arithmetik
        """
        self.prime = prime
        self.values: List[int] = [1]  # Index 0 ist immer 1
        self.metadata: Dict[int, WitnessVariable] = {
            0: WitnessVariable(0, "ONE", 1, is_public=True, var_type="constant")
        }
        self.num_public = 1  # Die Konstante 1 zählt als öffentlich
        
        # Tracking für Sparsity-Analyse
        self.activation_indices: Dict[int, List[int]] = {}  # layer -> [indices]
        self.zero_activations: List[int] = []  # Indices von Aktivierungen = 0
    
    def allocate(
        self,
        value: int,
        name: str = "",
        is_public: bool = False,
        layer: Optional[int] = None,
        neuron: Optional[int] = None,
        var_type: str = "generic"
    ) -> int:
        """
        Allokiert eine neue Variable im Witness.
        
        Returns:
            Der Index der neuen Variable
        """
        idx = len(self.values)
        normalized_value = value % self.prime
        
        self.values.append(normalized_value)
        self.metadata[idx] = WitnessVariable(
            index=idx,
            name=name,
            value=normalized_value,
            is_public=is_public,
            layer=layer,
            neuron=neuron,
            var_type=var_type
        )
        
        if is_public:
            self.num_public += 1
        
        # Tracking für Aktivierungen
        if var_type == "activation":
            if layer not in self.activation_indices:
                self.activation_indices[layer] = []
            self.activation_indices[layer].append(idx)
            
            if normalized_value == 0:
                self.zero_activations.append(idx)
        
        return idx
    
    def get(self, idx: int) -> int:
        """Gibt den Wert an einem Index zurück."""
        return self.values[idx]
    
    def set(self, idx: int, value: int) -> None:
        """Setzt den Wert an einem Index."""
        self.values[idx] = value % self.prime
        if idx in self.metadata:
            self.metadata[idx].value = self.values[idx]
    
    def get_public(self) -> List[int]:
        """Gibt alle öffentlichen Werte zurück."""
        return [
            self.values[idx] 
            for idx, meta in self.metadata.items() 
            if meta.is_public
        ]
    
    def get_private(self) -> List[int]:
        """Gibt alle privaten Werte zurück."""
        return [
            self.values[idx] 
            for idx, meta in self.metadata.items() 
            if not meta.is_public
        ]
    
    def as_list(self) -> List[int]:
        """Gibt den gesamten Witness als Liste zurück."""
        return self.values.copy()
    
    def size(self) -> int:
        """Gibt die Größe des Witness zurück."""
        return len(self.values)
    
    def get_sparsity_info(self) -> Dict:
        """
        Analysiert die Sparsity des Witness.
        
        Returns:
            Dictionary mit Sparsity-Statistiken
        """
        total_activations = sum(len(indices) for indices in self.activation_indices.values())
        zero_count = len(self.zero_activations)
        
        return {
            "total_activations": total_activations,
            "zero_activations": zero_count,
            "sparsity": zero_count / total_activations if total_activations > 0 else 0,
            "zero_indices": self.zero_activations.copy(),
            "by_layer": {
                layer: {
                    "total": len(indices),
                    "zeros": sum(1 for idx in indices if self.values[idx] == 0)
                }
                for layer, indices in self.activation_indices.items()
            }
        }
    
    def get_variables_by_type(self, var_type: str) -> List[WitnessVariable]:
        """Gibt alle Variablen eines bestimmten Typs zurück."""
        return [meta for meta in self.metadata.values() if meta.var_type == var_type]
    
    def get_variables_by_layer(self, layer: int) -> List[WitnessVariable]:
        """Gibt alle Variablen eines bestimmten Layers zurück."""
        return [meta for meta in self.metadata.values() if meta.layer == layer]
    
    def to_json(self) -> str:
        """Serialisiert den Witness zu JSON."""
        return json.dumps({
            "prime": self.prime,
            "values": self.values,
            "num_public": self.num_public,
            "metadata": {
                str(idx): {
                    "name": meta.name,
                    "is_public": meta.is_public,
                    "layer": meta.layer,
                    "neuron": meta.neuron,
                    "var_type": meta.var_type
                }
                for idx, meta in self.metadata.items()
            }
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Witness':
        """Deserialisiert einen Witness aus JSON."""
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
                var_type=meta_data["var_type"]
            )
        
        return witness
    
    def __repr__(self) -> str:
        return f"Witness(size={self.size()}, public={self.num_public}, private={self.size() - self.num_public})"


class WitnessBuilder:
    """
    Builder für die strukturierte Erstellung von Witnesses.
    
    Besonders nützlich für Neural Networks, wo Variablen nach Layer
    und Typ organisiert werden müssen.
    """
    
    def __init__(self, prime: int):
        self.witness = Witness(prime)
        self.current_layer = 0
    
    def set_layer(self, layer: int) -> 'WitnessBuilder':
        """Setzt den aktuellen Layer für nachfolgende Allokationen."""
        self.current_layer = layer
        return self
    
    def add_input(self, value: int, name: str = "", public: bool = True) -> int:
        """Fügt eine Eingabe hinzu."""
        return self.witness.allocate(
            value, name, is_public=public, 
            layer=0, var_type="input"
        )
    
    def add_weight(self, value: int, name: str = "") -> int:
        """Fügt ein Gewicht hinzu."""
        return self.witness.allocate(
            value, name, is_public=False,
            layer=self.current_layer, var_type="weight"
        )
    
    def add_bias(self, value: int, name: str = "") -> int:
        """Fügt einen Bias hinzu."""
        return self.witness.allocate(
            value, name, is_public=False,
            layer=self.current_layer, var_type="bias"
        )
    
    def add_activation(self, value: int, neuron: int, name: str = "") -> int:
        """Fügt eine Aktivierung hinzu."""
        return self.witness.allocate(
            value, name, is_public=False,
            layer=self.current_layer, neuron=neuron, var_type="activation"
        )
    
    def add_intermediate(self, value: int, name: str = "") -> int:
        """Fügt einen Zwischenwert hinzu."""
        return self.witness.allocate(
            value, name, is_public=False,
            layer=self.current_layer, var_type="intermediate"
        )
    
    def add_output(self, value: int, name: str = "", public: bool = True) -> int:
        """Fügt eine Ausgabe hinzu."""
        return self.witness.allocate(
            value, name, is_public=public,
            layer=self.current_layer, var_type="output"
        )
    
    def build(self) -> Witness:
        """Gibt den fertigen Witness zurück."""
        return self.witness


# Tests
if __name__ == "__main__":
    print("=== Witness Tests ===\n")
    
    # Einfacher Test
    witness = Witness(prime=101)
    
    x_idx = witness.allocate(3, "x", is_public=True, var_type="input")
    y_idx = witness.allocate(4, "y", is_public=True, var_type="input")
    z_idx = witness.allocate(12, "z", is_public=False, var_type="intermediate")
    
    print(f"Witness: {witness}")
    print(f"Values: {witness.as_list()}")
    print(f"Public: {witness.get_public()}")
    print(f"Private: {witness.get_private()}")
    
    # Neural Network Test
    print("\n--- Neural Network Witness ---")
    
    builder = WitnessBuilder(prime=101)
    
    # Eingaben
    builder.set_layer(0)
    in1 = builder.add_input(5, "input_0")
    in2 = builder.add_input(3, "input_1")
    
    # Layer 1
    builder.set_layer(1)
    w1 = builder.add_weight(2, "w1")
    w2 = builder.add_weight(3, "w2")
    b1 = builder.add_bias(1, "b1")
    
    # Aktivierungen (simuliert: eine ist 0)
    act1 = builder.add_activation(14, neuron=0, name="act_0")  # 5*2 + 3*3 + 1 = 14 (aktiv)
    act2 = builder.add_activation(0, neuron=1, name="act_1")   # Angenommen 0 (inaktiv)
    
    nn_witness = builder.build()
    
    print(f"NN Witness: {nn_witness}")
    print(f"Sparsity Info: {nn_witness.get_sparsity_info()}")
    print(f"Weights: {[v.name for v in nn_witness.get_variables_by_type('weight')]}")
    print(f"Layer 1 vars: {[v.name for v in nn_witness.get_variables_by_layer(1)]}")
