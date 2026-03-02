"""
PLONK Circuit Compiler für zkML

Dieses Modul übersetzt neuronale Netze in optimierte PLONK-Circuits.

Kernfunktionen:
1. Network → PLONK Circuit Konvertierung
2. GELU-Aktivierung als effiziente Polynom-Gates
3. Sparse-Optimierung: Inaktive Neuronen werden übersprungen

Die Integration erfolgt auf Circuit-Ebene, nicht auf Proof-Ebene.
Das bedeutet: Der Circuit selbst ist bereits optimiert, bevor der Prover läuft.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zkml_system.crypto.bn254.field import Fr
from zkml_system.plonk.polynomial import Polynomial


class GateType(Enum):
    """Typen von PLONK-Gates."""
    ADD = "add"           # a + b = c
    MUL = "mul"           # a * b = c
    CONST = "const"       # a = constant
    ZERO = "zero"         # a = 0 (für Sparse-Neuronen)
    GELU_SQUARE = "gelu_sq"    # x² für GELU
    GELU_CUBIC = "gelu_cub"    # x³ für GELU
    GELU_OUTPUT = "gelu_out"   # Finale GELU-Berechnung
    RELU_SIGN = "relu_sign"    # Boolean constraint for ReLU sign bit


@dataclass
class Wire:
    """Ein Draht im Circuit."""
    index: int
    value: Optional[Fr] = None
    is_public: bool = False
    name: str = ""


@dataclass
class Gate:
    """Ein Gate im PLONK-Circuit."""
    gate_type: GateType
    left: int       # Index des linken Drahts
    right: int      # Index des rechten Drahts
    output: int     # Index des Ausgabe-Drahts
    
    # Selektoren (für PLONK-Gleichung: q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0)
    q_L: Fr = field(default_factory=lambda: Fr.zero())
    q_R: Fr = field(default_factory=lambda: Fr.zero())
    q_O: Fr = field(default_factory=lambda: Fr.zero())
    q_M: Fr = field(default_factory=lambda: Fr.zero())
    q_C: Fr = field(default_factory=lambda: Fr.zero())
    
    # Metadaten
    layer_idx: int = -1
    neuron_idx: int = -1
    is_sparse_zero: bool = False


@dataclass
class CompiledCircuit:
    """Ein kompilierter PLONK-Circuit."""
    gates: List[Gate]
    wires: List[Wire]
    num_public_inputs: int
    num_public_outputs: int
    
    # Statistiken
    total_gates: int = 0
    sparse_gates: int = 0
    gelu_gates: int = 0
    mul_gates: int = 0
    add_gates: int = 0
    
    # Sparse-Info
    sparsity_map: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_gates = len(self.gates)
        self.sparse_gates = sum(1 for g in self.gates if g.is_sparse_zero)
        self.gelu_gates = sum(1 for g in self.gates if g.gate_type in 
                             [GateType.GELU_SQUARE, GateType.GELU_CUBIC, GateType.GELU_OUTPUT])
        self.mul_gates = sum(1 for g in self.gates if g.gate_type == GateType.MUL)
        self.add_gates = sum(1 for g in self.gates if g.gate_type == GateType.ADD)
    
    def get_selectors(self) -> Dict[str, List[Fr]]:
        """Extrahiert die Selektor-Polynome für PLONK."""
        n = len(self.gates)
        return {
            'q_L': [g.q_L for g in self.gates],
            'q_R': [g.q_R for g in self.gates],
            'q_O': [g.q_O for g in self.gates],
            'q_M': [g.q_M for g in self.gates],
            'q_C': [g.q_C for g in self.gates],
        }
    
    def get_wire_assignments(self) -> Tuple[List[Fr], List[Fr], List[Fr]]:
        """Extrahiert die Wire-Zuweisungen (a, b, c) für jeden Gate."""
        a_wires = []
        b_wires = []
        c_wires = []
        
        for gate in self.gates:
            a_wires.append(self.wires[gate.left].value or Fr.zero())
            b_wires.append(self.wires[gate.right].value or Fr.zero())
            c_wires.append(self.wires[gate.output].value or Fr.zero())
        
        return a_wires, b_wires, c_wires
    
    def stats_summary(self) -> str:
        """Gibt eine Zusammenfassung der Circuit-Statistiken zurück."""
        sparse_pct = (self.sparse_gates / self.total_gates * 100) if self.total_gates > 0 else 0
        return (
            f"CompiledCircuit(\n"
            f"  total_gates: {self.total_gates}\n"
            f"  sparse_gates: {self.sparse_gates} ({sparse_pct:.1f}%)\n"
            f"  gelu_gates: {self.gelu_gates}\n"
            f"  mul_gates: {self.mul_gates}\n"
            f"  add_gates: {self.add_gates}\n"
            f"  wires: {len(self.wires)}\n"
            f")"
        )


class CircuitCompiler:
    """
    Kompiliert neuronale Netze in optimierte PLONK-Circuits.
    
    Optimierungen:
    1. GELU-Aktivierung: 3 Gates statt 255 (für Bit-Dekomposition bei ReLU)
    2. Sparse-Optimierung: Inaktive Neuronen → 1 Zero-Gate statt voller Berechnung
    """
    
    def __init__(self, use_sparse: bool = True, use_gelu: bool = True):
        """
        Args:
            use_sparse: Aktiviert Sparse-Optimierung (überspringt inaktive Neuronen)
            use_gelu: Verwendet GELU-Gates statt ReLU-Bit-Dekomposition
        """
        self.use_sparse = use_sparse
        self.use_gelu = use_gelu
        
        self.gates: List[Gate] = []
        self.wires: List[Wire] = []
        self.wire_counter = 0
        
        # Tracking für Sparse-Optimierung
        self.sparsity_map: Dict[Tuple[int, int], bool] = {}  # (layer, neuron) -> is_active
        self.activation_values: Dict[Tuple[int, int], Fr] = {}  # (layer, neuron) -> value
    
    def _new_wire(self, name: str = "", is_public: bool = False) -> int:
        """Erstellt einen neuen Draht und gibt seinen Index zurück."""
        wire = Wire(index=self.wire_counter, name=name, is_public=is_public)
        self.wires.append(wire)
        self.wire_counter += 1
        return wire.index
    
    def _set_wire_value(self, wire_idx: int, value: Fr):
        """Setzt den Wert eines Drahts."""
        self.wires[wire_idx].value = value
    
    def _const_wire(self, value: Fr, layer_idx: int = -1, neuron_idx: int = -1) -> int:
        """Create a wire constrained to a constant value.
        
        Unlike _set_wire_value alone, this also adds a CONST gate
        that cryptographically binds the wire to the given value.
        Without this, a malicious prover can set arbitrary values.
        """
        wire = self._new_wire(f"const_{value.value if hasattr(value, 'value') else id(value)}")
        self._set_wire_value(wire, value)
        self._add_const_gate(wire, value, layer_idx, neuron_idx)
        return wire
    
    def _add_gate(self, gate: Gate):
        """Fügt ein Gate zum Circuit hinzu."""
        self.gates.append(gate)
    
    def _add_mul_gate(
        self, 
        left: int, 
        right: int, 
        output: int,
        layer_idx: int = -1,
        neuron_idx: int = -1
    ) -> Gate:
        """
        Fügt ein Multiplikations-Gate hinzu: left * right = output
        
        PLONK-Gleichung: q_M * a * b - q_O * c = 0
        → q_M = 1, q_O = -1
        """
        gate = Gate(
            gate_type=GateType.MUL,
            left=left,
            right=right,
            output=output,
            q_M=Fr.one(),
            q_O=Fr(Fr.MODULUS - 1),  # -1
            layer_idx=layer_idx,
            neuron_idx=neuron_idx
        )
        self._add_gate(gate)
        return gate
    
    def _add_add_gate(
        self,
        left: int,
        right: int,
        output: int,
        layer_idx: int = -1,
        neuron_idx: int = -1
    ) -> Gate:
        """
        Fügt ein Additions-Gate hinzu: left + right = output
        
        PLONK-Gleichung: q_L * a + q_R * b - q_O * c = 0
        → q_L = 1, q_R = 1, q_O = -1
        """
        gate = Gate(
            gate_type=GateType.ADD,
            left=left,
            right=right,
            output=output,
            q_L=Fr.one(),
            q_R=Fr.one(),
            q_O=Fr(Fr.MODULUS - 1),  # -1
            layer_idx=layer_idx,
            neuron_idx=neuron_idx
        )
        self._add_gate(gate)
        return gate
    
    def _add_const_gate(
        self,
        wire: int,
        constant: Fr,
        layer_idx: int = -1,
        neuron_idx: int = -1
    ) -> Gate:
        """
        Fügt ein Konstanten-Gate hinzu: wire = constant
        
        PLONK-Gleichung: q_L * a + q_C = 0
        → q_L = 1, q_C = -constant
        """
        gate = Gate(
            gate_type=GateType.CONST,
            left=wire,
            right=wire,  # Dummy
            output=wire,  # Dummy
            q_L=Fr.one(),
            q_C=Fr.zero() - constant,  # -constant (proper field negation)
            layer_idx=layer_idx,
            neuron_idx=neuron_idx
        )
        self._add_gate(gate)
        return gate
    
    def _add_zero_gate(
        self,
        wire: int,
        layer_idx: int = -1,
        neuron_idx: int = -1
    ) -> Gate:
        """
        Fügt ein Zero-Gate hinzu: wire = 0
        
        Dies ist die Sparse-Optimierung: Ein inaktives Neuron wird mit
        einem einzigen Gate bewiesen, statt der vollen Berechnung.
        
        PLONK-Gleichung: q_L * a = 0
        → q_L = 1, und a muss 0 sein
        """
        gate = Gate(
            gate_type=GateType.ZERO,
            left=wire,
            right=wire,  # Dummy
            output=wire,  # Dummy
            q_L=Fr.one(),
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
            is_sparse_zero=True
        )
        self._add_gate(gate)
        return gate
    
    def _compile_gelu_activation(
        self,
        input_wire: int,
        output_wire: int,
        layer_idx: int,
        neuron_idx: int
    ) -> List[Gate]:
        """
        Compile GELU activation into PLONK gates.

        GELU(x) ≈ 0.5·x + 0.398942·x² + 0.0535161·x³

        Uses fixed-point arithmetic with SCALE = 10^6:
          a_coeff = int(0.5       * SCALE) = 500_000
          b_coeff = int(0.398942  * SCALE) = 398_942
          c_coeff = int(0.0535161 * SCALE) =  53_516

        Gate layout (5 gates per neuron):
          Gate 1 (GELU_SQUARE): x * x = sq
          Gate 2 (GELU_CUBIC):  sq * x = cub
          Gate 3 (GELU_OUTPUT): a_coeff * x - partial1 = 0  (scaled linear term)
          Gate 4 (GELU_OUTPUT): partial1 + b_coeff * sq - partial2 = 0
          Gate 5 (GELU_OUTPUT): partial2 + c_coeff * cub - output = 0

        The output is rescaled back to base units (divided by SCALE)
        so downstream layers receive unscaled values.
        """
        SCALE = 1_000_000
        a_coeff = Fr(500_000)       # 0.5 * SCALE
        b_coeff = Fr(398_942)       # 0.398942 * SCALE
        c_coeff = Fr(53_516)        # 0.0535161 * SCALE
        SCALE_INV = Fr(SCALE).inverse() # modular inverse for rescaling
        NEG_ONE = Fr(Fr.MODULUS - 1)

        gates = []
        x_val = self.wires[input_wire].value or Fr.zero()

        # Gate 1: sq = x * x
        sq_wire = self._new_wire(f"gelu_sq_L{layer_idx}_N{neuron_idx}")
        sq_val = x_val * x_val
        self._set_wire_value(sq_wire, sq_val)
        g1 = Gate(
            gate_type=GateType.GELU_SQUARE,
            left=input_wire,
            right=input_wire,
            output=sq_wire,
            q_M=Fr.one(),
            q_O=NEG_ONE,
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
        )
        self._add_gate(g1)
        gates.append(g1)

        # Gate 2: cub = sq * x
        cub_wire = self._new_wire(f"gelu_cub_L{layer_idx}_N{neuron_idx}")
        cub_val = sq_val * x_val
        self._set_wire_value(cub_wire, cub_val)
        g2 = Gate(
            gate_type=GateType.GELU_CUBIC,
            left=sq_wire,
            right=input_wire,
            output=cub_wire,
            q_M=Fr.one(),
            q_O=NEG_ONE,
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
        )
        self._add_gate(g2)
        gates.append(g2)

        # Gate 3: partial1 = a_coeff * x
        # Constraint: a_coeff * a - c = 0  →  q_L = a_coeff, q_O = -1
        p1_wire = self._new_wire(f"gelu_p1_L{layer_idx}_N{neuron_idx}")
        p1_val = a_coeff * x_val
        self._set_wire_value(p1_wire, p1_val)
        g3 = Gate(
            gate_type=GateType.GELU_OUTPUT,
            left=input_wire,
            right=input_wire,   # unused
            output=p1_wire,
            q_L=a_coeff,
            q_O=NEG_ONE,
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
        )
        self._add_gate(g3)
        gates.append(g3)

        # Gate 4: partial2 = partial1 + b_coeff * sq
        # Constraint: q_L * partial1 + q_R * sq - q_O * partial2 = 0
        #   q_L = 1, q_R = b_coeff, q_O = -1
        # But standard PLONK gate: q_L*a + q_R*b + q_O*c + q_M*a*b + q_C = 0
        # We need: 1*partial1 + b_coeff*sq + (-1)*partial2 = 0
        p2_wire = self._new_wire(f"gelu_p2_L{layer_idx}_N{neuron_idx}")
        p2_val = p1_val + b_coeff * sq_val
        self._set_wire_value(p2_wire, p2_val)
        g4 = Gate(
            gate_type=GateType.GELU_OUTPUT,
            left=p1_wire,
            right=sq_wire,
            output=p2_wire,
            q_L=Fr.one(),
            q_R=b_coeff,
            q_O=NEG_ONE,
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
        )
        self._add_gate(g4)
        gates.append(g4)

        # Gate 5: gelu_scaled = partial2 + c_coeff * cub
        # Same structure: 1*partial2 + c_coeff*cub + (-1)*gelu_scaled = 0
        scaled_wire = self._new_wire(f"gelu_scaled_L{layer_idx}_N{neuron_idx}")
        scaled_val = p2_val + c_coeff * cub_val
        self._set_wire_value(scaled_wire, scaled_val)
        g5 = Gate(
            gate_type=GateType.GELU_OUTPUT,
            left=p2_wire,
            right=cub_wire,
            output=scaled_wire,
            q_L=Fr.one(),
            q_R=c_coeff,
            q_O=NEG_ONE,
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
        )
        self._add_gate(g5)
        gates.append(g5)

        # Gate 6 (FIX-H1): Rescale back to base units
        # output = gelu_scaled * SCALE_INV
        # Constraint: q_M * a * b + q_O * c = 0 → a*b = c
        out_val = scaled_val * SCALE_INV
        self._set_wire_value(output_wire, out_val)
        g6 = self._add_mul_gate(scaled_wire, self._const_wire(SCALE_INV, layer_idx, neuron_idx),
                                output_wire, layer_idx, neuron_idx)
        gates.append(g6)

        return gates

    def _compile_relu_activation(
        self,
        input_wire: int,
        output_wire: int,
        layer_idx: int,
        neuron_idx: int
    ) -> List[Gate]:
        """
        Compile ReLU activation into PLONK gates.

        ReLU(x) = max(0, x) = s · x, where s ∈ {0, 1}.

        Sign convention: x is "positive" if x.to_int() < MODULUS // 2,
        "negative" if x.to_int() >= MODULUS // 2 (upper half = negatives).

        Gate layout (2 gates per neuron):
          Gate 1 (RELU_SIGN): s · (s - 1) = 0  (boolean constraint)
              → q_M=1, q_L=-1 on wire s (self-multiply gate)
          Gate 2 (MUL): s · x = output

        Cost: 2 gates per neuron (vs 255 for full bit-decomposition).
        Trade-off: Prover determines sign — no in-circuit range proof
        on x. Suitable when inputs come from constrained prior layers.
        """
        NEG_ONE = Fr(Fr.MODULUS - 1)

        gates = []
        x_val = self.wires[input_wire].value or Fr.zero()

        # Determine sign: positive if in lower half of field
        x_int = x_val.to_int()
        is_positive = 1 if x_int < Fr.MODULUS // 2 else 0

        # Wire for sign bit s
        s_wire = self._new_wire(f"relu_sign_L{layer_idx}_N{neuron_idx}")
        self._set_wire_value(s_wire, Fr(is_positive))

        # Gate 1: Boolean constraint — s · (s - 1) = 0
        # PLONK form: q_M · a · b + q_L · a = 0
        # With a=b=s: q_M · s² + q_L · s = 0 → s² - s = 0 → s(s-1) = 0
        g1 = Gate(
            gate_type=GateType.RELU_SIGN,
            left=s_wire,
            right=s_wire,
            output=s_wire,  # unused/self-referencing
            q_M=Fr.one(),   # s²
            q_L=NEG_ONE,    # -s
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
        )
        self._add_gate(g1)
        gates.append(g1)

        # Gate 2: output = s * x
        relu_val = Fr(is_positive) * x_val
        self._set_wire_value(output_wire, relu_val)
        g2 = self._add_mul_gate(s_wire, input_wire, output_wire, layer_idx, neuron_idx)
        gates.append(g2)

        return gates
    
    def _compile_dense_layer(
        self,
        input_wires: List[int],
        weights: List[List[int]],  # [output_size][input_size]
        biases: List[int],
        activation: str,
        layer_idx: int,
        activation_values: Optional[List[int]] = None
    ) -> List[int]:
        """
        Kompiliert einen Dense-Layer in PLONK-Gates.
        
        Für jedes Neuron:
        1. Lineare Kombination: z = Σ w_i * x_i + b
        2. Aktivierung: y = f(z)
        
        Mit Sparse-Optimierung:
        - Wenn activation_values[j] == 0: Nur ein Zero-Gate
        - Sonst: Volle Berechnung
        """
        output_size = len(weights)
        input_size = len(input_wires)
        output_wires = []
        
        for j in range(output_size):
            # Prüfe Sparsity
            is_active = True
            if self.use_sparse and activation_values is not None:
                is_active = activation_values[j] != 0
                self.sparsity_map[(layer_idx, j)] = is_active
            
            if not is_active:
                # Sparse-Optimierung: Nur ein Zero-Gate
                output_wire = self._new_wire(f"sparse_zero_L{layer_idx}_N{j}")
                self._set_wire_value(output_wire, Fr.zero())
                self._add_zero_gate(output_wire, layer_idx, j)
                output_wires.append(output_wire)
                continue
            
            # Volle Berechnung für aktives Neuron
            
            # Schritt 1: Gewichtete Summe
            # z = Σ w_i * x_i
            accumulator = None
            
            for i in range(input_size):
                # w_i * x_i
                product_wire = self._new_wire(f"prod_L{layer_idx}_N{j}_I{i}")
                
                # FIX-H2: Constrain weight as constant
                weight_wire = self._const_wire(Fr(weights[j][i]), layer_idx, j)
                
                # Multiplikation
                self._add_mul_gate(weight_wire, input_wires[i], product_wire, layer_idx, j)
                
                if accumulator is None:
                    accumulator = product_wire
                else:
                    # Akkumuliere
                    new_acc = self._new_wire(f"acc_L{layer_idx}_N{j}_I{i}")
                    self._add_add_gate(accumulator, product_wire, new_acc, layer_idx, j)
                    accumulator = new_acc
            
            # Schritt 2: FIX-H2: Constrain bias as constant
            pre_activation = self._new_wire(f"pre_act_L{layer_idx}_N{j}")
            bias_wire = self._const_wire(Fr(biases[j]), layer_idx, j)
            self._add_add_gate(accumulator, bias_wire, pre_activation, layer_idx, j)
            
            # Schritt 3: Aktivierung
            output_wire = self._new_wire(f"output_L{layer_idx}_N{j}")
            
            if activation.lower() in ['gelu', 'swish'] and self.use_gelu:
                self._compile_gelu_activation(pre_activation, output_wire, layer_idx, j)
            elif activation.lower() == 'relu':
                self._compile_relu_activation(pre_activation, output_wire, layer_idx, j)
            else:
                # Linear: output = pre_activation (identity via constrained zero add)
                zero_wire = self._const_wire(Fr.zero(), layer_idx, j)
                self._add_add_gate(pre_activation, zero_wire, output_wire, layer_idx, j)
            
            output_wires.append(output_wire)
        
        return output_wires
    
    def compile_network(
        self,
        layer_configs: List[Dict[str, Any]],
        input_values: List[int],
        activation_values_per_layer: Optional[List[List[int]]] = None
    ) -> CompiledCircuit:
        """
        Kompiliert ein neuronales Netz in einen PLONK-Circuit.
        
        Args:
            layer_configs: Liste von Layer-Konfigurationen
                [{'type': 'dense', 'weights': [[...]], 'biases': [...], 'activation': 'gelu'}, ...]
            input_values: Eingabewerte für den Circuit
            activation_values_per_layer: Optional, für Sparse-Optimierung
                [[layer0_neuron_values], [layer1_neuron_values], ...]
        
        Returns:
            CompiledCircuit mit allen Gates und Wires
        """
        # Reset
        self.gates = []
        self.wires = []
        self.wire_counter = 0
        self.sparsity_map = {}
        
        # Eingabe-Wires erstellen
        input_wires = []
        for i, val in enumerate(input_values):
            wire = self._new_wire(f"input_{i}", is_public=True)
            self._set_wire_value(wire, Fr(val))
            input_wires.append(wire)
        
        # Layer kompilieren
        current_wires = input_wires
        
        for layer_idx, config in enumerate(layer_configs):
            if config['type'] == 'dense':
                activation_values = None
                if activation_values_per_layer and layer_idx < len(activation_values_per_layer):
                    activation_values = activation_values_per_layer[layer_idx]
                
                current_wires = self._compile_dense_layer(
                    input_wires=current_wires,
                    weights=config['weights'],
                    biases=config['biases'],
                    activation=config.get('activation', 'linear'),
                    layer_idx=layer_idx,
                    activation_values=activation_values
                )
        
        # Ausgabe-Wires markieren
        for wire_idx in current_wires:
            self.wires[wire_idx].is_public = True
        
        return CompiledCircuit(
            gates=self.gates,
            wires=self.wires,
            num_public_inputs=len(input_values),
            num_public_outputs=len(current_wires),
            sparsity_map=self.sparsity_map
        )


def compile_from_network(
    network,  # Network from network/builder.py
    inputs: List[int],
    use_sparse: bool = True,
    use_gelu: bool = True
) -> CompiledCircuit:
    """
    Convenience-Funktion: Kompiliert ein Network-Objekt in einen PLONK-Circuit.
    
    Args:
        network: Ein Network-Objekt aus network/builder.py
        inputs: Eingabewerte
        use_sparse: Aktiviert Sparse-Optimierung
        use_gelu: Verwendet GELU-Gates
    
    Returns:
        CompiledCircuit
    """
    compiler = CircuitCompiler(use_sparse=use_sparse, use_gelu=use_gelu)
    
    # Forward-Pass durchführen, um Aktivierungswerte zu erhalten
    outputs, witness, stats = network.forward(inputs)
    
    # Layer-Konfigurationen extrahieren
    layer_configs = []
    activation_values_per_layer = []
    
    for layer in network.hidden_layers:
        config = {
            'type': 'dense',
            'weights': [[w for w in row] for row in layer.weights.weights],
            'biases': [b for b in layer.weights.biases],
            'activation': layer.config.activation
        }
        layer_configs.append(config)
        
        # Aktivierungswerte aus dem Witness extrahieren (vereinfacht)
        # In einer vollständigen Implementierung würden wir die genauen Werte aus dem Witness lesen
        layer_size = layer.config.output_size
        activation_values = [0] * layer_size  # Placeholder
        activation_values_per_layer.append(activation_values)
    
    return compiler.compile_network(
        layer_configs=layer_configs,
        input_values=inputs,
        activation_values_per_layer=activation_values_per_layer if use_sparse else None
    )


# Test
if __name__ == "__main__":
    print("=" * 80)
    print("CIRCUIT COMPILER TEST")
    print("=" * 80)
    
    # Einfaches Netzwerk: 4 → 3 → 2
    layer_configs = [
        {
            'type': 'dense',
            'weights': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            'biases': [1, 2, 3],
            'activation': 'gelu'
        },
        {
            'type': 'dense',
            'weights': [[1, 2, 3], [4, 5, 6]],
            'biases': [1, 2],
            'activation': 'gelu'
        }
    ]
    
    inputs = [100, 200, 300, 400]
    
    # Test 1: Ohne Optimierungen (ReLU + Dense)
    print("\n--- Test 1: ReLU + Dense (keine Optimierungen) ---")
    compiler = CircuitCompiler(use_sparse=False, use_gelu=False)
    circuit = compiler.compile_network(layer_configs, inputs)
    print(circuit.stats_summary())
    
    # Test 2: Nur GELU
    print("\n--- Test 2: GELU + Dense ---")
    compiler = CircuitCompiler(use_sparse=False, use_gelu=True)
    circuit = compiler.compile_network(layer_configs, inputs)
    print(circuit.stats_summary())
    
    # Test 3: GELU + Sparse (50% Sparsity)
    print("\n--- Test 3: GELU + Sparse (50% Sparsity) ---")
    activation_values = [
        [100, 0, 200],  # Layer 0: Neuron 1 ist inaktiv
        [0, 150]        # Layer 1: Neuron 0 ist inaktiv
    ]
    compiler = CircuitCompiler(use_sparse=True, use_gelu=True)
    circuit = compiler.compile_network(layer_configs, inputs, activation_values)
    print(circuit.stats_summary())
    
    # Test 4: GELU + Sparse (100% Sparsity)
    print("\n--- Test 4: GELU + Sparse (100% Sparsity - alle inaktiv) ---")
    activation_values = [
        [0, 0, 0],
        [0, 0]
    ]
    compiler = CircuitCompiler(use_sparse=True, use_gelu=True)
    circuit = compiler.compile_network(layer_configs, inputs, activation_values)
    print(circuit.stats_summary())
    
    print("\n" + "=" * 80)
    print("VERGLEICH: Gate-Anzahl")
    print("=" * 80)
    print("""
    Konfiguration                    | Gates (geschätzt)
    ---------------------------------|------------------
    ReLU + Dense (Baseline)          | ~1300 (255 pro Neuron für ReLU)
    GELU + Dense                     | ~100 (3 pro Neuron für GELU)
    GELU + Sparse (50%)              | ~60 (Hälfte übersprungen)
    GELU + Sparse (100%)             | ~5 (nur Zero-Gates)
    """)
