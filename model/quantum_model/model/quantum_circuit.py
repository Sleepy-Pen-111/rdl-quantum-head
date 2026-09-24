from __future__ import annotations

import itertools
import math
import os
from typing import Optional

import torch
from torch import Tensor

try:
    import pennylane as qml
except ImportError:
    qml = None

from quantum_model.model.quantum_ansatz import (
    get_q_ansatz_weight_shapes,
    apply_q_ansatz,
    resolve_q_ansatz_type,
)


VALID_Q_READOUT_MODES = {"z_only", "z_pairwise", "z_all", "probs"}
VALID_Q_CIRCUIT_TYPES = {"angle", "rxry", "arctan", "amplitude"}


def resolve_q_readout_mode(q_readout_mode: Optional[str] = "z_pairwise") -> str:
    if q_readout_mode is None:
        q_readout_mode = "z_pairwise"

    q_readout_mode = str(q_readout_mode).lower()
    if q_readout_mode not in VALID_Q_READOUT_MODES:
        raise ValueError(
            f"Unknown q_readout_mode={q_readout_mode!r}. "
            f"Choose from {sorted(VALID_Q_READOUT_MODES)}."
        )
    return q_readout_mode


def resolve_q_circuit_type(q_circuit_type: str = "angle") -> str:
    q_circuit_type = str(q_circuit_type).lower()
    if q_circuit_type not in VALID_Q_CIRCUIT_TYPES:
        raise ValueError(
            f"Unknown q_circuit_type={q_circuit_type!r}. "
            f"Choose from {sorted(VALID_Q_CIRCUIT_TYPES)}."
        )
    return q_circuit_type


def get_q_readout_dim(n_qubits: int, q_readout_mode: str) -> int:
    q_readout_mode = resolve_q_readout_mode(q_readout_mode)

    if q_readout_mode == "z_only":
        return n_qubits
    if q_readout_mode == "z_pairwise":
        return n_qubits + n_qubits * (n_qubits - 1) // 2
    if q_readout_mode == "z_all":
        return 2 ** n_qubits - 1
    if q_readout_mode == "probs":
        return 2 ** n_qubits

    raise ValueError(f"Unknown q_readout_mode={q_readout_mode!r}")


def make_z_string_observable(wires_subset):
    obs = qml.PauliZ(wires_subset[0])
    for w in wires_subset[1:]:
        obs = obs @ qml.PauliZ(w)
    return obs


def quantum_measurement(n_qubits: int, q_readout_mode: str):
    q_readout_mode = resolve_q_readout_mode(q_readout_mode)

    if q_readout_mode == "probs":
        return qml.probs(wires=list(range(n_qubits)))

    measurements = []

    if q_readout_mode == "z_only":
        orders = [1]
    elif q_readout_mode == "z_pairwise":
        orders = [1, 2]
    elif q_readout_mode == "z_all":
        orders = range(1, n_qubits + 1)
    else:
        raise ValueError(f"Unknown q_readout_mode={q_readout_mode!r}")

    for order in orders:
        for subset in itertools.combinations(range(n_qubits), order):
            measurements.append(qml.expval(make_z_string_observable(subset)))

    return measurements


class QuantumCircuit(torch.nn.Module):
    """Reusable quantum circuit block for heads, recurrent cells, and aggregators."""

    _exported_circuits = set()

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_qubits: int = 8,
        n_q_layers: int = 1,
        n_heads: Optional[int] = None,
        q_readout_mode: str = "z_pairwise",
        rotation: str = "X",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_dropout: float = 0.0,
        dev_name: str = "default.qubit",
        export_circuit: bool = True,
        circuit_export_dir: str = os.path.expanduser("~/circuit_exports"),
        circuit_name: str = "quantum_circuit",
    ):
        super().__init__()

        if qml is None:
            raise ImportError(
                "pennylane is required to use QuantumCircuit. "
                "Please install pennylane before constructing quantum modules."
            )

        assert rotation in {"X", "Y", "Z"}
        q_circuit_type = resolve_q_circuit_type(q_circuit_type)
        q_ansatz_type = resolve_q_ansatz_type(q_ansatz_type)
        assert q_angle_activation in {"tanh", "atan", "none"}

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.n_heads = 1 if n_heads is None else int(n_heads)
        self.q_readout_mode = resolve_q_readout_mode(q_readout_mode)
        self.rotation = rotation
        self.q_circuit_type = q_circuit_type
        self.q_ansatz_type = q_ansatz_type
        self.q_angle_activation = q_angle_activation
        self.q_use_angle_affine = q_use_angle_affine
        self.q_dropout = float(q_dropout)

        if q_circuit_type == "rxry":
            self.quantum_input_dim = 2 * n_qubits
        elif q_circuit_type == "amplitude":
            self.quantum_input_dim = 2**n_qubits
        else:
            self.quantum_input_dim = n_qubits

        self.single_readout_dim = get_q_readout_dim(
            n_qubits=n_qubits,
            q_readout_mode=self.q_readout_mode,
        )
        self.total_readout_dim = self.single_readout_dim * self.n_heads

        self.pre_nets = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, self.quantum_input_dim) for _ in range(self.n_heads)]
        )

        if q_use_angle_affine:
            self.angle_scales = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(self.quantum_input_dim)) for _ in range(self.n_heads)]
            )
            self.angle_biases = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(self.quantum_input_dim)) for _ in range(self.n_heads)]
            )
        else:
            self.angle_scales = None
            self.angle_biases = None

        self.post_net = torch.nn.Linear(self.total_readout_dim, output_dim)
        self.readout_dropout = torch.nn.Dropout(self.q_dropout)
        self.q_layers = torch.nn.ModuleList()

        for head_idx in range(self.n_heads):
            dev = qml.device(dev_name, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                if q_circuit_type == "angle":
                    qml.AngleEmbedding(
                        inputs,
                        wires=range(n_qubits),
                        rotation=rotation,
                    )
                elif q_circuit_type == "rxry":
                    for i in range(n_qubits):
                        qml.RY(inputs[..., 2 * i], wires=i)
                        qml.RZ(inputs[..., 2 * i + 1], wires=i)
                elif q_circuit_type == "arctan":
                    for i in range(n_qubits):
                        qml.RY(inputs[..., i], wires=i)
                        qml.RZ(inputs[..., i] ** 2, wires=i)
                elif q_circuit_type == "amplitude":
                    qml.AmplitudeEmbedding(
                        inputs,
                        wires=range(n_qubits),
                        normalize=False,
                        pad_with=0.0,
                    )

                apply_q_ansatz(
                    qml,
                    q_ansatz_type=q_ansatz_type,
                    weights=weights,
                    n_qubits=n_qubits,
                    n_q_layers=n_q_layers,
                )

                return quantum_measurement(n_qubits, self.q_readout_mode)

            weight_shapes = get_q_ansatz_weight_shapes(
                q_ansatz_type,
                n_q_layers=n_q_layers,
                n_qubits=n_qubits,
            )
            self.q_layers.append(qml.qnn.TorchLayer(circuit, weight_shapes))

            export_key = (
                f"{circuit_name}_{q_circuit_type}_{q_ansatz_type}_"
                f"{self.q_readout_mode}_head{head_idx}"
            )
            if (
                export_circuit
                and head_idx == 0
                and export_key not in self._exported_circuits
            ):
                os.makedirs(circuit_export_dir, exist_ok=True)
                sample_inputs = self.build_sample_quantum_inputs()
                sample_weights = torch.zeros(weight_shapes["weights"])
                png_path = os.path.join(circuit_export_dir, f"{export_key}.png")
                txt_path = os.path.join(circuit_export_dir, f"{export_key}.txt")

                try:
                    fig, _ = qml.draw_mpl(circuit)(sample_inputs, sample_weights)
                    fig.savefig(png_path, bbox_inches="tight", dpi=300)
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass

                    print(f"[QuantumCircuit] Circuit image exported to: {png_path}")
                    print(
                        "[QuantumCircuit] "
                        f"input_dim={input_dim}, "
                        f"output_dim={output_dim}, "
                        f"n_qubits={n_qubits}, "
                        f"n_heads={self.n_heads}, "
                        f"q_circuit_type={q_circuit_type}, "
                        f"q_ansatz_type={q_ansatz_type}, "
                        f"q_readout_mode={self.q_readout_mode}, "
                        f"q_angle_activation={q_angle_activation}, "
                        f"q_use_angle_affine={q_use_angle_affine}, "
                        f"q_dropout={self.q_dropout}, "
                        f"quantum_input_dim={self.quantum_input_dim}, "
                        f"single_readout_dim={self.single_readout_dim}, "
                        f"total_readout_dim={self.total_readout_dim}"
                    )
                except Exception as e:
                    circuit_txt = qml.draw(circuit)(sample_inputs, sample_weights)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(circuit_txt)

                    print(f"[QuantumCircuit] Exported TXT circuit to: {txt_path}")
                    print(f"[QuantumCircuit] PNG export error: {e}")

                self._exported_circuits.add(export_key)

    def reset_parameters(self):
        for pre_net in self.pre_nets:
            pre_net.reset_parameters()

        self.post_net.reset_parameters()

        if self.q_use_angle_affine:
            for scale in self.angle_scales:
                torch.nn.init.ones_(scale)
            for bias in self.angle_biases:
                torch.nn.init.zeros_(bias)

        for q_layer in self.q_layers:
            for param in q_layer.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1)

    def apply_angle_activation(self, s: Tensor) -> Tensor:
        if self.q_angle_activation == "tanh":
            return torch.tanh(s) * math.pi
        if self.q_angle_activation == "atan":
            return 2.0 * torch.atan(s)
        if self.q_angle_activation == "none":
            return s
        raise ValueError(f"Unknown q_angle_activation: {self.q_angle_activation}")

    def normalize_amplitudes(self, s: Tensor) -> Tensor:
        eps = 1e-12
        norm = torch.linalg.vector_norm(s, ord=2, dim=-1, keepdim=True)
        safe_norm = torch.where(norm > eps, norm, torch.ones_like(norm))
        normalized = s / safe_norm

        fallback = torch.zeros_like(normalized)
        fallback[..., 0] = 1.0
        return torch.where(norm > eps, normalized, fallback)

    def prepare_quantum_inputs(self, s: Tensor, head_idx: int) -> Tensor:
        if self.q_use_angle_affine:
            s = s * self.angle_scales[head_idx] + self.angle_biases[head_idx]

        if self.q_circuit_type == "amplitude":
            return self.normalize_amplitudes(s)

        return self.apply_angle_activation(s)

    def build_sample_quantum_inputs(self) -> Tensor:
        sample_inputs = torch.zeros(self.quantum_input_dim)
        if self.q_circuit_type == "amplitude":
            sample_inputs[0] = 1.0
        return sample_inputs

    def forward(self, x: Tensor) -> Tensor:
        head_outputs = []

        for head_idx, (pre_net, q_layer) in enumerate(zip(self.pre_nets, self.q_layers)):
            s = pre_net(x)
            quantum_inputs = self.prepare_quantum_inputs(s, head_idx)
            q_out = q_layer(quantum_inputs)

            if q_out.dim() > 2:
                q_out = q_out.view(q_out.size(0), -1)

            head_outputs.append(q_out)

        q_out_all = torch.cat(head_outputs, dim=-1)
        q_out_all = self.readout_dropout(q_out_all)
        return self.post_net(q_out_all)
