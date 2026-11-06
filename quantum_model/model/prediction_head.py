from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import MLP


VALID_PREDICTION_HEAD_TYPES = {
    "classical",
    "quantum",
    "residual_quantum",
}


def resolve_prediction_head_type(head_type: str | None) -> str:
    if head_type is None:
        return "classical"

    normalized = str(head_type).strip().lower()
    if normalized not in VALID_PREDICTION_HEAD_TYPES:
        raise ValueError(
            f"Unknown prediction head type {head_type!r}. "
            f"Choose from {sorted(VALID_PREDICTION_HEAD_TYPES)}."
        )
    return normalized


class PredictionHeadBase(torch.nn.Module):
    def _set_all_requires_grad(self, requires_grad: bool) -> None:
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def freeze_quantum(self) -> None:
        return

    def unfreeze_quantum(self) -> None:
        return

    def freeze_classical(self) -> None:
        self._set_all_requires_grad(False)

    def unfreeze_classical(self) -> None:
        self._set_all_requires_grad(True)


class ClassicalPredictionHead(PredictionHeadBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm: str | None = "batch_norm",
    ) -> None:
        super().__init__()

        num_layers = max(int(num_layers), 1)
        hidden_dim = input_dim if hidden_dim is None else int(hidden_dim)

        channel_list = [int(input_dim)]
        if num_layers > 1:
            channel_list.extend([hidden_dim] * (num_layers - 1))
        channel_list.append(int(output_dim))

        self.mlp = MLP(
            channel_list=channel_list,
            norm=norm,
            dropout=float(dropout),
        )

    def reset_parameters(self) -> None:
        self.mlp.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


def build_prediction_head(
    *,
    head_type: str = "classical",
    input_dim: int,
    output_dim: int,
    norm: str = "batch_norm",
    hidden_dim: Optional[int] = None,
    num_layers: int = 1,
    dropout: float = 0.0,
    n_qubits: int = 4,
    n_q_layers: int = 2,
    n_heads: Optional[int] = None,
    q_readout_mode: str = "z_pairwise",
    q_circuit_type: str = "angle",
    q_ansatz_type: str = "rot_cnot_ring",
    q_angle_activation: str = "tanh",
    q_use_angle_affine: bool = False,
    q_dropout: float = 0.0,
    q_residual_mode: str = "add",
    q_alpha_init: float = 0.0,
    freeze_quantum_at_init: bool = False,
    freeze_alpha_with_quantum: bool = True,
    export_circuit: bool = False,
) -> PredictionHeadBase:
    head_type = resolve_prediction_head_type(head_type)

    if head_type == "classical":
        return ClassicalPredictionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
        )

    from quantum_model.model.quantum_prediction_head import (
        QuantumPredictionHead,
        ResidualQuantumPredictionHead,
    )

    if head_type == "quantum":
        return QuantumPredictionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            n_heads=n_heads,
            q_readout_mode=q_readout_mode,
            q_circuit_type=q_circuit_type,
            q_ansatz_type=q_ansatz_type,
            q_angle_activation=q_angle_activation,
            q_use_angle_affine=q_use_angle_affine,
            q_dropout=q_dropout,
            freeze_quantum_at_init=freeze_quantum_at_init,
            export_circuit=export_circuit,
        )

    if head_type == "residual_quantum":
        return ResidualQuantumPredictionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            n_heads=n_heads,
            q_readout_mode=q_readout_mode,
            q_circuit_type=q_circuit_type,
            q_ansatz_type=q_ansatz_type,
            q_angle_activation=q_angle_activation,
            q_use_angle_affine=q_use_angle_affine,
            q_dropout=q_dropout,
            q_residual_mode=q_residual_mode,
            q_alpha_init=q_alpha_init,
            freeze_quantum_at_init=freeze_quantum_at_init,
            freeze_alpha_with_quantum=freeze_alpha_with_quantum,
            export_circuit=export_circuit,
        )

    raise ValueError(f"Unsupported prediction head type {head_type!r}")
