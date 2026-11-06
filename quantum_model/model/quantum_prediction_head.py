from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from quantum_model.model.prediction_head import (
    ClassicalPredictionHead,
    PredictionHeadBase,
)
from quantum_model.model.quantum_circuit import QuantumCircuit


class QuantumPredictionHead(PredictionHeadBase):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        n_heads: Optional[int] = None,
        q_readout_mode: str = "z_pairwise",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_dropout: float = 0.0,
        freeze_quantum_at_init: bool = False,
        export_circuit: bool = False,
    ) -> None:
        super().__init__()
        self.quantum_head = QuantumCircuit(
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
            export_circuit=export_circuit,
            circuit_name="prediction_head_quantum",
        )
        self.freeze_quantum_at_init = bool(freeze_quantum_at_init)
        self._quantum_trainable = True
        self._classical_trainable = True

        if self.freeze_quantum_at_init:
            self.freeze_quantum()

    def reset_parameters(self) -> None:
        self.quantum_head.reset_parameters()
        if self.freeze_quantum_at_init:
            self.freeze_quantum()

    def _set_quantum_requires_grad(self, requires_grad: bool) -> None:
        for q_layer in self.quantum_head.q_layers:
            for param in q_layer.parameters():
                param.requires_grad_(requires_grad)

    def _set_classical_requires_grad(self, requires_grad: bool) -> None:
        quantum_param_ids = {
            id(param)
            for q_layer in self.quantum_head.q_layers
            for param in q_layer.parameters()
        }
        for param in self.quantum_head.parameters():
            if id(param) not in quantum_param_ids:
                param.requires_grad_(requires_grad)

    def freeze_quantum(self) -> None:
        self._set_quantum_requires_grad(False)
        self._quantum_trainable = False

    def unfreeze_quantum(self) -> None:
        self._set_quantum_requires_grad(True)
        self._quantum_trainable = True

    def freeze_classical(self) -> None:
        self._set_classical_requires_grad(False)
        self._classical_trainable = False
        if self._quantum_trainable:
            self.unfreeze_quantum()

    def unfreeze_classical(self) -> None:
        self._set_classical_requires_grad(True)
        self._classical_trainable = True
        if not self._quantum_trainable:
            self.freeze_quantum()

    def forward(self, x: Tensor) -> Tensor:
        return self.quantum_head(x)


class ResidualQuantumPredictionHead(PredictionHeadBase):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm: str | None = "batch_norm",
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
    ) -> None:
        super().__init__()
        self.q_residual_mode = str(q_residual_mode).lower()
        if self.q_residual_mode not in {"add", "concat"}:
            raise ValueError(
                f"Unknown q_residual_mode={q_residual_mode!r}. "
                "Choose from ['add', 'concat']."
            )

        self.freeze_quantum_at_init = bool(freeze_quantum_at_init)
        self.freeze_alpha_with_quantum = bool(freeze_alpha_with_quantum)
        self.q_alpha_init = float(q_alpha_init)
        self._quantum_trainable = True
        self._classical_trainable = True

        self.classical_head = ClassicalPredictionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
        )
        self.quantum_head = QuantumPredictionHead(
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
            freeze_quantum_at_init=False,
            export_circuit=export_circuit,
        )

        if self.q_residual_mode == "concat":
            self.mix = torch.nn.Linear(2 * output_dim, output_dim)
            self.alpha = None
        else:
            self.mix = None
            self.alpha = torch.nn.Parameter(
                torch.full((output_dim,), self.q_alpha_init)
            )

        if self.freeze_quantum_at_init:
            self.freeze_quantum()

    def reset_parameters(self) -> None:
        self.classical_head.reset_parameters()
        self.quantum_head.reset_parameters()
        if self.mix is not None:
            self.mix.reset_parameters()
        if self.alpha is not None:
            torch.nn.init.constant_(self.alpha, self.q_alpha_init)
        if self.freeze_quantum_at_init:
            self.freeze_quantum()

    def _sync_alpha_requires_grad(self) -> None:
        if self.alpha is None:
            return
        if self.freeze_alpha_with_quantum:
            self.alpha.requires_grad_(self._quantum_trainable)
        else:
            self.alpha.requires_grad_(self._classical_trainable)

    def freeze_quantum(self) -> None:
        self.quantum_head.freeze_quantum()
        self._quantum_trainable = False
        self._sync_alpha_requires_grad()

    def unfreeze_quantum(self) -> None:
        self.quantum_head.unfreeze_quantum()
        self._quantum_trainable = True
        self._sync_alpha_requires_grad()

    def freeze_classical(self) -> None:
        self.classical_head.freeze_classical()
        self.quantum_head.freeze_classical()
        if self.mix is not None:
            for param in self.mix.parameters():
                param.requires_grad_(False)
        self._classical_trainable = False
        self._sync_alpha_requires_grad()

    def unfreeze_classical(self) -> None:
        self.classical_head.unfreeze_classical()
        self.quantum_head.unfreeze_classical()
        if self.mix is not None:
            for param in self.mix.parameters():
                param.requires_grad_(True)
        self._classical_trainable = True
        if not self._quantum_trainable:
            self.quantum_head.freeze_quantum()
        self._sync_alpha_requires_grad()

    def forward(self, x: Tensor) -> Tensor:
        classical_out = self.classical_head(x)
        quantum_out = self.quantum_head(x)

        if self.q_residual_mode == "concat":
            return self.mix(torch.cat([classical_out, quantum_out], dim=-1))

        alpha = torch.sigmoid(self.alpha)
        return classical_out + alpha * quantum_out
