from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv, GATv2Conv
from torch_geometric.nn.aggr import LSTMAggregation
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import sort_edge_index

from numpy._core.multiarray import scalar
torch.serialization.add_safe_globals([scalar])
from numpy import dtype
torch.serialization.add_safe_globals([dtype])

NodeType = str
EdgeType = Tuple[str, str, str]
RELATION_WEIGHTED_SUM_FUSION_NAMES = {
    "weighted_sum",
    "relation_weighted_sum",
    "edge_type_weighted_sum",
}


class TypeGRUFusion(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        channels: int,
        gru_type: str = "gru",
        n_qubits: int = 8,
        n_q_layers: int = 1,
        n_heads: Optional[int] = None,
        gat_heads: int = 4,
        q_readout_mode: str = "z_pairwise",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_residual_mode: str = "add",
        q_alpha_init: float = 0.0,
        export_circuit: bool = True,
        freeze_quantum_at_init: bool = False,
        freeze_alpha_with_quantum: bool = True,
    ):
        super().__init__()

        if gru_type != "gru":
            raise ValueError(f"Only the classical GRU cell is supported, got {gru_type!r}.")
        self.grus = torch.nn.ModuleDict({
            node_type: torch.nn.GRUCell(channels, channels)
            for node_type in node_types
        })

    def reset_parameters(self):
        for gru in self.grus.values():
            if hasattr(gru, "reset_parameters"):
                gru.reset_parameters()

    def freeze_quantum(self) -> None:
        for gru in self.grus.values():
            if hasattr(gru, "freeze_quantum"):
                gru.freeze_quantum()

    def unfreeze_quantum(self) -> None:
        for gru in self.grus.values():
            if hasattr(gru, "unfreeze_quantum"):
                gru.unfreeze_quantum()

    def freeze_classical(self) -> None:
        for gru in self.grus.values():
            if hasattr(gru, "freeze_classical"):
                gru.freeze_classical()

    def unfreeze_classical(self) -> None:
        for gru in self.grus.values():
            if hasattr(gru, "unfreeze_classical"):
                gru.unfreeze_classical()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        rel_msg_dict: Dict[NodeType, Dict[str, Tensor]],
    ) -> Dict[NodeType, Tensor]:

        msg_dict = {}

        for node_type, h_ref in x_dict.items():
            messages = list(rel_msg_dict.get(node_type, {}).values())

            if len(messages) == 0:
                msg_dict[node_type] = torch.zeros_like(h_ref)
                continue

            if len(messages) == 1:
                msg = messages[0]
                if msg.dim() == 3:
                    if msg.size(0) < msg.size(1):
                        msg = msg.sum(dim=0)
                    else:
                        msg = msg.sum(dim=1)
                msg_dict[node_type] = msg
                continue

            h = torch.zeros_like(h_ref)

            for msg in messages:
                if msg.dim() == 3:
                    if msg.size(0) < msg.size(1):
                        msg = msg.sum(dim=0)
                    else:
                        msg = msg.sum(dim=1)

                h = self.grus[node_type](msg, h)

            msg_dict[node_type] = h

        return msg_dict


class RelationWeightedSumFusion(torch.nn.Module):
    """Learnable relation-level weighted-sum fusion.

    For each destination node type, all incoming relation messages are arranged
    in a fixed order derived from ``edge_types``. Each incoming edge type
    ``(src_type, rel_type, dst_type)`` gets one dedicated learnable scalar
    logit per layer. During fusion, logits for relations that share the same
    destination node type are normalized with softmax, and the fused message is:

        msg_dst = sum_r softmax(logits_dst)[r] * msg_r

    Missing relations in a sampled mini-batch are represented by zero tensors.
    This keeps the shape stable while still allowing the model to learn global
    relation-level importance.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        use_softmax: bool = True,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.channels = channels
        self.use_softmax = use_softmax

        self.incoming_edge_types: Dict[NodeType, List[EdgeType]] = {
            node_type: [
                edge_type for edge_type in edge_types
                if edge_type[2] == node_type
            ]
            for node_type in node_types
        }

        self.edge_type_to_key = {
            edge_type: self._edge_type_to_key(edge_type)
            for edge_type in edge_types
        }

        self.logits = torch.nn.ParameterDict()
        for edge_type in edge_types:
            self.logits[self.edge_type_to_key[edge_type]] = torch.nn.Parameter(
                torch.zeros(())
            )

    @staticmethod
    def _safe_key(name: str) -> str:
        # ParameterDict keys cannot contain dots.
        return str(name).replace(".", "_DOT_")

    @classmethod
    def _edge_type_to_key(cls, edge_type: EdgeType) -> str:
        src_type, rel_type, dst_type = edge_type
        return "__".join([
            cls._safe_key(src_type),
            cls._safe_key(rel_type),
            cls._safe_key(dst_type),
        ])

    @staticmethod
    def _ensure_2d_msg(msg: Tensor) -> Tensor:
        if msg.dim() == 2:
            return msg

        if msg.dim() == 3:
            if msg.size(0) < msg.size(1):
                return msg.sum(dim=0)
            return msg.sum(dim=1)

        raise RuntimeError(f"Unexpected msg shape: {tuple(msg.shape)}")

    def reset_parameters(self) -> None:
        for param in self.logits.values():
            torch.nn.init.zeros_(param)

    def _value_to_relation_messages(
        self,
        value: object,
        incoming_edge_types: List[EdgeType],
    ) -> Dict[EdgeType, Tensor]:
        rel_to_msg: Dict[EdgeType, Tensor] = {}

        if isinstance(value, (list, tuple)):
            for edge_type, msg in zip(incoming_edge_types, value):
                rel_to_msg[edge_type] = self._ensure_2d_msg(msg)

        elif isinstance(value, dict):
            for key, msg in value.items():
                if isinstance(key, tuple):
                    edge_type = key
                else:
                    parts = str(key).split("__")
                    edge_type = tuple(parts) if len(parts) == 3 else None

                if edge_type in incoming_edge_types:
                    rel_to_msg[edge_type] = self._ensure_2d_msg(msg)

        elif torch.is_tensor(value):
            if len(incoming_edge_types) == 1:
                rel_to_msg[incoming_edge_types[0]] = self._ensure_2d_msg(value)
            elif len(incoming_edge_types) > 1:
                # Safe fallback. This should not normally happen when
                # HeteroConv(aggr=None) is used with multiple incoming relations.
                rel_to_msg[incoming_edge_types[0]] = self._ensure_2d_msg(value)

        return rel_to_msg

    def forward(
        self,
        old_x_dict: Dict[NodeType, Tensor],
        conv_out: Dict[NodeType, object],
    ) -> Dict[NodeType, Tensor]:

        msg_dict: Dict[NodeType, Tensor] = {}

        for dst_type, h_ref in old_x_dict.items():
            incoming_edge_types = self.incoming_edge_types.get(dst_type, [])

            if len(incoming_edge_types) == 0:
                msg_dict[dst_type] = torch.zeros_like(h_ref)
                continue

            value = conv_out.get(dst_type, None)
            rel_to_msg = self._value_to_relation_messages(
                value=value,
                incoming_edge_types=incoming_edge_types,
            )

            logits = torch.stack([
                self.logits[self.edge_type_to_key[edge_type]]
                for edge_type in incoming_edge_types
            ])

            if self.use_softmax:
                weights = torch.softmax(logits, dim=0)
            else:
                weights = logits

            fused = torch.zeros_like(h_ref)

            for rel_idx, edge_type in enumerate(incoming_edge_types):
                msg = rel_to_msg.get(edge_type, None)

                if msg is None:
                    msg = torch.zeros_like(h_ref)
                elif msg.size(0) != h_ref.size(0):
                    raise RuntimeError(
                        f"Weighted-sum fusion size mismatch for dst_type={dst_type}, "
                        f"edge_type={edge_type}: msg={tuple(msg.shape)}, "
                        f"h_ref={tuple(h_ref.shape)}"
                    )

                fused = fused + weights[rel_idx] * msg

            msg_dict[dst_type] = fused

        return msg_dict


class HeteroRecurrentGNN(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        intra_aggr: str = "mean",
        type_fusion: str = "sum",
        node_update: str = "mlp",
        num_layers: int = 2,
        use_norm: bool = True,
        use_relu: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
        residual_weight: float = 0.5,
        gru_type: str = "gru",
        n_qubits: int = 8,
        n_q_layers: int = 1,
        n_heads: Optional[int] = None,
        gat_heads: int = 4,
        q_readout_mode: str = "z_pairwise",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_residual_mode: str = "add",
        q_alpha_init: float = 0.0,
        export_circuit: bool = True,
        freeze_quantum_at_init: bool = False,
        freeze_alpha_with_quantum: bool = True,
    ):
        super().__init__()

        intra_aggr = self._normalize_intra_aggr_name(intra_aggr)
        type_fusion = self._normalize_type_fusion_name(type_fusion)

        assert intra_aggr in {
            "mean",
            "mean_project",
            "max_project",
            "max",
            "lstm",
            "gatv2",
        }

        assert type_fusion in {
            "sum",
            "mean",
            "concat",
            "weighted_sum",
            "gru",
        }

        assert node_update in {
            "mlp",
            "gru",
        }

        self.node_types = node_types
        self.edge_types = edge_types
        self.channels = channels
        self.intra_aggr = intra_aggr
        self.type_fusion = type_fusion
        self.node_update_type = node_update

        self.num_layers = num_layers
        self.use_norm = use_norm
        self.use_relu = use_relu
        self.residual = residual

        self.gru_type = gru_type
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.n_heads = n_heads
        self.gat_heads = gat_heads
        self.q_readout_mode = q_readout_mode
        self.q_circuit_type = q_circuit_type
        self.q_ansatz_type = q_ansatz_type
        self.q_angle_activation = q_angle_activation
        self.q_use_angle_affine = q_use_angle_affine
        self.q_residual_mode = q_residual_mode
        self.q_alpha_init = q_alpha_init
        self.export_circuit = export_circuit

        self.freeze_quantum_at_init = freeze_quantum_at_init
        self.freeze_alpha_with_quantum = freeze_alpha_with_quantum
        self._quantum_trainable = True

        self.residual_weights = torch.nn.ParameterDict({
            nt: torch.nn.Parameter(torch.tensor(residual_weight))
            for nt in node_types
        })

        project = self._parse_sage_project(intra_aggr)

        self.type_fusion_is_recurrent = type_fusion == "gru"
        self.type_fusion_is_concat = type_fusion == "concat"
        self.type_fusion_is_weighted_sum = type_fusion == "weighted_sum"

        # For recurrent fusion and concat fusion we need HeteroConv to return
        # one message tensor per edge type/relation. Therefore aggr must be None.
        hetero_aggr = None if (self.type_fusion_is_recurrent or self.type_fusion_is_concat or self.type_fusion_is_weighted_sum) else type_fusion

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: self._build_relation_conv(
                        intra_aggr=intra_aggr,
                        channels=channels,
                        project=project,
                        n_qubits=n_qubits,
                        n_q_layers=n_q_layers,
                        n_heads=n_heads,
                        q_readout_mode=q_readout_mode,
                        q_circuit_type=q_circuit_type,
                        q_ansatz_type=q_ansatz_type,
                        q_angle_activation=q_angle_activation,
                        q_use_angle_affine=q_use_angle_affine,
                        q_residual_mode=q_residual_mode,
                        q_alpha_init=q_alpha_init,
                        export_circuit=export_circuit,
                        freeze_quantum_at_init=freeze_quantum_at_init,
                        freeze_alpha_with_quantum=freeze_alpha_with_quantum,
                        gatv2_heads=gat_heads,
                        gatv2_dropout=dropout,
                    )
                    for edge_type in edge_types
                },
                aggr=hetero_aggr,
            )
            self.convs.append(conv)


        if self.type_fusion_is_concat:
            self.concat_projs = torch.nn.ModuleList()
            for _ in range(num_layers):
                proj_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    incoming_count = len([
                        edge_type for edge_type in edge_types
                        if edge_type[2] == node_type
                    ])
                    # Some node types may have no incoming edge type in a sampled graph.
                    # Keep input dim at least channels to make the module valid.
                    in_dim = max(1, incoming_count) * channels
                    proj_dict[node_type] = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(channels, channels),
                    )
                self.concat_projs.append(proj_dict)
        else:
            self.concat_projs = None

        if self.type_fusion_is_weighted_sum:
            self.weighted_sum_fusions = torch.nn.ModuleList([
                RelationWeightedSumFusion(
                    node_types=node_types,
                    edge_types=edge_types,
                    channels=channels,
                    use_softmax=True,
                )
                for _ in range(num_layers)
            ])
        else:
            self.weighted_sum_fusions = None

        if self.type_fusion_is_recurrent:
            fusion_gru_type = self._parse_recurrent_cell_type(type_fusion)

            self.type_fusions = torch.nn.ModuleList([
                TypeGRUFusion(
                    node_types=node_types,
                    channels=channels,
                    gru_type=fusion_gru_type,
                    n_qubits=n_qubits,
                    n_q_layers=n_q_layers,
                    n_heads=n_heads,
                    q_readout_mode=q_readout_mode,
                    q_circuit_type=q_circuit_type,
                    q_ansatz_type=q_ansatz_type,
                    q_angle_activation=q_angle_activation,
                    q_use_angle_affine=q_use_angle_affine,
                    q_residual_mode=q_residual_mode,
                    q_alpha_init=q_alpha_init,
                    export_circuit=export_circuit,
                    freeze_quantum_at_init=freeze_quantum_at_init,
                    freeze_alpha_with_quantum=freeze_alpha_with_quantum,
                )
                for _ in range(num_layers)
            ])
        else:
            self.type_fusions = None

        self.node_updates = None
        self.node_update_cell = None

        if node_update == "mlp":
            self.node_updates = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(2 * channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                )
                for _ in range(num_layers)
            ])

        elif node_update == "gru":
            self.node_update_cell = torch.nn.GRUCell(channels, channels)

        else:
            raise ValueError(f"Unknown node_update={node_update}")

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            if use_norm:
                for node_type in node_types:
                    norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

        self.act = torch.nn.ReLU() if use_relu else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def _normalize_intra_aggr_name(name: str) -> str:
        name = str(name).lower().replace("-", "_")
        if name == "gat":
            return "gatv2"
        return name

    @staticmethod
    def _normalize_type_fusion_name(name: str) -> str:
        name = str(name).lower()
        if name in RELATION_WEIGHTED_SUM_FUSION_NAMES:
            return "weighted_sum"
        return name

    @staticmethod
    def _parse_sage_project(intra_aggr: str) -> bool:
        if intra_aggr in {"mean_project", "max_project"}:
            return True
        return False

    @staticmethod
    def _build_relation_conv(
        intra_aggr: str,
        channels: int,
        project: bool,
        n_qubits: int = 8,
        n_q_layers: int = 1,
        n_heads: Optional[int] = None,
        q_readout_mode: str = "z_pairwise",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_residual_mode: str = "add",
        q_alpha_init: float = -4.0,
        export_circuit: bool = True,
        freeze_quantum_at_init: bool = False,
        freeze_alpha_with_quantum: bool = True,
        gatv2_heads: int = 4,
        gatv2_dropout: float = 0.0,
    ):
        if intra_aggr == "gatv2":
            return GATv2Conv(
                (channels, channels),
                channels,
                heads=gatv2_heads,
                concat=False,
                dropout=gatv2_dropout,
                add_self_loops=False,
            )

        return SAGEConv(
            (channels, channels),
            channels,
            aggr=HeteroRecurrentGNN._build_sage_aggr(
                intra_aggr=intra_aggr,
                channels=channels,
                n_qubits=n_qubits,
                n_q_layers=n_q_layers,
                n_heads=n_heads,
                q_readout_mode=q_readout_mode,
                q_circuit_type=q_circuit_type,
                q_ansatz_type=q_ansatz_type,
                q_angle_activation=q_angle_activation,
                q_use_angle_affine=q_use_angle_affine,
                q_residual_mode=q_residual_mode,
                q_alpha_init=q_alpha_init,
                export_circuit=export_circuit,
                freeze_quantum_at_init=freeze_quantum_at_init,
                freeze_alpha_with_quantum=freeze_alpha_with_quantum,
            ),
            project=project,
            root_weight=False,
            normalize=False,
        )

    @staticmethod
    def _build_sage_aggr(
        intra_aggr: str,
        channels: int,
        n_qubits: int = 8,
        n_q_layers: int = 1,
        n_heads: Optional[int] = None,
        q_readout_mode: str = "z_pairwise",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_residual_mode: str = "add",
        q_alpha_init: float = -4.0,
        export_circuit: bool = True,
        freeze_quantum_at_init: bool = False,
        freeze_alpha_with_quantum: bool = True,
    ):
        if intra_aggr == "mean":
            return "mean"
        if intra_aggr == "max":
            return "max"
        if intra_aggr == "mean_project":
            return "mean"
        if intra_aggr == "max_project":
            return "max"

        if intra_aggr == "lstm":
            return LSTMAggregation(channels, channels)

        raise ValueError(f"Unknown intra_aggr: {intra_aggr}")

    @staticmethod
    def _parse_recurrent_cell_type(name: str) -> str:
        name = str(name).lower()

        if name == "gru":
            return "gru"

        raise ValueError(f"Unknown recurrent cell type: {name}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.type_fusions is not None:
            for fusion in self.type_fusions:
                fusion.reset_parameters()

        if self.concat_projs is not None:
            for proj_dict in self.concat_projs:
                for proj in proj_dict.values():
                    for module in proj.modules():
                        if hasattr(module, "reset_parameters"):
                            module.reset_parameters()

        if self.weighted_sum_fusions is not None:
            for fusion in self.weighted_sum_fusions:
                fusion.reset_parameters()

        if self.node_updates is not None:
            for update in self.node_updates:
                for module in update.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

        if self.node_update_cell is not None:
            if hasattr(self.node_update_cell, "reset_parameters"):
                self.node_update_cell.reset_parameters()

        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

        # Important:
        # reset_parameters may reset requires_grad through child modules in some cases.
        # Re-apply initial freeze policy after reset.
        if self.freeze_quantum_at_init:
            self.freeze_quantum()

    def _set_all_requires_grad(self, requires_grad: bool) -> None:
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def freeze_quantum(self) -> None:
        for conv in self.convs:
            for module in conv.modules():
                if module is not conv and hasattr(module, "freeze_quantum"):
                    module.freeze_quantum()

        if self.type_fusions is not None:
            for fusion in self.type_fusions:
                if hasattr(fusion, "freeze_quantum"):
                    fusion.freeze_quantum()

        if self.node_update_cell is not None:
            if hasattr(self.node_update_cell, "freeze_quantum"):
                self.node_update_cell.freeze_quantum()

        self._quantum_trainable = False

    def unfreeze_quantum(self) -> None:
        for conv in self.convs:
            for module in conv.modules():
                if module is not conv and hasattr(module, "unfreeze_quantum"):
                    module.unfreeze_quantum()

        if self.type_fusions is not None:
            for fusion in self.type_fusions:
                if hasattr(fusion, "unfreeze_quantum"):
                    fusion.unfreeze_quantum()

        if self.node_update_cell is not None:
            if hasattr(self.node_update_cell, "unfreeze_quantum"):
                self.node_update_cell.unfreeze_quantum()

        self._quantum_trainable = True

    def freeze_classical(self) -> None:
        # Freeze the whole GNN first, then restore the quantum branch if it is
        # currently meant to be trainable. This keeps "quantum-only finetune"
        # aligned with the latest quantum freeze/unfreeze schedule.
        self._set_all_requires_grad(False)

        if self._quantum_trainable:
            self.unfreeze_quantum()

    def unfreeze_classical(self) -> None:
        self._set_all_requires_grad(True)

        if not self._quantum_trainable:
            self.freeze_quantum()



    @staticmethod
    def _clean_edge_index_dict(
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[EdgeType, Tensor]:
        clean_edge_index_dict = {}

        for edge_type, edge_index in edge_index_dict.items():
            if (
                edge_index is not None
                and edge_index.numel() > 0
                and edge_index.size(1) > 0
            ):
                clean_edge_index_dict[edge_type] = sort_edge_index(
                    edge_index,
                    sort_by_row=False,
                )

        return clean_edge_index_dict

    @staticmethod
    def _ensure_2d_msg(msg: Tensor) -> Tensor:
        if msg.dim() == 2:
            return msg

        if msg.dim() == 3:
            if msg.size(0) < msg.size(1):
                return msg.sum(dim=0)
            return msg.sum(dim=1)

        raise RuntimeError(f"Unexpected msg shape: {tuple(msg.shape)}")

    def _fuse_with_gru(
        self,
        old_x_dict: Dict[NodeType, Tensor],
        conv_out: Dict[NodeType, object],
        layer_idx: int,
    ) -> Dict[NodeType, Tensor]:

        rel_msg_dict: Dict[NodeType, Dict[str, Tensor]] = {
            node_type: {}
            for node_type in old_x_dict.keys()
        }

        for dst_type, value in conv_out.items():
            if isinstance(value, (list, tuple)):
                incoming_edge_types = [
                    edge_type
                    for edge_type in self.edge_types
                    if edge_type[2] == dst_type
                ]

                for edge_type, msg in zip(incoming_edge_types, value):
                    rel_msg_dict[dst_type]["__".join(edge_type)] = msg

            elif isinstance(value, dict):
                for edge_type, msg in value.items():
                    key = (
                        "__".join(edge_type)
                        if isinstance(edge_type, tuple)
                        else str(edge_type)
                    )
                    rel_msg_dict[dst_type][key] = msg

            elif torch.is_tensor(value):
                rel_msg_dict[dst_type]["single"] = value

        return self.type_fusions[layer_idx](
            x_dict=old_x_dict,
            rel_msg_dict=rel_msg_dict,
        )

    def _fuse_with_concat(
        self,
        old_x_dict: Dict[NodeType, Tensor],
        conv_out: Dict[NodeType, object],
        layer_idx: int,
    ) -> Dict[NodeType, Tensor]:
        """Fuse relation-wise messages by fixed-order concat + zero padding.

        HeteroConv(aggr=None) returns, for each destination node type, either
        a list/tuple of message tensors or a dictionary keyed by edge type.
        We reconstruct a fixed relation order from self.edge_types. For each
        dst node type, every incoming edge type contributes one [N_dst, C]
        block. Missing relations in the sampled mini-batch are filled with zeros.
        The concatenated [N_dst, C * num_incoming_edge_types] tensor is projected
        back to [N_dst, C], so downstream node_update keeps the same interface.
        """
        if self.concat_projs is None:
            raise RuntimeError("concat_projs is None, but type_fusion='concat'.")

        msg_dict: Dict[NodeType, Tensor] = {}

        for dst_type, h_ref in old_x_dict.items():
            incoming_edge_types = [
                edge_type for edge_type in self.edge_types
                if edge_type[2] == dst_type
            ]

            if len(incoming_edge_types) == 0:
                msg_dict[dst_type] = torch.zeros_like(h_ref)
                continue

            value = conv_out.get(dst_type, None)
            rel_to_msg: Dict[EdgeType, Tensor] = {}

            if isinstance(value, (list, tuple)):
                for edge_type, msg in zip(incoming_edge_types, value):
                    rel_to_msg[edge_type] = self._ensure_2d_msg(msg)

            elif isinstance(value, dict):
                for key, msg in value.items():
                    if isinstance(key, tuple):
                        edge_type = key
                    else:
                        # Fallback for string keys such as "src__rel__dst".
                        parts = str(key).split("__")
                        edge_type = tuple(parts) if len(parts) == 3 else None
                    if edge_type in incoming_edge_types:
                        rel_to_msg[edge_type] = self._ensure_2d_msg(msg)

            elif torch.is_tensor(value):
                # This should not normally happen when aggr=None and multiple
                # relations exist. Keep a safe fallback for single-relation cases.
                rel_to_msg[incoming_edge_types[0]] = self._ensure_2d_msg(value)

            blocks = []
            for edge_type in incoming_edge_types:
                msg = rel_to_msg.get(edge_type, None)

                if msg is None:
                    msg = torch.zeros_like(h_ref)
                elif msg.size(0) != h_ref.size(0):
                    raise RuntimeError(
                        f"Concat fusion size mismatch for dst_type={dst_type}, "
                        f"edge_type={edge_type}: msg={tuple(msg.shape)}, "
                        f"h_ref={tuple(h_ref.shape)}"
                    )

                blocks.append(msg)

            concat_msg = torch.cat(blocks, dim=-1)
            msg_dict[dst_type] = self.concat_projs[layer_idx][dst_type](concat_msg)

        return msg_dict

    def _fuse_with_weighted_sum(
        self,
        old_x_dict: Dict[NodeType, Tensor],
        conv_out: Dict[NodeType, object],
        layer_idx: int,
    ) -> Dict[NodeType, Tensor]:
        """Fuse relation-wise messages by learnable scalar weights.

        This uses HeteroConv(aggr=None), so relation-wise outputs are still
        available. For each destination node type, a fixed relation order is
        used and one learnable logit per incoming relation is normalized with
        softmax.
        """
        if self.weighted_sum_fusions is None:
            raise RuntimeError(
                "weighted_sum_fusions is None, but type_fusion='weighted_sum'."
            )

        return self.weighted_sum_fusions[layer_idx](
            old_x_dict=old_x_dict,
            conv_out=conv_out,
        )

    def _update_nodes(
        self,
        old_x_dict: Dict[NodeType, Tensor],
        msg_dict: Dict[NodeType, Tensor],
        layer_idx: int,
    ) -> Dict[NodeType, Tensor]:

        valid_node_types = []
        all_msgs = []
        all_h_prevs = []
        split_sizes = []

        for node_type, h_prev in old_x_dict.items():
            if h_prev.size(0) == 0:
                continue

            msg = msg_dict.get(node_type, torch.zeros_like(h_prev))
            msg = self._ensure_2d_msg(msg)

            if msg.size(0) != h_prev.size(0):
                raise RuntimeError(
                    f"Message/node size mismatch for node_type={node_type}: "
                    f"msg={tuple(msg.shape)}, h_prev={tuple(h_prev.shape)}"
                )

            valid_node_types.append(node_type)
            all_msgs.append(msg)
            all_h_prevs.append(h_prev)
            split_sizes.append(h_prev.size(0))

        x_dict = {
            node_type: old_x_dict[node_type]
            for node_type in old_x_dict
            if old_x_dict[node_type].size(0) == 0
        }

        if valid_node_types:
            batched_msg = torch.cat(all_msgs, dim=0)
            batched_h_prev = torch.cat(all_h_prevs, dim=0)

            if self.node_update_type == "mlp":
                if self.node_updates is None:
                    raise RuntimeError(
                        "node_updates is None, but node_update_type='mlp'."
                    )

                batched_h_new = self.node_updates[layer_idx](
                    torch.cat([batched_h_prev, batched_msg], dim=-1)
                )

            elif self.node_update_type == "gru":
                if self.node_update_cell is None:
                    raise RuntimeError(
                        "node_update_cell is None, but node_update_type is recurrent."
                    )

                batched_h_new = self.node_update_cell(
                    batched_msg,
                    batched_h_prev,
                )

            else:
                raise ValueError(f"Unknown node_update: {self.node_update_type}")

            h_new_list = torch.split(batched_h_new, split_sizes, dim=0)

            for node_type, h_prev, h_new in zip(
                valid_node_types,
                all_h_prevs,
                h_new_list,
            ):
                if self.residual and h_new.shape == h_prev.shape:
                    alpha = torch.sigmoid(self.residual_weights[node_type])
                    h_new = (1.0 - alpha) * h_prev + alpha * h_new

                x_dict[node_type] = h_new

        return x_dict

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:

        for i, conv in enumerate(self.convs):
            old_x_dict = x_dict

            clean_edge_index_dict = self._clean_edge_index_dict(edge_index_dict)

            conv_out = conv(
                old_x_dict,
                clean_edge_index_dict,
            )

            if self.type_fusion_is_recurrent:
                msg_dict = self._fuse_with_gru(
                    old_x_dict=old_x_dict,
                    conv_out=conv_out,
                    layer_idx=i,
                )
            elif self.type_fusion_is_concat:
                msg_dict = self._fuse_with_concat(
                    old_x_dict=old_x_dict,
                    conv_out=conv_out,
                    layer_idx=i,
                )
            elif self.type_fusion_is_weighted_sum:
                msg_dict = self._fuse_with_weighted_sum(
                    old_x_dict=old_x_dict,
                    conv_out=conv_out,
                    layer_idx=i,
                )
            else:
                msg_dict = {
                    node_type: conv_out.get(
                        node_type,
                        torch.zeros_like(old_x_dict[node_type]),
                    )
                    for node_type in old_x_dict.keys()
                }

            x_dict = self._update_nodes(
                old_x_dict=old_x_dict,
                msg_dict=msg_dict,
                layer_idx=i,
            )

            if self.use_norm:
                x_dict = {
                    node_type: (
                        self.norms[i][node_type](x)
                        if node_type in self.norms[i]
                        else x
                    )
                    for node_type, x in x_dict.items()
                }

            if self.act is not None:
                x_dict = {
                    node_type: self.act(x)
                    for node_type, x in x_dict.items()
                }

            if self.dropout is not None:
                x_dict = {
                    node_type: self.dropout(x)
                    for node_type, x in x_dict.items()
                }

        return x_dict



# -----------------------------------------------------------------------------
# One-step graph encoders for relational database learning.
#
# These encoders intentionally do not use recurrent aggregation. They are meant
# for the quantum prediction-head experiments, where the graph encoder is a
# classical relational backbone and the quantum/classical head is attached after
# the target entity/node embedding is produced.
# -----------------------------------------------------------------------------

ONE_STEP_ENCODER_NAMES = {
    "hgt",
    "graphormer",
}


def _normalize_encoder_family(name: str) -> str:
    name = str(name).lower().replace("-", "_")
    if name in {"one_step", "onestep", "one_step_encoder"}:
        return "one_step"
    if name in {"staged", "modular", "message_passing"}:
        return "staged"
    raise ValueError(f"Unknown encoder_family={name}")


def _normalize_encoder_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


class TypeSpecificInputProjection(torch.nn.Module):
    """Project heterogeneous raw node features into a shared hidden space."""

    def __init__(
        self,
        node_types: List[NodeType],
        in_channels_dict: Dict[NodeType, int],
        hidden_channels: int,
        use_activation: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_types = node_types
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        self.use_activation = use_activation

        missing = [nt for nt in node_types if nt not in in_channels_dict]
        if missing:
            raise ValueError(f"Missing in_channels_dict entries for node types: {missing}")

        self.projs = torch.nn.ModuleDict({
            nt: torch.nn.Linear(in_channels_dict[nt], hidden_channels)
            for nt in node_types
        })
        self.act = torch.nn.ReLU() if use_activation else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def reset_parameters(self) -> None:
        for proj in self.projs.values():
            proj.reset_parameters()

    def forward(self, x_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        out = {}
        for nt, x in x_dict.items():
            if nt not in self.projs:
                continue
            h = self.projs[nt](x)
            if self.act is not None:
                h = self.act(h)
            if self.dropout is not None:
                h = self.dropout(h)
            out[nt] = h
        return out


class TypeProjectedEncoder(torch.nn.Module):
    """Wrapper that adds type-specific input projection before an encoder.

    This is useful for the existing staged encoder, because ``HeteroRecurrentGNN``
    assumes that every node type is already represented in ``channels`` hidden
    dimensions.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        node_types: List[NodeType],
        in_channels_dict: Dict[NodeType, int],
        hidden_channels: int,
        projection_dropout: float = 0.0,
        projection_activation: bool = False,
    ):
        super().__init__()
        self.input_projection = TypeSpecificInputProjection(
            node_types=node_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            use_activation=projection_activation,
            dropout=projection_dropout,
        )
        self.encoder = encoder

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()

    def freeze_quantum(self) -> None:
        if hasattr(self.encoder, "freeze_quantum"):
            self.encoder.freeze_quantum()

    def unfreeze_quantum(self) -> None:
        if hasattr(self.encoder, "unfreeze_quantum"):
            self.encoder.unfreeze_quantum()

    def freeze_classical(self) -> None:
        if hasattr(self.encoder, "freeze_classical"):
            self.encoder.freeze_classical()

    def unfreeze_classical(self) -> None:
        if hasattr(self.encoder, "unfreeze_classical"):
            self.encoder.unfreeze_classical()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        x_dict = self.input_projection(x_dict)
        return self.encoder(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            num_sampled_nodes_dict=num_sampled_nodes_dict,
            num_sampled_edges_dict=num_sampled_edges_dict,
        )


class HGTOneStepEncoder(torch.nn.Module):
    """Local one-step heterogeneous transformer encoder based on HGTConv.

    HGT is placed in the ``one_step`` family because type-aware aggregation,
    relation interaction, and node update are integrated inside each HGT layer.
    It is still local: attention is computed only over existing typed edges in
    ``edge_index_dict``.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        in_channels_dict: Dict[NodeType, int],
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        use_norm: bool = True,
        use_relu: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.metadata = (node_types, edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.use_norm = use_norm
        self.use_relu = use_relu
        self.residual = residual

        self.input_projection = TypeSpecificInputProjection(
            node_types=node_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
        )

        self.convs = torch.nn.ModuleList([
            HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=self.metadata,
                heads=heads,
            )
            for _ in range(num_layers)
        ])

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            if use_norm:
                for nt in node_types:
                    norm_dict[nt] = LayerNorm(hidden_channels, mode="node")
            self.norms.append(norm_dict)

        self.act = torch.nn.ReLU() if use_relu else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

        if self.out_channels != hidden_channels:
            self.output_projs = torch.nn.ModuleDict({
                nt: torch.nn.Linear(hidden_channels, self.out_channels)
                for nt in node_types
            })
        else:
            self.output_projs = None

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        if self.output_projs is not None:
            for proj in self.output_projs.values():
                proj.reset_parameters()

    @staticmethod
    def _clean_edge_index_dict(
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[EdgeType, Tensor]:
        clean_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0:
                clean_edge_index_dict[edge_type] = sort_edge_index(
                    edge_index,
                    sort_by_row=False,
                )
        return clean_edge_index_dict

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        x_dict = self.input_projection(x_dict)
        clean_edge_index_dict = self._clean_edge_index_dict(edge_index_dict)

        for layer_idx, conv in enumerate(self.convs):
            old_x_dict = x_dict
            conv_out = conv(old_x_dict, clean_edge_index_dict)

            new_x_dict = {}
            for nt, old_x in old_x_dict.items():
                h = conv_out.get(nt, old_x)
                if self.residual and h.shape == old_x.shape:
                    h = h + old_x
                if self.use_norm and nt in self.norms[layer_idx]:
                    h = self.norms[layer_idx][nt](h)
                if self.act is not None:
                    h = self.act(h)
                if self.dropout is not None:
                    h = self.dropout(h)
                new_x_dict[nt] = h
            x_dict = new_x_dict

        if self.output_projs is not None:
            x_dict = {
                nt: self.output_projs[nt](x)
                for nt, x in x_dict.items()
                if nt in self.output_projs
            }

        return x_dict


class HeteroGATv2OneStepEncoder(torch.nn.Module):
    """Local one-step heterogeneous GATv2 encoder.

    This restores the old direct GAT baseline under the current one-step
    encoder interface. Attention is edge-local, with one typed GATv2Conv per
    heterogeneous relation and hetero aggregation handled by HeteroConv.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        in_channels_dict: Dict[NodeType, int],
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        use_norm: bool = True,
        use_relu: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.use_norm = use_norm
        self.use_relu = use_relu
        self.residual = residual

        self.input_projection = TypeSpecificInputProjection(
            node_types=node_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
        )

        self.convs = torch.nn.ModuleList([
            HeteroConv(
                {
                    edge_type: GATv2Conv(
                        (hidden_channels, hidden_channels),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            for _ in range(num_layers)
        ])

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            if use_norm:
                for nt in node_types:
                    norm_dict[nt] = LayerNorm(hidden_channels, mode="node")
            self.norms.append(norm_dict)

        self.act = torch.nn.ReLU() if use_relu else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

        if self.out_channels != hidden_channels:
            self.output_projs = torch.nn.ModuleDict({
                nt: torch.nn.Linear(hidden_channels, self.out_channels)
                for nt in node_types
            })
        else:
            self.output_projs = None

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        if self.output_projs is not None:
            for proj in self.output_projs.values():
                proj.reset_parameters()

    @staticmethod
    def _clean_edge_index_dict(
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[EdgeType, Tensor]:
        clean_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0:
                clean_edge_index_dict[edge_type] = sort_edge_index(
                    edge_index,
                    sort_by_row=False,
                )
        return clean_edge_index_dict

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        x_dict = self.input_projection(x_dict)
        clean_edge_index_dict = self._clean_edge_index_dict(edge_index_dict)

        for layer_idx, conv in enumerate(self.convs):
            old_x_dict = x_dict
            conv_out = conv(old_x_dict, clean_edge_index_dict)

            new_x_dict = {}
            for nt, old_x in old_x_dict.items():
                h = conv_out.get(nt, old_x)
                if self.residual and h.shape == old_x.shape:
                    h = h + old_x
                if self.use_norm and nt in self.norms[layer_idx]:
                    h = self.norms[layer_idx][nt](h)
                if self.act is not None:
                    h = self.act(h)
                if self.dropout is not None:
                    h = self.dropout(h)
                new_x_dict[nt] = h
            x_dict = new_x_dict

        if self.output_projs is not None:
            x_dict = {
                nt: self.output_projs[nt](x)
                for nt, x in x_dict.items()
                if nt in self.output_projs
            }

        return x_dict


class HeteroGraphormerLayer(torch.nn.Module):
    """Full self-attention layer over a flattened heterogeneous subgraph."""

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float = 0.0,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(
                f"hidden_channels={channels} must be divisible by heads={heads}"
            )

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = torch.nn.LayerNorm(channels)
        self.norm2 = torch.nn.LayerNorm(channels)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(channels, ffn_multiplier * channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
            torch.nn.Linear(ffn_multiplier * channels, channels),
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def reset_parameters(self) -> None:
        self.attn._reset_parameters()
        for module in self.ffn.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(self, tokens: Tensor, attn_bias: Tensor) -> Tensor:
        # tokens: [N, C]; attn_bias: [N, N], additive attention mask.
        attn_out, _ = self.attn(
            tokens.unsqueeze(0),
            tokens.unsqueeze(0),
            tokens.unsqueeze(0),
            attn_mask=attn_bias,
            need_weights=False,
        )
        attn_out = attn_out.squeeze(0)
        tokens = self.norm1(tokens + self.dropout(attn_out))
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_out))
        return tokens


class HeteroGraphormerOneStepEncoder(torch.nn.Module):
    """Global one-step heterogeneous Graphormer-style encoder.

    This encoder flattens all nodes in the sampled heterogeneous subgraph into a
    single token sequence and applies full self-attention. Therefore, a node can
    attend to non-neighbor nodes in the same sampled subgraph. The graph
    structure is injected as additive attention bias.

    First implementation level:
        - node-type embedding
        - self-pair bias
        - direct relation bias for typed edges
        - non-edge bias for all other token pairs

    Optional future extension:
        - shortest-path distance bias
        - edge-path encoding
        - centrality/degree encoding
    """

    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        in_channels_dict: Dict[NodeType, int],
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        ffn_multiplier: int = 4,
        use_node_type_embedding: bool = True,
        bidirectional_relation_bias: bool = True,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.use_node_type_embedding = use_node_type_embedding
        self.bidirectional_relation_bias = bidirectional_relation_bias

        self.input_projection = TypeSpecificInputProjection(
            node_types=node_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
        )

        self.node_type_to_idx = {nt: idx for idx, nt in enumerate(node_types)}
        if use_node_type_embedding:
            self.node_type_embedding = torch.nn.Embedding(len(node_types), hidden_channels)
        else:
            self.node_type_embedding = None

        self.self_bias = torch.nn.Parameter(torch.zeros(()))
        self.non_edge_bias = torch.nn.Parameter(torch.zeros(()))
        self.edge_type_to_key = {
            edge_type: RelationWeightedSumFusion._edge_type_to_key(edge_type)
            for edge_type in edge_types
        }
        self.relation_bias = torch.nn.ParameterDict({
            self.edge_type_to_key[edge_type]: torch.nn.Parameter(torch.zeros(()))
            for edge_type in edge_types
        })

        self.layers = torch.nn.ModuleList([
            HeteroGraphormerLayer(
                channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                ffn_multiplier=ffn_multiplier,
            )
            for _ in range(num_layers)
        ])

        if self.out_channels != hidden_channels:
            self.output_projs = torch.nn.ModuleDict({
                nt: torch.nn.Linear(hidden_channels, self.out_channels)
                for nt in node_types
            })
        else:
            self.output_projs = None

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        if self.node_type_embedding is not None:
            self.node_type_embedding.reset_parameters()
        torch.nn.init.zeros_(self.self_bias)
        torch.nn.init.zeros_(self.non_edge_bias)
        for param in self.relation_bias.values():
            torch.nn.init.zeros_(param)
        for layer in self.layers:
            layer.reset_parameters()
        if self.output_projs is not None:
            for proj in self.output_projs.values():
                proj.reset_parameters()

    def _flatten_x_dict(
        self,
        x_dict: Dict[NodeType, Tensor],
    ) -> Tuple[Tensor, Dict[NodeType, Tuple[int, int]]]:
        tokens = []
        offsets: Dict[NodeType, Tuple[int, int]] = {}
        cursor = 0

        for nt in self.node_types:
            if nt not in x_dict:
                continue
            x = x_dict[nt]
            n = x.size(0)
            offsets[nt] = (cursor, cursor + n)
            cursor += n

            if self.node_type_embedding is not None:
                type_idx = torch.full(
                    (n,),
                    self.node_type_to_idx[nt],
                    device=x.device,
                    dtype=torch.long,
                )
                x = x + self.node_type_embedding(type_idx)

            tokens.append(x)

        if not tokens:
            raise RuntimeError("Cannot run Graphormer encoder on an empty x_dict.")

        return torch.cat(tokens, dim=0), offsets

    def _unflatten_tokens(
        self,
        tokens: Tensor,
        offsets: Dict[NodeType, Tuple[int, int]],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {}
        for nt, (start, end) in offsets.items():
            x_dict[nt] = tokens[start:end]
        return x_dict

    def _build_attention_bias(
        self,
        tokens: Tensor,
        offsets: Dict[NodeType, Tuple[int, int]],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Tensor:
        n_total = tokens.size(0)
        device = tokens.device
        dtype = tokens.dtype

        attn_bias = torch.zeros((n_total, n_total), device=device, dtype=dtype)
        attn_bias = attn_bias + self.non_edge_bias.to(device=device, dtype=dtype)

        diag_idx = torch.arange(n_total, device=device)
        attn_bias[diag_idx, diag_idx] = self.self_bias.to(device=device, dtype=dtype)

        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in self.edge_type_to_key:
                continue
            src_type, _rel_type, dst_type = edge_type
            if src_type not in offsets or dst_type not in offsets:
                continue
            if edge_index is None or edge_index.numel() == 0 or edge_index.size(1) == 0:
                continue

            src_start, src_end = offsets[src_type]
            dst_start, dst_end = offsets[dst_type]
            row = edge_index[0].to(device=device, dtype=torch.long) + src_start
            col = edge_index[1].to(device=device, dtype=torch.long) + dst_start

            valid = (
                (row >= src_start) & (row < src_end) &
                (col >= dst_start) & (col < dst_end)
            )
            if not torch.any(valid):
                continue

            row = row[valid]
            col = col[valid]
            bias_value = self.relation_bias[self.edge_type_to_key[edge_type]].to(
                device=device,
                dtype=dtype,
            )

            # MultiheadAttention uses attn_mask[target_query, source_key].
            # To let destination nodes attend to their source nodes, write
            # bias at [dst_global, src_global].
            attn_bias[col, row] = bias_value

            if self.bidirectional_relation_bias:
                attn_bias[row, col] = bias_value

        return attn_bias

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        x_dict = self.input_projection(x_dict)
        tokens, offsets = self._flatten_x_dict(x_dict)
        attn_bias = self._build_attention_bias(
            tokens=tokens,
            offsets=offsets,
            edge_index_dict=edge_index_dict,
        )

        for layer in self.layers:
            tokens = layer(tokens, attn_bias)

        x_dict = self._unflatten_tokens(tokens, offsets)

        if self.output_projs is not None:
            x_dict = {
                nt: self.output_projs[nt](x)
                for nt, x in x_dict.items()
                if nt in self.output_projs
            }

        return x_dict


def build_graph_encoder(
    encoder_family: str,
    encoder_name: str,
    node_types: List[NodeType],
    edge_types: List[EdgeType],
    in_channels_dict: Dict[NodeType, int],
    hidden_channels: int,
    out_channels: Optional[int] = None,
    num_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.0,
    # staged options
    intra_aggr: str = "mean",
    type_fusion: str = "weighted_sum",
    node_update: str = "mlp",
    use_norm: bool = True,
    use_relu: bool = True,
    residual: bool = True,
    residual_weight: float = 0.5,
    # quantum/recurrent options for staged variants
    gru_type: str = "gru",
    n_qubits: int = 8,
    n_q_layers: int = 1,
    n_heads: Optional[int] = None,
    q_readout_mode: str = "z_pairwise",
    q_circuit_type: str = "angle",
    q_ansatz_type: str = "rot_cnot_ring",
    q_angle_activation: str = "tanh",
    q_use_angle_affine: bool = False,
    q_residual_mode: str = "add",
    q_alpha_init: float = 0.0,
    export_circuit: bool = True,
    freeze_quantum_at_init: bool = False,
    freeze_alpha_with_quantum: bool = True,
    # graphormer options
    ffn_multiplier: int = 4,
    bidirectional_relation_bias: bool = True,
    use_node_type_embedding: bool = True,
) -> torch.nn.Module:
    """Build a graph encoder using the staged/one-step taxonomy.

    Args:
        encoder_family:
            ``"staged"`` keeps the existing explicit aggregation/fusion/update
            design. ``"one_step"`` uses integrated one-step encoders.
        encoder_name:
            For staged encoders, use names such as ``"sage"`` for the current
            SAGE-based staged implementation. For one-step encoders, use
            ``"hgt"`` or the canonical ``"graphormer"`` spelling.
    """
    encoder_family = _normalize_encoder_family(encoder_family)
    encoder_name = _normalize_encoder_name(encoder_name)

    if encoder_family == "staged":
        if encoder_name not in {"sage", "hetero_sage", "hetero_recurrent_gnn"}:
            raise ValueError(
                "The current staged implementation supports encoder_name "
                f"'sage'/'hetero_sage'. Got encoder_name={encoder_name}."
            )

        staged_encoder = HeteroRecurrentGNN(
            node_types=node_types,
            edge_types=edge_types,
            channels=hidden_channels,
            intra_aggr=intra_aggr,
            type_fusion=type_fusion,
            node_update=node_update,
            num_layers=num_layers,
            use_norm=use_norm,
            use_relu=use_relu,
            dropout=dropout,
            residual=residual,
            residual_weight=residual_weight,
            gru_type=gru_type,
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            n_heads=n_heads,
            gat_heads=4,
            q_readout_mode=q_readout_mode,
            q_circuit_type=q_circuit_type,
            q_ansatz_type=q_ansatz_type,
            q_angle_activation=q_angle_activation,
            q_use_angle_affine=q_use_angle_affine,
            q_residual_mode=q_residual_mode,
            q_alpha_init=q_alpha_init,
            export_circuit=export_circuit,
            freeze_quantum_at_init=freeze_quantum_at_init,
            freeze_alpha_with_quantum=freeze_alpha_with_quantum,
        )

        return TypeProjectedEncoder(
            encoder=staged_encoder,
            node_types=node_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
        )

    if encoder_family == "one_step":
        if encoder_name == "hgt":
            return HGTOneStepEncoder(
                node_types=node_types,
                edge_types=edge_types,
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                use_norm=use_norm,
                use_relu=use_relu,
                residual=residual,
            )

        if encoder_name == "graphormer":
            return HeteroGraphormerOneStepEncoder(
                node_types=node_types,
                edge_types=edge_types,
                in_channels_dict=in_channels_dict,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                ffn_multiplier=ffn_multiplier,
                use_node_type_embedding=use_node_type_embedding,
                bidirectional_relation_bias=bidirectional_relation_bias,
            )

    raise ValueError(
        f"Unknown encoder_family={encoder_family}, encoder_name={encoder_name}"
    )
