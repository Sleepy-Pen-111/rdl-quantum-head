from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from relbench.modeling.nn import (
    HeteroEncoder,
    HeteroTemporalEncoder,
)

from quantum_model.model.nn_new import (
    HeteroRecurrentGNN,
    HGTOneStepEncoder,
    HeteroGraphormerOneStepEncoder,
)
from quantum_model.model.prediction_head import build_prediction_head

# 将所有相关的 numpy 类型一次性添加到安全白名单
from numpy._core.multiarray import scalar
torch.serialization.add_safe_globals([scalar])
from numpy import dtype
torch.serialization.add_safe_globals([dtype])


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
        gnn: str = "staged",

        # New recurrent hetero GNN arguments
        intra_aggr: str = "mean",
        type_fusion: str = "sum",
        node_update: str = "mlp",
        n_qubits: int = 4,
        n_q_layers: int = 2,
        n_heads: int | None = None,
        q_readout_mode: str = "z_pairwise",
        q_circuit_type: str = "angle",
        q_ansatz_type: str = "rot_cnot_ring",
        q_angle_activation: str = "tanh",
        q_use_angle_affine: bool = False,
        q_residual_mode: str = "concat",
        q_alpha_init: float = 0.0,
        
        q_freeze_quantum_at_init: bool = False,
        q_freeze_alpha_with_quantum: bool = True,
        q_quantum_unfreeze_epoch: int | None = None,
        q_quantum_only_finetune: bool = False,

        prediction_head: str = "classical",
        prediction_head_hidden_dim: int | None = None,
        prediction_head_num_layers: int = 1,
        prediction_head_dropout: float = 0.0,
        prediction_head_n_qubits: int | None = None,
        prediction_head_n_q_layers: int | None = None,
        prediction_head_n_heads: int | None = None,
        prediction_head_q_readout_mode: str | None = None,
        prediction_head_q_circuit_type: str | None = None,
        prediction_head_q_ansatz_type: str | None = None,
        prediction_head_q_angle_activation: str | None = None,
        prediction_head_q_use_angle_affine: bool | None = None,
        prediction_head_q_dropout: float = 0.0,
        prediction_head_q_residual_mode: str = "add",
        prediction_head_q_alpha_init: float = 0.0,
        prediction_head_freeze_quantum_at_init: bool | None = None,
        prediction_head_freeze_alpha_with_quantum: bool | None = None,
        prediction_head_export_circuit: bool = False,

        # GAT arguments
        gat_heads: int = 4,
        gat_dropout: float = 0.0,


    ):
        super().__init__()
        
        self.q_quantum_unfreeze_epoch = q_quantum_unfreeze_epoch
        self.q_quantum_only_finetune = q_quantum_only_finetune
        
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        gnn = str(gnn).lower().replace("-", "_")
        intra_aggr = str(intra_aggr).lower().replace("-", "_")
        in_channels_dict = {node_type: channels for node_type in data.node_types}

        if gnn == "staged":
            # Staged encoder: explicit relation aggregation -> type fusion -> node update.
            # This preserves the old HeteroRecurrentGNN path and lets GATv2
            # participate inside the same staged design space.
            self.gnn = HeteroRecurrentGNN(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                intra_aggr=intra_aggr,
                type_fusion=type_fusion,
                node_update=node_update,
                num_layers=num_layers,
                use_norm=True,
                use_relu=True,
                dropout=gat_dropout,
                residual=False,
                residual_weight=0.5,
                n_qubits=n_qubits,
                n_q_layers=n_q_layers,
                n_heads=n_heads,
                gat_heads=gat_heads,
                q_readout_mode=q_readout_mode,
                q_circuit_type=q_circuit_type,
                q_ansatz_type=q_ansatz_type,
                q_angle_activation=q_angle_activation,
                q_use_angle_affine=q_use_angle_affine,
                q_residual_mode=q_residual_mode,
                q_alpha_init=q_alpha_init,
                export_circuit=True,
                freeze_quantum_at_init=q_freeze_quantum_at_init,
                freeze_alpha_with_quantum=q_freeze_alpha_with_quantum,
            )

        elif gnn == "hgt":
            # One-step local heterogeneous transformer. Attention is still edge-local.
            self.gnn = HGTOneStepEncoder(
                node_types=data.node_types,
                edge_types=data.edge_types,
                in_channels_dict=in_channels_dict,
                hidden_channels=channels,
                out_channels=channels,
                num_layers=num_layers,
                heads=gat_heads,
                dropout=gat_dropout,
                use_norm=True,
                use_relu=True,
                residual=True,
            )

        elif gnn == "graphormer":
            # One-step global Graphormer-style encoder over the sampled subgraph.
            # Canonical spelling is graphormer; graphformer is intentionally not accepted.
            self.gnn = HeteroGraphormerOneStepEncoder(
                node_types=data.node_types,
                edge_types=data.edge_types,
                in_channels_dict=in_channels_dict,
                hidden_channels=channels,
                out_channels=channels,
                num_layers=num_layers,
                heads=gat_heads,
                dropout=gat_dropout,
                ffn_multiplier=4,
                use_node_type_embedding=True,
                bidirectional_relation_bias=True,
            )

        else:
            raise ValueError(
                f"Unknown gnn={gnn!r}. Expected one of: 'staged', 'hgt', 'graphormer'."
            )

        head_n_qubits = n_qubits if prediction_head_n_qubits is None else prediction_head_n_qubits
        head_n_q_layers = n_q_layers if prediction_head_n_q_layers is None else prediction_head_n_q_layers
        head_n_heads = n_heads if prediction_head_n_heads is None else prediction_head_n_heads
        head_q_readout_mode = (
            q_readout_mode
            if prediction_head_q_readout_mode is None
            else prediction_head_q_readout_mode
        )
        head_q_circuit_type = (
            q_circuit_type
            if prediction_head_q_circuit_type is None
            else prediction_head_q_circuit_type
        )
        head_q_ansatz_type = (
            q_ansatz_type
            if prediction_head_q_ansatz_type is None
            else prediction_head_q_ansatz_type
        )
        head_q_angle_activation = (
            q_angle_activation
            if prediction_head_q_angle_activation is None
            else prediction_head_q_angle_activation
        )
        head_q_use_angle_affine = (
            q_use_angle_affine
            if prediction_head_q_use_angle_affine is None
            else prediction_head_q_use_angle_affine
        )
        head_freeze_quantum_at_init = (
            q_freeze_quantum_at_init
            if prediction_head_freeze_quantum_at_init is None
            else prediction_head_freeze_quantum_at_init
        )
        head_freeze_alpha_with_quantum = (
            q_freeze_alpha_with_quantum
            if prediction_head_freeze_alpha_with_quantum is None
            else prediction_head_freeze_alpha_with_quantum
        )

        self.head = build_prediction_head(
            head_type=prediction_head,
            input_dim=channels,
            output_dim=out_channels,
            norm=norm,
            hidden_dim=prediction_head_hidden_dim,
            num_layers=prediction_head_num_layers,
            dropout=prediction_head_dropout,
            n_qubits=head_n_qubits,
            n_q_layers=head_n_q_layers,
            n_heads=head_n_heads,
            q_readout_mode=head_q_readout_mode,
            q_circuit_type=head_q_circuit_type,
            q_ansatz_type=head_q_ansatz_type,
            q_angle_activation=head_q_angle_activation,
            q_use_angle_affine=head_q_use_angle_affine,
            q_dropout=prediction_head_q_dropout,
            q_residual_mode=prediction_head_q_residual_mode,
            q_alpha_init=prediction_head_q_alpha_init,
            freeze_quantum_at_init=head_freeze_quantum_at_init,
            freeze_alpha_with_quantum=head_freeze_alpha_with_quantum,
            export_circuit=prediction_head_export_circuit,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()
             
    @staticmethod
    def _set_module_requires_grad(module: torch.nn.Module | None, requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(requires_grad)


    def reset_prediction_head(self) -> None:
        """Reset only the prediction head.

        This is used by two-stage experiments: load a shared backbone checkpoint,
        discard the old pretraining head, and train a newly initialized head.
        """
        if not hasattr(self, "head") or self.head is None:
            raise RuntimeError("Model has no prediction head to reset.")
        if not hasattr(self.head, "reset_parameters"):
            raise RuntimeError("Prediction head does not implement reset_parameters().")
        self.head.reset_parameters()

    def freeze_backbone(self) -> None:
        """Freeze all modules except the prediction head."""
        self._set_module_requires_grad(self.encoder, False)
        self._set_module_requires_grad(self.temporal_encoder, False)
        self._set_module_requires_grad(self.gnn, False)
        self._set_module_requires_grad(self.embedding_dict, False)
        self._set_module_requires_grad(self.id_awareness_emb, False)
        self._set_module_requires_grad(self.head, True)

    def unfreeze_backbone(self) -> None:
        """Make the graph backbone trainable again.

        This keeps the prediction head trainable as well, enabling end-to-end fine-tuning.
        """
        self._set_module_requires_grad(self.encoder, True)
        self._set_module_requires_grad(self.temporal_encoder, True)
        self._set_module_requires_grad(self.gnn, True)
        self._set_module_requires_grad(self.embedding_dict, True)
        self._set_module_requires_grad(self.id_awareness_emb, True)
        self._set_module_requires_grad(self.head, True)

    def trainable_parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(param.numel() for param in self.parameters()),
            "trainable": sum(
                param.numel() for param in self.parameters() if param.requires_grad
            ),
        }

    def freeze_quantum(self) -> None:
        if hasattr(self.gnn, "freeze_quantum"):
            self.gnn.freeze_quantum()
        if hasattr(self.head, "freeze_quantum"):
            self.head.freeze_quantum()

    def unfreeze_quantum(self) -> None:
        if hasattr(self.gnn, "unfreeze_quantum"):
            self.gnn.unfreeze_quantum()
        if hasattr(self.head, "unfreeze_quantum"):
            self.head.unfreeze_quantum()

    def freeze_classical(self) -> None:
        self._set_module_requires_grad(self.encoder, False)
        self._set_module_requires_grad(self.temporal_encoder, False)
        self._set_module_requires_grad(self.embedding_dict, False)
        self._set_module_requires_grad(self.id_awareness_emb, False)

        if hasattr(self.gnn, "freeze_classical"):
            self.gnn.freeze_classical()
        else:
            self._set_module_requires_grad(self.gnn, False)

        if hasattr(self.head, "freeze_classical"):
            self.head.freeze_classical()
        else:
            self._set_module_requires_grad(self.head, False)

    def unfreeze_classical(self) -> None:
        self._set_module_requires_grad(self.encoder, True)
        self._set_module_requires_grad(self.temporal_encoder, True)
        self._set_module_requires_grad(self.embedding_dict, True)
        self._set_module_requires_grad(self.id_awareness_emb, True)

        if hasattr(self.gnn, "unfreeze_classical"):
            self.gnn.unfreeze_classical()
        else:
            self._set_module_requires_grad(self.gnn, True)

        if hasattr(self.head, "unfreeze_classical"):
            self.head.unfreeze_classical()
        else:
            self._set_module_requires_grad(self.head, True)

             
    def apply_quantum_schedule(self, epoch: int) -> None:
        if self.q_quantum_unfreeze_epoch is None:
            return

        if epoch == self.q_quantum_unfreeze_epoch:
            self.unfreeze_quantum()

            if self.q_quantum_only_finetune:
                self.freeze_classical()    

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])
