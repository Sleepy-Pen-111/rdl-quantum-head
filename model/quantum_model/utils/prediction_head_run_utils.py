from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


# These fields are used only by this launcher for naming and CSV grouping.
# They must NOT be passed to gnn_entity_balanced.py / gnn_entity_2stage.py.
LAUNCHER_ONLY_ARG_KEYS = {
    "head_training_mode",
    "graph_family",
    "graph_config_tag",
    "matched_param_tolerance_abs",
    "matched_param_tolerance_rel",
}

DEFAULT_MATCHED_PARAM_TOLERANCE_REL = 0.05
DEFAULT_MATCHED_PARAM_TOLERANCE_ABS = 100


# ============================================================
# Parameter counting helpers
# ============================================================


def readout_dim(n_qubits: int, readout_mode: str) -> int:
    if readout_mode == "z_only":
        return n_qubits
    if readout_mode == "z_pairwise":
        return n_qubits + n_qubits * (n_qubits - 1) // 2
    if readout_mode == "z_all":
        return 2**n_qubits - 1
    if readout_mode == "probs":
        return 2**n_qubits
    raise ValueError(f"Unknown readout mode: {readout_mode}")


def quantum_input_dim(n_qubits: int, circuit_type: str) -> int:
    if circuit_type == "rxry":
        return 2 * n_qubits
    if circuit_type == "amplitude":
        return 2**n_qubits
    return n_qubits


def is_feasible_quantum_config(
    *,
    n_qubits: int,
    circuit_type: str,
    readout_mode: str,
    max_quantum_input_dim: int,
    max_readout_dim: int,
) -> tuple[bool, str | None]:
    q_input_dim = quantum_input_dim(n_qubits, circuit_type)
    q_output_dim = readout_dim(n_qubits, readout_mode)

    if q_input_dim > max_quantum_input_dim:
        return False, f"quantum_input_dim={q_input_dim} exceeds limit {max_quantum_input_dim}"

    if q_output_dim > max_readout_dim:
        return False, f"readout_dim={q_output_dim} exceeds limit {max_readout_dim}"

    return True, None


def exceeds_param_limit(param_count: int, max_params: int) -> bool:
    return max_params > 0 and param_count > max_params


def classical_head_param_count(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.0,
    norm: str | None = "batch_norm",
) -> Any:
    del dropout  # Dropout does not change trainable parameter count.

    num_layers = max(int(num_layers), 1)
    hidden_dim = int(hidden_dim)

    channel_list = [int(input_dim)]
    if num_layers > 1:
        channel_list.extend([hidden_dim] * (num_layers - 1))
    channel_list.append(int(output_dim))

    linear_params = 0
    for in_dim, out_dim in zip(channel_list[:-1], channel_list[1:]):
        linear_params += in_dim * out_dim + out_dim

    norm_params = 0
    if norm is not None:
        # torch_geometric.nn.MLP(..., plain_last=True) applies normalization only
        # on hidden layers, and common norms here use affine weight+bias.
        for hidden_channels in channel_list[1:-1]:
            norm_params += 2 * hidden_channels

    return linear_params + norm_params


def quantum_layer_param_count(
    *,
    n_qubits: int,
    n_q_layers: int,
    q_ansatz_type: str,
) -> int:
    from quantum_model.model.quantum_ansatz import get_q_ansatz_weight_shapes

    weight_shapes = get_q_ansatz_weight_shapes(
        q_ansatz_type,
        n_q_layers=n_q_layers,
        n_qubits=n_qubits,
    )
    return sum(
        _shape_numel(shape)
        for shape in weight_shapes.values()
    )


def _shape_numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def quantum_prediction_head_param_count(
    *,
    input_dim: int,
    output_dim: int,
    n_qubits: int,
    n_q_layers: int,
    n_heads: int | None,
    q_readout_mode: str,
    q_circuit_type: str,
    q_ansatz_type: str,
    q_use_angle_affine: bool,
) -> int:
    head_count = 1 if n_heads is None else int(n_heads)
    q_input_dim = quantum_input_dim(n_qubits, q_circuit_type)
    q_readout_dim = readout_dim(n_qubits, q_readout_mode)

    pre_net_params_per_head = input_dim * q_input_dim + q_input_dim
    affine_params_per_head = 2 * q_input_dim if q_use_angle_affine else 0
    q_layer_params_per_head = quantum_layer_param_count(
        n_qubits=n_qubits,
        n_q_layers=n_q_layers,
        q_ansatz_type=q_ansatz_type,
    )

    total_readout_dim = head_count * q_readout_dim
    post_net_params = total_readout_dim * output_dim + output_dim

    return head_count * (
        pre_net_params_per_head
        + affine_params_per_head
        + q_layer_params_per_head
    ) + post_net_params


def residual_quantum_prediction_head_param_count(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    norm: str | None,
    n_qubits: int,
    n_q_layers: int,
    n_heads: int | None,
    q_readout_mode: str,
    q_circuit_type: str,
    q_ansatz_type: str,
    q_use_angle_affine: bool,
    q_residual_mode: str,
) -> int:
    classical_params = classical_head_param_count(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        norm=norm,
    )
    quantum_params = quantum_prediction_head_param_count(
        input_dim=input_dim,
        output_dim=output_dim,
        n_qubits=n_qubits,
        n_q_layers=n_q_layers,
        n_heads=n_heads,
        q_readout_mode=q_readout_mode,
        q_circuit_type=q_circuit_type,
        q_ansatz_type=q_ansatz_type,
        q_use_angle_affine=q_use_angle_affine,
    )

    if str(q_residual_mode).lower() == "concat":
        mix_params = (2 * output_dim) * output_dim + output_dim
        return classical_params + quantum_params + mix_params

    alpha_params = output_dim
    return classical_params + quantum_params + alpha_params


def build_quantum_head_for_counting(
    *,
    head_type: str,
    input_dim: int,
    output_dim: int,
    n_qubits: int,
    n_q_layers: int,
    q_readout_mode: str,
    q_circuit_type: str,
    q_ansatz_type: str,
    q_angle_activation: str,
    q_use_angle_affine: bool,
    q_dropout: float = 0.0,
    q_residual_mode: str | None = None,
) -> Any:
    del q_angle_activation, q_dropout

    if head_type == "quantum":
        return quantum_prediction_head_param_count(
            input_dim=input_dim,
            output_dim=output_dim,
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            n_heads=None,
            q_readout_mode=q_readout_mode,
            q_circuit_type=q_circuit_type,
            q_ansatz_type=q_ansatz_type,
            q_use_angle_affine=q_use_angle_affine,
        )

    if head_type == "residual_quantum":
        return residual_quantum_prediction_head_param_count(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=input_dim,
            num_layers=1,
            dropout=0.0,
            norm="batch_norm",
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            n_heads=None,
            q_readout_mode=q_readout_mode,
            q_circuit_type=q_circuit_type,
            q_ansatz_type=q_ansatz_type,
            q_use_angle_affine=q_use_angle_affine,
            q_residual_mode=q_residual_mode or "add",
        )

    raise ValueError(f"Unsupported quantum counting head_type={head_type!r}")


def find_param_matched_hidden_dim(
    *,
    input_dim: int,
    output_dim: int,
    num_layers: int,
    target_param_count: int,
    max_hidden_dim: int,
    dropout: float = 0.0,
) -> tuple[int, int]:
    best_hidden_dim = 1
    best_param_count = classical_head_param_count(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=1,
        num_layers=num_layers,
        dropout=dropout,
    )
    best_gap = abs(best_param_count - target_param_count)

    for hidden_dim in range(2, max_hidden_dim + 1):
        param_count = classical_head_param_count(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        gap = abs(param_count - target_param_count)

        if gap < best_gap:
            best_hidden_dim = hidden_dim
            best_param_count = param_count
            best_gap = gap

    return best_hidden_dim, best_param_count


def matched_param_tolerance(
    *,
    target_param_count: int,
    rel_tolerance: float,
    abs_tolerance: int,
) -> int:
    relative_window = abs(int(target_param_count)) * max(float(rel_tolerance), 0.0)
    return max(int(abs_tolerance), int(round(relative_window)))


def find_reusable_matched_mlp(
    *,
    cache: Sequence[dict[str, Any]],
    match_signature: tuple[Any, ...],
    target_param_count: int,
    rel_tolerance: float,
    abs_tolerance: int,
) -> dict[str, Any] | None:
    tolerance = matched_param_tolerance(
        target_param_count=target_param_count,
        rel_tolerance=rel_tolerance,
        abs_tolerance=abs_tolerance,
    )
    best_entry = None
    best_gap = None

    for entry in cache:
        if entry["match_signature"] != match_signature:
            continue

        gap = abs(int(entry["target_param_count"]) - int(target_param_count))
        if gap <= tolerance and (best_gap is None or gap < best_gap):
            best_entry = entry
            best_gap = gap

    return best_entry


# ============================================================
# Naming helpers
# ============================================================

def sanitize_fragment(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("/", "-").replace("\\", "-").replace(" ", "-")
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9.+-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "none"


def runtime_safe_run_name(value: Any) -> str:
    return sanitize_fragment(value)


NAME_ALIASES = {
    "angle": "ang",
    "arctan": "atan",
    "amplitude": "amp",
    "rxry": "rxry",

    "rot-cnot-ring": "rcr",
    "rot-cnot-chain": "rcc",
    "rot-cz-ring": "rcz",
    "rot-none": "r0",

    "z-only": "zo",
    "z-pairwise": "zp",
    "z-all": "za",
    "probs": "pr",

    "tanh": "th",
    "atan": "at",
    "none": "id",

    "staged": "stg",
    "hgt": "hgt",
    "graphormer": "gormer",
    "recurrent": "rec",
    "graphsage": "sage",
    "sage": "sage",
    "gat": "gat",
    "gatv2": "gatv2",
    "gat-v2": "gatv2",
    "transformer": "trf",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",

    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph-transformer": "gtr",
    "graph_transformer": "gtr",
    "gtrans": "gtrans",

    "max-project": "xproj",
    "mean-project": "mproj",
    "projectmax": "pmax",
    "projectmin": "pmin",
    "lastmin": "lastmin",
    "lasta": "lasta",
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "mlp": "mlp",
    "gru": "gru",

    "classical": "mlp",
    "quantum": "qhead",
    "residual-quantum": "rqhead",

    "add": "add",
    "concat": "cat",

    "uniform": "uni",
    "last": "last",
    "total": "tot",
    "per-edge-type": "pet",

    "head-only": "headonly",
    "end-to-end": "endtoend",
}


def short_value(value: Any) -> str:
    text = sanitize_fragment(value)
    return NAME_ALIASES.get(text, text)


def build_graph_family_tag(args: argparse.Namespace) -> str:
    """Return a short, human-readable graph backbone family name.

    Canonical launcher-level GNN names are now:
    - staged: explicit relation aggregation / type fusion / node update
    - hgt: local one-step heterogeneous transformer
    - graphormer: global Graphormer-style one-step encoder
    """
    gnn = sanitize_fragment(getattr(args, "gnn", "staged"))

    if gnn == "graphormer":
        return "Graphormer"
    if gnn == "hgt":
        return "HGT"
    if gnn == "staged":
        return "Staged"

    # Defensive fallback for old summaries/plans that may still be present.
    if gnn == "recurrent":
        return "Staged"
    if gnn in {"sage", "graphsage"}:
        return "GraphSage"

    return runtime_safe_run_name(gnn)


def build_graph_config_tag(args: argparse.Namespace) -> str:
    """Return a graph configuration tag fine-grained enough for grouping.

    For one-step encoders, the family name is already the configuration. For
    staged encoders, include the three staged choices so different aggregation /
    fusion / node-update variants do not collapse into one summary bucket.
    """
    family = build_graph_family_tag(args)
    gnn = sanitize_fragment(getattr(args, "gnn", "staged"))

    if gnn in {"staged", "recurrent"}:
        intra_aggr = sanitize_fragment(getattr(args, "intra_aggr", "mean"))
        bits = [
            family,
            short_value(intra_aggr),
            short_value(getattr(args, "type_fusion", "sum")),
            short_value(getattr(args, "node_update", "mlp")),
        ]
        return runtime_safe_run_name("-".join(bits))

    return family


def normalize_head_training_mode(mode: str) -> str:
    """Canonical mode names used by the launcher.

    end_to_end is kept as a backward-compatible alias for finetune.
    """
    if mode == "end_to_end":
        return "finetune"
    return mode


def uses_pretrained_backbone(args: argparse.Namespace) -> bool:
    return (
        args.two_stage_head_search
        and normalize_head_training_mode(args.head_training_mode) in {"head_only", "finetune"}
    )


def build_head_training_mode_tag(args: argparse.Namespace) -> str:
    mode = normalize_head_training_mode(args.head_training_mode)
    if mode == "head_only":
        return "headonly"
    if mode == "finetune":
        return "finetune"
    if mode == "scratch":
        return "scratch"
    return runtime_safe_run_name(mode)


def _uses_legacy_staged_storage_layout(args: argparse.Namespace) -> bool:
    """Preserve old folder names for the historical default staged setting.

    Earlier launcher runs stored all staged variants under the family-only
    folder name. To keep previously produced default runs reusable, we retain
    that old layout only for the long-used default staged configuration.
    """
    gnn = sanitize_fragment(getattr(args, "gnn", "staged"))
    if gnn not in {"staged", "recurrent"}:
        return False

    intra_aggr = short_value(getattr(args, "intra_aggr", "mean"))
    type_fusion = short_value(getattr(args, "type_fusion", "sum"))
    node_update = short_value(getattr(args, "node_update", "mlp"))

    return (
        intra_aggr == "xproj"
        and type_fusion == "sum"
        and node_update == "mlp"
    )


def build_graph_storage_tag(args: argparse.Namespace) -> str:
    """Return the folder tag used for experiment/backbone storage.

    New staged variants use the full graph configuration tag so different
    aggregation/fusion/update settings do not collide. The historical default
    staged setting keeps the old family-only layout for backward compatibility.
    """
    if _uses_legacy_staged_storage_layout(args):
        return runtime_safe_run_name(build_graph_family_tag(args))
    return runtime_safe_run_name(build_graph_config_tag(args))


def get_graph_root(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_dir)
        / sanitize_fragment(args.dataset)
        / sanitize_fragment(args.task)
        / build_graph_storage_tag(args)
    )


def get_experiment_root(args: argparse.Namespace) -> Path:
    folder_name = runtime_safe_run_name(
        f"{build_graph_storage_tag(args)}-{build_head_training_mode_tag(args)}"
    )
    return (
        Path(args.output_dir)
        / sanitize_fragment(args.dataset)
        / sanitize_fragment(args.task)
        / folder_name
    )


def get_shared_backbone_root(args: argparse.Namespace) -> Path:
    return get_graph_root(args) / "shared-backbone"


def build_q_config_tag(
    *,
    n_qubits: int,
    n_q_layers: int,
    q_circuit_type: str,
    q_ansatz_type: str,
    q_readout_mode: str,
    q_angle_activation: str,
    q_use_angle_affine: bool,
    q_dropout: float = 0.0,
) -> str:
    bits = [
        f"q{n_qubits}x{n_q_layers}",
        short_value(q_circuit_type),
        short_value(q_ansatz_type),
        short_value(q_readout_mode),
        short_value(q_angle_activation),
    ]
    if q_use_angle_affine:
        bits.append("aff")
    if float(q_dropout) > 0.0:
        bits.append(f"qd{short_value(q_dropout)}")
    return runtime_safe_run_name("-".join(bits))


def build_backbone_run_name(args: argparse.Namespace, param_count: int) -> str:
    pieces = [
        "backbone",
        build_graph_family_tag(args),
        f"h{args.backbone_hidden_dim}-l{args.backbone_head_layers}",
        f"p{param_count}",
        f"seed{args.seed}",
    ]
    return runtime_safe_run_name("-".join(pieces))


def build_run_name(
    *,
    role: str,
    args: argparse.Namespace,
    q_config_tag: str | None = None,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    param_count: int | None = None,
    target_param_count: int | None = None,
    residual_mode: str | None = None,
) -> str:
    pieces: list[str] = [
        role,
        build_graph_family_tag(args),
        build_head_training_mode_tag(args),
    ]

    if q_config_tag is not None:
        if role == "matched-mlp":
            pieces.append(f"for-{q_config_tag}")
        else:
            pieces.append(q_config_tag)

    if residual_mode is not None:
        pieces.append(f"res-{short_value(residual_mode)}")

    if hidden_dim is not None and num_layers is not None:
        pieces.append(f"h{hidden_dim}-l{num_layers}")

    if dropout is not None:
        try:
            dropout_value = float(dropout)
        except (TypeError, ValueError):
            dropout_value = None

        if dropout_value is None or dropout_value > 0.0:
            pieces.append(f"do{short_value(dropout)}")

    if param_count is not None:
        param_tag = f"p{param_count}"
        if target_param_count is not None:
            param_tag += f"-t{target_param_count}"
        pieces.append(param_tag)

    pieces.append(f"seed{args.seed}")

    return runtime_safe_run_name("-".join(pieces))


# ============================================================
# Experiment spec
# ============================================================

@dataclass
class ExperimentSpec:
    name: str
    role: str
    group_id: str
    base_quantum_name: str | None
    args: dict[str, Any]
    param_count: int
    matched_target_param_count: int | None = None
    q_config_tag: str | None = None
    matched_baseline_name: str | None = None


# ============================================================
# Base args and matrix construction
# ============================================================

def build_base_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "task": args.task,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "early_stopping_patience": getattr(args, "early_stopping_patience", None),
        "early_stopping_min_delta": getattr(args, "early_stopping_min_delta", 0.0),

        "gnn": args.gnn,
        "intra_aggr": args.intra_aggr,
        "type_fusion": args.type_fusion,
        "node_update": args.node_update,
        "num_layers": args.num_layers,

        "num_neighbors": args.num_neighbors,
        "neighbor_sampling_mode": args.neighbor_sampling_mode,
        "temporal_strategy": args.temporal_strategy,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "num_workers": args.num_workers,
        "seed": args.seed,

        "cache_dir": args.cache_dir,
        "output_dir": str(get_experiment_root(args)),

        "prediction_head_dropout": 0.0,
        "prediction_head_q_dropout": 0.0,
        "regression_tune_metric": getattr(args, "regression_tune_metric", "r2"),

        "resume": args.resume,

        # Launcher-only metadata. This is kept in plan/summary, but filtered
        # before calling the trainer.
        "head_training_mode": getattr(args, "head_training_mode", "scratch"),
        "graph_family": build_graph_family_tag(args),
        "graph_config_tag": build_graph_config_tag(args),
    }


def unique_preserving_order(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_backbone_pretrain_spec(args: argparse.Namespace) -> ExperimentSpec:
    base_args = build_base_args(args)
    base_args["epochs"] = args.backbone_epochs
    base_args["resume"] = args.resume_backbone
    base_args["output_dir"] = str(get_shared_backbone_root(args))

    param_count = classical_head_param_count(
        input_dim=args.channels,
        output_dim=1,
        hidden_dim=args.backbone_hidden_dim,
        num_layers=args.backbone_head_layers,
    )

    run_name = build_backbone_run_name(args, param_count)

    run_args = dict(base_args)
    run_args.update(
        {
            "run_name": run_name,
            "prediction_head": "classical",
            "prediction_head_hidden_dim": args.backbone_hidden_dim,
            "prediction_head_num_layers": args.backbone_head_layers,
        }
    )

    return ExperimentSpec(
        name=run_name,
        role="backbone_pretrain",
        group_id=f"shared_backbone_seed{args.seed}",
        base_quantum_name=None,
        args=run_args,
        param_count=param_count,
        matched_target_param_count=None,
        q_config_tag=None,
        matched_baseline_name=None,
    )


def build_backbone_checkpoint_path(args: argparse.Namespace) -> Path:
    backbone_spec = build_backbone_pretrain_spec(args)
    return get_shared_backbone_root(args) / backbone_spec.name / "best_checkpoint.pt"


def apply_head_training_mode_args(
    *,
    run_args: dict[str, Any],
    args: argparse.Namespace,
    backbone_checkpoint_path: Path | None,
) -> dict[str, Any]:
    """Inject trainer arguments for the selected head-training protocol.

    Modes:
    - head_only: load MLP-pretrained backbone, reset head, freeze backbone.
    - finetune: load MLP-pretrained backbone, reset head, train backbone + head.
    - scratch: do not load any pretrained checkpoint; train full model from random init.

    The old name end_to_end is accepted as an alias for finetune.
    """
    updated = dict(run_args)
    mode = normalize_head_training_mode(args.head_training_mode)

    # All candidate-head runs use head_epochs. In scratch mode this is the
    # total full-model training budget from random initialization.
    updated["epochs"] = args.head_epochs
    updated["resume"] = args.resume_heads

    if not args.two_stage_head_search:
        return updated

    if mode == "scratch":
        # True full end-to-end from random initialization.
        # No init_backbone_checkpoint, no load_backbone_only, no freeze.
        return updated

    if mode == "head_only":
        freeze_backbone = True
    elif mode == "finetune":
        freeze_backbone = False
    else:
        raise ValueError(f"Unknown head_training_mode: {args.head_training_mode}")

    if backbone_checkpoint_path is None:
        raise ValueError(f"Mode {mode} requires a backbone checkpoint path.")

    updated.update(
        {
            "init_backbone_checkpoint": str(backbone_checkpoint_path),
            "load_backbone_only": True,
            "freeze_backbone": freeze_backbone,
            "reset_prediction_head": True,
        }
    )

    return updated


def build_experiment_matrix(args: argparse.Namespace) -> tuple[list[ExperimentSpec], list[str]]:
    experiments: list[ExperimentSpec] = []
    skipped: list[str] = []
    base_args = build_base_args(args)
    backbone_checkpoint_path = build_backbone_checkpoint_path(args)

    seen_matched_mlp_keys: set[tuple[Any, ...]] = set()
    classical_hidden_dims = unique_preserving_order(
        getattr(args, "classical_head_hidden_dims", [args.standard_mlp_hidden_dim])
    )
    classical_depths = unique_preserving_order(
        getattr(args, "classical_head_depths", [args.standard_mlp_num_layers])
    )
    classical_dropouts = unique_preserving_order(
        getattr(args, "classical_head_dropouts", [0.0])
    )
    matched_classical_depths = unique_preserving_order(
        getattr(args, "matched_classical_head_depths", [args.classical_num_layers])
    )
    matched_classical_dropouts = unique_preserving_order(
        getattr(args, "matched_classical_head_dropouts", [0.0])
    )
    matched_param_tolerance_rel = getattr(
        args,
        "matched_param_tolerance_rel",
        DEFAULT_MATCHED_PARAM_TOLERANCE_REL,
    )
    matched_param_tolerance_abs = getattr(
        args,
        "matched_param_tolerance_abs",
        DEFAULT_MATCHED_PARAM_TOLERANCE_ABS,
    )
    quantum_head_dropouts = unique_preserving_order(
        getattr(args, "prediction_head_q_dropouts", [0.0])
    )
    matched_mlp_cache: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # 0. Fixed standard MLP baseline as a candidate head
    # ------------------------------------------------------------
    if args.include_standard_mlp:
        for hidden_dim in classical_hidden_dims:
            for num_layers in classical_depths:
                for dropout in classical_dropouts:
                    standard_mlp_param_count = classical_head_param_count(
                        input_dim=args.channels,
                        output_dim=1,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )

                    if exceeds_param_limit(standard_mlp_param_count, args.max_prediction_head_params):
                        skipped.append(
                            "skip base-mlp: "
                            f"h{hidden_dim}-l{num_layers}-do{dropout} "
                            f"param_count={standard_mlp_param_count} exceeds "
                            f"max_prediction_head_params={args.max_prediction_head_params}"
                        )
                        continue

                    standard_mlp_name = build_run_name(
                        role="base-mlp-head",
                        args=args,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        param_count=standard_mlp_param_count,
                    )

                    standard_mlp_args = dict(base_args)
                    standard_mlp_args.update(
                        {
                            "run_name": standard_mlp_name,
                            "prediction_head": "classical",
                            "prediction_head_hidden_dim": hidden_dim,
                            "prediction_head_num_layers": num_layers,
                            "prediction_head_dropout": dropout,
                        }
                    )
                    standard_mlp_args = apply_head_training_mode_args(
                        run_args=standard_mlp_args,
                        args=args,
                        backbone_checkpoint_path=backbone_checkpoint_path,
                    )

                    experiments.append(
                        ExperimentSpec(
                            name=standard_mlp_name,
                            role="base_mlp",
                            group_id=f"baseline-h{hidden_dim}-l{num_layers}-do{dropout}",
                            base_quantum_name=None,
                            args=standard_mlp_args,
                            param_count=standard_mlp_param_count,
                            matched_target_param_count=None,
                            q_config_tag=None,
                            matched_baseline_name=None,
                        )
                    )

    # ------------------------------------------------------------
    # 1. Quantum configuration groups
    # ------------------------------------------------------------
    for n_qubits in args.prediction_head_qubits:
        for q_circuit_type in args.prediction_head_encodings:
            for q_ansatz_type in args.prediction_head_q_ansatz_types:
                for q_readout_mode in args.prediction_head_readouts:
                    for q_dropout in quantum_head_dropouts:
                        feasible, reason = is_feasible_quantum_config(
                            n_qubits=n_qubits,
                            circuit_type=q_circuit_type,
                            readout_mode=q_readout_mode,
                            max_quantum_input_dim=args.max_quantum_input_dim,
                            max_readout_dim=args.max_readout_dim,
                        )
                        if not feasible:
                            skipped.append(
                                "skip "
                                f"q={n_qubits}, "
                                f"encoding={q_circuit_type}, "
                                f"ansatz={q_ansatz_type}, "
                                f"readout={q_readout_mode}, "
                                f"q_dropout={q_dropout}: {reason}"
                            )
                            continue

                        q_config_tag = build_q_config_tag(
                            n_qubits=n_qubits,
                            n_q_layers=args.prediction_head_n_q_layers,
                            q_circuit_type=q_circuit_type,
                            q_ansatz_type=q_ansatz_type,
                            q_readout_mode=q_readout_mode,
                            q_angle_activation=args.prediction_head_q_angle_activation,
                            q_use_angle_affine=args.prediction_head_q_use_angle_affine,
                            q_dropout=q_dropout,
                        )
                        group_id = q_config_tag

                        # ------------------------------------------------------------
                        # 1.1 Pure quantum head
                        # ------------------------------------------------------------
                        try:
                            quantum_param_count = build_quantum_head_for_counting(
                                head_type="quantum",
                                input_dim=args.channels,
                                output_dim=1,
                                n_qubits=n_qubits,
                                n_q_layers=args.prediction_head_n_q_layers,
                                q_readout_mode=q_readout_mode,
                                q_circuit_type=q_circuit_type,
                                q_ansatz_type=q_ansatz_type,
                                q_angle_activation=args.prediction_head_q_angle_activation,
                                q_use_angle_affine=args.prediction_head_q_use_angle_affine,
                                q_dropout=q_dropout,
                            )
                        except Exception as exc:
                            skipped.append(
                                f"skip qhead {q_config_tag}: failed to build for counting: {exc}"
                            )
                            continue

                        if exceeds_param_limit(quantum_param_count, args.max_prediction_head_params):
                            skipped.append(
                                "skip "
                                f"qhead group {q_config_tag}: "
                                f"param_count={quantum_param_count} exceeds "
                                f"max_prediction_head_params={args.max_prediction_head_params}. "
                                "matched_mlp for this quantum config is also skipped."
                            )
                            continue

                        quantum_name = build_run_name(
                            role="qhead",
                            args=args,
                            q_config_tag=q_config_tag,
                            param_count=quantum_param_count,
                        )

                        quantum_args = dict(base_args)
                        quantum_args.update(
                            {
                                "run_name": quantum_name,
                                "prediction_head": "quantum",
                                "prediction_head_n_qubits": n_qubits,
                                "prediction_head_n_q_layers": args.prediction_head_n_q_layers,
                                "prediction_head_q_circuit_type": q_circuit_type,
                                "prediction_head_q_ansatz_type": q_ansatz_type,
                                "prediction_head_q_readout_mode": q_readout_mode,
                                "prediction_head_q_angle_activation": args.prediction_head_q_angle_activation,
                                "prediction_head_q_use_angle_affine": args.prediction_head_q_use_angle_affine,
                                "prediction_head_q_dropout": q_dropout,
                                "prediction_head_export_circuit": False,
                            }
                        )
                        quantum_args = apply_head_training_mode_args(
                            run_args=quantum_args,
                            args=args,
                            backbone_checkpoint_path=backbone_checkpoint_path,
                        )

                        # ------------------------------------------------------------
                        # 1.2 Parameter-matched MLP, de-duplicated per seed/mode
                        # ------------------------------------------------------------
                        matched_mlp_name = None
                        matched_mlp_best_gap = None
                        for requested_layers in matched_classical_depths:
                            classical_layers = requested_layers
                            if classical_layers <= 0:
                                classical_layers = args.prediction_head_n_q_layers + 2

                            for dropout in matched_classical_dropouts:
                                match_signature = (
                                    args.dataset,
                                    args.task,
                                    build_graph_family_tag(args),
                                    args.head_training_mode,
                                    int(classical_layers),
                                    float(dropout),
                                    int(args.seed),
                                )
                                cached = find_reusable_matched_mlp(
                                    cache=matched_mlp_cache,
                                    match_signature=match_signature,
                                    target_param_count=quantum_param_count,
                                    rel_tolerance=matched_param_tolerance_rel,
                                    abs_tolerance=matched_param_tolerance_abs,
                                )
                                if cached is None:
                                    hidden_dim, classical_param_count = find_param_matched_hidden_dim(
                                        input_dim=args.channels,
                                        output_dim=1,
                                        num_layers=classical_layers,
                                        target_param_count=quantum_param_count,
                                        max_hidden_dim=args.max_classical_hidden_dim,
                                        dropout=dropout,
                                    )
                                    current_matched_mlp_name = build_run_name(
                                        role="matched-mlp",
                                        args=args,
                                        hidden_dim=hidden_dim,
                                        num_layers=classical_layers,
                                        dropout=dropout,
                                        param_count=classical_param_count,
                                    )
                                    matched_mlp_cache.append(
                                        {
                                            "match_signature": match_signature,
                                            "target_param_count": quantum_param_count,
                                            "name": current_matched_mlp_name,
                                            "hidden_dim": hidden_dim,
                                            "param_count": classical_param_count,
                                        }
                                    )
                                else:
                                    current_matched_mlp_name = cached["name"]
                                    hidden_dim = cached["hidden_dim"]
                                    classical_param_count = cached["param_count"]

                                if not exceeds_param_limit(
                                    classical_param_count,
                                    args.max_prediction_head_params,
                                ):
                                    matched_gap = abs(classical_param_count - quantum_param_count)
                                    if matched_mlp_best_gap is None or matched_gap < matched_mlp_best_gap:
                                        matched_mlp_best_gap = matched_gap
                                        matched_mlp_name = current_matched_mlp_name

                                matched_key = (
                                    args.dataset,
                                    args.task,
                                    build_graph_family_tag(args),
                                    args.head_training_mode,
                                    hidden_dim,
                                    classical_layers,
                                    float(dropout),
                                    classical_param_count,
                                    args.seed,
                                )

                                if args.include_matched_mlp:
                                    if exceeds_param_limit(classical_param_count, args.max_prediction_head_params):
                                        skipped.append(
                                            "skip "
                                            f"matched-mlp for {q_config_tag}: "
                                            f"h{hidden_dim}-l{classical_layers}-do{dropout} "
                                            f"param_count={classical_param_count} exceeds "
                                            f"max_prediction_head_params={args.max_prediction_head_params}"
                                        )
                                    elif matched_key in seen_matched_mlp_keys:
                                        skipped.append(
                                            "deduplicate "
                                            f"matched-mlp h{hidden_dim}-l{classical_layers}-do{dropout}-p{classical_param_count} "
                                            f"for q_config={q_config_tag}; using {current_matched_mlp_name}"
                                        )
                                    else:
                                        seen_matched_mlp_keys.add(matched_key)

                                        matched_mlp_args = dict(base_args)
                                        matched_mlp_args.update(
                                            {
                                                "run_name": current_matched_mlp_name,
                                                "prediction_head": "classical",
                                                "prediction_head_hidden_dim": hidden_dim,
                                                "prediction_head_num_layers": classical_layers,
                                                "prediction_head_dropout": dropout,
                                            }
                                        )
                                        matched_mlp_args = apply_head_training_mode_args(
                                            run_args=matched_mlp_args,
                                            args=args,
                                            backbone_checkpoint_path=backbone_checkpoint_path,
                                        )

                                        experiments.append(
                                            ExperimentSpec(
                                                name=current_matched_mlp_name,
                                                role="matched_mlp",
                                                group_id=f"matched-h{hidden_dim}-l{classical_layers}-do{dropout}-p{classical_param_count}",
                                                base_quantum_name=None,
                                                args=matched_mlp_args,
                                                param_count=classical_param_count,
                                                matched_target_param_count=quantum_param_count,
                                                q_config_tag=None,
                                                matched_baseline_name=None,
                                            )
                                        )

                        if args.include_quantum:
                            experiments.append(
                                ExperimentSpec(
                                    name=quantum_name,
                                    role="qhead",
                                    group_id=group_id,
                                    base_quantum_name=quantum_name,
                                    args=quantum_args,
                                    param_count=quantum_param_count,
                                    matched_target_param_count=None,
                                    q_config_tag=q_config_tag,
                                    matched_baseline_name=matched_mlp_name,
                                )
                            )

                        # ------------------------------------------------------------
                        # 1.3 Residual / hybrid quantum head
                        # ------------------------------------------------------------
                        if args.include_residual_quantum:
                            try:
                                residual_param_count = build_quantum_head_for_counting(
                                    head_type="residual_quantum",
                                    input_dim=args.channels,
                                    output_dim=1,
                                    n_qubits=n_qubits,
                                    n_q_layers=args.prediction_head_n_q_layers,
                                    q_readout_mode=q_readout_mode,
                                    q_circuit_type=q_circuit_type,
                                    q_ansatz_type=q_ansatz_type,
                                    q_angle_activation=args.prediction_head_q_angle_activation,
                                    q_use_angle_affine=args.prediction_head_q_use_angle_affine,
                                    q_dropout=q_dropout,
                                    q_residual_mode=args.prediction_head_q_residual_mode,
                                )
                            except Exception as exc:
                                skipped.append(
                                    "skip "
                                    f"rqhead {q_config_tag}: failed to build residual quantum head "
                                    f"for counting. If prediction_head.py does not support "
                                    f"residual_quantum yet, run with --no-include_residual_quantum. "
                                    f"Original error: {exc}"
                                )
                                continue

                            if exceeds_param_limit(
                                residual_param_count,
                                args.max_prediction_head_params,
                            ):
                                skipped.append(
                                    "skip "
                                    f"rqhead {q_config_tag}: "
                                    f"param_count={residual_param_count} exceeds "
                                    f"max_prediction_head_params={args.max_prediction_head_params}"
                                )
                                continue

                            residual_name = build_run_name(
                                role="rqhead",
                                args=args,
                                q_config_tag=q_config_tag,
                                param_count=residual_param_count,
                                residual_mode=args.prediction_head_q_residual_mode,
                            )

                            residual_args = dict(base_args)
                            residual_args.update(
                                {
                                    "run_name": residual_name,
                                    "prediction_head": "residual_quantum",
                                    "prediction_head_n_qubits": n_qubits,
                                    "prediction_head_n_q_layers": args.prediction_head_n_q_layers,
                                    "prediction_head_q_circuit_type": q_circuit_type,
                                    "prediction_head_q_ansatz_type": q_ansatz_type,
                                    "prediction_head_q_readout_mode": q_readout_mode,
                                    "prediction_head_q_angle_activation": args.prediction_head_q_angle_activation,
                                    "prediction_head_q_use_angle_affine": args.prediction_head_q_use_angle_affine,
                                    "prediction_head_q_dropout": q_dropout,
                                    "prediction_head_q_residual_mode": args.prediction_head_q_residual_mode,
                                    "prediction_head_export_circuit": False,
                                }
                            )
                            residual_args = apply_head_training_mode_args(
                                run_args=residual_args,
                                args=args,
                                backbone_checkpoint_path=backbone_checkpoint_path,
                            )

                            experiments.append(
                                ExperimentSpec(
                                    name=residual_name,
                                    role="rqhead",
                                    group_id=group_id,
                                    base_quantum_name=quantum_name,
                                    args=residual_args,
                                    param_count=residual_param_count,
                                    matched_target_param_count=None,
                                    q_config_tag=q_config_tag,
                                    matched_baseline_name=matched_mlp_name,
                                )
                            )

    return experiments, skipped


# ============================================================
# CLI command helpers
# ============================================================

def to_cli_args(config: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []

    for key, value in config.items():
        if key in LAUNCHER_ONLY_ARG_KEYS:
            continue

        flag = f"--{key}"

        if isinstance(value, bool):
            cli_args.append(flag if value else f"--no-{key}")
        elif value is not None:
            cli_args.extend([flag, str(value)])

    return cli_args


def build_command(config: dict[str, Any], trainer_script: Path) -> list[str]:
    return [sys.executable, str(trainer_script), *to_cli_args(config)]


# ============================================================
# Plan writing
# ============================================================

def write_plan(
    *,
    output_dir: Path,
    channel: str,
    experiments: Sequence[ExperimentSpec],
    skipped: Sequence[str],
    trainer_script: Path,
    backbone_specs: Sequence[ExperimentSpec] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / f"{sanitize_fragment(channel)}_plan.json"

    payload = {
        "channel": channel,
        "num_head_experiments": len(experiments),
        "num_backbone_pretrains": len(backbone_specs or []),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backbone_pretrains": [
            {
                **asdict(spec),
                "command": build_command(spec.args, trainer_script=trainer_script),
            }
            for spec in (backbone_specs or [])
        ],
        "experiments": [
            {
                **asdict(spec),
                "command": build_command(spec.args, trainer_script=trainer_script),
            }
            for spec in experiments
        ],
        "skipped": list(skipped),
    }

    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return plan_path


def print_summary(
    *,
    channel: str,
    experiments: Sequence[ExperimentSpec],
    skipped: Sequence[str],
    plan_path: Path,
    backbone_specs: Sequence[ExperimentSpec] | None = None,
) -> None:
    print(f"Prediction-head experiment channel: {channel}")
    print(f"Plan written to: {plan_path}")

    if backbone_specs:
        print("")
        print(f"[BACKBONES] {len(backbone_specs)}")
        for spec in backbone_specs:
            print(
                f"[BB  ] {spec.name} | seed={spec.args.get('seed')} | "
                f"params={spec.param_count}"
            )
            print(f"      output_dir={spec.args.get('output_dir')}")

    print("")
    print(f"Head experiments: {len(experiments)}")

    if skipped:
        print("")
        print("Skipped configurations:")
        for item in skipped:
            print(f"  - {item}")

    print("")
    for spec in experiments:
        if spec.role == "base_mlp":
            print(f"[BASE] {spec.name} | params={spec.param_count}")
        elif spec.role == "matched_mlp":
            print(f"[MMLP] {spec.name} | params={spec.param_count}")
        elif spec.role == "qhead":
            print(
                f"[QHD ] {spec.name} | params={spec.param_count} | "
                f"matched={spec.matched_baseline_name}"
            )
        elif spec.role == "rqhead":
            print(
                f"[RQHD] {spec.name} | params={spec.param_count} | "
                f"matched={spec.matched_baseline_name}"
            )
        else:
            print(f"[????] {spec.name} | role={spec.role} | params={spec.param_count}")


# ============================================================
# Result parsing
# ============================================================

def _safe_json_load(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []

    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)

    return result


def _candidate_result_paths(spec: ExperimentSpec) -> list[Path]:
    output_dir = Path(spec.args["output_dir"])

    raw_spec_name = str(spec.name)
    safe_spec_name = runtime_safe_run_name(spec.name)

    raw_arg_run_name = str(spec.args.get("run_name", spec.name))
    safe_arg_run_name = runtime_safe_run_name(raw_arg_run_name)

    run_names = [
        raw_spec_name,
        safe_spec_name,
        raw_arg_run_name,
        safe_arg_run_name,
    ]

    paths: list[Path] = []

    for run_name in run_names:
        run_dir = output_dir / run_name
        paths.append(run_dir / "summary.json")
        paths.append(run_dir / "history.json")

    for run_name in run_names:
        paths.append(output_dir / f"{run_name}_summary.json")
        paths.append(output_dir / f"{run_name}_history.json")
        paths.append(output_dir / f"{run_name}.json")

    return _unique_paths(paths)


def load_summary_for_spec(spec: ExperimentSpec) -> dict[str, Any]:
    candidates = _candidate_result_paths(spec)

    for path in candidates:
        if path.name != "summary.json":
            continue

        payload = _safe_json_load(path)
        if not isinstance(payload, dict):
            continue

        payload["summary_found"] = True
        payload["summary_path"] = str(path)
        payload["summary_format"] = "summary_json"
        return payload

    for path in candidates:
        if path.name != "history.json":
            continue

        payload = _safe_json_load(path)
        if not isinstance(payload, dict):
            continue

        payload["summary_found"] = True
        payload["summary_path"] = str(path)
        payload["summary_format"] = "history_json"
        payload.setdefault("status", "loaded_from_history")
        return payload

    return {
        "summary_found": False,
        "status": "missing_summary",
        "summary_path": str(candidates[0]) if candidates else None,
        "searched_paths": [str(path) for path in candidates],
    }


def is_completed_run(spec: ExperimentSpec) -> bool:
    """
    Only check this exact spec's own run directory.

    Do not scan other existing runs here; otherwise seed43/seed44 may be
    incorrectly skipped because seed42 has a similar completed run.
    """
    output_dir = Path(spec.args["output_dir"])
    run_name = runtime_safe_run_name(spec.args.get("run_name", spec.name))
    run_dir = output_dir / run_name
    checkpoint_path = run_dir / "best_checkpoint.pt"

    for path in [run_dir / "summary.json", run_dir / "history.json"]:
        payload = _safe_json_load(path)
        if not isinstance(payload, dict):
            continue

        status = str(payload.get("status", "")).lower()
        if status == "completed":
            if spec.role == "backbone_pretrain" and not checkpoint_path.exists():
                continue
            return True

        if payload.get("best_test_metrics") is not None:
            if spec.role == "backbone_pretrain" and not checkpoint_path.exists():
                continue
            return True

    return False


def get_completed_summary_for_spec(spec: ExperimentSpec) -> dict[str, Any] | None:
    summary = load_summary_for_spec(spec)

    if not summary.get("summary_found"):
        return None

    status = str(summary.get("status", "")).lower()
    has_test_metrics = summary.get("best_test_metrics") is not None

    if status == "completed" or has_test_metrics:
        return summary

    return None


def select_late_best_experiments_for_extension(
    *,
    experiments: Sequence[ExperimentSpec],
    best_epoch_min: int,
    extra_epochs: int,
    eligible_roles: set[str],
) -> tuple[list[ExperimentSpec], list[str]]:
    selected: list[ExperimentSpec] = []
    skipped: list[str] = []

    for spec in experiments:
        if spec.role not in eligible_roles:
            skipped.append(
                f"skip extension {spec.name}: role={spec.role} not in eligible_roles"
            )
            continue

        summary = get_completed_summary_for_spec(spec)
        if summary is None:
            skipped.append(f"skip extension {spec.name}: no completed summary found")
            continue

        best_epoch = summary.get("best_epoch")
        try:
            best_epoch_int = int(best_epoch)
        except Exception:
            skipped.append(f"skip extension {spec.name}: invalid best_epoch={best_epoch}")
            continue

        if best_epoch_int < best_epoch_min:
            skipped.append(
                f"skip extension {spec.name}: best_epoch={best_epoch_int} < {best_epoch_min}"
            )
            continue

        current_epochs = spec.args.get("epochs", 0)
        try:
            current_epochs_int = int(current_epochs)
        except Exception:
            current_epochs_int = 0

        extended_spec = copy.deepcopy(spec)
        extended_spec.args["epochs"] = current_epochs_int + int(extra_epochs)
        extended_spec.args["resume"] = True

        # Keep the same run_name. This is important:
        # same run_name + resume=True means continue original run.
        extended_spec.name = spec.name
        extended_spec.role = f"{spec.role}_extended"

        selected.append(extended_spec)

    return selected, skipped


def load_summary_from_run_dir(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.json"

    payload = _safe_json_load(summary_path)
    if isinstance(payload, dict):
        payload["summary_found"] = True
        payload["summary_path"] = str(summary_path)
        payload["summary_format"] = "summary_json_scanned"
        payload["_scanned_run_dir"] = str(run_dir)
        return payload

    payload = _safe_json_load(history_path)
    if isinstance(payload, dict):
        payload["summary_found"] = True
        payload["summary_path"] = str(history_path)
        payload["summary_format"] = "history_json_scanned"
        payload.setdefault("status", "loaded_from_history")
        payload["_scanned_run_dir"] = str(run_dir)
        return payload

    return None


def scan_existing_run_summaries(output_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    scanned: list[dict[str, Any]] = []
    ignored_names = {"_launcher_logs", "__pycache__"}

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue

        for run_dir in sorted(output_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name in ignored_names:
                continue
            if run_dir.name.startswith("."):
                continue

            summary = load_summary_from_run_dir(run_dir)
            if summary is not None:
                scanned.append(summary)

    return scanned


def _get_nested_metric(obj: dict[str, Any], outer: str, inner: str) -> Any:
    nested = obj.get(outer, {})
    if isinstance(nested, dict):
        return nested.get(inner)
    return None


def _get_last_history_metric(obj: dict[str, Any], split: str, metric: str) -> Any:
    metrics = obj.get(f"{split}_metrics", {})
    if not isinstance(metrics, dict):
        return None

    values = metrics.get(metric)
    if isinstance(values, list) and values:
        return values[-1]

    return values


def _get_last_list_value(obj: dict[str, Any], key: str) -> Any:
    values = obj.get(key)
    if isinstance(values, list) and values:
        return values[-1]
    return values


COMMON_SUMMARY_METRICS = [
    "r2",
    "mae",
    "rmse",
    "roc_auc",
    "auroc",
    "auc",
    "accuracy",
    "acc",
    "f1",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "precision",
    "macro_precision",
    "micro_precision",
    "recall",
    "macro_recall",
    "micro_recall",
    "average_precision",
    "auprc",
    "multilabel_auprc_macro",
    "multiclass_f1",
    "mrr",
]

TUNE_METRIC_PRIORITY = [
    "r2",
    "roc_auc",
    "auroc",
    "auc",
    "multilabel_auprc_macro",
    "multiclass_f1",
    "macro_f1",
    "f1",
    "micro_f1",
    "accuracy",
    "mrr",
    "mae",
    "rmse",
]


def _metric_names_from_summary(summary: dict[str, Any]) -> list[str]:
    names: set[str] = set(COMMON_SUMMARY_METRICS)

    for outer in ["best_val_metrics", "best_test_metrics", "val_metrics", "test_metrics"]:
        value = summary.get(outer, {})
        if isinstance(value, dict):
            names.update(str(key) for key in value.keys())

    return sorted(names)


def _get_metric_value(summary: dict[str, Any], split: str, metric: str) -> Any:
    value = _get_nested_metric(summary, f"best_{split}_metrics", metric)
    if value is not None:
        return value

    value = _get_last_history_metric(summary, split, metric)
    if value is not None:
        return value

    # Some older summary writers may store flat keys such as val_f1/test_f1.
    return summary.get(f"{split}_{metric}")


def _infer_tune_metric(summary: dict[str, Any], run_args: dict[str, Any]) -> str | None:
    explicit = summary.get("tune_metric")
    if explicit is not None:
        return str(explicit)

    regression_metric = run_args.get("regression_tune_metric")
    metric_names = set(_metric_names_from_summary(summary))
    if regression_metric is not None and str(regression_metric) in metric_names:
        return str(regression_metric)

    best_val_metrics = summary.get("best_val_metrics", {})
    best_test_metrics = summary.get("best_test_metrics", {})
    available = set()
    if isinstance(best_val_metrics, dict):
        available.update(str(key) for key in best_val_metrics.keys())
    if isinstance(best_test_metrics, dict):
        available.update(str(key) for key in best_test_metrics.keys())

    for metric in TUNE_METRIC_PRIORITY:
        if metric in available:
            return metric

    return None


def _row_from_summary(
    *,
    summary: dict[str, Any],
    spec: ExperimentSpec | None,
) -> dict[str, Any]:
    run_args = summary.get("args", {}) if isinstance(summary, dict) else {}

    if spec is not None:
        role = spec.role
        group_id = spec.group_id
        q_config_tag = spec.q_config_tag
        name = spec.name
        base_quantum_name = spec.base_quantum_name
        matched_baseline_name = spec.matched_baseline_name
        param_count = spec.param_count
        matched_target_param_count = spec.matched_target_param_count
        spec_args = spec.args
    else:
        role = "existing_run"
        group_id = None
        q_config_tag = None
        name = summary.get("run_name") or Path(summary.get("_scanned_run_dir", "unknown")).name
        base_quantum_name = None
        matched_baseline_name = None
        param_count = None
        matched_target_param_count = None
        spec_args = {}

    train_loss = _get_last_list_value(summary, "train_loss")
    tune_metric = _infer_tune_metric(summary, run_args)
    test_tune_metric = (
        _get_metric_value(summary, "test", tune_metric)
        if tune_metric is not None
        else None
    )

    row: dict[str, Any] = {
        "name": name,
        "best_epoch": summary.get("best_epoch"),
        "best_val_metric": summary.get("best_val_metric"),
        "test_tune_metric": test_tune_metric,
        "role": role,

        "head_training_mode": run_args.get(
            "head_training_mode",
            spec_args.get("head_training_mode"),
        ),
        "graph_family": run_args.get(
            "graph_family",
            spec_args.get("graph_family"),
        ),
        "graph_config_tag": run_args.get(
            "graph_config_tag",
            spec_args.get("graph_config_tag"),
        ),

        "train_loss": train_loss,

        "group_id": group_id,
        "q_config_tag": q_config_tag,
        "status": summary.get("status"),
        "summary_found": summary.get("summary_found"),
        "summary_format": summary.get("summary_format"),

        "epoch": summary.get("epoch"),
        "runtime_run_name": summary.get("run_name"),
        "base_quantum_name": base_quantum_name,
        "matched_baseline_name": matched_baseline_name,

        "param_count": param_count,
        "matched_target_param_count": matched_target_param_count,

        "dataset": run_args.get("dataset", spec_args.get("dataset")),
        "task": run_args.get("task", spec_args.get("task")),
        "tune_metric": tune_metric,
        "seed": run_args.get("seed", spec_args.get("seed")),

        "gnn": run_args.get("gnn", spec_args.get("gnn")),
        "intra_aggr": run_args.get("intra_aggr", spec_args.get("intra_aggr")),
        "type_fusion": run_args.get("type_fusion", spec_args.get("type_fusion")),
        "node_update": run_args.get("node_update", spec_args.get("node_update")),
        "channels": run_args.get("channels", spec_args.get("channels")),
        "early_stopping_patience": run_args.get(
            "early_stopping_patience",
            spec_args.get("early_stopping_patience"),
        ),
        "early_stopping_min_delta": run_args.get(
            "early_stopping_min_delta",
            spec_args.get("early_stopping_min_delta"),
        ),
        "num_layers": run_args.get("num_layers", spec_args.get("num_layers")),
        "num_neighbors": run_args.get("num_neighbors", spec_args.get("num_neighbors")),
        "neighbor_sampling_mode": run_args.get(
            "neighbor_sampling_mode",
            spec_args.get("neighbor_sampling_mode"),
        ),
        "temporal_strategy": run_args.get(
            "temporal_strategy",
            spec_args.get("temporal_strategy"),
        ),

        "prediction_head": run_args.get("prediction_head", spec_args.get("prediction_head")),
        "prediction_head_hidden_dim": run_args.get(
            "prediction_head_hidden_dim",
            spec_args.get("prediction_head_hidden_dim"),
        ),
        "prediction_head_num_layers": run_args.get(
            "prediction_head_num_layers",
            spec_args.get("prediction_head_num_layers"),
        ),
        "prediction_head_dropout": run_args.get(
            "prediction_head_dropout",
            spec_args.get("prediction_head_dropout"),
        ),
        "prediction_head_n_qubits": run_args.get(
            "prediction_head_n_qubits",
            spec_args.get("prediction_head_n_qubits"),
        ),
        "prediction_head_n_q_layers": run_args.get(
            "prediction_head_n_q_layers",
            spec_args.get("prediction_head_n_q_layers"),
        ),
        "prediction_head_q_circuit_type": run_args.get(
            "prediction_head_q_circuit_type",
            spec_args.get("prediction_head_q_circuit_type"),
        ),
        "prediction_head_q_ansatz_type": run_args.get(
            "prediction_head_q_ansatz_type",
            spec_args.get("prediction_head_q_ansatz_type"),
        ),
        "prediction_head_q_readout_mode": run_args.get(
            "prediction_head_q_readout_mode",
            spec_args.get("prediction_head_q_readout_mode"),
        ),
        "prediction_head_q_angle_activation": run_args.get(
            "prediction_head_q_angle_activation",
            spec_args.get("prediction_head_q_angle_activation"),
        ),
        "prediction_head_q_use_angle_affine": run_args.get(
            "prediction_head_q_use_angle_affine",
            spec_args.get("prediction_head_q_use_angle_affine"),
        ),
        "prediction_head_q_dropout": run_args.get(
            "prediction_head_q_dropout",
            spec_args.get("prediction_head_q_dropout"),
        ),
        "prediction_head_q_residual_mode": run_args.get(
            "prediction_head_q_residual_mode",
            spec_args.get("prediction_head_q_residual_mode"),
        ),

        "init_backbone_checkpoint": run_args.get(
            "init_backbone_checkpoint",
            spec_args.get("init_backbone_checkpoint"),
        ),
        "load_backbone_only": run_args.get(
            "load_backbone_only",
            spec_args.get("load_backbone_only"),
        ),
        "freeze_backbone": run_args.get(
            "freeze_backbone",
            spec_args.get("freeze_backbone"),
        ),
        "reset_prediction_head": run_args.get(
            "reset_prediction_head",
            spec_args.get("reset_prediction_head"),
        ),

        "summary_path": summary.get("summary_path"),
        "searched_paths": json.dumps(summary.get("searched_paths", []), ensure_ascii=False),
        "message": summary.get("message"),
    }

    # Add every metric we can find, not just regression metrics. This makes
    # classification summaries (F1/AUROC/precision/accuracy/etc.) show up in
    # the CSV and aggregate CSV without hard-coding the task type.
    for metric_name in _metric_names_from_summary(summary):
        row[f"val_{metric_name}"] = _get_metric_value(summary, "val", metric_name)
        row[f"test_{metric_name}"] = _get_metric_value(summary, "test", metric_name)

    return row

def build_summary_rows(
    experiments: Sequence[ExperimentSpec],
    *,
    scan_dirs: Sequence[Path],
    include_existing_runs: bool = True,
    backbone_specs: Sequence[ExperimentSpec] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_summary_paths: set[str] = set()
    seen_run_names: set[str] = set()

    specs: list[ExperimentSpec] = []
    if backbone_specs is not None:
        specs.extend(list(backbone_specs))
    specs.extend(list(experiments))

    for spec in specs:
        summary = load_summary_for_spec(spec)
        row = _row_from_summary(summary=summary, spec=spec)
        rows.append(row)

        if row.get("summary_path"):
            seen_summary_paths.add(str(row["summary_path"]))
        if row.get("runtime_run_name"):
            seen_run_names.add(str(row["runtime_run_name"]))
        if row.get("name"):
            seen_run_names.add(str(row["name"]))

    if include_existing_runs:
        for summary in scan_existing_run_summaries(scan_dirs):
            summary_path = str(summary.get("summary_path", ""))
            runtime_run_name = str(summary.get("run_name", ""))

            if summary_path and summary_path in seen_summary_paths:
                continue
            if runtime_run_name and runtime_run_name in seen_run_names:
                continue

            row = _row_from_summary(summary=summary, spec=None)
            rows.append(row)

            if row.get("summary_path"):
                seen_summary_paths.add(str(row["summary_path"]))
            if row.get("runtime_run_name"):
                seen_run_names.add(str(row["runtime_run_name"]))
            if row.get("name"):
                seen_run_names.add(str(row["name"]))

    return rows


def write_run_summary(
    *,
    output_dir: Path,
    experiments: Sequence[ExperimentSpec],
    scan_dirs: Sequence[Path],
    backbone_specs: Sequence[ExperimentSpec] | None = None,
) -> Path:
    rows = build_summary_rows(
        experiments,
        scan_dirs=scan_dirs,
        include_existing_runs=True,
        backbone_specs=backbone_specs,
    )

    csv_path = output_dir / "run_summary.csv"

    if not rows:
        return csv_path

    preferred_cols = [
        "name",
        "best_epoch",
        "best_val_metric",
        "test_tune_metric",
        "role",
        "seed",
        "head_training_mode",
        "graph_family",
        "graph_config_tag",

        "val_r2",
        "test_r2",
        "val_mae",
        "test_mae",
        "val_rmse",
        "test_rmse",
        "val_roc_auc",
        "test_roc_auc",
        "val_auroc",
        "test_auroc",
        "val_auc",
        "test_auc",
        "val_accuracy",
        "test_accuracy",
        "val_f1",
        "test_f1",
        "val_macro_f1",
        "test_macro_f1",
        "val_micro_f1",
        "test_micro_f1",
        "val_multiclass_f1",
        "test_multiclass_f1",
        "val_precision",
        "test_precision",
        "val_recall",
        "test_recall",
        "val_average_precision",
        "test_average_precision",
        "val_auprc",
        "test_auprc",
        "val_multilabel_auprc_macro",
        "test_multilabel_auprc_macro",
        "val_mrr",
        "test_mrr",
        "train_loss",

        "status",
        "summary_found",
        "summary_format",

        "group_id",
        "q_config_tag",
        "runtime_run_name",
        "base_quantum_name",
        "matched_baseline_name",

        "param_count",
        "matched_target_param_count",

        "prediction_head",
        "prediction_head_hidden_dim",
        "prediction_head_num_layers",
        "prediction_head_dropout",
        "prediction_head_n_qubits",
        "prediction_head_n_q_layers",
        "prediction_head_q_circuit_type",
        "prediction_head_q_ansatz_type",
        "prediction_head_q_readout_mode",
        "prediction_head_q_angle_activation",
        "prediction_head_q_use_angle_affine",
        "prediction_head_q_dropout",
        "prediction_head_q_residual_mode",

        "dataset",
        "task",
        "tune_metric",
        "gnn",
        "intra_aggr",
        "type_fusion",
        "node_update",
        "channels",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "num_layers",
        "num_neighbors",
        "neighbor_sampling_mode",
        "temporal_strategy",

        "init_backbone_checkpoint",
        "load_backbone_only",
        "freeze_backbone",
        "reset_prediction_head",

        "summary_path",
        "searched_paths",
        "message",
    ]

    all_cols = {key for row in rows for key in row.keys()}
    ordered_cols = [col for col in preferred_cols if col in all_cols]
    ordered_cols += sorted(col for col in all_cols if col not in ordered_cols)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return csv_path


def strip_seed_from_name(name: str) -> str:
    return re.sub(r"-seed\d+", "-seedX", str(name))


def write_aggregate_summary(*, summary_csv_path: Path) -> Path:
    import pandas as pd

    agg_path = summary_csv_path.with_name("run_summary_agg.csv")

    if not summary_csv_path.exists():
        return agg_path

    df = pd.read_csv(summary_csv_path)

    if df.empty:
        df.to_csv(agg_path, index=False)
        return agg_path

    if "summary_found" in df.columns:
        df = df[df["summary_found"].astype(str).str.lower().isin(["true", "1"])].copy()

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("completed")].copy()

    if df.empty:
        df.to_csv(agg_path, index=False)
        return agg_path

    df["config_key"] = df["name"].apply(strip_seed_from_name)

    group_cols = [
        "config_key",
        "role",
        "head_training_mode",
        "graph_family",
        "graph_config_tag",
        "prediction_head",
        "prediction_head_hidden_dim",
        "prediction_head_num_layers",
        "prediction_head_dropout",
        "prediction_head_n_qubits",
        "prediction_head_n_q_layers",
        "prediction_head_q_circuit_type",
        "prediction_head_q_ansatz_type",
        "prediction_head_q_readout_mode",
        "prediction_head_q_angle_activation",
        "prediction_head_q_dropout",
        "prediction_head_q_residual_mode",
        "param_count",
    ]
    group_cols = [col for col in group_cols if col in df.columns]

    metric_cols = [
        col for col in df.columns
        if (
            col in {"best_val_metric", "test_tune_metric"}
            or col.startswith("val_")
            or col.startswith("test_")
        )
    ]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    metric_cols = [col for col in metric_cols if df[col].notna().any()]

    agg_dict = {col: ["mean", "std", "min", "max", "count"] for col in metric_cols}

    agg_df = df.groupby(group_cols, dropna=False).agg(agg_dict)
    agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    rename_map = {
        "best_val_metric_mean": "mean_best_val_metric",
        "best_val_metric_std": "std_best_val_metric",
        "test_tune_metric_mean": "mean_test_tune_metric",
        "test_tune_metric_std": "std_test_tune_metric",
        "test_tune_metric_count": "n",
    }
    agg_df = agg_df.rename(columns=rename_map)

    agg_df.to_csv(agg_path, index=False)
    return agg_path



# ============================================================
# Seed-level matrix construction
# ============================================================

def build_seed_specs(
    args: argparse.Namespace,
) -> tuple[list[int], list[ExperimentSpec], list[ExperimentSpec], list[str]]:
    """Build backbone and head experiment specs for one or more seeds.

    This belongs in utils because it constructs ExperimentSpec objects and expands
    the launcher grid. The run file should only define the front-end grid,
    parse arguments, and orchestrate the high-level execution flow.
    """
    seed_list = args.seeds if args.seeds is not None else [args.seed]

    all_experiments: list[ExperimentSpec] = []
    all_backbone_specs: list[ExperimentSpec] = []
    all_skipped: list[str] = []

    for seed in seed_list:
        seed_args = copy.copy(args)
        seed_args.seed = int(seed)

        if uses_pretrained_backbone(seed_args):
            seed_backbone_spec = build_backbone_pretrain_spec(seed_args)
            all_backbone_specs.append(seed_backbone_spec)

        seed_experiments, seed_skipped = build_experiment_matrix(seed_args)

        all_experiments.extend(seed_experiments)
        all_skipped.extend([f"seed={seed}: {item}" for item in seed_skipped])

    return list(seed_list), all_backbone_specs, all_experiments, all_skipped


# ============================================================
# Execution
# ============================================================

def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def execute_plan(
    experiments: Sequence[ExperimentSpec],
    *,
    project_root: Path,
    trainer_script: Path,
    status_path: Path,
    summary_output_dir: Path,
    scan_dirs: Sequence[Path],
    all_summary_experiments: Sequence[ExperimentSpec] | None = None,
    backbone_specs: Sequence[ExperimentSpec] | None = None,
    skip_completed: bool = True,
) -> None:
    launcher_log_dir = status_path.parent / "_launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    summary_specs = list(all_summary_experiments) if all_summary_experiments is not None else list(experiments)

    for index, spec in enumerate(experiments, start=1):
        if skip_completed and is_completed_run(spec):
            print("")
            print(f"[{index}/{len(experiments)}] Skip completed run: {spec.name}")
            append_jsonl(
                status_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "skip_completed",
                    "index": index,
                    "name": spec.name,
                    "role": spec.role,
                    "group_id": spec.group_id,
                },
            )
            continue

        command = build_command(spec.args, trainer_script=trainer_script)

        stdout_path = launcher_log_dir / f"{index:04d}_{spec.name}_stdout.log"
        stderr_path = launcher_log_dir / f"{index:04d}_{spec.name}_stderr.log"

        append_jsonl(
            status_path,
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "event": "start",
                "index": index,
                "name": spec.name,
                "role": spec.role,
                "group_id": spec.group_id,
                "command": command,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            },
        )

        print("")
        print(f"[{index}/{len(experiments)}] Running {spec.role}: {spec.name}")
        print(" ".join(command))

        return_code: int | None = None

        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                result = subprocess.run(
                    command,
                    cwd=project_root,
                    check=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )

            return_code = int(result.returncode)

            append_jsonl(
                status_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "end",
                    "index": index,
                    "name": spec.name,
                    "role": spec.role,
                    "group_id": spec.group_id,
                    "return_code": return_code,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                },
            )

            if return_code != 0:
                print(f"Run failed with return code {return_code}: {spec.name}")
                print(f"  stderr: {stderr_path}")
                if spec.role == "backbone_pretrain":
                    raise RuntimeError(
                        f"Backbone pretraining failed with return code {return_code}. "
                        f"See stderr: {stderr_path}"
                    )
            else:
                print(f"Run finished: {spec.name}")
                print(f"  stdout: {stdout_path}")

        except Exception as exc:
            append_jsonl(
                status_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "exception",
                    "index": index,
                    "name": spec.name,
                    "role": spec.role,
                    "group_id": spec.group_id,
                    "message": str(exc),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                },
            )
            print(f"Run crashed: {spec.name}")
            print(str(exc))
            raise

        try:
            csv_path = write_run_summary(
                output_dir=summary_output_dir,
                experiments=summary_specs,
                scan_dirs=scan_dirs,
                backbone_specs=backbone_specs,
            )
            agg_path = write_aggregate_summary(summary_csv_path=csv_path)

            print("")
            print(f"Incremental run summary updated after run {index}:")
            print(f"  CSV : {csv_path}")
            print(f"  AGG : {agg_path}")

            append_jsonl(
                status_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "summary_updated",
                    "index": index,
                    "name": spec.name,
                    "role": spec.role,
                    "group_id": spec.group_id,
                    "return_code": return_code,
                    "csv_path": str(csv_path),
                    "agg_path": str(agg_path),
                },
            )

        except Exception as exc:
            print(f"Failed to refresh run summary after run {index}: {exc}")
            append_jsonl(
                status_path,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "summary_update_failed",
                    "index": index,
                    "name": spec.name,
                    "role": spec.role,
                    "group_id": spec.group_id,
                    "message": str(exc),
                },
            )


