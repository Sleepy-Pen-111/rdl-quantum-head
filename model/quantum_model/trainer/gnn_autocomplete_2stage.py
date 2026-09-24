import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

RELBENCH_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RELBENCH_REPO_ROOT.parent
for path in (str(RELBENCH_REPO_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from quantum_model.trainer.model import Model
from quantum_model.text_embedder import GloveTextEmbedding

from relbench.base import Dataset, EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from quantum_model.model.training_runtime import (
    build_auto_run_name,
    GPURequiredError,
    RuntimeDeviceGuard,
    resolve_run_name,
    TrainingRunManager,
    add_runtime_args,
    exit_for_gpu_error,
    normalize_q_readout_mode,
    resolve_device,
)
from quantum_model.model.quantum_ansatz import VALID_Q_ANSATZ_TYPES
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

parser = argparse.ArgumentParser()

# Common experiment arguments aligned with gnn_entity_balanced.py.
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="results-position")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument(
    "--regression_tune_metric",
    type=str,
    default="r2",
    choices=["r2", "mae", "rmse"],
    help="Validation metric used to select the best checkpoint for regression tasks.",
)
parser.add_argument(
    "--early_stopping_patience",
    type=int,
    default=None,
    help="Stop after this many epochs without validation improvement. Disabled when unset.",
)
parser.add_argument(
    "--early_stopping_min_delta",
    type=float,
    default=0.0,
    help="Minimum validation improvement required to reset early-stopping patience.",
)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument(
    "--gnn",
    type=str,
    default="staged",
    choices=["staged", "hgt", "graphormer"],
    help=(
        "Graph encoder choice. "
        "'staged' uses the explicit aggregation/fusion/update pipeline "
        "controlled by --intra_aggr, --type_fusion, and --node_update; "
        "'hgt' and 'graphormer' use one-step encoders."
    ),
)
parser.add_argument(
    "--intra_aggr",
    type=str,
    default="mean",
    choices=["mean", "max", "mean_project", "max_project", "lstm", "gatv2"],
)
parser.add_argument(
    "--type_fusion",
    type=str,
    default="sum",
    choices=[
        "sum",
        "mean",
        "weighted_sum",
        "relation_weighted_sum",
        "edge_type_weighted_sum",
        "gru",
    ],
)
parser.add_argument(
    "--node_update",
    type=str,
    default="mlp",
    choices=["mlp", "gru"],
)
parser.add_argument("--n_qubits", type=int, default=4)
parser.add_argument("--n_q_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=None)
parser.add_argument(
    "--q_readout_mode",
    type=str,
    default="z_pairwise",
    choices=["z_pairwise", "z_only", "z_all", "probs"],
)
parser.add_argument(
    "--q_circuit_type",
    type=str,
    default="angle",
    choices=["angle", "rxry", "arctan", "amplitude"],
    help=(
        "Quantum data encoding type. "
        "'angle': one angle per qubit; "
        "'rxry': two angles per qubit using RX/RY; "
        "'arctan': RY(theta), RZ(theta^2); "
        "'amplitude': amplitude embedding."
    ),
)
parser.add_argument(
    "--q_ansatz_type",
    type=str,
    default="rot_cnot_ring",
    choices=sorted(VALID_Q_ANSATZ_TYPES),
    help="Quantum ansatz applied after encoding and before measurement.",
)
parser.add_argument(
    "--q_angle_activation",
    type=str,
    default="tanh",
    choices=["tanh", "atan", "none"],
    help=(
        "Activation applied before quantum rotations. "
        "'tanh': tanh(x)*pi; "
        "'atan': 2*atan(x); "
        "'none': no activation."
    ),
)
parser.add_argument(
    "--q_use_angle_affine",
    action="store_true",
    help=(
        "Enable learnable affine transform "
        "theta = scale * s + bias "
        "before quantum angle activation."
    ),
)
parser.add_argument(
    "--q_residual_mode",
    type=str,
    default="concat",
    choices=["add", "concat"],
)
parser.add_argument("--q_alpha_init", type=float, default=0.0)
parser.add_argument(
    "--prediction_head",
    type=str,
    default="classical",
    choices=["classical", "quantum", "residual_quantum"],
)
parser.add_argument("--prediction_head_hidden_dim", type=int, default=None)
parser.add_argument("--prediction_head_num_layers", type=int, default=1)
parser.add_argument("--prediction_head_dropout", type=float, default=0.0)
parser.add_argument("--prediction_head_n_qubits", type=int, default=None)
parser.add_argument("--prediction_head_n_q_layers", type=int, default=None)
parser.add_argument("--prediction_head_n_heads", type=int, default=None)
parser.add_argument(
    "--prediction_head_q_readout_mode",
    type=str,
    default=None,
    choices=["z_pairwise", "z_only", "z_all", "probs"],
)
parser.add_argument(
    "--prediction_head_q_circuit_type",
    type=str,
    default=None,
    choices=["angle", "rxry", "arctan", "amplitude"],
)
parser.add_argument(
    "--prediction_head_q_ansatz_type",
    type=str,
    default=None,
    choices=sorted(VALID_Q_ANSATZ_TYPES),
)
parser.add_argument(
    "--prediction_head_q_angle_activation",
    type=str,
    default=None,
    choices=["tanh", "atan", "none"],
)
parser.add_argument(
    "--prediction_head_q_use_angle_affine",
    action=argparse.BooleanOptionalAction,
    default=None,
)
parser.add_argument(
    "--Q_Pred_Head_Dropout",
    "--prediction_head_q_dropout",
    dest="prediction_head_q_dropout",
    type=float,
    default=0.0,
    help="Dropout applied to quantum prediction-head readout features before the final post-net.",
)
parser.add_argument(
    "--prediction_head_q_residual_mode",
    type=str,
    default="add",
    choices=["add", "concat"],
)
parser.add_argument("--prediction_head_q_alpha_init", type=float, default=0.0)
parser.add_argument(
    "--prediction_head_freeze_quantum_at_init",
    action=argparse.BooleanOptionalAction,
    default=None,
)
parser.add_argument(
    "--prediction_head_freeze_alpha_with_quantum",
    action=argparse.BooleanOptionalAction,
    default=None,
)
parser.add_argument(
    "--prediction_head_export_circuit",
    action=argparse.BooleanOptionalAction,
    default=False,
)

parser.add_argument(
    "--init_backbone_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional checkpoint path used to initialize the model before training. "
        "For prediction-head experiments, this is usually the shared backbone best_checkpoint.pt."
    ),
)
parser.add_argument(
    "--load_backbone_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "When loading init_backbone_checkpoint, ignore prediction-head parameters "
        "so the current run keeps its newly initialized head."
    ),
)
parser.add_argument(
    "--freeze_backbone",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Freeze encoder/temporal encoder/GNN/shallow embeddings and train only "
        "the prediction head."
    ),
)
parser.add_argument(
    "--reset_prediction_head",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Reset the prediction head after optional checkpoint loading. "
        "Useful when reusing a shared backbone but changing the head."
    ),
)

parser.add_argument(
    "--freeze_quantum_at_init",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--freeze_alpha_with_quantum",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--quantum_unfreeze_epoch", type=int, default=None)
parser.add_argument(
    "--quantum_only_finetune",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument(
    "--neighbor_sampling_mode",
    type=str,
    default="total",
    choices=["total", "per_edge_type"],
    help=(
        "total: original PyG behavior, each hop samples num_neighbors globally. "
        "per_edge_type: split num_neighbors over all edge types and sample at least "
        "floor(num_neighbors / num_edge_types) per edge type."
    ),
)
parser.add_argument("--gat_heads", type=int, default=4)
parser.add_argument("--gat_dropout", type=float, default=0.0)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
add_runtime_args(parser, default_run_name="relbench_autocomplete")

# Autocomplete-specific arguments.
parser.add_argument(
    "--task_type",
    type=str,
    default="REGRESSION",
    choices=["BINARY_CLASSIFICATION", "REGRESSION", "MULTILABEL_CLASSIFICATION"],
)
args = parser.parse_args()


def resolve_gnn_encoder_mode(args: argparse.Namespace) -> None:
    """Normalize the compact --gnn interface.

    User-facing choices are intentionally kept minimal:
        --gnn staged     -> old three-step/staged encoder
        --gnn hgt        -> one-step local heterogeneous transformer
        --gnn graphormer -> one-step global Graphormer-style encoder

    The derived attributes are useful for logging and for Model implementations
    that route to a graph encoder factory. The run script still passes args.gnn
    to Model, so the public interface remains a single --gnn argument.
    """
    gnn = str(args.gnn).lower()

    if gnn == "staged":
        args.encoder_family = "staged"
        args.encoder_name = "staged"
        args.one_step_encoder_name = None
    elif gnn == "hgt":
        args.encoder_family = "one_step"
        args.encoder_name = "hgt"
        args.one_step_encoder_name = "hgt"
    elif gnn == "graphormer":
        args.encoder_family = "one_step"
        args.encoder_name = "graphormer"
        args.one_step_encoder_name = "graphormer"
    else:
        raise ValueError(f"Unknown --gnn value: {args.gnn}")

    args.gnn = gnn


resolve_gnn_encoder_mode(args)
args.q_readout_mode = normalize_q_readout_mode(args.q_readout_mode)
if args.prediction_head_q_readout_mode is not None:
    args.prediction_head_q_readout_mode = normalize_q_readout_mode(
        args.prediction_head_q_readout_mode
    )


try:
    device = resolve_device(args.torch_device, args.require_cuda)
except GPURequiredError as exc:
    exit_for_gpu_error(exc)

if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)


try:
    task: EntityTask = get_task(args.dataset, args.task, download=bool(args.download))
except Exception:
    if bool(args.download):
        print(
            "Download failed. If you already have the dataset cached, re-run with --no-download.",
            file=sys.stderr,
        )
    raise
dataset: Dataset = task.dataset

stypes_cache_path = Path(
    f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/stypes.json"
)
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

# Remove the target column from the col_to_stype_dict if it exists
if task.target_col in col_to_stype_dict[task.entity_table]:
    del col_to_stype_dict[task.entity_table][task.target_col]
for col in dataset.remove_columns:
    if col in col_to_stype_dict[task.entity_table]:
        del col_to_stype_dict[task.entity_table][col]

data, col_stats_dict = make_pkey_fkey_graph(
    # NOTE: Important!: Do not use `upto_test_timestamp=True` here, as this will
    # cause task rows with timestamps after the test timestamp to dangle.
    dataset.get_db(
        upto_test_timestamp=False,
    ),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized",
)

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = args.regression_tune_metric
    higher_is_better = tune_metric == "r2"
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
    out_channels = task.num_classes
    loss_fn = CrossEntropyLoss()
    tune_metric = "mrr"  # "macro_f1"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

def build_num_neighbors(data, args):
    """Build NeighborLoader num_neighbors configuration.

    total: keep the original list format.
    per_edge_type: use PyG dict format and assign the same per-relation budget
    to every edge type.
    """
    base_per_hop = [
        max(1, int(args.num_neighbors / 2**i))
        for i in range(args.num_layers)
    ]

    if args.neighbor_sampling_mode == "total":
        return base_per_hop

    num_edge_types = max(1, len(data.edge_types))
    per_edge_type_per_hop = [
        max(1, int(n / num_edge_types))
        for n in base_per_hop
    ]
    return {edge_type: per_edge_type_per_hop for edge_type in data.edge_types}


num_neighbors_cfg = build_num_neighbors(data, args)

loader_dict: Dict[str, object] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]

    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors_cfg,
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train() -> float:
    model.train()
    device_guard.preflight(model)

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    pbar = tqdm(loader_dict["train"], total=total_steps, mininterval=10.0)
    for batch in pbar:
        batch = batch.to(device)
        device_guard.maybe_check(model=model, batch=batch, step=steps + 1)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = loss_fn(pred.float(), batch[entity_table].y)
        else:
            loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        pbar.set_description(f"Loss: {loss_accum / count_accum:.4f}")
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()
    device_guard.preflight(model)

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=1)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()



# ============================================================
# Two-stage prediction-head experiment helpers
# ============================================================

def _extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint object type: {type(checkpoint).__name__}"
        )

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint model state_dict must be a dict, got {type(state_dict).__name__}"
        )

    return state_dict


def _is_prediction_head_key(key: str) -> bool:
    # In quantum_model.trainer.model.Model, the prediction head is stored as self.head.
    return key.startswith("head.")


def initialize_from_backbone_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
    load_backbone_only: bool,
) -> None:
    if checkpoint_path is None:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"init_backbone_checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = _extract_model_state_dict(checkpoint)

    if load_backbone_only:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not _is_prediction_head_key(key)
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded backbone-only checkpoint from: {path}")
        print(f"Loaded tensors: {len(state_dict)}")
        print(f"Missing keys after backbone-only load: {len(missing)}")
        print(f"Unexpected keys after backbone-only load: {len(unexpected)}")
    else:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded full checkpoint from: {path}")


def apply_head_experiment_controls(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if args.reset_prediction_head:
        if hasattr(model, "reset_prediction_head"):
            model.reset_prediction_head()
        elif hasattr(model, "head") and hasattr(model.head, "reset_parameters"):
            model.head.reset_parameters()
        else:
            raise RuntimeError(
                "reset_prediction_head=True, but the model has no reset_prediction_head() "
                "method and model.head.reset_parameters() is unavailable."
            )
        print("Prediction head has been reset.")

    if args.freeze_backbone:
        if hasattr(model, "freeze_backbone"):
            model.freeze_backbone()
        else:
            raise RuntimeError(
                "freeze_backbone=True, but the model does not implement freeze_backbone()."
            )
        print("Backbone has been frozen. Only prediction-head parameters remain trainable.")
    else:
        if hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
        print("Backbone is trainable.")

    trainable_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    total_count = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable_count} / {total_count}")

    if trainable_count <= 0:
        raise RuntimeError("No trainable parameters left after applying freeze controls.")


def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("Cannot build optimizer: no trainable parameters.")
    return torch.optim.Adam(trainable_params, lr=lr)


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",

    gnn=args.gnn,

    intra_aggr=args.intra_aggr,
    type_fusion=args.type_fusion,
    node_update=args.node_update,

    gat_heads=args.gat_heads,
    gat_dropout=args.gat_dropout,

    n_qubits=args.n_qubits,
    n_q_layers=args.n_q_layers,
    n_heads=args.n_heads,

    q_readout_mode=args.q_readout_mode,
    q_circuit_type=args.q_circuit_type,
    q_ansatz_type=args.q_ansatz_type,
    q_angle_activation=args.q_angle_activation,
    q_use_angle_affine=args.q_use_angle_affine,
    q_residual_mode=args.q_residual_mode,
    q_alpha_init=args.q_alpha_init,

    q_freeze_quantum_at_init=args.freeze_quantum_at_init,
    q_freeze_alpha_with_quantum=args.freeze_alpha_with_quantum,
    q_quantum_unfreeze_epoch=args.quantum_unfreeze_epoch,
    q_quantum_only_finetune=args.quantum_only_finetune,
    prediction_head=args.prediction_head,
    prediction_head_hidden_dim=args.prediction_head_hidden_dim,
    prediction_head_num_layers=args.prediction_head_num_layers,
    prediction_head_dropout=args.prediction_head_dropout,
    prediction_head_n_qubits=args.prediction_head_n_qubits,
    prediction_head_n_q_layers=args.prediction_head_n_q_layers,
    prediction_head_n_heads=args.prediction_head_n_heads,
    prediction_head_q_readout_mode=args.prediction_head_q_readout_mode,
    prediction_head_q_circuit_type=args.prediction_head_q_circuit_type,
    prediction_head_q_ansatz_type=args.prediction_head_q_ansatz_type,
    prediction_head_q_angle_activation=args.prediction_head_q_angle_activation,
    prediction_head_q_use_angle_affine=args.prediction_head_q_use_angle_affine,
    prediction_head_q_dropout=args.prediction_head_q_dropout,
    prediction_head_q_residual_mode=args.prediction_head_q_residual_mode,
    prediction_head_q_alpha_init=args.prediction_head_q_alpha_init,
    prediction_head_freeze_quantum_at_init=args.prediction_head_freeze_quantum_at_init,
    prediction_head_freeze_alpha_with_quantum=args.prediction_head_freeze_alpha_with_quantum,
    prediction_head_export_circuit=args.prediction_head_export_circuit,
).to(device)

initialize_from_backbone_checkpoint(
    model=model,
    checkpoint_path=args.init_backbone_checkpoint,
    device=device,
    load_backbone_only=args.load_backbone_only,
)
apply_head_experiment_controls(model, args)
optimizer = build_optimizer(model, args.lr)
# =========================
# logging / history
# =========================

output_dir = getattr(args, "output_dir", "training_logs")
os.makedirs(output_dir, exist_ok=True)

run_name = resolve_run_name(
    args.run_name,
    default_run_name="relbench_autocomplete",
    auto_run_name=build_auto_run_name(args, task_family="autocomplete"),
)

run_manager = TrainingRunManager(
    output_dir=output_dir,
    run_name=run_name,
    args=args,
    script_path=__file__,
)
device_guard = RuntimeDeviceGuard(
    device=device,
    require_cuda=args.require_cuda,
    check_interval=args.device_check_interval,
)

history = {
    "epoch": [],
    "train_loss": [],
    "val_metrics": {},
    "run_name": run_name,
    "args": vars(args),
    "tune_metric": tune_metric,
}

best_epoch = -1

best_val_metric = -math.inf if higher_is_better else math.inf
epochs_without_improvement = 0
resume_state = run_manager.resume_if_possible(
    model=model,
    optimizer=optimizer,
    history=history,
    best_epoch=best_epoch,
    best_val_metric=best_val_metric,
    device=device,
    resume_enabled=args.resume,
)
history = resume_state.history
best_epoch = resume_state.best_epoch
best_val_metric = resume_state.best_val_metric
last_completed_epoch = max(resume_state.start_epoch - 1, 0)
history.setdefault("tune_metric", tune_metric)
history.setdefault("early_stopping_patience", args.early_stopping_patience)
history.setdefault("early_stopping_min_delta", args.early_stopping_min_delta)

try:
    for epoch in range(resume_state.start_epoch, args.epochs + 1):
        if hasattr(model, "apply_quantum_schedule"):
            model.apply_quantum_schedule(epoch)

        train_loss = train()
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))

        for metric_name, metric_value in val_metrics.items():
            if metric_name not in history["val_metrics"]:
                history["val_metrics"][metric_name] = []

            history["val_metrics"][metric_name].append(float(metric_value))

        print(
            f"Epoch: {epoch:02d}, "
            f"Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        current_val_metric = float(val_metrics[tune_metric])
        improved = (
            (higher_is_better and current_val_metric > best_val_metric + args.early_stopping_min_delta)
            or
            (not higher_is_better and current_val_metric < best_val_metric - args.early_stopping_min_delta)
        )
        tied_best = (
            (higher_is_better and current_val_metric >= best_val_metric)
            or
            (not higher_is_better and current_val_metric <= best_val_metric)
        )

        if improved or (best_epoch < 0 and tied_best):
            best_val_metric = current_val_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            run_manager.save_best_checkpoint(
                epoch=epoch,
                model=model,
                best_val_metric=best_val_metric,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        if args.checkpoint_every_epoch > 0 and epoch % args.checkpoint_every_epoch == 0:
            run_manager.save_epoch_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                best_epoch=best_epoch,
                best_val_metric=best_val_metric,
            )

        last_completed_epoch = epoch
        if (
            args.early_stopping_patience is not None
            and args.early_stopping_patience >= 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"patience={args.early_stopping_patience}, "
                f"min_delta={args.early_stopping_min_delta}, "
                f"tune_metric={tune_metric}"
            )
            break
except GPURequiredError as exc:
    run_manager.save_epoch_checkpoint(
        epoch=last_completed_epoch,
        model=model,
        optimizer=optimizer,
        history=history,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        status="interrupted_gpu",
        message=str(exc),
    )
    run_manager.mark_status(
        status="interrupted_gpu",
        epoch=last_completed_epoch,
        history=history,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        message=str(exc),
    )
    exit_for_gpu_error(exc)
except KeyboardInterrupt:
    run_manager.save_epoch_checkpoint(
        epoch=last_completed_epoch,
        model=model,
        optimizer=optimizer,
        history=history,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        status="interrupted",
        message="KeyboardInterrupt",
    )
    run_manager.mark_status(
        status="interrupted",
        epoch=last_completed_epoch,
        history=history,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        message="KeyboardInterrupt",
    )
    raise
except Exception as exc:
    run_manager.save_epoch_checkpoint(
        epoch=last_completed_epoch,
        model=model,
        optimizer=optimizer,
        history=history,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        status="failed",
        message=repr(exc),
    )
    run_manager.mark_status(
        status="failed",
        epoch=last_completed_epoch,
        history=history,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        message=repr(exc),
    )
    raise

if not run_manager.load_best_model(model, device=device):
    print("Warning: best checkpoint is missing, using current model weights.")

val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))

print(f"Best epoch: {best_epoch}")
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)

print(f"Best test metrics: {test_metrics}")

# =========================
# save history
# =========================

history["best_epoch"] = best_epoch
history["best_val_metric"] = float(best_val_metric)
history["best_val_metrics"] = {k: float(v) for k, v in val_metrics.items()}
history["best_test_metrics"] = {k: float(v) for k, v in test_metrics.items()}
history["tune_metric"] = tune_metric
history["early_stopping_patience"] = args.early_stopping_patience
history["early_stopping_min_delta"] = args.early_stopping_min_delta

history_path = os.path.join(
    output_dir,
    f"{run_name}_history.json",
)

with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

run_manager.save_epoch_checkpoint(
    epoch=last_completed_epoch,
    model=model,
    optimizer=optimizer,
    history=history,
    best_epoch=best_epoch,
    best_val_metric=best_val_metric,
    status="completed",
)
run_manager.mark_status(
    status="completed",
    epoch=last_completed_epoch,
    history=history,
    best_epoch=best_epoch,
    best_val_metric=best_val_metric,
    best_val_metrics=history["best_val_metrics"],
    best_test_metrics=history["best_test_metrics"],
)

print(f"Saved history to: {history_path}")
