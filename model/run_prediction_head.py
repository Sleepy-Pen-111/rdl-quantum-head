from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DEFAULT_TRAINER_SCRIPT = (
    PROJECT_ROOT / "quantum_model" / "trainer" / "gnn_autocomplete_2stage.py"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_logs" / "prediction_head"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_model.model.quantum_ansatz import VALID_Q_ANSATZ_TYPES


# ============================================================
# Experiment Grid Configuration
# ============================================================
# Edit this block for most experiment changes. CLI arguments below use these
# values as defaults, so you can still override them from the command line.

EXPERIMENT_GRID = {
    # Basic experiment identity
    "channel": "prediction_head",
    "dataset": "rel-hm",
    "task": "transactions-price",

    # Random seeds. Example: [42, 43, 44]
    "seed": 42,
    "seeds": [42, 43, 44],

    # Training basics
    "lr": 0.001,
    "epochs": 60,
    "batch_size": 512,
    "channels": 128,

    # Early stopping.
    # Stop if validation metric does not improve for this many consecutive epochs.
    # Set patience <= 0 to disable early stopping.
    #"early_stopping_patience": 10,
    #"early_stopping_min_delta": 0.0,

    # Graph backbone configuration
    "gnn": "staged",
    "intra_aggr": "max_project",
    "type_fusion": "sum",
    "node_update": "mlp",
    "num_layers": 1,
    "num_neighbors": 128,
    "neighbor_sampling_mode": "total",
    "temporal_strategy": "uniform",
    "max_steps_per_epoch": 2000,
    "num_workers": 0,

    # Two-stage / end-to-end protocol
    # choices: "head_only", "finetune", "scratch", "end_to_end"
    "two_stage_head_search": True,
    "head_training_mode": "head_only",
    "backbone_epochs": 100,
    "head_epochs": 50,
    "backbone_hidden_dim": 128,
    "backbone_head_layers": 2,

    # High-level prediction-head selection.
    # "mlp" expands to standard_mlp + matched_mlp.
    # Other valid entries: "standard_mlp", "matched_mlp", "quantum", "residual_quantum".
    "prediction_heads": ["mlp", "quantum"],

    # Fixed standard MLP baseline
    "standard_mlp_hidden_dim": 128,
    "standard_mlp_num_layers": 2,
    "classical_head_hidden_dims": [128],
    "classical_head_depths": [2, 4, 8],
    "classical_head_dropouts": [0.0, 0.1, 0.2],

    # Quantum-head grid
    "prediction_head_qubits": [5, 10],
    "prediction_head_encodings": ["angle", "rxry", "arctan", "amplitude"],
    "prediction_head_readouts": ["z_only", "z_pairwise", "z_all", "probs"],
    "prediction_head_q_ansatz_types": ["rot_cnot_ring"],
    "prediction_head_n_q_layers": 2,
    "prediction_head_q_angle_activation": "tanh",
    "prediction_head_q_use_angle_affine": False,
    "prediction_head_q_dropouts": [0.0, 0.1, 0.2],
    "prediction_head_q_residual_mode": "add",

    # Matched MLP settings
    "classical_num_layers": 4,
    "max_classical_hidden_dim": 512,
    "matched_classical_head_depths": [2],
    "matched_classical_head_dropouts": [0.0],
    "matched_param_tolerance_rel": 0.05,
    "matched_param_tolerance_abs": 100,

    # Feasibility guards
    "max_quantum_input_dim": 4096,
    "max_readout_dim": 4096,
    "max_prediction_head_params": 150000,

    # Resume / execution controls
    "resume": False,
    "resume_backbone": True,
    "resume_heads": False,
    "skip_completed": True,
    "force_retrain_backbone": False,
    "extend_late_best": False,
    "extend_best_epoch_min": 90,
    "extend_extra_epochs": 50,
    "extend_roles": ["base_mlp", "matched_mlp", "qhead"],

    # Metrics / paths
    #"regression_tune_metric": "r2",
    "cache_dir": str(Path.home() / ".cache" / "relbench_examples"),
}


def grid_default(key: str, default):
    return EXPERIMENT_GRID.get(key, default)


VALID_GNN_CHOICES = ["staged", "hgt", "graphormer"]
VALID_INTRA_AGGR_CHOICES = [
    "mean",
    "max",
    "mean_project",
    "max_project",
    "lstm",
    "gatv2",
]
VALID_TYPE_FUSION_CHOICES = [
    "sum",
    "mean",
    "weighted_sum",
    "relation_weighted_sum",
    "edge_type_weighted_sum",
    "gru",
]
VALID_NODE_UPDATE_CHOICES = ["mlp", "gru"]
VALID_NEIGHBOR_SAMPLING_MODES = ["total", "per_edge_type"]
VALID_REGRESSION_TUNE_METRICS = ["r2", "mae", "rmse"]
VALID_Q_CIRCUIT_TYPES = ["angle", "rxry", "arctan", "amplitude"]
VALID_Q_READOUT_MODES = ["z_pairwise", "z_only", "z_all", "probs"]
VALID_Q_ANGLE_ACTIVATIONS = ["tanh", "atan", "none"]
VALID_EXTEND_ROLES = ["base_mlp", "matched_mlp", "qhead", "rqhead"]


_HEAD_ALIASES = {
    "mlp": "mlp",
    "classical": "mlp",
    "standard_mlp": "standard_mlp",
    "base_mlp": "standard_mlp",
    "matched_mlp": "matched_mlp",
    "param_matched_mlp": "matched_mlp",
    "quantum": "quantum",
    "qhead": "quantum",
    "residual_quantum": "residual_quantum",
    "rqhead": "residual_quantum",
    "hybrid_quantum": "residual_quantum",
}


def _validate_mlp_depths_at_least_two(
    args: argparse.Namespace,
    *,
    attr_name: str,
    label: str,
) -> None:
    depths = getattr(args, attr_name, None)
    if depths is None:
        return

    invalid_depths = [int(depth) for depth in depths if int(depth) < 2]
    if invalid_depths:
        raise ValueError(
            f"{label} must use at least two layers for the controlled MLP grid. "
            f"Got {attr_name}={list(depths)!r}."
        )


def apply_prediction_head_selection(args: argparse.Namespace) -> argparse.Namespace:
    """Translate the front-end prediction_heads list into internal include flags."""
    raw_heads = args.prediction_heads or []
    selected: set[str] = set()

    for item in raw_heads:
        key = str(item).strip().lower().replace("-", "_")
        if key not in _HEAD_ALIASES:
            valid = ", ".join(sorted(_HEAD_ALIASES))
            raise ValueError(f"Unknown prediction head '{item}'. Valid entries: {valid}")
        selected.add(_HEAD_ALIASES[key])

    args.include_standard_mlp = "mlp" in selected or "standard_mlp" in selected
    args.include_matched_mlp = "mlp" in selected or "matched_mlp" in selected
    args.include_quantum = "quantum" in selected
    args.include_residual_quantum = "residual_quantum" in selected

    if args.include_standard_mlp:
        _validate_mlp_depths_at_least_two(
            args,
            attr_name="classical_head_depths",
            label="Standard MLP depths",
        )
    if args.include_matched_mlp:
        _validate_mlp_depths_at_least_two(
            args,
            attr_name="matched_classical_head_depths",
            label="Matched MLP depths",
        )
        if args.matched_param_tolerance_rel < 0.0:
            raise ValueError("--matched_param_tolerance_rel must be non-negative.")
        if args.matched_param_tolerance_abs < 0:
            raise ValueError("--matched_param_tolerance_abs must be non-negative.")

    return args


from quantum_model.utils.prediction_head_run_utils import (
    build_backbone_checkpoint_path,
    build_seed_specs,
    execute_plan,
    get_experiment_root,
    print_summary,
    select_late_best_experiments_for_extension,
    uses_pretrained_backbone,
    write_aggregate_summary,
    write_plan,
    write_run_summary,
)


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, execute, and summarize a controlled two-stage experiment matrix "
            "for prediction-head comparisons."
        )
    )

    parser.add_argument("--channel", type=str, default=EXPERIMENT_GRID["channel"])
    parser.add_argument("--dataset", type=str, default=EXPERIMENT_GRID["dataset"])
    parser.add_argument("--task", type=str, default=EXPERIMENT_GRID["task"])

    parser.add_argument("--lr", type=float, default=EXPERIMENT_GRID["lr"])
    parser.add_argument("--epochs", type=int, default=EXPERIMENT_GRID["epochs"])
    parser.add_argument("--batch_size", type=int, default=EXPERIMENT_GRID["batch_size"])
    parser.add_argument("--channels", type=int, default=EXPERIMENT_GRID["channels"])
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=grid_default("early_stopping_patience", None),
        help=(
            "Stop training if the validation metric does not improve for this many "
            "consecutive epochs. Use <=0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=grid_default("early_stopping_min_delta", 0.0),
        help=(
            "Minimum validation-metric improvement required to reset early-stopping patience."
        ),
    )

    parser.add_argument(
        "--gnn",
        type=str,
        default=EXPERIMENT_GRID["gnn"],
        choices=VALID_GNN_CHOICES,
        help=(
            "Graph encoder family. "
            "'staged' uses the explicit aggregation/fusion/update pipeline. "
            "'hgt' and 'graphormer' use one-step encoders."
        ),
    )
    parser.add_argument(
        "--intra_aggr",
        type=str,
        default=EXPERIMENT_GRID["intra_aggr"],
        choices=VALID_INTRA_AGGR_CHOICES,
    )
    parser.add_argument(
        "--type_fusion",
        type=str,
        default=EXPERIMENT_GRID["type_fusion"],
        choices=VALID_TYPE_FUSION_CHOICES,
    )
    parser.add_argument(
        "--node_update",
        type=str,
        default=EXPERIMENT_GRID["node_update"],
        choices=VALID_NODE_UPDATE_CHOICES,
    )
    parser.add_argument("--num_layers", type=int, default=EXPERIMENT_GRID["num_layers"])

    parser.add_argument("--num_neighbors", type=int, default=EXPERIMENT_GRID["num_neighbors"])
    parser.add_argument(
        "--neighbor_sampling_mode",
        type=str,
        default=EXPERIMENT_GRID["neighbor_sampling_mode"],
        choices=VALID_NEIGHBOR_SAMPLING_MODES,
    )
    parser.add_argument("--temporal_strategy", type=str, default=EXPERIMENT_GRID["temporal_strategy"])
    parser.add_argument("--max_steps_per_epoch", type=int, default=EXPERIMENT_GRID["max_steps_per_epoch"])
    parser.add_argument("--num_workers", type=int, default=EXPERIMENT_GRID["num_workers"])
    parser.add_argument("--seed", type=int, default=EXPERIMENT_GRID["seed"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=EXPERIMENT_GRID["seeds"],
        help=(
            "Run the same experiment matrix for multiple seeds. "
            "Example: --seeds 42 43 44. If omitted, the single --seed value is used."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--trainer_script",
        type=Path,
        default=DEFAULT_TRAINER_SCRIPT,
        help=(
            "Trainer script called by this launcher. Use this if your two-stage "
            "trainer is named gnn_entity_2stage.py."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=EXPERIMENT_GRID["cache_dir"],
    )

    parser.add_argument(
        "--regression_tune_metric",
        type=str,
        default=grid_default("regression_tune_metric", "r2"),
        choices=VALID_REGRESSION_TUNE_METRICS,
    )

    # Resume policy
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["resume"],
        help="Generic resume flag used by non-two-stage mode.",
    )
    parser.add_argument(
        "--resume_backbone",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["resume_backbone"],
        help="Resume the shared backbone pretraining run if checkpoint exists.",
    )
    parser.add_argument(
        "--resume_heads",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["resume_heads"],
        help="Resume individual head runs if checkpoint exists.",
    )
    parser.add_argument(
        "--skip_completed",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["skip_completed"],
        help="Skip runs whose summary/status already indicates completion.",
    )

    # Two-stage protocol
    parser.add_argument(
        "--two_stage_head_search",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["two_stage_head_search"],
        help="Train backbone once per seed, then load it for different prediction heads.",
    )
    parser.add_argument(
        "--head_training_mode",
        type=str,
        default=EXPERIMENT_GRID["head_training_mode"],
        choices=["head_only", "finetune", "scratch", "end_to_end"],
        help=(
            "Prediction-head training protocol. "
            "'head_only' loads an MLP-pretrained backbone, freezes it, and trains only the head. "
            "'finetune' loads an MLP-pretrained backbone and jointly fine-tunes backbone + head. "
            "'scratch' does not load a pretrained backbone and trains the full model from random initialization. "
            "'end_to_end' is kept as a backward-compatible alias for 'finetune'."
        ),
    )
    parser.add_argument(
        "--backbone_epochs",
        type=int,
        default=EXPERIMENT_GRID["backbone_epochs"],
        help="Epochs for the shared backbone pretraining stage.",
    )
    parser.add_argument(
        "--head_epochs",
        type=int,
        default=EXPERIMENT_GRID["head_epochs"],
        help="Epochs for each prediction-head run after loading the shared backbone.",
    )
    parser.add_argument(
        "--backbone_hidden_dim",
        type=int,
        default=EXPERIMENT_GRID["backbone_hidden_dim"],
        help="Hidden dimension of the base MLP head used during backbone pretraining.",
    )
    parser.add_argument(
        "--backbone_head_layers",
        type=int,
        default=EXPERIMENT_GRID["backbone_head_layers"],
        help="Number of MLP layers in the base head used during backbone pretraining.",
    )
    parser.add_argument(
        "--force_retrain_backbone",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["force_retrain_backbone"],
        help="Retrain the shared backbone even if best_checkpoint.pt already exists.",
    )

    # Late-best extension
    parser.add_argument(
        "--extend_late_best",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["extend_late_best"],
        help=(
            "Only continue completed runs whose best_epoch is close to the previous "
            "training limit. Uses the same run_name and resume=True."
        ),
    )
    parser.add_argument(
        "--extend_best_epoch_min",
        type=int,
        default=EXPERIMENT_GRID["extend_best_epoch_min"],
        help="Only extend runs whose best_epoch is >= this threshold.",
    )
    parser.add_argument(
        "--extend_extra_epochs",
        type=int,
        default=EXPERIMENT_GRID["extend_extra_epochs"],
        help="Number of additional epochs to train selected late-best runs.",
    )
    parser.add_argument(
        "--extend_roles",
        type=str,
        nargs="+",
        default=EXPERIMENT_GRID["extend_roles"],
        choices=VALID_EXTEND_ROLES,
        help="Which roles are eligible for extension.",
    )

    parser.add_argument(
        "--prediction_heads",
        type=str,
        nargs="+",
        default=EXPERIMENT_GRID["prediction_heads"],
        help=(
            "High-level prediction-head selection. Examples: "
            "--prediction_heads mlp quantum, or --prediction_heads standard_mlp matched_mlp residual_quantum."
        ),
    )

    # Which experiment groups to include
    parser.add_argument(
        "--include_standard_mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Internal compatibility flag. Prefer --prediction_heads or EXPERIMENT_GRID['prediction_heads'].",
    )
    parser.add_argument(
        "--include_matched_mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Internal compatibility flag. Prefer --prediction_heads or EXPERIMENT_GRID['prediction_heads'].",
    )
    parser.add_argument(
        "--include_quantum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Internal compatibility flag. Prefer --prediction_heads or EXPERIMENT_GRID['prediction_heads'].",
    )
    parser.add_argument(
        "--include_residual_quantum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Internal compatibility flag. Prefer --prediction_heads or EXPERIMENT_GRID['prediction_heads'].",
    )

    # Fixed standard MLP baseline
    parser.add_argument(
        "--standard_mlp_hidden_dim",
        type=int,
        default=EXPERIMENT_GRID["standard_mlp_hidden_dim"],
        help="Hidden dimension for the fixed standard MLP baseline.",
    )
    parser.add_argument(
        "--standard_mlp_num_layers",
        type=int,
        default=EXPERIMENT_GRID["standard_mlp_num_layers"],
        help="Legacy single-depth setting for the fixed standard MLP baseline.",
    )
    parser.add_argument(
        "--classical_head_hidden_dims",
        type=int,
        nargs="+",
        default=EXPERIMENT_GRID["classical_head_hidden_dims"],
        help="Hidden dimensions to sweep for the fixed classical MLP baseline.",
    )
    parser.add_argument(
        "--classical_head_depths",
        type=int,
        nargs="+",
        default=EXPERIMENT_GRID["classical_head_depths"],
        help="Depths to sweep for the fixed classical MLP baseline.",
    )
    parser.add_argument(
        "--classical_head_dropouts",
        type=float,
        nargs="+",
        default=EXPERIMENT_GRID["classical_head_dropouts"],
        help="Dropout values to sweep for the fixed classical MLP baseline.",
    )

    # Quantum-head search space
    parser.add_argument(
        "--prediction_head_qubits",
        type=int,
        nargs="+",
        default=EXPERIMENT_GRID["prediction_head_qubits"],
    )
    parser.add_argument(
        "--prediction_head_encodings",
        type=str,
        nargs="+",
        default=EXPERIMENT_GRID["prediction_head_encodings"],
        choices=VALID_Q_CIRCUIT_TYPES,
    )
    parser.add_argument(
        "--prediction_head_readouts",
        type=str,
        nargs="+",
        default=EXPERIMENT_GRID["prediction_head_readouts"],
        choices=VALID_Q_READOUT_MODES,
    )
    parser.add_argument(
        "--prediction_head_q_ansatz_types",
        type=str,
        nargs="+",
        default=EXPERIMENT_GRID["prediction_head_q_ansatz_types"],
        choices=sorted(VALID_Q_ANSATZ_TYPES),
    )
    parser.add_argument("--prediction_head_n_q_layers", type=int, default=EXPERIMENT_GRID["prediction_head_n_q_layers"])
    parser.add_argument(
        "--prediction_head_q_angle_activation",
        type=str,
        default=EXPERIMENT_GRID["prediction_head_q_angle_activation"],
        choices=VALID_Q_ANGLE_ACTIVATIONS,
    )
    parser.add_argument(
        "--prediction_head_q_use_angle_affine",
        action=argparse.BooleanOptionalAction,
        default=EXPERIMENT_GRID["prediction_head_q_use_angle_affine"],
    )
    parser.add_argument(
        "--Q_Pred_Head_Dropout",
        "--prediction_head_q_dropouts",
        dest="prediction_head_q_dropouts",
        type=float,
        nargs="+",
        default=EXPERIMENT_GRID["prediction_head_q_dropouts"],
        help=(
            "Dropout values to sweep for quantum prediction-head readout features "
            "before the final post-net."
        ),
    )
    parser.add_argument(
        "--prediction_head_q_residual_mode",
        type=str,
        default=EXPERIMENT_GRID["prediction_head_q_residual_mode"],
        choices=["add", "concat"],
    )

    # Matched MLP settings
    parser.add_argument(
        "--classical_num_layers",
        type=int,
        default=EXPERIMENT_GRID["classical_num_layers"],
        help=(
            "Legacy single-depth setting for the matched MLP baseline. "
            "Use a non-positive value to auto-set it to n_q_layers + 2."
        ),
    )
    parser.add_argument("--max_classical_hidden_dim", type=int, default=EXPERIMENT_GRID["max_classical_hidden_dim"])
    parser.add_argument(
        "--matched_classical_head_depths",
        type=int,
        nargs="+",
        default=EXPERIMENT_GRID["matched_classical_head_depths"],
        help=(
            "Depths to sweep for the parameter-matched classical MLP baseline. "
            "Each value must be at least 2."
        ),
    )
    parser.add_argument(
        "--matched_classical_head_dropouts",
        type=float,
        nargs="+",
        default=EXPERIMENT_GRID["matched_classical_head_dropouts"],
        help="Dropout values to sweep for the parameter-matched classical MLP baseline.",
    )
    parser.add_argument(
        "--matched_param_tolerance_rel",
        type=float,
        default=EXPERIMENT_GRID["matched_param_tolerance_rel"],
        help=(
            "Relative tolerance for reusing an existing matched MLP target. "
            "A quantum head can share a matched MLP when target parameter counts "
            "differ by at most max(abs_tolerance, rel_tolerance * target_params)."
        ),
    )
    parser.add_argument(
        "--matched_param_tolerance_abs",
        type=int,
        default=EXPERIMENT_GRID["matched_param_tolerance_abs"],
        help="Absolute minimum tolerance for reusing an existing matched MLP target.",
    )

    # Feasibility guards
    parser.add_argument(
        "--max_quantum_input_dim",
        type=int,
        default=EXPERIMENT_GRID["max_quantum_input_dim"],
        help="Skip quantum configs whose pre-circuit width exceeds this limit.",
    )
    parser.add_argument(
        "--max_readout_dim",
        type=int,
        default=EXPERIMENT_GRID["max_readout_dim"],
        help="Skip quantum configs whose measurement width exceeds this limit.",
    )
    parser.add_argument(
        "--max_prediction_head_params",
        type=int,
        default=EXPERIMENT_GRID["max_prediction_head_params"],
        help=(
            "Skip prediction-head configs whose trainable parameter count exceeds "
            "this limit. Use <=0 to disable this filter."
        ),
    )

    # Execution
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch all experiments in sequence.",
    )

    args = parser.parse_args()
    return apply_prediction_head_selection(args)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    trainer_script = Path(args.trainer_script)

    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

    experiment_root = get_experiment_root(args)
    experiment_root.mkdir(parents=True, exist_ok=True)

    seed_list, backbone_specs, experiments, skipped = build_seed_specs(args)

    if not experiments:
        raise RuntimeError("No feasible prediction-head experiments were generated.")

    scan_dirs: list[Path] = [experiment_root]
    for backbone_spec in backbone_specs:
        backbone_output_dir = Path(backbone_spec.args["output_dir"])
        if backbone_output_dir not in scan_dirs:
            scan_dirs.append(backbone_output_dir)

    plan_path = write_plan(
        output_dir=experiment_root,
        channel=args.channel,
        experiments=experiments,
        skipped=skipped,
        trainer_script=trainer_script,
        backbone_specs=backbone_specs,
    )

    print_summary(
        channel=args.channel,
        experiments=experiments,
        skipped=skipped,
        plan_path=plan_path,
        backbone_specs=backbone_specs,
    )

    status_path = experiment_root / "run_status.jsonl"

    if args.execute:
        if uses_pretrained_backbone(args):
            for backbone_spec in backbone_specs:
                seed_value = int(backbone_spec.args.get("seed"))
                seed_args = copy.copy(args)
                seed_args.seed = seed_value

                backbone_checkpoint_path = build_backbone_checkpoint_path(seed_args)

                if args.force_retrain_backbone or not backbone_checkpoint_path.exists():
                    print("")
                    print(f"Shared backbone checkpoint not found for seed={seed_value}.")
                    print("Starting shared backbone pretraining first.")
                    print(f"Expected checkpoint: {backbone_checkpoint_path}")

                    # Avoid resuming from stale last_checkpoint.pt when best_checkpoint.pt is missing.
                    backbone_spec.args["resume"] = False

                    execute_plan(
                        [backbone_spec],
                        project_root=PROJECT_ROOT,
                        trainer_script=trainer_script,
                        status_path=status_path,
                        summary_output_dir=experiment_root,
                        scan_dirs=scan_dirs,
                        all_summary_experiments=experiments,
                        backbone_specs=backbone_specs,
                        skip_completed=False,
                    )
                else:
                    print("")
                    print(
                        f"Using existing shared backbone checkpoint for seed={seed_value}: "
                        f"{backbone_checkpoint_path}"
                    )

                if not backbone_checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Backbone checkpoint not found after pretraining "
                        f"for seed={seed_value}: {backbone_checkpoint_path}"
                    )

            if args.extend_late_best:
                extension_specs, extension_skipped = select_late_best_experiments_for_extension(
                    experiments=experiments,
                    best_epoch_min=args.extend_best_epoch_min,
                    extra_epochs=args.extend_extra_epochs,
                    eligible_roles=set(args.extend_roles),
                )

                print("")
                print("Late-best extension mode enabled.")
                print(f"Selected runs for extension: {len(extension_specs)}")
                print(f"Extension skipped: {len(extension_skipped)}")
                for item in extension_skipped[:80]:
                    print(f"  - {item}")
                if len(extension_skipped) > 80:
                    print(f"  ... {len(extension_skipped) - 80} more skipped entries")

                if extension_specs:
                    execute_plan(
                        extension_specs,
                        project_root=PROJECT_ROOT,
                        trainer_script=trainer_script,
                        status_path=status_path,
                        summary_output_dir=experiment_root,
                        scan_dirs=scan_dirs,
                        all_summary_experiments=experiments,
                        backbone_specs=backbone_specs,
                        skip_completed=False,
                    )
                else:
                    print("No runs selected for extension.")
            else:
                execute_plan(
                    experiments,
                    project_root=PROJECT_ROOT,
                    trainer_script=trainer_script,
                    status_path=status_path,
                    summary_output_dir=experiment_root,
                    scan_dirs=scan_dirs,
                    all_summary_experiments=experiments,
                    backbone_specs=backbone_specs,
                    skip_completed=args.skip_completed,
                )

        else:
            if args.extend_late_best:
                extension_specs, extension_skipped = select_late_best_experiments_for_extension(
                    experiments=experiments,
                    best_epoch_min=args.extend_best_epoch_min,
                    extra_epochs=args.extend_extra_epochs,
                    eligible_roles=set(args.extend_roles),
                )
                print("")
                print("Late-best extension mode enabled.")
                print(f"Selected runs for extension: {len(extension_specs)}")
                print(f"Extension skipped: {len(extension_skipped)}")
                if extension_specs:
                    execute_plan(
                        extension_specs,
                        project_root=PROJECT_ROOT,
                        trainer_script=trainer_script,
                        status_path=status_path,
                        summary_output_dir=experiment_root,
                        scan_dirs=scan_dirs,
                        all_summary_experiments=experiments,
                        backbone_specs=None,
                        skip_completed=False,
                    )
                else:
                    print("No runs selected for extension.")
            else:
                execute_plan(
                    experiments,
                    project_root=PROJECT_ROOT,
                    trainer_script=trainer_script,
                    status_path=status_path,
                    summary_output_dir=experiment_root,
                    scan_dirs=scan_dirs,
                    all_summary_experiments=experiments,
                    backbone_specs=None,
                    skip_completed=args.skip_completed,
                )

    csv_path = write_run_summary(
        output_dir=experiment_root,
        experiments=experiments,
        scan_dirs=scan_dirs,
        backbone_specs=backbone_specs,
    )
    agg_path = write_aggregate_summary(summary_csv_path=csv_path)

    print("")
    print(f"Seeds: {seed_list}")
    print(f"Run summary CSV written to: {csv_path}")
    print(f"Aggregate summary CSV written to: {agg_path}")

    if args.execute:
        print(f"Run status written to: {status_path}")
    else:
        print("")
        print("Dry-run only: no training was launched.")
        print("Existing summaries were parsed if available.")
        print("To execute all experiments, rerun with:")
        print("  python run_prediction_head.py --execute")


if __name__ == "__main__":
    main()
