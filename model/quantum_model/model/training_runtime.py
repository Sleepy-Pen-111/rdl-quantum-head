import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


GPU_REQUIRED_EXIT_CODE = 86
RUNTIME_NAME_ARG_KEYS = {
    "run_name",
    "output_dir",
    "resume",
    "checkpoint_every_epoch",
    "device_check_interval",
    "torch_device",
    "require_cuda",
    "cache_dir",
    "download",
}

NAME_VALUE_ALIASES = {
    "q_readout_mode": {
        "z_pairwise": "zp",
        "z_only": "zo",
        "z_all": "za",
        "probs": "pr",
    },
    "q_circuit_type": {
        "angle": "ang",
        "rxry": "rxry",
        "arctan": "atan",
        "amplitude": "amp",
    },
    "q_ansatz_type": {
        "rot_cnot_ring": "rcr",
        "rot_cnot_chain": "rcc",
        "rot_cz_ring": "rcz",
        "rot_none": "r0",
    },
    "q_angle_activation": {
        "tanh": "th",
        "atan": "at",
        "none": "id",
    },
    "q_residual_mode": {
        "concat": "cat",
        "add": "add",
    },
    "temporal_strategy": {
        "uniform": "uni",
        "last": "last",
    },
    "neighbor_sampling_mode": {
        "total": "tot",
        "per_edge_type": "pet",
    },
    "task_type": {
        "REGRESSION": "reg",
        "BINARY_CLASSIFICATION": "bin",
        "MULTILABEL_CLASSIFICATION": "mlc",
        "MULTICLASS_CLASSIFICATION": "mcc",
    },
    "task_family": {
        "entity": "ent",
        "autocomplete": "auto",
        "recommendation": "rec",
    },
    "gnn": {
        "recurrent": "rec",
        "sage": "sage",
        "gat": "gat",
    },
    "prediction_head": {
        "classical": "mlp",
        "quantum": "qhead",
        "residual_quantum": "rqhead",
    },
    "intra_aggr": {
        "mean_project": "mproj",
        "max_project": "xproj",
    },
}


class GPURequiredError(RuntimeError):
    """Raised when a run is configured to require CUDA but CUDA is unavailable."""


def add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_run_name: str,
    default_output_dir: str = "training_logs",
    default_device: str = "cuda",
    default_require_cuda: bool = False,
) -> None:
    parser.add_argument("--run_name", type=str, default=default_run_name)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument(
        "--torch_device",
        type=str,
        default=default_device,
        choices=["auto", "cpu", "cuda"],
        help="Requested torch device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--require_cuda",
        action=argparse.BooleanOptionalAction,
        default=default_require_cuda,
        help="Abort the run instead of silently falling back to CPU.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest on-disk checkpoint for the same run if it exists.",
    )
    parser.add_argument(
        "--checkpoint_every_epoch",
        type=int,
        default=1,
        help="Save a resumable checkpoint every N epochs.",
    )
    parser.add_argument(
        "--device_check_interval",
        type=int,
        default=200,
        help="When CUDA is required, re-check the runtime device every N train steps.",
    )


def resolve_device(torch_device: str, require_cuda: bool) -> torch.device:
    requested = str(torch_device).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        if require_cuda:
            raise GPURequiredError(
                "CUDA was required, but torch.cuda.is_available() is False. "
                "The run has been stopped before training started."
            )
        requested = "cpu"

    device = torch.device(requested)

    if require_cuda and device.type != "cuda":
        raise GPURequiredError(
            f"CUDA was required, but the resolved device is '{device.type}'."
        )

    return device


def normalize_q_readout_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "z_pairwise" if value else "z_only"
    if value is None:
        return "z_pairwise"

    normalized = str(value).strip().lower()
    legacy_map = {
        "true": "z_pairwise",
        "false": "z_only",
        "1": "z_pairwise",
        "0": "z_only",
    }
    return legacy_map.get(normalized, normalized)


def sanitize_name_component(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip().lower()
    text = text.replace("/", "-").replace("\\", "-").replace(" ", "-")
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9.+-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "none"


def compact_number(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return format(value, ".3g").replace("+", "")
        return str(value).lower()
    return sanitize_name_component(value)


def normalize_name_value(key: str, value: Any) -> str:
    if key == "q_readout_mode":
        value = normalize_q_readout_mode(value)
    alias_map = NAME_VALUE_ALIASES.get(key, {})
    normalized = value if isinstance(value, str) else sanitize_name_component(value)
    normalized = str(normalized)
    if normalized in alias_map:
        return alias_map[normalized]
    if normalized.upper() in alias_map:
        return alias_map[normalized.upper()]
    return sanitize_name_component(normalized)


def _name_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: sanitize_for_json(value)
        for key, value in config.items()
        if key not in RUNTIME_NAME_ARG_KEYS
    }


def _config_from_args(args_or_dict: Any) -> Dict[str, Any]:
    if isinstance(args_or_dict, dict):
        return dict(args_or_dict)
    return dict(vars(args_or_dict))


def build_auto_run_name(args_or_dict: Any, *, task_family: str) -> str:
    config = _config_from_args(args_or_dict)
    payload = _name_payload(config)
    payload["task_family"] = task_family

    hash_suffix = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]

    parts = [
        normalize_name_value("task_family", task_family),
        sanitize_name_component(config.get("dataset")),
        sanitize_name_component(config.get("task")),
    ]

    arch_bits = [
        normalize_name_value("gnn", config.get("gnn")),
        normalize_name_value("intra_aggr", config.get("intra_aggr")),
        sanitize_name_component(config.get("type_fusion")),
        sanitize_name_component(config.get("node_update")),
    ]
    parts.append("a-" + "-".join(arch_bits))

    if "task_type" in config:
        parts.append(f"tt-{normalize_name_value('task_type', config.get('task_type'))}")

    if "lr" in config:
        parts.append(f"lr-{compact_number(config.get('lr'))}")

    model_bits = []
    if "channels" in config:
        model_bits.append(f"c{compact_number(config.get('channels'))}")
    if "num_layers" in config:
        model_bits.append(f"l{compact_number(config.get('num_layers'))}")
    if "batch_size" in config:
        model_bits.append(f"b{compact_number(config.get('batch_size'))}")
    if "num_neighbors" in config:
        model_bits.append(f"n{compact_number(config.get('num_neighbors'))}")
    if model_bits:
        parts.append("s-" + "".join(model_bits))

    extra_bits = []
    if "temporal_strategy" in config:
        extra_bits.append(f"ts{normalize_name_value('temporal_strategy', config.get('temporal_strategy'))}")

    if "neighbor_sampling_mode" in config:
        extra_bits.append(
            f"sm{normalize_name_value('neighbor_sampling_mode', config.get('neighbor_sampling_mode'))}"
        )

    if "eval_epochs_interval" in config:
        extra_bits.append(f"ev{compact_number(config.get('eval_epochs_interval'))}")

    if "share_same_time" in config:
        extra_bits.append(f"sst{compact_number(config.get('share_same_time'))}")

    if "use_shallow" in config:
        extra_bits.append(f"sh{compact_number(config.get('use_shallow'))}")
    if extra_bits:
        parts.append("x-" + "-".join(extra_bits))

    prediction_head_type = normalize_name_value(
        "prediction_head",
        config.get("prediction_head", "classical"),
    )
    prediction_head_bits = [prediction_head_type]
    if config.get("prediction_head_hidden_dim") is not None:
        prediction_head_bits.append(
            f"h{compact_number(config.get('prediction_head_hidden_dim'))}"
        )
    if int(config.get("prediction_head_num_layers", 1)) != 1:
        prediction_head_bits.append(
            f"l{compact_number(config.get('prediction_head_num_layers'))}"
        )
    if float(config.get("prediction_head_dropout", 0.0)) != 0.0:
        prediction_head_bits.append(
            f"d{compact_number(config.get('prediction_head_dropout'))}"
        )
    if prediction_head_bits != ["mlp"]:
        parts.append("head-" + "-".join(prediction_head_bits))

    prediction_head_quantum_types = {"quantum", "residual_quantum"}
    if config.get("prediction_head") in prediction_head_quantum_types:
        head_q_bits = []
        if (
            config.get("prediction_head_n_qubits") is not None
            or config.get("prediction_head_n_q_layers") is not None
        ):
            head_q_bits.append(
                "q"
                f"{compact_number(config.get('prediction_head_n_qubits'))}"
                "x"
                f"{compact_number(config.get('prediction_head_n_q_layers'))}"
            )
        if config.get("prediction_head_n_heads") is not None:
            head_q_bits.append(f"h{compact_number(config.get('prediction_head_n_heads'))}")
        if config.get("prediction_head_q_circuit_type") is not None:
            head_q_bits.append(
                normalize_name_value(
                    "q_circuit_type",
                    config.get("prediction_head_q_circuit_type"),
                )
            )
        if config.get("prediction_head_q_ansatz_type") is not None:
            head_q_bits.append(
                normalize_name_value(
                    "q_ansatz_type",
                    config.get("prediction_head_q_ansatz_type"),
                )
            )
        if config.get("prediction_head_q_readout_mode") is not None:
            head_q_bits.append(
                normalize_name_value(
                    "q_readout_mode",
                    config.get("prediction_head_q_readout_mode"),
                )
            )
        if config.get("prediction_head_q_angle_activation") is not None:
            head_q_bits.append(
                normalize_name_value(
                    "q_angle_activation",
                    config.get("prediction_head_q_angle_activation"),
                )
            )
        if config.get("prediction_head_q_use_angle_affine"):
            head_q_bits.append("aff")
        if float(config.get("prediction_head_q_dropout", 0.0)) != 0.0:
            head_q_bits.append(
                f"qd{compact_number(config.get('prediction_head_q_dropout'))}"
            )
        if config.get("prediction_head") == "residual_quantum":
            head_q_bits.append(
                normalize_name_value(
                    "q_residual_mode",
                    config.get("prediction_head_q_residual_mode", "add"),
                )
            )
        if config.get("prediction_head_freeze_quantum_at_init"):
            head_q_bits.append("fq")
        if head_q_bits:
            parts.append("headq-" + "-".join(head_q_bits))

    parts.append(hash_suffix)
    return "__".join(parts)


def resolve_run_name(
    requested_run_name: str,
    *,
    default_run_name: str,
    auto_run_name: str,
) -> str:
    if requested_run_name and requested_run_name != default_run_name:
        return sanitize_name_component(requested_run_name)
    return auto_run_name


def _iter_tensor_devices(obj: Any):
    if torch.is_tensor(obj):
        yield obj.device
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensor_devices(value)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensor_devices(value)


class RuntimeDeviceGuard:
    def __init__(
        self,
        *,
        device: torch.device,
        require_cuda: bool,
        check_interval: int = 50,
    ) -> None:
        self.device = device
        self.require_cuda = bool(require_cuda)
        self.check_interval = max(int(check_interval), 1)

    def _assert_cuda_available(self) -> None:
        if self.require_cuda and not torch.cuda.is_available():
            raise GPURequiredError(
                "CUDA disappeared during training. The current run has been stopped so "
                "you can reconnect to a GPU and resume later."
            )

    def assert_model_device(self, model: torch.nn.Module) -> None:
        self._assert_cuda_available()
        if not self.require_cuda:
            return

        for name, param in model.named_parameters():
            if param.device.type != self.device.type:
                raise GPURequiredError(
                    f"Parameter '{name}' is on '{param.device.type}' instead of "
                    f"required device '{self.device.type}'."
                )
            break

    def assert_batch_device(self, batch: Any) -> None:
        self._assert_cuda_available()
        if not self.require_cuda:
            return

        for tensor_device in _iter_tensor_devices(batch):
            if tensor_device.type != self.device.type:
                raise GPURequiredError(
                    f"A training tensor is on '{tensor_device.type}' instead of "
                    f"required device '{self.device.type}'."
                )
                # pragma: no cover - structurally unreachable
            break

    def preflight(self, model: torch.nn.Module) -> None:
        self.assert_model_device(model)

    def maybe_check(
        self,
        *,
        model: torch.nn.Module,
        batch: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.require_cuda:
            return

        if step is not None and step > 1 and step % self.check_interval != 0:
            return

        self.assert_model_device(model)
        if batch is not None:
            self.assert_batch_device(batch)


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, indent=2)
    tmp_path.replace(path)


def load_trusted_checkpoint(
    path: Path,
    *,
    map_location: torch.device | str,
) -> Dict[str, Any]:
    # These checkpoints are written by this runtime via torch.save(...), and they
    # intentionally store small Python metadata alongside the state_dicts.
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), which rejects some
    # of that metadata during unpickling, so we opt into full loading here.
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass
class ResumeState:
    start_epoch: int
    best_epoch: int
    best_val_metric: float
    history: Dict[str, Any]


class TrainingRunManager:
    def __init__(
        self,
        *,
        output_dir: str,
        run_name: str,
        args: Any,
        script_path: str,
    ) -> None:
        self.run_name = run_name
        self.args = args
        self.script_path = str(script_path)
        self.run_dir = Path(output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_path = self.run_dir / "last_checkpoint.pt"
        self.best_checkpoint_path = self.run_dir / "best_checkpoint.pt"
        self.history_path = self.run_dir / "history.json"
        self.summary_path = self.run_dir / "summary.json"

    def resume_if_possible(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict[str, Any],
        best_epoch: int,
        best_val_metric: float,
        device: torch.device,
        resume_enabled: bool,
    ) -> ResumeState:
        if not resume_enabled or not self.last_checkpoint_path.exists():
            return ResumeState(
                start_epoch=1,
                best_epoch=best_epoch,
                best_val_metric=best_val_metric,
                history=history,
            )

        checkpoint = load_trusted_checkpoint(
            self.last_checkpoint_path,
            map_location="cpu",
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_to_device(optimizer, device)

        restored_history = checkpoint.get("history", history)
        restored_best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        restored_best_val_metric = float(
            checkpoint.get("best_val_metric", best_val_metric)
        )
        last_completed_epoch = int(checkpoint.get("epoch", 0))

        print(
            f"Resuming run '{self.run_name}' from epoch {last_completed_epoch + 1} "
            f"(last completed epoch: {last_completed_epoch})."
        )

        return ResumeState(
            start_epoch=last_completed_epoch + 1,
            best_epoch=restored_best_epoch,
            best_val_metric=restored_best_val_metric,
            history=restored_history,
        )

    def save_epoch_checkpoint(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict[str, Any],
        best_epoch: int,
        best_val_metric: float,
        status: str = "running",
        message: Optional[str] = None,
    ) -> None:
        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_epoch": int(best_epoch),
                "best_val_metric": float(best_val_metric),
                "status": status,
                "message": message,
                "run_name": self.run_name,
                "script_path": self.script_path,
                "args": sanitize_for_json(vars(self.args)),
            },
            self.last_checkpoint_path,
        )
        atomic_write_json(self.history_path, history)

    def save_best_checkpoint(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        best_val_metric: float,
        val_metrics: Dict[str, Any],
    ) -> None:
        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "best_val_metric": float(best_val_metric),
                "val_metrics": sanitize_for_json(val_metrics),
                "run_name": self.run_name,
            },
            self.best_checkpoint_path,
        )

    def load_best_model(self, model: torch.nn.Module, *, device: torch.device) -> bool:
        if not self.best_checkpoint_path.exists():
            return False

        checkpoint = load_trusted_checkpoint(
            self.best_checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return True

    def mark_status(
        self,
        *,
        status: str,
        epoch: int,
        history: Dict[str, Any],
        best_epoch: int,
        best_val_metric: float,
        message: Optional[str] = None,
        best_val_metrics: Optional[Dict[str, Any]] = None,
        best_test_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        atomic_write_json(
            self.summary_path,
            {
                "status": status,
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_val_metric": float(best_val_metric),
                "best_val_metrics": best_val_metrics or {},
                "best_test_metrics": best_test_metrics or {},
                "message": message,
                "run_name": self.run_name,
                "script_path": self.script_path,
                "run_dir": str(self.run_dir),
                "args": vars(self.args),
            },
        )
        atomic_write_json(self.history_path, history)


def exit_for_gpu_error(exc: GPURequiredError) -> None:
    print(str(exc), file=sys.stderr)
    raise SystemExit(GPU_REQUIRED_EXIT_CODE) from exc
