## QGNN Training Notes

This repository currently has three training entrypoints that matter for the maintained prediction-head workflow:

- `run_prediction_head.py`
  Main launcher for shared-backbone plus candidate-head experiments.
- `quantum_model/trainer/gnn_entity_2stage.py`
  Direct two-stage trainer for entity tasks.
- `quantum_model/trainer/gnn_autocomplete_2stage.py`
  Direct two-stage trainer for autocomplete tasks.


### Backbone Arguments

These are the current maintained graph-backbone arguments for the two-stage trainers.

| Argument | Choices / type | Launcher default | Trainer default |
| --- | --- | --- | --- |
| `--gnn` | `staged`, `hgt`, `graphormer` | `staged` | `staged` |
| `--intra_aggr` | `mean`, `max`, `mean_project`, `max_project`, `lstm`, `gatv2` | `max_project` | `mean` |
| `--type_fusion` | `sum`, `mean`, `weighted_sum`, `relation_weighted_sum`, `edge_type_weighted_sum`, `gru` | `sum` | `sum` |
| `--node_update` | `mlp`, `gru` | `mlp` | `mlp` |
| `--num_layers` | integer | `1` | `2` |
| `--num_neighbors` | integer | `128` | `128` |
| `--neighbor_sampling_mode` | `total`, `per_edge_type` | `total` | `total` |
| `--gat_heads` | integer | not exposed by `run_prediction_head.py` | `4` |
| `--gat_dropout` | float | not exposed by `run_prediction_head.py` | `0.0` |
| `--temporal_strategy` | forwarded to PyG `NeighborLoader` | `uniform` | `uniform` |
| `--max_steps_per_epoch` | integer | `2000` | `2000` |
| `--num_workers` | integer | `0` | `0` |

### Shared Runtime And Optimization Arguments

These are used by the maintained two-stage trainers.

| Argument | Choices / type | `gnn_entity_2stage.py` default | `gnn_autocomplete_2stage.py` default |
| --- | --- | --- | --- |
| `--dataset` | string | `rel-hm` | `rel-f1` |
| `--task` | string | `item-sales` | `results-position` |
| `--lr` | float | `0.002` | `0.005` |
| `--epochs` | integer | `10` | `10` |
| `--batch_size` | integer | `512` | `512` |
| `--channels` | integer | `128` | `128` |
| `--seed` | integer | `42` | `42` |
| `--download` / `--no-download` | boolean | `True` | `True` |
| `--cache_dir` | path | `~/.cache/relbench_examples` | `~/.cache/relbench_examples` |
| `--run_name` | string | `relbench_run` | `relbench_autocomplete` |
| `--output_dir` | path | `training_logs` | `training_logs` |
| `--torch_device` | `auto`, `cpu`, `cuda` | `cuda` | `cuda` |
| `--require_cuda` / `--no-require_cuda` | boolean | `False` | `False` |
| `--resume` / `--no-resume` | boolean | `True` | `True` |
| `--checkpoint_every_epoch` | integer | `1` | `1` |
| `--device_check_interval` | integer | `200` | `200` |

Regression-only controls:

| Argument | Choices / type | Default |
| --- | --- | --- |
| `--regression_tune_metric` | `r2`, `mae`, `rmse` | `r2` |
| `--early_stopping_patience` | integer or `None` | `None` |
| `--early_stopping_min_delta` | float | `0.0` |

| `--freeze_alpha_with_quantum` / `--no-freeze_alpha_with_quantum` | boolean | `True` |
| `--quantum_unfreeze_epoch` | integer or `None` | `None` |
| `--quantum_only_finetune` / `--no-quantum_only_finetune` | boolean | `False` |

### Prediction-Head Arguments In Direct Trainers

The direct trainer interface uses `prediction_head_*` names.

| Argument | Choices / type | Direct-trainer default | Wiring note |
| --- | --- | --- | --- |
| `--prediction_head` | `classical`, `quantum`, `residual_quantum` | `classical` | selects the head builder |
| `--prediction_head_hidden_dim` | integer or `None` | `None` | classical head width |
| `--prediction_head_num_layers` | integer | `1` | classical head depth |
| `--prediction_head_dropout` | float | `0.0` | classical head dropout |
| `--prediction_head_n_qubits` | integer or `None` | `None` | `None` inherits backbone `n_qubits` |
| `--prediction_head_n_q_layers` | integer or `None` | `None` | `None` inherits backbone `n_q_layers` |
| `--prediction_head_n_heads` | integer or `None` | `None` | `None` inherits backbone `n_heads` |
| `--prediction_head_q_readout_mode` | `z_pairwise`, `z_only`, `z_all`, `probs` | `None` | `None` inherits backbone readout |
| `--prediction_head_q_circuit_type` | `angle`, `rxry`, `arctan`, `amplitude` | `None` | `None` inherits backbone encoding |
| `--prediction_head_q_ansatz_type` | `rot_cnot_ring`, `rot_cnot_chain`, `rot_cz_ring`, `rot_none` | `None` | `None` inherits backbone ansatz |
| `--prediction_head_q_angle_activation` | `tanh`, `atan`, `none` or `None` | `None` | `None` inherits backbone activation |
| `--prediction_head_q_use_angle_affine` / `--no-prediction_head_q_use_angle_affine` | boolean or `None` | `None` | `None` inherits backbone affine setting |
| `--Q_Pred_Head_Dropout` / `--prediction_head_q_dropout` | float | `0.0` | dropout on quantum readout features before the prediction-head post-net |
| `--prediction_head_q_residual_mode` | `add`, `concat` | `add` | residual-quantum heads only |
| `--prediction_head_q_alpha_init` | float | `0.0` | residual-quantum heads only |
| `--prediction_head_freeze_quantum_at_init` / `--no-prediction_head_freeze_quantum_at_init` | boolean or `None` | `None` | `None` inherits backbone freeze setting |
| `--prediction_head_freeze_alpha_with_quantum` / `--no-prediction_head_freeze_alpha_with_quantum` | boolean or `None` | `None` | `None` inherits backbone alpha-freeze setting |
| `--prediction_head_export_circuit` / `--no-prediction_head_export_circuit` | boolean | `False` | export helper artifacts when supported |

### Backbone Reuse Arguments In Direct Trainers

These matter when a run should reuse a shared backbone checkpoint.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--init_backbone_checkpoint` | path or `None` | `None` | initialize from an existing checkpoint |
| `--load_backbone_only` / `--no-load_backbone_only` | boolean | `False` | load backbone weights but ignore old head weights |
| `--freeze_backbone` / `--no-freeze_backbone` | boolean | `False` | train only the head |
| `--reset_prediction_head` / `--no-reset_prediction_head` | boolean | `False` | reinitialize the head after loading |

### `run_prediction_head.py`

`run_prediction_head.py` is the maintained batch launcher for prediction-head experiments. Its CLI defaults come from `EXPERIMENT_GRID`.

#### Current launcher defaults

| Argument | Default |
| --- | --- |
| `--channel` | `prediction_head` |
| `--dataset` | `rel-hm` |
| `--task` | `transactions-price` |
| `--trainer_script` | `quantum_model/trainer/gnn_autocomplete_2stage.py` |
| `--output_dir` | `training_logs/prediction_head` |
| `--seed` | `42` |
| `--seeds` | `42 43 44` |
| `--lr` | `0.001` |
| `--epochs` | `60` |
| `--batch_size` | `512` |
| `--channels` | `128` |
| `--gnn` | `staged` |
| `--intra_aggr` | `max_project` |
| `--type_fusion` | `sum` |
| `--node_update` | `mlp` |
| `--num_layers` | `1` |
| `--num_neighbors` | `128` |
| `--neighbor_sampling_mode` | `total` |
| `--temporal_strategy` | `uniform` |
| `--max_steps_per_epoch` | `2000` |
| `--num_workers` | `0` |
| `--two_stage_head_search` | `True` |
| `--head_training_mode` | `head_only` |
| `--backbone_epochs` | `100` |
| `--head_epochs` | `50` |
| `--backbone_hidden_dim` | `128` |
| `--backbone_head_layers` | `2` |
| `--prediction_heads` | `mlp quantum` |
| `--standard_mlp_hidden_dim` | `128` |
| `--standard_mlp_num_layers` | `2` |
| `--classical_head_hidden_dims` | `128` |
| `--classical_head_depths` | `2 4 8` |
| `--classical_head_dropouts` | `0.0 0.1 0.2` |
| `--prediction_head_qubits` | `5 10` |
| `--prediction_head_encodings` | `angle rxry arctan amplitude` |
| `--prediction_head_readouts` | `z_only z_pairwise z_all probs` |
| `--prediction_head_q_ansatz_types` | `rot_cnot_ring` |
| `--prediction_head_n_q_layers` | `2` |
| `--prediction_head_q_angle_activation` | `tanh` |
| `--prediction_head_q_use_angle_affine` | `False` |
| `--Q_Pred_Head_Dropout` | `0.0` |
| `--prediction_head_q_residual_mode` | `add` |
| `--classical_num_layers` | `4` |
| `--max_classical_hidden_dim` | `512` |
| `--matched_classical_head_depths` | `2` |
| `--matched_classical_head_dropouts` | `0.0` |
| `--matched_param_tolerance_rel` | `0.05` |
| `--matched_param_tolerance_abs` | `100` |
| `--max_quantum_input_dim` | `4096` |
| `--max_readout_dim` | `4096` |
| `--max_prediction_head_params` | `150000` |
| `--resume` | `False` |
| `--resume_backbone` | `True` |
| `--resume_heads` | `False` |
| `--skip_completed` | `True` |
| `--force_retrain_backbone` | `False` |
| `--extend_late_best` | `False` |
| `--extend_best_epoch_min` | `90` |
| `--extend_extra_epochs` | `50` |
| `--extend_roles` | `base_mlp matched_mlp qhead` |
| `--cache_dir` | `~/.cache/relbench_examples` |
| `--regression_tune_metric` | `r2` |

Launcher limitation:

- `run_prediction_head.py` forwards `gnn`, `intra_aggr`, `type_fusion`, `node_update`, `num_layers`, `num_neighbors`, `neighbor_sampling_mode`, `temporal_strategy`, and general training/runtime arguments.
- It does not currently expose direct-trainer backbone knobs such as `gat_heads` or `gat_dropout`.
- It does expose prediction-head sweep controls such as `prediction_head_qubits`, `prediction_head_encodings`, `prediction_head_readouts`, and related `prediction_head_*` options.

#### High-level launcher arguments

| Argument | Choices / type | Notes |
| --- | --- | --- |
| `--prediction_heads` | accepted families and aliases | canonical families are `mlp`, `standard_mlp`, `matched_mlp`, `quantum`, `residual_quantum`; `mlp` expands to both classical baselines |
| `--head_training_mode` | `head_only`, `finetune`, `scratch`, `end_to_end` | `end_to_end` is a backward-compatible alias for `finetune` |
| `--resume_backbone` | boolean | resume shared-backbone pretraining run |
| `--resume_heads` | boolean | resume generated head runs |
| `--skip_completed` | boolean | skip runs whose summary already marks completion |
| `--force_retrain_backbone` | boolean | ignore an existing shared backbone and retrain it |
| `--execute` | flag | without this flag, the launcher only writes the plan |

#### Sweep controls

| Argument | Choices / type | Notes |
| --- | --- | --- |
| `--prediction_head_qubits` | one or more integers | quantum / residual-quantum sweep |
| `--prediction_head_encodings` | one or more of `angle`, `rxry`, `arctan`, `amplitude` | quantum / residual-quantum sweep |
| `--prediction_head_readouts` | one or more of `z_pairwise`, `z_only`, `z_all`, `probs` | quantum / residual-quantum sweep |
| `--prediction_head_q_ansatz_types` | one or more of `rot_cnot_ring`, `rot_cnot_chain`, `rot_cz_ring`, `rot_none` | quantum / residual-quantum sweep |
| `--prediction_head_n_q_layers` | integer | shared quantum-head depth |
| `--prediction_head_q_angle_activation` | `tanh`, `atan`, `none` | shared quantum-head activation |
| `--prediction_head_q_use_angle_affine` | boolean | shared quantum-head affine preprocessing |
| `--Q_Pred_Head_Dropout` | one or more floats | quantum / residual-quantum dropout sweep |
| `--prediction_head_q_residual_mode` | `add`, `concat` | residual-quantum only |
| `--classical_head_hidden_dims` | one or more integers | fixed classical baseline sweep |
| `--classical_head_depths` | one or more integers, each >= 2 | fixed classical baseline sweep |
| `--classical_head_dropouts` | one or more floats | fixed classical baseline sweep |
| `--matched_classical_head_depths` | one or more integers, each >= 2 | parameter-matched classical sweep |
| `--matched_classical_head_dropouts` | one or more floats | parameter-matched classical sweep |
| `--matched_param_tolerance_rel` | float >= 0 | relative parameter-count tolerance for reusing a matched MLP |
| `--matched_param_tolerance_abs` | integer >= 0 | absolute minimum parameter-count tolerance for reusing a matched MLP |
| `--max_classical_hidden_dim` | integer | search ceiling for parameter-matched baselines |
| `--max_quantum_input_dim` | integer | feasibility guard |
| `--max_readout_dim` | integer | feasibility guard |
| `--max_prediction_head_params` | integer | skip oversized heads; use `<=0` to disable |

The default `matched_mlp` is a compact capacity-control baseline: it uses a two-layer MLP with dropout fixed at `0.0`, then reuses one matched MLP for quantum heads whose target parameter counts are within `max(abs_tolerance, rel_tolerance * target_params)`.

#### Late-best extension

| Argument | Choices / type | Notes |
| --- | --- | --- |
| `--extend_late_best` | boolean | continue only runs that already completed and peaked near the old budget |
| `--extend_best_epoch_min` | integer | minimum old `best_epoch` required for extension |
| `--extend_extra_epochs` | integer | extra epochs to append |
| `--extend_roles` | one or more of `base_mlp`, `matched_mlp`, `qhead`, `rqhead` | which role families are eligible |

### Task-Specific Arguments

#### `gnn_entity_2stage.py`

| Argument | Choices / type | Default |
| --- | --- | --- |
| `--include_task_tables` | `all`, `current_only`, `none` | `none` |

#### `gnn_autocomplete_2stage.py`

| Argument | Choices / type | Default |
| --- | --- | --- |
| `--task_type` | `BINARY_CLASSIFICATION`, `REGRESSION`, `MULTILABEL_CLASSIFICATION` | `REGRESSION` |

### Example Commands

Prediction-head dry-run:

```bash
python run_prediction_head.py \
  --dataset rel-f1 \
  --task driver-position \
  --trainer_script quantum_model/trainer/gnn_entity_2stage.py \
  --gnn staged \
  --intra_aggr max_project \
  --type_fusion sum \
  --node_update mlp \
  --prediction_heads mlp quantum
```

Prediction-head execution with one-step Graphormer backbone:

```bash
python run_prediction_head.py \
  --execute \
  --dataset rel-f1 \
  --task driver-position \
  --trainer_script quantum_model/trainer/gnn_entity_2stage.py \
  --gnn graphormer \
  --head_training_mode head_only \
  --prediction_heads standard_mlp matched_mlp quantum \
  --prediction_head_qubits 5 10 \
  --prediction_head_encodings angle arctan amplitude \
  --prediction_head_readouts z_pairwise probs
```

Direct two-stage trainer call with a quantum prediction head:

```bash
python quantum_model/trainer/gnn_entity_2stage.py \
  --dataset rel-f1 \
  --task driver-position \
  --gnn staged \
  --intra_aggr max_project \
  --type_fusion sum \
  --node_update mlp \
  --prediction_head quantum \
  --prediction_head_n_qubits 5 \
  --prediction_head_n_q_layers 2 \
  --prediction_head_q_circuit_type angle \
  --prediction_head_q_ansatz_type rot_cnot_ring \
  --prediction_head_q_readout_mode z_pairwise
```


This is the single-experiment building block for the next step, where multiple experiment dashboards can be compared across tasks.

For a live local task browser that can load experiment folders directly from the page, open:

`analysis_dashboards/task-browser.html`

That entrypoint does not require a prebuilt `dashboard-data.js`. Use `Add task`, choose one experiment directory, and the page will parse `run_summary.csv`, optional `run_summary_agg.csv`, and `json_files/*.json` locally in the browser.
