# Configuration

Training recipes are small YAML files under `configs/training/`. They inherit
common model, data, optimization, hardware, and verifier settings from
`configs/defaults/bird.yaml`. Archive settings live in `configs/archive/bird.yaml`.
The method recipe lists only its own objective settings and overrides.

```yaml
extends:
  - ../defaults/bird.yaml
  - ../archive/bird.yaml
method: EliteReplay
training:
  replay_per_iteration: 128
method_options:
  sft_weight: 0.05
  sft_reduction: response_mean
  replay_sampling: mode_cycles
```

Parent paths are relative to the YAML file containing `extends`. Parents merge
in order, followed by the child: mappings merge recursively; scalar values and
lists replace their parents. Cycles and unknown settings are rejected. Data and
output paths inside the resolved configuration are relative to the repository.

To customize a run, create a new file beside the method recipes:

```yaml
extends: elitereplay.yaml
output_root: artifacts/my-run
training:
  learning_rate: 0.000001
method_options:
  sft_weight: 0.05
```

Inspect it before launching:

```bash
elitereplay config --config configs/training/my-run.yaml
elitereplay config --config configs/training/my-run.yaml --plan
elitereplay train --config configs/training/my-run.yaml --check
elitereplay train --config configs/training/my-run.yaml --quick-check
elitereplay train --config configs/training/my-run.yaml
```

| Setting | Meaning |
| --- | --- |
| `training.rollout_iterations` | Logged training iterations and checkpoint counter |
| `training.prompts_per_iteration` | On-policy question groups generated per iteration |
| `training.prompts_per_optimizer_update` | Question groups consumed per optimizer update |
| `training.rollouts_per_prompt` | Samples per on-policy question |
| `training.replay_per_iteration` | Archive/reference responses replayed per iteration |
| `runtime.trainer_gpus` / `rollout_gpus` | GPUs assigned to optimization / generation |
| `output_root` | Parent directory for this run's state and checkpoints |

Optimizer updates per iteration equal prompts per iteration divided by prompts
per optimizer update. The validator checks divisibility and replay allocation
across trainer ranks. Changing a run's inputs after it starts requires a new
output root; checkpoint resume requires the original frozen inputs.

Evaluation configs under `configs/evaluation/` are self-contained and specify
benchmark data, model exports, sampling, and verification. The data-preparation
config is `configs/data/bird.yaml`. Historical training settings are retained
under `configs/historical/`, separately from the public default recipes.
