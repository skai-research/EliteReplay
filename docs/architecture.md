# Code organization

The installed `elitereplay` CLI is the public entry point. Heavy GPU imports are
lazy, so help, configuration inspection, and SQL metrics work on CPU machines.
`python -m elitereplay` exposes the same commands.

| Package | Responsibility |
| --- | --- |
| `elitereplay.archive` | Candidate generation, taxonomy, descriptors, verified archive insertion, replay export |
| `elitereplay.training` | Configuration, split preparation, replay sampling, source snapshots, launch and recovery |
| `elitereplay.training.objectives` | GRPO, MaxRL, SetPO, DPH, and combined RL/replay losses |
| `elitereplay.integrations` | Trinity registrations and workflow, veRL helper code, hosted labeler client |
| `elitereplay.evaluation` | Resumable rollout generation, correctness checks, metrics, and model export validation |
| `elitereplay.evaluation.descriptors` | Structural and AST representations used by evaluation |
| `elitereplay.evaluation.verifiers` | BIRD execution worker and reward handling |
| `elitereplay.data` | BIRD schema extraction and prompt preparation shared by training and evaluation |
| `elitereplay.cli` | Command routing, environment checks, and setup orchestration |

## Runtime integration

`vendor/runtime/` contains complete modified upstream Trinity/veRL files.
`runtime/locks/` records exact upstream revisions, file hashes, kernel downloads,
and optional baseline references. `elitereplay setup` installs the two framework
checkouts and kernel packages. It verifies existing files and rejects unrelated
local modifications. Training does not depend on cloning baseline references.

A training run copies the required Python code and framework sources into a
frozen bundle. The plugin entry imports `elitereplay.integrations.trinity`, which
registers the SQL workflow, replay sampler, and combined policy loss. Numerical
objectives remain separate from this framework adapter.

## Extending the project

- **Change training settings:** add a recipe under `configs/training/` using
  `extends`; inspect it with `elitereplay config`.
- **Add an objective:** implement it in `training/objectives/`, then wire its
  options and estimator into the config compiler and Trinity integration.
- **Change archive search:** start with `archive/pipeline.py`, `archive/archive.py`,
  and `archive/providers.py`. Preserve verification and export contracts.
- **Add a metric or SQL representation:** use `evaluation/diversity.py` and
  `evaluation/descriptors/`; keep the representation identified in output.

`scripts/` contains task runners grouped under `setup/`, `data/`, `archive/`,
`training/`, and `evaluation/`. Their numerical implementation lives in the
package. Training commands need a source checkout for configs, lock files, and
runtime overlays. A wheel installation supports standalone metrics.
