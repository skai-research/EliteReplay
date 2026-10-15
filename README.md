# EliteReplay

Code for *EliteReplay: Preserving Diverse Solutions in RL Post-Training via Quality-Diversity Search*.

EliteReplay builds a quality-diversity archive of verified SQL solutions with a frozen base model, then trains with GRPO and a supervised replay loss over that fixed archive.

This repository contains archive creation, BIRD training, BIRD/Spider execution evaluation, and the U@k, effective-diversity, and strategy-retention scorers. It includes the released archive: **2,708 occupied cells across 780 questions**, and **1,500 replay examples**. Model weights will be published separately; they are not included here.

## Quick start: evaluation metrics on CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,data]'
pytest -q
elitereplay-metrics examples/scored_sql.jsonl --k 1 2 4 --bootstrap 0
```

The example uses synthetic SQL and supplied correctness labels to demonstrate the metric interface. It does not run a database or represent benchmark results.

For real results, retain all generated attempts, including failures, and run the SQL execution evaluator first. The default metric protocol uses coarsened structural descriptors for U@k/retention and alias-normalized AST identity for effective diversity. These are separate from the archive's labeler-induced search taxonomy. See [evaluation](docs/evaluation.md) for definitions, protocol options, bootstrap intervals, and rollout conversion.

## Archive creation and training

Start with [the training guide](docs/training.md). It covers:

1. Building the pinned GPU environment and installing Trinity/veRL source overlays.
2. Downloading the base model and preparing the exact 5,000-question BIRD split.
3. Importing the supplied archive or building it with eight seeds and eight rounds of eight mutations per question.
4. Running the required training smoke check, full training, checkpoint resume, and HF-format export.

Training uses **four A100 80 GB GPUs**, split into two trainer GPUs and two rollout GPUs. The standalone SQL evaluator defaults to one GPU. No cluster scheduler or project account is required by the public commands.

The core supervised objective is implemented directly in [`objective.py`](src/evorl/baselines/objective.py):

\[
L=(1-\mu)L_{\mathrm{GRPO}}+\mu L_{\mathrm{SFT}},\qquad\mu=0.05.
\]

The SFT term averages negative log likelihood within each response, then over replay responses. Replay cycles across questions and occupied modes. [`integration.py`](src/evorl/baselines/integration.py) registers the objective and sampler with Trinity-RFT. The complete modified upstream files are in [`vendor/runtime`](vendor/runtime), with licenses, pinned revisions, and hashes. Installation does not require applying diffs or rewriting source at launch.

## Reproduction status

The default `configs/*.yaml` encode the manuscript's **100-iteration recipe**. The historical EliteReplay run used a 200-iteration configured schedule, two optimizer updates per iteration, and was evaluated at checkpoint 110. These counters must not be conflated. The historical configuration is provided separately, and the differences are described in [reproducibility notes](docs/reproducibility.md).

This release adds runnable tooling and validation to the original submission package. It does **not** claim a new full GPU rerun or exact regeneration of every numerical claim in the manuscript. The [validation record](docs/validation.md) distinguishes completed checks from work that requires GPU allocations and benchmark assets.

## Layout

| Path | Purpose |
| --- | --- |
| `src/evorl/EliteReplay/` | Search, descriptors, verified archive, mutation providers, export |
| `src/evorl/baselines/` | Training objectives, replay sampler, data contracts, resume |
| `plugins/elitereplay_trinity.py` | SQL rollout workflow and Trinity integration |
| `src/evorl/evaluation/` | SQL generation, correctness verification, diversity metrics |
| `configs/` | EliteReplay, IID replay, GRPO, MaxRL, SetPO, DPH-F recipes |
| `elite_archive/` | Fixed archive, replay export, taxonomy, provenance |
| `vendor/runtime/` | Complete modified Trinity and veRL files |
| `results/reference_metrics.json` | Saved analysis reference values and source hashes |

The internal Python namespace `evorl` is retained to keep saved runtime contracts compatible. The method is named **EliteReplay** in the public interface. Historical `qd_chord` identifiers in provenance refer to EliteReplay.

Code is released under [Apache-2.0](LICENSE), subject to the original notices for third-party components. See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Dataset and model licenses remain separate.
