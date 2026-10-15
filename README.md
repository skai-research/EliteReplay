# EliteReplay

**[Paper (arXiv)](https://arxiv.org/abs/ARXIV_ID) · [Model (Hugging Face)](https://huggingface.co/HF_ORG/EliteReplay-BIRD) · [Training guide](docs/training.md) · [Evaluation guide](docs/evaluation.md)**

*The paper and model links are placeholders and will be updated when available.*

Official code and artifacts for **EliteReplay: Preserving Diverse Solutions in RL Post-Training via Quality-Diversity Search**.

EliteReplay preserves diverse solutions during reinforcement learning by combining GRPO with supervised replay from a quality-diversity archive. A frozen base model builds the archive through strategy-guided search, and execution verification keeps correct SQL solutions for replay.

This project provides the archive builder, the released BIRD archive, EliteReplay and baseline training, BIRD/Spider evaluation, and tools for measuring solution diversity.

## What's included

| Component | What it provides | Where to start |
| --- | --- | --- |
| **Archive builder** | Strategy labeling, mutation, SQL verification, MAP-Elites insertion, and replay-set selection | [Search code](src/evorl/EliteReplay/) · [Build an archive](docs/training.md#3-choose-an-archive) |
| **Released BIRD archive** | 2,708 occupied cells across 780 questions, with 1,500 selected replay examples | [Archive files](elite_archive/) · [Import instructions](docs/training.md#use-the-released-archive) |
| **Training** | EliteReplay, IID replay, GRPO, MaxRL, SetPO, and DPH-F objectives and configurations | [Training code](src/evorl/baselines/) · [Configs](configs/) · [Guide](docs/training.md) |
| **SQL evaluation** | Resumable generation and execution checking on BIRD and Spider | [Evaluation code](src/evorl/evaluation/) · [Guide](docs/evaluation.md#generate-and-verify-sql) |
| **Diversity metrics** | U@k, effective diversity, strategy retention, pass@k, and bootstrap intervals | [Metric implementation](src/evorl/evaluation/diversity.py) · [Definitions](docs/evaluation.md#definitions-and-protocols) |

## Get started

Clone the repository and install the Python package:

```bash
git clone https://github.com/skai-research/EliteReplay.git
cd EliteReplay
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,data]'
```

This installation supports CPU analysis and data tools. GPU training and generation use the environment described in the [training guide](docs/training.md#1-runtime).

### Build or use an archive

The [archive builder](src/evorl/EliteReplay/) samples initial SQL candidates, selects target strategies, and mutates verified parents. Each verified candidate enters the cell for the strategy and structure it actually uses. The final export selects examples across questions and occupied cells for replay.

After [setting up the runtime and BIRD data](docs/training.md), choose one of these paths:

```bash
# Use the released archive.
python scripts/import_elite_archive.py --config configs/elitereplay.yaml

# Or build a new archive with a configured labeler and a separate output root.
python scripts/baselines/run.py --config configs/elitereplay.yaml prepare
```

The released files include the [full archive](elite_archive/map_elites_archive.json), [replay examples](elite_archive/archive.jsonl), and [strategy taxonomy](elite_archive/taxonomy.json). Fresh search uses eight initial samples followed by eight rounds of eight mutations per question. See [archive setup](docs/training.md#3-choose-an-archive) for labeler configuration and export options.

### Train with EliteReplay

Training uses veRL and Trinity-RFT for policy optimization and supervised replay, with vLLM for rollout generation. The default hardware layout uses four A100 80 GB GPUs: two for training and two for generation.

Once the runtime, data, and archive are ready:

```bash
python scripts/baselines/run.py --config configs/elitereplay.yaml smoke
python scripts/baselines/run.py --config configs/elitereplay.yaml train
# Resume an interrupted run:
python scripts/baselines/run.py --config configs/elitereplay.yaml resume
```

Edit [the EliteReplay config](configs/elitereplay.yaml) to set training parameters and output locations. The [objective](src/evorl/baselines/objective.py) combines GRPO with a response-mean supervised replay loss, using a default replay weight of 0.05. [Comparison configs](configs/) use the same runner for the included baselines.

The [runtime integration](vendor/runtime/) includes complete modified Trinity/veRL source files, installed over pinned upstream versions. See the [training guide](docs/training.md) for environment setup, data preparation, checkpointing, and model export.

### Evaluate SQL models

The evaluator generates SQL and checks execution results on BIRD and Spider. Set the dataset and model paths in the [evaluation configs](configs/evaluation/), then run:

```bash
python scripts/evaluation/run.py --config configs/evaluation/bird.yaml prepare
python scripts/evaluation/run.py --config configs/evaluation/bird.yaml --model elitereplay run
```

Use `spider.yaml` for Spider. Generation and verification are resumable; failed and truncated attempts remain in the sample pool. The [evaluation guide](docs/evaluation.md) covers dataset setup, model exports, and rollout formats.

### Measure solution diversity

The metric CLI takes scored SQL samples and reports pass@k, U@k, effective diversity, and strategy retention. Try it on the small synthetic example:

```bash
elitereplay-metrics examples/scored_sql.jsonl --k 1 2 4 --bootstrap 0
```

For model outputs, first [convert verified rollouts](docs/evaluation.md#convert-existing-rollout-files) into the shared JSONL format, then run:

```bash
elitereplay-metrics bird_samples.jsonl --output bird_metrics.json
```

The CLI computes all metrics from the same input samples. Descriptor choices and bootstrap settings are configurable; see [metric definitions](docs/evaluation.md#definitions-and-protocols). Correctness labels come from the SQL evaluator, not the metric CLI.

## Development and experiment notes

Run the tests manually with `pytest -q`. The internal Python namespace is `evorl`; historical `qd_chord` identifiers refer to EliteReplay.

[Experiment notes](docs/reproducibility.md) document configuration history, archive provenance, and saved analysis protocols. [Validation notes](docs/validation.md) describe the checks performed for this release. Model weights will be published separately on Hugging Face.

## License

Original EliteReplay code is released under the [MIT License](LICENSE). Included third-party code retains its upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Dataset and model terms are separate.
