<div align="center">

# EliteReplay

### Preserving Diverse Solutions in RL Post-Training via Quality-Diversity Search

<a href="https://arxiv.org/abs/ARXIV_ID"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge" alt="Paper on arXiv (coming soon)"></a>
<a href="https://huggingface.co/HF_ORG/EliteReplay-BIRD"><img src="https://img.shields.io/badge/Weights-Hugging_Face-FFD21E?style=for-the-badge" alt="Model weights on Hugging Face (coming soon)"></a>
<a href="https://github.com/skai-research/EliteReplay"><img src="https://img.shields.io/badge/Code-GitHub-24292F?style=for-the-badge" alt="Code on GitHub"></a>
<a href="docs/training.md"><img src="https://img.shields.io/badge/Docs-Getting_Started-2563EB?style=for-the-badge" alt="Getting started"></a>

*Paper and model links are placeholders; both will be updated when available.*

</div>

Official implementation and artifacts for **EliteReplay**.

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

## Installation

Clone the repository and install the Python package:

```bash
git clone https://github.com/skai-research/EliteReplay.git
cd EliteReplay
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,data]'
```

This installs the CPU analysis and data tools. For training and SQL generation, use the GPU environment below on your training machine:

```bash
docker build -f containers/Dockerfile -t elitereplay:runtime .
docker run --gpus all --ipc=host --shm-size=32g -it \
  -v "$PWD":/workspace -w /workspace elitereplay:runtime
```

Inside the container, install the project and its training integration:

```bash
pip install -e '.[test,data]'
python scripts/baselines/setup_runtime.py --install
python scripts/baselines/fetch_upstream.py --clone
python scripts/baselines/setup_kernels.py --install
```

The default training configuration uses **4 × A100 80 GB GPUs**, split between two training GPUs and two rollout GPUs. The environment uses Python 3.12, PyTorch 2.11, CUDA 13.0, and vLLM 0.20.2. See [runtime setup](docs/training.md#1-runtime) for dependency and kernel-build details.

## Training on BIRD

### 1. Prepare the data

Download the base model and question data:

```bash
python scripts/fetch_assets.py --models
```

Place the official BIRD training databases at `external/bird-data/train_databases/<db_id>/<db_id>.sqlite`, then prepare the training split:

```bash
python scripts/baselines/prepare_data.py --config experiments/baselines/data/bird.yaml
```

This prepares 5,000 training questions, a 1,000-question archive subset, and 668 internal validation questions. See [data setup](docs/training.md#2-data-and-base-model) for download and directory details.

### 2. Choose an archive

The [archive builder](src/evorl/EliteReplay/) samples initial SQL candidates, selects target strategies, and mutates verified parents. Each verified candidate enters the cell for the strategy and structure it actually uses. The final export selects examples across questions and occupied cells for replay.

Use the released archive to start training:

```bash
python scripts/import_elite_archive.py --config configs/elitereplay.yaml
```

The released files include the [full archive](elite_archive/map_elites_archive.json), [replay examples](elite_archive/archive.jsonl), and [strategy taxonomy](elite_archive/taxonomy.json). Fresh search uses eight initial samples followed by eight rounds of eight mutations per question. See [archive setup](docs/training.md#3-choose-an-archive) for labeler configuration and export options.

To build your own archive, configure a labeler and choose a separate output root as described in [archive setup](docs/training.md#rebuild-the-archive), then run:

```bash
python scripts/baselines/run.py --config configs/elitereplay.yaml prepare
```

### 3. Launch training

Training uses veRL and Trinity-RFT for policy optimization and supervised replay, with vLLM for rollout generation.

Once the runtime, data, and archive are ready:

```bash
python scripts/baselines/run.py --config configs/elitereplay.yaml smoke
python scripts/baselines/run.py --config configs/elitereplay.yaml train
# Resume an interrupted run:
python scripts/baselines/run.py --config configs/elitereplay.yaml resume
```

Edit [the EliteReplay config](configs/elitereplay.yaml) to set training parameters and output locations. The [objective](src/evorl/baselines/objective.py) combines GRPO with a response-mean supervised replay loss, using a default replay weight of 0.05. [Comparison configs](configs/) use the same runner for the included baselines.

The [runtime integration](vendor/runtime/) includes complete modified Trinity/veRL source files, installed over pinned upstream versions. See the [training guide](docs/training.md) for environment setup, data preparation, checkpointing, and model export.

## Evaluation

### BIRD and Spider

The evaluator generates SQL and checks execution results on BIRD and Spider. Set the dataset and model paths in the [evaluation configs](configs/evaluation/), then run:

```bash
python scripts/evaluation/run.py --config configs/evaluation/bird.yaml prepare
python scripts/evaluation/run.py --config configs/evaluation/bird.yaml --model elitereplay run
```

Use `spider.yaml` for Spider. Generation and verification are resumable; failed and truncated attempts remain in the sample pool. The [evaluation guide](docs/evaluation.md) covers dataset setup, model exports, and rollout formats.

### Diversity metrics

The metric CLI takes scored SQL samples and reports pass@k, U@k, effective diversity, and strategy retention. Try it on the small synthetic example:

```bash
elitereplay-metrics examples/scored_sql.jsonl --k 1 2 4 --bootstrap 0
```

For model outputs, first [convert verified rollouts](docs/evaluation.md#convert-existing-rollout-files) into the shared JSONL format, then run:

```bash
elitereplay-metrics bird_samples.jsonl --output bird_metrics.json
```

The CLI computes all metrics from the same input samples. Descriptor choices and bootstrap settings are configurable; see [metric definitions](docs/evaluation.md#definitions-and-protocols). Correctness labels come from the SQL evaluator, not the metric CLI.

## Documentation and support

- [Training guide](docs/training.md): environment, data, archive construction, training, and checkpoint export.
- [Evaluation guide](docs/evaluation.md): SQL verification, rollout conversion, metrics, and bootstrap intervals.
- [Experiment notes](docs/reproducibility.md): configuration history, archive provenance, and analysis protocols.
- [Validation notes](docs/validation.md): checks performed for this release.

For questions or bugs, please [open an issue](https://github.com/skai-research/EliteReplay/issues). To run the tests locally, use `pytest -q`.

## Acknowledgements

EliteReplay builds on [veRL](https://github.com/verl-project/verl), [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT), and [vLLM](https://github.com/vllm-project/vllm). We thank their contributors and the authors of the baseline implementations included in this project. Source attribution and licenses are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

The paper link and BibTeX entry will be added when the arXiv version is available.

## License

Original EliteReplay code is released under the [MIT License](LICENSE). Included third-party code retains its upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Dataset and model terms are separate.
