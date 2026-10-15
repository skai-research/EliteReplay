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
| **Archive builder** | Strategy labeling, mutation, SQL verification, MAP-Elites insertion, and replay-set selection | [Search code](src/elitereplay/archive/) · [Build an archive](docs/training.md#3-choose-an-archive) |
| **Released BIRD archive** | 2,708 occupied cells across 780 questions, with 1,500 selected replay examples | [Archive files](elite_archive/) · [Import instructions](docs/training.md#use-the-released-archive) |
| **Training** | EliteReplay, IID replay, GRPO, MaxRL, SetPO, and DPH-F objectives and configurations | [Training code](src/elitereplay/training/) · [Configs](configs/) · [Guide](docs/training.md) |
| **SQL evaluation** | Resumable generation and execution checking on BIRD and Spider | [Evaluation code](src/elitereplay/evaluation/) · [Guide](docs/evaluation.md#generate-and-verify-sql) |
| **Diversity metrics** | U@k, effective diversity, strategy retention, pass@k, and bootstrap intervals | [Metric implementation](src/elitereplay/evaluation/diversity.py) · [Definitions](docs/evaluation.md#definitions-and-protocols) |

## Installation

```bash
git clone https://github.com/skai-research/EliteReplay.git
cd EliteReplay
```

**For training and SQL generation**, build the GPU environment and open a shell in it:

```bash
docker build -f containers/Dockerfile -t elitereplay:runtime .
docker run --gpus all --ipc=host --shm-size=32g -it \
  -v "$PWD":/workspace -w /workspace elitereplay:runtime
```

Inside the container:

```bash
pip install -e '.[data]'
elitereplay setup
elitereplay doctor --training
```

`setup` installs the pinned Trinity/veRL integration and compiles the training kernels. `doctor` reports missing dependencies and checks the GPU allocation. The default training configuration uses **4 × A100 80 GB GPUs**: two for training and two for rollout generation. See the [training guide](docs/training.md) for environment details.

**For CPU metrics only**, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
elitereplay doctor
elitereplay metrics examples/scored_sql.jsonl --k 1 2 4 --bootstrap 0
```

## Training on BIRD

### 1. Prepare the data

```bash
elitereplay data fetch --models
```

Download the official BIRD training databases and place them at `external/bird-data/train_databases/<db_id>/<db_id>.sqlite`, then run:

```bash
elitereplay data prepare
```

This prepares 5,000 training questions, a 1,000-question archive subset, and 668 internal validation questions. [Data setup details →](docs/training.md#2-data-and-base-model)

### 2. Choose an archive

Use the released archive:

```bash
elitereplay archive import
```

Or configure a labeler and a separate output root to build your own:

```bash
elitereplay archive build --config configs/training/elitereplay.yaml
```

The archive builder samples SQL, mutates verified parents toward target strategies, and selects verified examples across questions and occupied cells. The release includes the [full archive](elite_archive/map_elites_archive.json), [1,500 replay examples](elite_archive/archive.jsonl), and [strategy taxonomy](elite_archive/taxonomy.json). [Archive setup details →](docs/training.md#3-choose-an-archive)

### 3. Launch training

```bash
elitereplay train --check
elitereplay train --smoke
elitereplay train
# Resume after an interruption:
elitereplay train --resume
```

The default is [EliteReplay](configs/training/elitereplay.yaml). Select another method with `--config`, for example:

```bash
elitereplay train --config configs/training/grpo.yaml --smoke
elitereplay train --config configs/training/grpo.yaml
```

Each [method config](configs/training/) contains its own settings and inherits [shared BIRD defaults](configs/defaults/bird.yaml). Inspect the full configuration or generation budget with:

```bash
elitereplay config
elitereplay config --plan
```

[Configuration guide →](docs/configuration.md)

## Evaluation

### BIRD and Spider

Set the dataset and model paths in the [evaluation configs](configs/evaluation/), then run:

```bash
elitereplay evaluate --config configs/evaluation/bird.yaml --prepare
elitereplay evaluate --config configs/evaluation/bird.yaml --model elitereplay --smoke
elitereplay evaluate --config configs/evaluation/bird.yaml --model elitereplay
```

Use `spider.yaml` for Spider. Evaluation resumes completed generation and verification blocks. [Dataset and model setup →](docs/evaluation.md#generate-and-verify-sql)

### Diversity metrics

Convert verified rollouts to the scored JSONL format, then compute pass@k, U@k, effective diversity, and strategy retention:

```bash
elitereplay rollouts --dataset bird \
  --run base=artifacts/evaluation/bird/official_dev/base/main \
  --run elitereplay=artifacts/evaluation/bird/official_dev/elitereplay/main \
  --output bird_samples.jsonl
elitereplay metrics bird_samples.jsonl --output bird_metrics.json
```

All metrics use the same input sample pool. Descriptor choices and bootstrap settings are configurable. [Metric definitions →](docs/evaluation.md#definitions-and-protocols)

## Code organization

```text
src/elitereplay/
  archive/                 # Search, strategy labels, mutations, archive export
  training/                # Data splits, replay, run state, and checkpoint recovery
    objectives/            # EliteReplay and baseline loss implementations
  evaluation/              # Generation, execution checking, and metrics
    descriptors/           # SQL strategy and AST representations
    verifiers/             # Isolated BIRD SQL execution
  integrations/            # Trinity workflow, veRL helpers, labeler client
  data/                    # Shared BIRD schema and prompt preparation
  cli/                     # Public commands, setup, and environment checks
configs/                   # Shared defaults, per-method recipes, and evaluation
runtime/locks/             # Framework, kernel, and reference-code versions
vendor/runtime/            # Complete modified upstream framework files
elite_archive/             # Released archive, replay examples, and taxonomy
```

The CLI is the main entry point; `scripts/` holds the underlying task runners. See the [architecture guide](docs/architecture.md) for extension points and the [training guide](docs/training.md) for checkpoint export. Use `elitereplay --help` to list commands.

For questions or bugs, please [open an issue](https://github.com/skai-research/EliteReplay/issues). Tests run manually with `pip install -e '.[test]'` and `pytest -q`. [Experiment notes](docs/reproducibility.md) and [validation notes](docs/validation.md) record configuration history and verification scope.

## Acknowledgements

EliteReplay builds on [veRL](https://github.com/verl-project/verl), [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT), and [vLLM](https://github.com/vllm-project/vllm). We thank their contributors and the authors of the baseline implementations included in this project. Source attribution and licenses are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

The paper link and BibTeX entry will be added when the arXiv version is available.

## License

Original EliteReplay code is released under the [MIT License](LICENSE). Included third-party code retains its upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Dataset and model terms are separate.
