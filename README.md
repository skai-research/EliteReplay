<div align="center">

# EliteReplay

### Preserving Diverse Solutions in RL Post-Training via Quality-Diversity Search

<a href="https://arxiv.org/abs/2610.22290"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge" alt="Paper"></a>
<a href="https://huggingface.co/hananour/EliteReplay-BIRD-4B"><img src="https://img.shields.io/badge/Weights-Hugging_Face-FFD21E?style=for-the-badge" alt="Weights"></a>
<a href="https://github.com/skai-research/EliteReplay"><img src="https://img.shields.io/badge/Code-GitHub-24292F?style=for-the-badge" alt="Code"></a>

</div>

EliteReplay uses quality-diversity search to build an archive of verified solutions with different strategies.
During RL post-training, replaying these solutions alongside on-policy updates helps preserve solution diversity.

## What's included

- **Archive creation code:** [search, labeling, verification, and replay selection](src/elitereplay/archive/).
- **Training code:** [EliteReplay training](src/elitereplay/training/) with veRL, Trinity-RFT, and vLLM.
- **Metrics:** [pass@k, U@k, effective diversity, and strategy retention](src/elitereplay/evaluation/).
- **Data:** the paper's [archive and 1,500 replay examples](elite_archive/), [training split](data/bird_split_identity.json), and [dataset preparation scripts](scripts/data/). Source databases are downloaded separately.

## Setup

```bash
git clone https://github.com/skai-research/EliteReplay.git
cd EliteReplay
docker build -f containers/Dockerfile -t elitereplay:runtime .
docker run --gpus all --ipc=host --shm-size=32g -it \
  -v "$PWD":/workspace -w /workspace elitereplay:runtime
# Inside the container:
pip install -e '.[data]'
elitereplay setup
```

The included training recipe uses **4 × A100 80 GB GPUs** (two for training, two for generation) and 32 CPU cores. Setup installs the pinned framework integration and compiles the required kernels.

## Build an archive

Start with tasks, a frozen generator, and a verifier that checks candidate solutions. Search samples initial solutions, mutates verified parents toward target strategies, and keeps verified elites in descriptor cells. The selected elites become the replay set.

Configure the model, data, output directory, and search budget in [the training recipe](configs/training/elitereplay.yaml) and its inherited [archive settings](configs/archive/bird.yaml). Set `offline.labeler_api_base` and `offline.labeler_model` for your OpenAI-compatible labeler endpoint, with `LITELLM_API_KEY` in your environment. Use a new `output_root` for each archive.

```bash
elitereplay archive build --config configs/training/elitereplay.yaml
```

The supplied data and training adapter are for BIRD. For another domain, implement the [DomainAdapter](src/elitereplay/archive/domain.py) for prompts, parsing, verification, and replay formatting; add its strategy descriptors and register it in the data and training pipeline. The search loop remains the same.

**To use the paper's BIRD data:**

```bash
elitereplay data fetch --models
# Place official BIRD training databases at:
# external/bird-data/train_databases/<db_id>/<db_id>.sqlite
elitereplay data prepare
```

Get databases from [BIRD](https://bird-bench.github.io/). Preparation creates 5,000 training questions, a 1,000-question archive subset within that training set, and 668 validation questions. You can then build an archive as above, or use the released archive without a labeler:

```bash
elitereplay archive import
```

## Train with EliteReplay

Once the data and archive are ready:

```bash
elitereplay train --config configs/training/elitereplay.yaml
# Resume an interrupted run:
elitereplay train --config configs/training/elitereplay.yaml --resume
```

The recipe combines on-policy GRPO with supervised replay at μ = 0.05. Edit the YAML for batch size, learning rate, training duration, and replay weight; `elitereplay config` shows the resolved settings. The default recipe runs 100 iterations. The released model's original schedule is recorded separately in [its configuration](configs/historical/elitereplay_schedule.yaml) and [model card](https://huggingface.co/hananour/EliteReplay-BIRD-4B).

To export a complete training checkpoint for inference:

```bash
PYTHONPATH="$PWD/verl_fork:$PWD/src" elitereplay export \
  --actor /path/to/checkpoint/actor --output models/elitereplay-bird
```

## Evaluation and metrics

[Evaluation configs](configs/evaluation/) specify model and dataset paths for BIRD and Spider 1.0. Acquire the official development databases separately. For Spider, import the official archive with `elitereplay data spider --archive /path/to/spider_data.zip`. To use our released weights:

```bash
hf download hananour/EliteReplay-BIRD-4B --local-dir models/elitereplay-bird
elitereplay evaluate --config configs/evaluation/bird.yaml --prepare
elitereplay evaluate --config configs/evaluation/bird.yaml --model base
elitereplay evaluate --config configs/evaluation/bird.yaml --model elitereplay
```

Convert verified rollouts and compute metrics:

```bash
elitereplay rollouts --dataset bird \
  --run base=artifacts/evaluation/bird/official_dev/base/main \
  --run elitereplay=artifacts/evaluation/bird/official_dev/elitereplay/main \
  --output bird_samples.jsonl
elitereplay metrics bird_samples.jsonl --output bird_metrics.json
```

Metrics run on CPU with `pip install -e .`. Input records contain `model`, `question_id`, `sample_id`, `correct`, and `sql`; keep failed attempts. U@k counts expected correct strategies, effective diversity measures correctness-weighted pairwise difference, and retention measures coverage of the base model's correct strategies. Defaults use `coarsened_v2` for U@k/retention and `bound_alias_ast` for effective diversity; these choices are recorded in the output. Use `elitereplay metrics --help` for options.

## Citation

```bibtex
@misc{moussa2026elitereplaypreservingdiversesolutions,
  title={EliteReplay: Preserving Diverse Solutions in RL Post-Training via Quality-Diversity Search},
  author={Hanane Nour Moussa and Doeun Lee and Sachin Kumar},
  year={2026},
  eprint={2610.22290},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.22290},
}
```

## License

Original code uses the [MIT License](LICENSE). Framework and evaluation dependencies retain their [upstream licenses](THIRD_PARTY_NOTICES.md); dataset and model terms are separate.
