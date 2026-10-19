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
## Building an archive

Start with subset of tasks from your train set, a frozen generator (e.g. the base model), and a verifier that checks candidate solutions. Search samples initial solutions, mutates verified parents toward target strategies, and keeps verified elites in descriptor cells. The selected elites become the replay set.

Configure the model, data, output directory, and search budget in [the training recipe](configs/training/elitereplay.yaml) and its inherited [archive settings](configs/archive/bird.yaml). Set `offline.labeler_api_base` and `offline.labeler_model` for your labeler model.

```bash
elitereplay archive build --config configs/training/elitereplay.yaml
```

## Training with EliteReplay

Once the data and archive are ready, train with EliteReplay, which combines GRPO with supervised replay:

```bash
elitereplay train --config configs/training/elitereplay.yaml
```

## Evaluation and metrics

Compute **U@k, effective diversity, strategy retention, and pass@k** from verified outputs in any domain. Provide a JSONL file with `model`, `question_id`, `sample_id`, `correct`, and `strategy` fields ([example](examples/scored_samples.jsonl)). Use your domain's verifier for correctness and a consistent strategy taxonomy across models. Keep failed attempts; their strategy can be `null`.

```bash
elitereplay metrics samples.jsonl --descriptor precomputed --base-model base \
  --k 1 2 4 8 16 32 64 128 --output metrics.json
```

Include the base model for retention, with the same tasks and sample counts for each model. Choose `k` values no larger than your samples per task. This mode measures diversity by distinct strategy labels. Metrics run on CPU; use `elitereplay metrics --help` for options.

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

Original code uses the [MIT License](LICENSE). Framework and evaluation dependencies retain their [own licenses](THIRD_PARTY_NOTICES.md). 
