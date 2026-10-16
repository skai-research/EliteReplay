# Archive creation and BIRD training

Run commands from a clone of this repository. Use a writable checkout and separate output roots for different experiments. The scripts store resolved configuration, input hashes, source snapshots, and checkpoint state. They refuse incompatible cache reuse.

## 1. Runtime

The recorded environment uses Python 3.12, PyTorch 2.11/CUDA 13.0, Transformers 5.8.0, vLLM 0.20.2, and Ray 2.53.0. This is a deliberately pinned integration; installing the latest Trinity package can select incompatible versions. The container recipe pins the public Linux amd64 vLLM 0.20.2 image manifest.

```bash
docker build -f containers/Dockerfile -t elitereplay:runtime .
docker run --gpus all --ipc=host --shm-size=32g -it \
  -v "$PWD":/workspace -w /workspace elitereplay:runtime
# Remaining commands execute inside the container:
pip install -e '.[data]'
elitereplay setup
elitereplay doctor --training
```

`elitereplay setup` checks pinned upstream revisions and copies complete modified source files from `vendor/runtime/`. It rejects unrelated local edits and verifies file hashes. veRL and Trinity are loaded from these source trees by the runner; do not `pip install -U` either framework over them.

The pinned Qwen3.5 kernels are installed into `external/qwen35-kernels/` with a content manifest. Build and verify them on a machine with the matching CUDA toolkit and a visible NVIDIA GPU; the installer defaults to four compilation workers. The trainer refuses the slow fallback. The `TransferQueue==0.1.7` metadata has a known NumPy<2 constraint, while this tested runtime uses NumPy 2.2.6. It is explicitly installed with `--no-deps` in the container, matching the recorded environment.

Training needs four visible A100 80 GB GPUs (2 trainer + 2 rollout), approximately 32 available CPU cores for the supplied configuration, and storage for the base model, databases, rollout journals, and checkpoints. The original runs used Apptainer; native mode inside the Docker container uses the same Python launch path. The Docker build and CPU integration checks pass; a fresh GPU quick check is still required on the deployment host. See [validation notes](validation.md).

## 2. Data and base model

```bash
elitereplay data fetch --models
# Add --embedding if running SetPO.
```

This fetches the exact filtered BIRD question release, pinned evaluator, and Qwen3.5-4B-Base snapshot. Acquire the BIRD **training** databases from [BIRD's official release](https://bird-bench.github.io/) under its terms, and place the SQLite files at:

```text
external/bird-data/train_databases/<db_id>/<db_id>.sqlite
```

The question file is downloaded separately to `data/raw/bird/`. Do not substitute development questions or change database schemas. The training preparation uses 5,000 questions, a 1,000-question archive subset **within those 5,000**, and a separate 668-question internal validation set from held-out training databases. No 1,000-question subtraction from the RL pool is performed.

```bash
elitereplay data prepare
```

`data/bird_split_identity.json` contains the ordered IDs and content hashes of the paper split. Archive import checks these semantic records rather than depending on byte-identical Parquet serialization. Preparation refuses an already populated output directory.

## 3. Choose an archive

### Use the released archive

```bash
elitereplay archive import
```

This verifies archive/taxonomy hashes, counts, and exact split membership, then stages an immutable replay set. The data can be re-serialized, but the question content and ordering must match. No labeler credentials or fresh generation are needed for this path.

### Rebuild the archive

Use a new `output_root` in a copy of `configs/training/elitereplay.yaml`; do not import the released archive there. Set `offline.labeler_api_base` to your HTTPS OpenAI-compatible endpoint and `offline.labeler_model` to the available Qwen3-Coder-30B-A3B-Instruct deployment. Export `LITELLM_API_KEY` securely in your environment; never add credentials to YAML or Git. Different hosted deployments can change search outputs even with the same taxonomy protocol.

```bash
elitereplay config --plan
elitereplay archive build
```

Search uses eight initial samples and eight rounds of eight mutation children for each of 1,000 questions: 72,000 requests. The frozen base model generates SQL; the labeler supplies the approach taxonomy and labels. Structural descriptors are deterministic. Targets prioritize missing approaches with attempt balancing; parent selection prefers a different primary approach when possible. Candidates are verified and inserted into the cell they actually realize. A candidate replaces an occupied cell's elite only when the archive's quality/tie-break rule improves it (among eligible SQL, shorter output wins). Search and export maintain append-only journals and resume completed requests.

The export selects 1,500 verified examples with question/mode coverage. New search need not recover the exact published archive; its hashes and receipts identify the actual output. See the shipped archive for exact reproduction of replay inputs.

## 4. Train and resume

```bash
elitereplay train --check
elitereplay train --quick-check
elitereplay train
# After interruption:
elitereplay train --resume
```

The two-iteration quick check has separate state and must complete with the same configuration/source identity before production training. It does not count toward the full run. Training stops naturally at the configured `rollout_iterations`. The `elitereplay config --plan` command prints optimizer-update counts separately. Set `output_root` and `runtime.max_checkpoints_to_keep` to suit available storage before starting.

The default recipe uses 128 questions × 8 on-policy responses, 128 replay responses, LR 1e-6, five warmup iterations, policy clipping [0.8,1.28], temperature 1, top-p .95, prompt/response limits 16,384/4,096, and response-mean replay weight .05. GRPO and EliteReplay do not add a generic base-model KL penalty. DPH-F's cached forward-KL term is separate.

All comparison configs use the same prepared data. GRPO/MaxRL/SetPO need no offline replay preparation. DPH-F and IID replay use `elitereplay train --config <method.yaml> --prepare`; IID samples independently and does not read the elite archive. SetPO also requires the embedding snapshot. Consult each YAML for objective-specific parameters.

## 5. Export weights for evaluation

Select an explicit complete checkpoint; the `actor/huggingface` directory can contain tokenizer/config files without weights. Merge the FSDP shards first:

```bash
PYTHONPATH="$PWD/verl_fork:$PWD/src" elitereplay export \
  --actor /path/to/global_step_100/actor --output models/elitereplay-bird
```

Use the actual checkpoint suffix, not a renamed counter. Export hashes source shards and records provenance. This command creates a local HF-format model; it does not upload it. No model upload is part of this code release.

## Python package and run names

The Python package is `elitereplay`, and new training runs use the
`elitereplay-baselines` checkpoint project directory. Environment overrides for
the evaluation repository root use `ELITEREPLAY_REPO_ROOT`.

When upgrading an existing installation, reinstall with `pip install -e .`.
Use a fresh runtime checkout and output root after changing package or runtime
versions; existing frozen run bundles should continue with their original code.

## Configuration and optional reference code

Method recipes live in `configs/training/` and inherit shared defaults. Read
[configuration](configuration.md) before changing batch sizes or the GPU layout.
`elitereplay config --config <file>` prints the full resolved YAML; every run
also saves its resolved configuration with its source snapshot.

`elitereplay setup --dry-run` prints installation commands without changing files.
`--skip-kernels` installs only the framework sources. `--references` additionally
fetches the baseline authors' reference repositories. These reference clones
are optional for training; Spider data import uses the DPH reference checkout.

The CLI wraps the task runners in `scripts/`; advanced diagnostics remain
available through `scripts/training/run.py --help`. CPU metric use does not
require framework clones, CUDA, or model weights.
