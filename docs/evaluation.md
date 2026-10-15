# BIRD/Spider evaluation and diversity

## Generate and verify SQL

Use the training container or an equivalent pinned vLLM environment. Evaluation is direct sampling, without archive lookup, reranking, or deduplication. Each question receives 128 independently seeded samples at temperature 1 and top-p .95. Prompt/response limits are 16,384/4,096 tokens. Failures and truncations stay in the denominator. The default evaluator uses one GPU and 16 bounded CPU verification workers; adjust GPU memory and concurrency to the device.

Acquire official BIRD dev (20240627, 1,534 questions) and Spider **1.0** dev (1,034 questions) separately. For BIRD, place the question JSON at `data/evaluation/v1/bird/official_dev.json` and databases at `external/bird-data/dev_20240627/dev_databases/<db>/<db>.sqlite`. Use the pinned evaluator fetched by `elitereplay data fetch`.

For Spider, this implementation imports the exact DPH-RL question order and official databases. After `elitereplay setup --skip-kernels --references`, download the official archive and run:

```bash
elitereplay data spider --archive /path/to/spider_data.zip
```

The importer verifies the expected archive SHA256 `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`, database contents, and DPH source hash. Its source includes the original public download URL. Do not substitute a different dataset version when reproducing saved results.

Edit model paths in `configs/evaluation/bird.yaml` and `spider.yaml` to complete HF exports. A config-only `actor/huggingface` folder is not an exported model. Each config contains `base` and `elitereplay`; add other methods with the same structure. Evaluation preparation requires the frozen training split for overlap checks.

```bash
elitereplay evaluate --config configs/evaluation/bird.yaml --prepare
elitereplay evaluate --config configs/evaluation/bird.yaml --model base
elitereplay evaluate --config configs/evaluation/bird.yaml --model elitereplay
# Substitute spider.yaml for Spider.
```

Repeating the evaluation command resumes complete generation/score blocks. It checks configuration, model, source, and dataset identity before reusing results. Hashing model weights can take time. `--summarize` reports only a complete split. BIRD uses the pinned result-set execution semantics; Spider uses the pinned DPH comparator. These checks establish agreement on the given database, not semantic equivalence on all databases. The resource-bounded SQLite workers are not a general-purpose Python sandbox.

## Convert existing rollout files

```bash
elitereplay rollouts --dataset bird \
  --run base=artifacts/evaluation/bird/official_dev/base/main \
  --run elitereplay=artifacts/evaluation/bird/official_dev/elitereplay/main \
  --output bird_samples.jsonl
elitereplay metrics bird_samples.jsonl --output bird_metrics.json
```

The converter also accepts the original experiment directories after restoration. It checks generated/scored content hashes, all sample IDs, equal coverage, and failure accounting. It refuses overwriting output. Evaluate BIRD and Spider separately. JSONL records have `model`, `question_id`, `sample_id`, `correct` (Boolean), and `sql` (string, including empty/invalid failures); `dataset` is optional. Supplied correctness labels must come from execution checking, not the metric script.

## Definitions and protocols

Let n be the sample count, c the number of correct samples, and c_s the correct samples in strategy class s. Each reported result is a uniform mean over questions, including unsolved questions.

- **pass@k:** `1 - C(n-c,k)/C(n,k)`.
- **U@k:** sum over s of `1 - C(n-c_s,k)/C(n,k)`. This is the expected number of correct strategies in a random size-k subset, not prefix coverage.
- **Effective diversity:** the mean of `r_i r_j (1-sim(i,j))` over distinct sample pairs. With binary strategy identity it is `[c(c-1) - sum_s c_s(c_s-1)]/[n(n-1)]`, the probability of two correct, different strategies sampled without replacement.
- **Retention:** for each question where the base model has at least two correct strategies, the fraction of its strategy set also found by the trained model; then average equally across eligible questions. No eligible questions yields `null`, not zero.

The default CLI reproduces the **saved analysis protocols**: `--descriptor coarsened_v2` for U@k/retention, and `--effective-descriptor bound_alias_ast` for effective diversity. The former coarsens joins, aggregation, subqueries, windows, set operators, top-k, and anti-joins. The latter normalizes relation aliases with scope-aware bindings while preserving literal, expression, and column distinctions. This distinction is intentional and recorded in output metadata. Saved retention eligibility is 825 BIRD / 562 Spider questions under coarsened_v2.

Available alternatives:

```bash
# Use the same alias-aware AST identity for all three metrics:
elitereplay metrics bird_samples.jsonl --descriptor bound_alias_ast \
  --effective-descriptor bound_alias_ast --output bird_ast_metrics.json
# Reproduce the anonymous code package's height-four multiset/Dice protocol:
elitereplay metrics bird_samples.jsonl --descriptor height4_dice \
  --effective-descriptor height4_dice --output bird_dice_metrics.json
```

`height4_dice` gives continuous correctness-weighted dissimilarity, not a binary different-strategy event. Different descriptors can produce different U@k, retention, and effective-diversity values. They must be labeled, not substituted silently. None of these structural identities proves semantic equivalence or measures an unobserved reasoning trace.

Output probabilities are in [0,1]; multiply by 100 to display percentages. U@k is an expected count. Default intervals use 4,000 question bootstrap resamples with seed 20261001, paired across methods. They condition on saved runs, do not account for training-seed or database-cluster variation, and are marginal intervals. Use `--bootstrap 0` for fast deterministic point estimates.

`results/reference_metrics.json` contains values from the saved analysis exports with source hashes. It is a reference for provenance, not a substitute for regenerating scores from raw model outputs. Exact regeneration of all paper figures has not been certified; see reproducibility notes.

The saved Spider GRPO reference values span two rollout cohorts; see the explicit
source records and `docs/reproducibility.md`. Use one chosen cohort per model
when computing a new comparison.

## Quick GPU check

After preparing the full benchmark and exporting a checkpoint, check generation
and verification before starting the full evaluation:

```bash
elitereplay evaluate --config configs/evaluation/bird.yaml --model elitereplay --smoke
```

This evaluates two questions with two samples each, writes to a separate
`smoke/` directory, and marks the result as diagnostic. Repeating the command
resumes its saved attempts. It cannot fill or overwrite the full `main/` run.
