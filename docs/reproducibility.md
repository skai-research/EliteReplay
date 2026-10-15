# Experiment notes

This release keeps the experiment's archive, objective implementations, verifier
contracts, and pinned framework integration available as source. It corrects
portability problems in the anonymous package without relabeling historical
checkpoints or pretending that a new recipe was the run used for a figure.

## Training counters

`configs/training/elitereplay.yaml` and the comparison recipes implement the manuscript's
100 logged-iteration recipe. With `prompts_per_optimizer_update: 128`, there is
one optimizer update per logged iteration. This is an explicit reproduction
recipe, not a description of every historical checkpoint.

The preserved EliteReplay training bundle has `rollout_iterations: 200`,
`prompts_per_optimizer_update: 64`, and 128 prompts per iteration. The evaluated
checkpoint is 110, corresponding to 220 optimizer updates. Its sanitized
configuration is `configs/historical/elitereplay_schedule.yaml`. It retains the
original schedule; changing it to 110 or 100 would change schedule semantics.
The labeler endpoint and filesystem paths are portable placeholders. Inspect
actual checkpoint provenance when comparing methods; do not assume every
baseline followed the identical history. In particular, both an earlier DPH-F
run and a later fixed-batch continuation exist in the experiment records.

The paper's statement of 100 steps is therefore not an exact historical run
counter. The release does not claim that running the new 100-iteration recipe
must yield the paper's numbers. Weights are intentionally deferred to a later
model release, where the selected checkpoint needs its own provenance.

## Archive and splits

The bundled archive is the same normalized/anonymized copy distributed with the
submission: 2,708 cells, 780 represented questions, and 1,500 selected replay
responses. Its source and release hashes are retained. Search uses 1,000 of the
5,000 RL training questions; they are not a disjoint withheld set. Seven training
databases provide the 668-question internal validation split.

The labeler induces approach labels, not every structural feature. The replay
set is frozen before training. Rebuilding search invokes hosted labeling and
GPU sampling and is not guaranteed to reproduce the exact archived bytes.
Importing the supplied archive is the reproducible fixed-input route. Data
import checks ordered record hashes, allowing equivalent Parquet serialization
while rejecting changed prompts, gold metadata, or question selection.

## Metrics

The anonymous package supplied a shared height-four AST multiset/Dice scorer.
The saved full analysis exports instead use `structural_coarsened_v2` for U@k
and retention, and `sqlite_bound_alias_ast_v1` for the binary effective-diversity
analysis. Both are available and explicitly identified in metric outputs.
These differ from the labeler-defined archive cells. Continuous Dice ED and
binary identity ED are distinct estimands and must not share an unlabeled number.

`results/reference_metrics.json` preserves saved aggregate analysis values and
source hashes. The reference retention panel has 825 BIRD and 562 Spider
questions. Some earlier paper/source audits found mismatches between manuscript
figures and older analysis exports; this code release does not certify every
figure by selecting whichever descriptor matches a desired result. Run the
scorer on the identified rollouts and report its protocol/version with results.

## What is and is not packaged

Included: archive construction and replay training, baseline objectives,
BIRD/Spider generation and execution checking, metric aggregation and bootstrap,
fixed archive, split identities, exact runtime overlays, and setup/test scripts.
Benchmark databases, full raw rollouts, model weights, hosted credentials,
cluster-specific launch controllers, and other-domain experiments are not part
of this Git repository. The experiment artifacts are preserved separately.

Use an editable clone for training: the runners deliberately refer to repository
configs, vendor overlays, and source snapshots. The installed wheel is sufficient
for metric calculation, not the entire training asset directory.

## Recorded cohort discrepancy

A release-time spot check found that the saved Spider GRPO U@k/retention export
uses the `spider_pass128_20260929` step-110 run, whereas the effective-diversity
export uses the collaborator's `evaluation/v1/spider/official_dev/grpo` run.
Both were labeled `grpo_s110` in analysis exports; that label is not sufficient
provenance. Their exact source records are included beside the reference values.
Do not combine those references as if they came from a single sampled pool.
The new CLI computes all metrics from the same input JSONL, with its hash.
