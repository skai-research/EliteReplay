# Release validation (2026-10-15)

The organized release was checked from a clean export of the Git index.

- Built the Docker recipe from the public, digest-pinned Linux amd64 vLLM
  0.20.2 image. Confirmed Python 3.12, PyTorch 2.11/CUDA 13.0 and the pinned
  dependencies, without the research environment's supplementary directories.
- Installed fresh pinned Trinity and veRL checkouts through `elitereplay setup`
  and verified every complete-file overlay hash. Repeated setup verifies
  existing files without overwriting unrelated local edits.
- **105 tests passed inside the fresh container**, including archive insertion
  and recovery, replay tokenization, numerical objectives, gradient invariance,
  descriptors, metrics, resumable evaluation, checkpoint-shard validation,
  config inheritance, and command dispatch. The CPU-only environment passed
  95 tests with the PyTorch-dependent module skipped.
- Imported the actual Trinity integration and loaded all six training recipes
  through Trinity's structured configuration loader. The refactor preserves
  the previous recipes' model, data, training settings and active objective
  parameters.
- Rebuilt the actual BIRD 5,000/1,000/668 splits in temporary storage from the
  pinned questions and original databases. All ordered record hashes matched.
  Imported the 1,500-example released archive and verified a repeat import.
- Built the source distribution and wheel. An isolated wheel installation ran
  `elitereplay doctor` and the metric CLI with bootstrap enabled outside the
  checkout, without framework or GPU dependencies. Both SQL execution workers
  correctly accepted and rejected test queries from that wheel installation.
- Checked tracked files for old project names, cluster-specific paths, broken
  documentation links, oversized artifacts, and whitespace errors. Runtime
  overlays and their third-party licenses are included in the source release.

The pinned causal-conv1d extension also compiled successfully from source in
the fresh image, and both kernel packages imported. The final Transformers
fast-path check requires a visible CUDA GPU; CPU-only verification did not pass
and no successful kernel receipt was written. The installer now checks for a
visible GPU before spending time compiling.

The container retains the recorded `TransferQueue==0.1.7` / NumPy 2.2.6
combination. `pip check` reports TransferQueue's NumPy<2 metadata constraint;
that package is explicitly installed with `--no-deps`, as explained in
[training.md](training.md). Other dependency constraints passed.

A fresh four-GPU training/checkpoint/export/evaluation smoke run has **not**
completed for this refactor. Run `elitereplay train --smoke` on the deployment
GPU allocation before production training; the runner enforces completion.
The tests above do not establish GPU memory fit or reproduce paper accuracy.

Earlier saved-result checks compared three questions each from BIRD and Spider
across six methods (128 samples each). Coarsened strategy counts agreed for all
36 model-question groups. Effective diversity agreed for all groups when
checked against their own recorded source cohorts. The Spider GRPO cohort
mismatch between historical U@k/retention and ED records is documented in
[reproducibility.md](reproducibility.md). This was not a full benchmark rerun.
