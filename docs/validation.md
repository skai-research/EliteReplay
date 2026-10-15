# Release validation (2026-10-15)

Completed locally before publication:

- Fresh clones of both pinned runtime repositories, complete-file overlay
  installation, and final source-hash verification.
- Fresh fetch/verification of all five public baseline reference repositories.
- 96 tests passed in the recorded Python 3.12 / PyTorch 2.11 / CUDA 13 container
  (CPU execution). These cover archive verification/insertion and request
  recovery, replay tokenization, numerical objectives and microbatch/distributed
  gradient invariance, descriptor behavior, pass@k accounting, interrupted
  evaluation recovery, tampered score/seed rejection, and complete model shards.
- The additional Git installer fixture passed on the host (97 tests covered
  across environments). It is skipped in the recorded container, which lacks
  Git; the new Docker recipe explicitly installs Git and compilation tools.
  This fixture checks refusal to overwrite local edits and repeat verification.
- The actual training plugin imported and registered in the pinned runtime.
  All six public training configs parsed with Trinity's structured config loader.
- Rebuilt the actual BIRD 5,000/1,000/668 split in temporary storage from the
  pinned filtered data and original databases. All ordered record hashes matched.
  The fixed 1,500-example archive imported successfully; repeat import was safe.
- The package built as a source distribution and wheel. The installed metric CLI
  ran the synthetic example with bootstrap enabled.
- Public files were checked for cluster-specific paths, credential patterns,
  oversized artifacts, and whitespace errors. No weights or credentials included.

The new release has not run a fresh full four-GPU training/evaluation campaign.
The new Docker recipe has not been rebuilt locally; runtime tests used the
recorded working Apptainer image plus the documented supplementary dependencies.
Kernel compilation and the required two-iteration training smoke check must be
performed in the deployment environment before a production run. These CPU
checks do not establish GPU memory fit on another device or reproduce paper
accuracy by themselves.

The saved-result spot check uses three questions each from BIRD and Spider and
six methods (128 samples each). Coarsened strategy counts agreed for all 36
model-question groups. Effective-diversity values also agreed for all 36 groups
when checked against their own recorded source cohorts. A Spider GRPO cohort mismatch was found between the
historical U@k/retention and ED source records and is documented in
`reproducibility.md`; it is not hidden by changing the scorer. This spot check
is not a full rerun of the 1,534/1,034-question benchmark analyses.

The package rename was checked from a clean export of the Git index: 96 tests
passed in the recorded container, with the Git fixture passing separately on
the host. The source distribution and wheel built successfully. An isolated
wheel install ran the metric CLI and both SQL verifier subprocesses on correct
and incorrect queries. All six training configurations loaded with the renamed
Trinity registrations. The clean export includes every runtime overlay file;
the model-weights ignore rule applies only to the top-level model directory.
