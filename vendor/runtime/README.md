# Complete runtime overlays

These are complete modified upstream files, not independent replacement packages.
`scripts/baselines/setup_runtime.py --install` clones the exact commits listed in
`experiments/baselines/runtime/runtime.lock.json`, verifies original hashes, and
copies these files into the checkouts. Run without `--install` to verify them.
Local edits and changed upstream revisions are rejected.

- Trinity-RFT: https://github.com/agentscope-ai/Trinity-RFT, commit `9a7e26e7d71b843d19e70344a2fe3b47efe2f418`.
- veRL: https://github.com/verl-project/verl, commit `7aed6b230776f963fa09509c10d9c3a767d1102c`.

Original licenses are included under each overlay. The files are modified by
EliteReplay contributors; upstream copyright notices remain intact. The lock
records both upstream and released file hashes.

The modifications combine the experiment's compatibility fixes and its former
launch-time transformations: Qwen3.5/vLLM compatibility, mixed-source loss fields,
optimizer minibatch control, raw log probabilities and truncation propagation,
reference-policy handling, memory-bounded logits, complete rollout-batch checks,
and error propagation. The actual EliteReplay objective and replay sampler live
in `src/evorl/baselines`; the overlays support their framework integration.

The run bundle copies these already reviewed sources. It does not use
`git apply` or text substitutions at training launch. This keeps the upstream
provenance explicit without redistributing entire framework repositories.
