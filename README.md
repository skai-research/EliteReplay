# EliteReplay

Code and the frozen elite archive for the BIRD experiments. Benchmark data and model checkpoints are not included.

- `configs/`: training configurations for EliteReplay, IID_Replay, and the comparison methods.
- `data/`: BIRD preprocessing code.
- `elite_archive/`: the verified MAP-Elites archive, replay examples, taxonomy, and source receipt.
- `experiments/`: pinned runtime dependencies and compatibility patches.
- `plugins/`: Trinity integration for archive creation and training.
- `scripts/`: archive import and experiment setup commands.
- `src/evorl/EliteReplay/`: archive search, verification, and export code.
- `src/evorl/baselines/`: training objectives, replay sampling, and runtime adapters.
- `src/evorl/reward/`: BIRD execution checking.
- `src/evorl/analysis/` and `metrics.py`: SQL AST representation and evaluation metrics.
- `tests/`: metric checks.
