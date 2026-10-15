"""Internal process entrypoint for a frozen run bundle."""
import json
from pathlib import Path
import sys
import tempfile
import yaml


def main():
    directory, action = Path(sys.argv[1]), sys.argv[2]
    c = yaml.safe_load((directory / "experiment.yaml").read_text())
    if action in ("prepare", "prepare-seeds"):
        from .offline import prepare
        print(json.dumps(prepare(c, seeds_only=action == "prepare-seeds"), indent=2))
        return
    import ray
    import torch
    from .kernels import require_fast_path
    require_fast_path()
    from trinity.utils.plugin_loader import load_plugins
    load_plugins()
    from trinity.common.config import load_config
    from trinity.cli.launcher import run_stage, _require_stage_success
    cfg = load_config(str(directory / "trinity.yaml"))
    cfg.check_and_update()
    required = c["runtime"]["trainer_gpus"] + c["runtime"]["rollout_gpus"]
    if torch.cuda.device_count() != required:
        raise RuntimeError(f"Runner expects {required} visible allocated GPUs, found {torch.cuda.device_count()}")
    torch.manual_seed(c["seed"])
    # Create and stop only this process's local cluster; never ray stop --force.
    with tempfile.TemporaryDirectory(prefix="evorl-ray-", dir="/tmp") as temporary:
        ray.init(address="local", num_cpus=c["runtime"]["cpus"], num_gpus=required,
                 namespace=cfg.ray_namespace, _temp_dir=temporary, include_dashboard=False,
                 runtime_env={"env_vars": cfg.get_envs()})
        try:
            _require_stage_success(run_stage(cfg))
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
