#!/usr/bin/env python3
"""Run one baseline/domain using only its editable YAML hyperparameters."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from elitereplay.training.config import DPH_METHODS, load, plan, run_dir, offline_dir, trinity_config
from elitereplay.training.runtime import (resolved, preflight, bundle, launch, reconcile,
                                    refresh_failed_runtime, require_completed_smoke,
                                    migrate_performance_runtime, migrate_batch_timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("action", choices=("plan", "check", "prepare", "prepare-seeds", "smoke", "train", "resume", "refresh-runtime", "migrate-runtime", "migrate-recovery-runtime", "migrate-batch-timeout", "stress-runtime"))
    args = parser.parse_args()
    c = resolved(load(args.config))
    if args.action == "prepare-seeds" and c["method"] != "EliteReplay":
        raise ValueError("prepare-seeds requires a EliteReplay configuration")
    if c["runtime"].get("validation_only") and args.action in ("prepare", "prepare-seeds", "train", "resume", "migrate-runtime", "migrate-recovery-runtime", "migrate-batch-timeout"):
        raise ValueError("validation_only configurations support diagnostics, not preparation or full training")
    if args.action == "plan":
        print(json.dumps(plan(c), indent=2))
        return
    report = preflight(c, require_offline=args.action not in ("prepare", "prepare-seeds", "refresh-runtime"))
    print(json.dumps(report, indent=2), flush=True)
    if not report["ready"]:
        raise SystemExit(2)
    if args.action == "check":
        return
    root = run_dir(c)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a+") as lock:
        # Different methods have independent locks; the same run cannot race.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "refresh-runtime":
            print(json.dumps(refresh_failed_runtime(c), indent=2), flush=True)
            return
        if args.action == "migrate-runtime":
            print(json.dumps(migrate_performance_runtime(c), indent=2), flush=True)
            return
        if args.action == "migrate-recovery-runtime":
            print(json.dumps(migrate_performance_runtime(c, from_contract=5), indent=2), flush=True)
            return
        if args.action == "migrate-batch-timeout":
            print(json.dumps(migrate_batch_timeout(c), indent=2), flush=True)
            return
        offline, smoke = args.action in ("prepare", "prepare-seeds"), args.action == "smoke"
        if offline and c["method"] not in ("EliteReplay", "IID_Replay", *DPH_METHODS):
            print("No offline preparation required for this method.")
            return
        directory = bundle(c, args.config, smoke=smoke, offline=offline)
        if args.action == "stress-runtime":
            launch(c, directory, stress=True)
            return
        if not offline and not smoke:
            require_completed_smoke(c, directory)
        trinity = yaml.safe_load((directory / "trinity.yaml").read_text())
        job = Path(trinity["checkpoint_root_dir"]) / trinity["project"] / trinity["group"] / trinity["name"]
        if args.action == "resume" or (smoke and job.exists()):
            result = reconcile(c, trinity)
            print(json.dumps(result, indent=2), flush=True)
            if result["status"] == "complete":
                return
        elif not offline and job.exists() and any(job.iterdir()):
            raise FileExistsError(f"Run already has state at {job}; use resume or a new output_root")
        if offline:
            # Different coefficient runs may share one immutable archive.
            # Serialize its single writer across their independent run locks.
            shared = offline_dir(c)
            shared.mkdir(parents=True, exist_ok=True)
            with (shared / ".prepare.lock").open("a+") as prepare_lock:
                fcntl.flock(prepare_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                launch(c, directory, offline=True, seeds_only=args.action == "prepare-seeds")
        else:
            launch(c, directory, offline=False)


if __name__ == "__main__":
    main()
