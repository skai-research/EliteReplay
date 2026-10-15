"""Reviewable run bundles, source pins, input checks and checkpoint recovery."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from .config import DPH_METHODS, ROOT, REPLAY_METHODS, path, run_dir, offline_dir, fingerprint, offline_fingerprint, trinity_config
from .data import sha256, validate_splits

TRINITY_COMMIT = "9a7e26e7d71b843d19e70344a2fe3b47efe2f418"
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
RUNTIME_CONTRACT = 6


def resolved(c):
    c = copy.deepcopy(c)
    for section, keys in {"model": ["path"], "data": ["root"], "runtime": ["image"],
                          "method_options": ["embedding_model"],
                          "verifier": ["bird_database_root", "bird_evaluator_root", "taco_image", "apptainer_image", "gurobipy_path", "highspy_path"]}.items():
        for key in keys:
            if key in c[section]:
                c[section][key] = str(path(c[section][key]))
    c["output_root"] = str(path(c["output_root"]))
    if "source_dir" in c["offline"]:
        c["offline"]["source_dir"] = str(path(c["offline"]["source_dir"]))
    if "seed_source_config" in c["offline"]:
        c["offline"]["seed_source_config"] = str(path(c["offline"]["seed_source_config"]))
    if "iid_taxonomy_path" in c["offline"]:
        c["offline"]["iid_taxonomy_path"] = str(path(c["offline"]["iid_taxonomy_path"]))
    if "kernel_directory" in c["runtime"]:
        c["runtime"]["kernel_directory"] = str(path(c["runtime"]["kernel_directory"]))
    return c


def upstream_check():
    spec = importlib.util.spec_from_file_location("baseline_upstream_check", ROOT / "scripts/baselines/fetch_upstream.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.check()


def runtime_check():
    spec = importlib.util.spec_from_file_location("baseline_runtime_check", ROOT / "scripts/baselines/setup_runtime.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.check()


def validate_optmath_archive(c, root, complete, count):
    """Check the frozen actual export size against its immutable generation ceiling."""
    effective_path = Path(root) / "effective_archive.json"
    effective = json.loads(effective_path.read_text())
    if (sha256(effective_path) != complete.get("effective_archive_sha256")
            or effective.get("effective_examples") != count
            or complete.get("effective_examples") != count
            or effective.get("archive_sha256") != complete["archive_sha256"]
            or effective.get("generation_config_sha256") != offline_fingerprint(c)
            or not c["offline"]["minimum_examples"] <= count <= c["offline"]["export_target"]):
        raise ValueError("OptMATH frozen effective archive identity/count mismatch")
    return effective_path


def inputs(c, *, require_offline=True):
    """Validate the prepared split contract; return content hashes for freezing."""
    import pyarrow.parquet as pq
    data = path(c["data"]["root"])
    from .shared_data import materialize_domain
    materialize_domain(data)
    manifest = json.loads((data / "split_manifest.json").read_text())
    if manifest.get("validation_only") and not c["runtime"].get("validation_only"):
        raise ValueError("Diagnostic data requires runtime.validation_only=true")
    if manifest["domain"] != c["domain"] or not manifest["archive_is_training_subset"]:
        raise ValueError("Wrong data domain/split protocol")
    files = {str(data / "split_manifest.json"): sha256(data / "split_manifest.json")}
    if c["domain"] == "optmath":
        if manifest.get("training_ready") is not True:
            raise ValueError("OptMATH release has not passed reference validation and freeze")
        for key in ("gurobipy_path", "highspy_path"):
            dependency_root = path(c["verifier"][key])
            dependencies = [f for f in dependency_root.rglob("*")
                            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"]
            if not dependencies:
                raise ValueError(f"Empty OptMATH solver dependency tree: {key}")
            files.update({str(f): sha256(f) for f in sorted(dependencies)})
        solver_image = path(c["verifier"]["apptainer_image"])
        files[str(solver_image)] = sha256(solver_image)
    if "seed_source_config" in c["offline"]:
        source = path(c["offline"]["seed_source_config"])
        files[str(source)] = sha256(source)
    if "iid_taxonomy_path" in c["offline"]:
        taxonomy = path(c["offline"]["iid_taxonomy_path"])
        files[str(taxonomy)] = sha256(taxonomy)
    splits = {}
    for name in ("train", "aux_search", "validation"):
        source = data / f"{name}.parquet"
        actual = sha256(source)
        if actual != manifest["parquet_sha256"][name]:
            raise ValueError(f"Split hash mismatch: {source}")
        files[str(source)] = actual
        splits[name] = pq.read_table(source).to_pylist()
    ids = validate_splits(splits)
    for name, rows in splits.items():
        if len(rows) != manifest["splits"][name]["rows"] or ids[name] != manifest["splits"][name]["problem_ids"]:
            raise ValueError(f"Split identity/count mismatch: {name}")
    if len(splits["train"]) != c["data"]["train_questions"] or len(splits["aux_search"]) != c["data"]["archive_questions"]:
        raise ValueError("Prepared question counts disagree with experiment config")
    if c["domain"] == "bird":
        train_db = {r["extra_info"]["db_id"] for r in splits["train"]}
        val_db = {r["extra_info"]["db_id"] for r in splits["validation"]}
        if train_db & val_db:
            raise ValueError("BIRD validation databases overlap training")
    if require_offline and c["method"] in REPLAY_METHODS:
        from .replay import Replay
        root = offline_dir(c)
        complete = json.loads((root / "complete.json").read_text())
        if complete.get("validation_only") and not c["runtime"].get("validation_only"):
            raise ValueError("Diagnostic replay requires runtime.validation_only=true")
        if complete["config_sha256"] != offline_fingerprint(c):
            raise ValueError("Offline artifacts were prepared with a different experiment config")
        for split, key in (("train", "train_sha256"), ("aux_search", "aux_sha256")):
            if complete[key] != files[str(data / f"{split}.parquet")]:
                raise ValueError("Offline artifacts belong to different training splits")
        filenames = {"references.jsonl": "references_sha256", "exploration.parquet": "exploration_sha256"} if c["method"] in DPH_METHODS else {"archive.jsonl": "archive_sha256"}
        for name, key in filenames.items():
            actual = sha256(root / name)
            if actual != complete[key]:
                raise ValueError(f"Offline file hash mismatch: {name}")
            files[str(root / name)] = actual
        replay_path = root / ("references.jsonl" if c["method"] in DPH_METHODS else "archive.jsonl")
        rows = [json.loads(line) for line in replay_path.read_text().splitlines() if line.strip()]
        if any(r.get("metadata", {}).get("validation_only") for r in rows) and not c["runtime"].get("validation_only"):
            raise ValueError("Diagnostic replay rows cannot be used for full training")
        kind = {"EliteReplay": "qd", "IID_Replay": "iid", "dph_js": "dph_js", "dph_f": "dph_f"}[c["method"]]
        replay = Replay(rows, method=kind, seed=c["seed"],
                        sampling=c["method_options"].get("replay_sampling", "question_random"))
        allowed = set(ids["train"] if c["method"] in DPH_METHODS else ids["aux_search"])
        if not set(replay.ids) <= allowed:
            raise ValueError("Replay contains a question outside its authorized training pool")
        if c["domain"] == "optmath" and c["method"] == "EliteReplay":
            effective_path = validate_optmath_archive(c, root, complete, len(rows))
            files[str(effective_path)] = sha256(effective_path)
        elif c["method"] not in DPH_METHODS and len(rows) != c["offline"]["export_target"]:
            raise ValueError("Archive has not reached the matched export target")
        if c["method"] in DPH_METHODS:
            exploration = pq.read_table(root / "exploration.parquet").to_pylist()
            exp_ids = {r["extra_info"]["problem_id"] for r in exploration}
            if exp_ids & set(replay.ids) or exp_ids | set(replay.ids) != allowed:
                raise ValueError("Invalid DPH exploration/preservation partition")
        files[str(root / "complete.json")] = sha256(root / "complete.json")
    return files


def preflight(c, *, require_offline=True):
    errors = []
    required = [path(c["model"]["path"]) / "config.json", path(c["model"]["path"]) / "tokenizer_config.json"]
    if c["runtime"]["launcher"] == "apptainer":
        required.append(path(c["runtime"]["image"]))
        if not shutil.which("apptainer"):
            errors.append("apptainer executable is unavailable")
    if c["domain"] == "bird":
        required += [path(c["verifier"]["bird_database_root"]), path(c["verifier"]["bird_evaluator_root"])]
    elif c["domain"] == "taco":
        required.append(path(c["verifier"]["taco_image"]))
    elif c["domain"] == "optmath":
        required += [path(c["verifier"][k]) for k in ("apptainer_image", "gurobipy_path", "highspy_path")]
        try:
            from .offline import adapter
            adapter(c).preflight()
        except (ValueError, OSError, RuntimeError) as error:
            errors.append(f"OptMATH verifier: {error}")
    if c["method"] == "setpo":
        required.append(path(c["method_options"]["embedding_model"]) / "config.json")
    for item in required:
        if not item.exists():
            errors.append(f"Missing {item}")
    model_path = path(c["model"]["path"])
    if model_path.name != c["model"]["revision"]:
        errors.append("model.path must point to the pinned Hugging Face snapshot revision")
    for directory, commit in ((ROOT / "Trinity-RFT", TRINITY_COMMIT), (ROOT / "verl_fork", VERL_COMMIT)):
        try:
            head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
            if head != commit:
                errors.append(f"Runtime checkout {directory} must be at {commit}")
        except subprocess.CalledProcessError:
            errors.append(f"Missing runtime checkout {directory}")
    try:
        upstream_check()
        runtime_check()
    except (ValueError, OSError) as error:
        errors.append(str(error))
    if require_offline:
        from . import kernels
        try:
            kernels.check(kernels.directory(c))
        except (ValueError, OSError, KeyError) as error:
            errors.append(str(error))
    try:
        files = inputs(c, require_offline=require_offline)
    except (ValueError, OSError, KeyError) as error:
        errors.append(f"Inputs: {error}. Prepare shared data, then run this method's prepare command if needed.")
        files = {}
    return {"ready": not errors, "errors": errors, "input_sha256": files}










def source_hashes(directory):
    return {str(f.relative_to(directory)): sha256(f) for f in sorted(directory.rglob("*"))
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"}








def bundle(c, original_config, *, smoke=False, offline=False):
    parent = run_dir(c)
    parent.mkdir(parents=True, exist_ok=True)
    name = "prepare-bundle" if offline else "smoke-bundle" if smoke else "train-bundle"
    target = parent / name
    expected = {"config_sha256": fingerprint(c), "input_sha256": inputs(c, require_offline=not offline)}
    if target.exists():
        manifest = json.loads((target / "manifest.json").read_text())
        if not offline and manifest.get("runtime_contract") != RUNTIME_CONTRACT:
            raise ValueError("This frozen training bundle predates the current runtime fixes. "
                             "Without a production checkpoint, run refresh-runtime, then smoke. "
                             "Production checkpoints require an explicit migration or their original runtime.")
        if any(manifest[k] != v for k, v in expected.items()):
            raise ValueError("Run bundle inputs changed; choose a new output_root")
        for filename, digest in manifest["source_sha256"].items():
            if sha256(target / filename) != digest:
                raise ValueError(f"Frozen run source changed: {filename}")
        return target
    staging = parent / (name + f".building-{os.getpid()}")
    staging.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for source, destination in ((ROOT / "src/evorl", "src/evorl"),
            (ROOT / "Trinity-RFT/trinity", "Trinity-RFT/trinity"), (ROOT / "verl_fork/verl", "verl_fork/verl")):
        shutil.copytree(source, staging / destination, ignore=ignore)
    (staging / "plugins").mkdir()
    shutil.copy2(ROOT / "plugins/elitereplay_trinity.py", staging / "plugins/elitereplay_trinity.py")
    (staging / "plugin_entry").mkdir()
    (staging / "plugin_entry/baseline_suite.py").write_text("from evorl.baselines.integration import *\n")
    (staging / "verl_fork/verl/version/version").write_text("0.8.0\n")
    if not offline:
        from . import kernels
        kernel_root = kernels.directory(c)
        kernels.check(kernel_root)
        shutil.copytree(kernel_root / "site-packages", staging / "kernel_deps/site-packages", ignore=ignore)
        shutil.copy2(kernel_root / "manifest.json", staging / "kernel_deps/manifest.json")
        shutil.copy2(kernels.LOCK, staging / "kernels.lock.json")
    shutil.copy2(original_config, staging / "experiment.original.yaml")
    (staging / "experiment.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    (staging / "trinity.yaml").write_text(yaml.safe_dump(trinity_config(c, smoke=smoke), sort_keys=False))
    shutil.copy2(ROOT / "experiments/baselines/upstream.lock.json", staging / "upstream.lock.json")
    shutil.copy2(ROOT / "experiments/baselines/runtime/runtime.lock.json", staging / "runtime.lock.json")
    expected.update(runtime_contract=RUNTIME_CONTRACT, source_sha256=source_hashes(staging),
                    runtime_base_commits={"trinity": TRINITY_COMMIT, "verl": VERL_COMMIT})
    (staging / "manifest.json").write_text(json.dumps(expected, indent=2) + "\n")
    staging.rename(target)
    return target


def refresh_failed_runtime(c):
    """Archive failed startup state, preserving offline data and all evidence.

    The caller must hold the run's exclusive .runner.lock. Preserve diagnostic
    smoke checkpoints, but never migrate completed production checkpoints.
    """
    root = run_dir(c)
    candidates = [root / name for name in ("smoke-bundle", "train-bundle", "smoke", "train")]
    for directory in (root / "train-bundle", root / "train"):
        if directory.exists() and (any(directory.rglob(".full_checkpoint")) or
                                   any(directory.rglob("latest_checkpointed_iteration.txt"))):
            raise ValueError(f"Completed checkpoint under {directory}; refusing runtime replacement")
    saved = root / "runtime-recovery" / str(time.time_ns())
    moved = []
    for directory in candidates:
        if directory.exists():
            saved.mkdir(parents=True, exist_ok=True)
            destination = saved / directory.name
            directory.rename(destination)
            moved.append(str(destination))
    return {"status": "failed_runtime_preserved", "archived": moved,
            "offline_data_unchanged": True}


def migrate_performance_runtime(c, *, from_contract=4):
    """Explicit 4 -> 5 or 5 -> 6 migration, under the caller's exclusive run lock.

    Preserve every production checkpoint and offline artifact in place. Archive
    the old executable bundles and smoke evidence, then require a new smoke run.
    Scientific config changes, different inputs and incomplete checkpoints fail
    before any rename. Never call this against a running experiment.
    """
    root = run_dir(c)
    old_bundle = root / "train-bundle"
    manifest = json.loads((old_bundle / "manifest.json").read_text())
    if manifest.get("runtime_contract") != from_contract or (from_contract, RUNTIME_CONTRACT) not in ((4, 5), (5, 6)):
        raise ValueError("Only reviewed runtime-contract 4 -> 5 and 5 -> 6 migrations are supported")
    expected = {"config_sha256": fingerprint(c), "input_sha256": inputs(c)}
    if any(manifest[k] != v for k, v in expected.items()):
        raise ValueError("Migration requires identical experiment configuration and input data")
    for filename, digest in manifest["source_sha256"].items():
        if sha256(old_bundle / filename) != digest:
            raise ValueError(f"Frozen source changed before migration: {filename}")
    trinity = trinity_config(c)
    if yaml.safe_load((old_bundle / "trinity.yaml").read_text()) != trinity:
        raise ValueError("Compiled training configuration changed; this is not a performance-only migration")
    job = Path(trinity["checkpoint_root_dir"]) / trinity["project"] / trinity["group"] / trinity["name"]
    step = int((job / "latest_checkpointed_iteration.txt").read_text().strip())
    checkpoint = job / f"global_step_{step}"
    if not 0 < step < trinity["trainer"]["total_steps"] or not (checkpoint / ".full_checkpoint").is_file():
        raise ValueError("Migration requires an unfinished run with a complete production checkpoint")
    world = c["runtime"]["trainer_gpus"]
    checkpoint_files = {}
    for rank in range(world):
        for kind in ("model", "optim", "extra_state"):
            file = checkpoint / "actor" / f"{kind}_world_size_{world}_rank_{rank}.pt"
            if not file.is_file() or file.stat().st_size == 0:
                raise ValueError(f"Missing checkpoint shard: {file}")
            checkpoint_files[str(file.relative_to(root))] = file.stat().st_size
    saved = root / "runtime-migrations" / str(time.time_ns())
    saved.mkdir(parents=True)
    record = {"status": "preparing", "from_contract": from_contract, "to_contract": RUNTIME_CONTRACT,
              "config_sha256": fingerprint(c), "checkpoint_iteration": step,
              "checkpoint_files_bytes": checkpoint_files, "archived": [],
              "production_checkpoints_unchanged": True, "offline_data_unchanged": True}
    def save():
        (saved / "migration.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        for name in ("train-bundle", "smoke-bundle", "smoke"):
            directory = root / name
            if directory.exists():
                directory.rename(saved / name)
                record["archived"].append(name)
                save()
    except BaseException:
        for name in reversed(record["archived"]):
            (saved / name).rename(root / name)
        record["status"] = "rolled_back"
        save()
        raise
    record["status"] = "requires_new_smoke"
    save()
    return {**record, "record": str(saved / "migration.json")}


def migrate_batch_timeout(c):
    """Change only a stopped run's batch wait budget; preserve optimizer/data state."""
    root = run_dir(c)
    old_bundle = root / "train-bundle"
    previous = yaml.safe_load((old_bundle / "experiment.yaml").read_text())
    comparable = copy.deepcopy(c)
    old_timeout = previous["runtime"].get("batch_timeout_seconds", previous["runtime"]["sync_timeout_seconds"])
    new_timeout = comparable["runtime"].pop("batch_timeout_seconds", None)
    prior_comparable = copy.deepcopy(previous)
    prior_comparable["runtime"].pop("batch_timeout_seconds", None)
    if comparable != prior_comparable or type(new_timeout) is not int or new_timeout <= old_timeout:
        raise ValueError("Timeout migration permits only an increased batch_timeout_seconds")
    manifest = json.loads((old_bundle / "manifest.json").read_text())
    if (manifest.get("runtime_contract") != RUNTIME_CONTRACT
            or manifest["config_sha256"] != fingerprint(previous)
            or manifest["input_sha256"] != inputs(c)):
        raise ValueError("Timeout migration requires unchanged inputs and current runtime")
    for filename, digest in manifest["source_sha256"].items():
        if sha256(old_bundle / filename) != digest:
            raise ValueError(f"Frozen source changed before migration: {filename}")
    old_trinity = yaml.safe_load((old_bundle / "trinity.yaml").read_text())
    new_trinity = trinity_config(c)
    allowed = copy.deepcopy(new_trinity)
    allowed["buffer"]["trainer_input"]["experience_buffer"]["max_read_timeout"] = old_timeout
    allowed["synchronizer"]["sync_timeout"] = old_timeout
    if old_trinity != allowed:
        raise ValueError("Compiled configuration changed beyond the two batch wait limits")
    job = Path(old_trinity["checkpoint_root_dir"]) / old_trinity["project"] / old_trinity["group"] / old_trinity["name"]
    step = int((job / "latest_checkpointed_iteration.txt").read_text().strip())
    checkpoint = job / f"global_step_{step}"
    if not 0 < step < c["training"]["rollout_iterations"] or not (checkpoint / ".full_checkpoint").is_file():
        raise ValueError("A complete unfinished production checkpoint is required")
    for rank in range(c["runtime"]["trainer_gpus"]):
        for kind in ("model", "optim", "extra_state"):
            shard = checkpoint / "actor" / f"{kind}_world_size_{c['runtime']['trainer_gpus']}_rank_{rank}.pt"
            if not shard.is_file() or not shard.stat().st_size:
                raise ValueError(f"Missing checkpoint shard: {shard}")
    saved = root / "runtime-migrations" / f"batch-timeout-{time.time_ns()}"
    saved.mkdir(parents=True)
    record = {"old_batch_timeout_seconds": old_timeout, "new_batch_timeout_seconds": new_timeout,
              "checkpoint_iteration": step, "archived": [], "production_checkpoints_unchanged": True,
              "offline_data_unchanged": True, "status": "preparing"}
    def save():
        (saved / "migration.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        for name in ("train-bundle", "smoke-bundle", "smoke", "prepare-bundle"):
            directory = root / name
            if directory.exists():
                directory.rename(saved / name)
                record["archived"].append(name)
                save()
    except BaseException:
        for name in reversed(record["archived"]):
            (saved / name).rename(root / name)
        record["status"] = "rolled_back"
        save()
        raise
    record["status"] = "requires_new_smoke"
    save()
    return {**record, "record": str(saved / "migration.json")}


def require_completed_smoke(c, train_bundle):
    """A full run must use the same tested inputs and code as its smoke run."""
    if c["runtime"].get("validation_only"):
        raise ValueError("Diagnostic configurations cannot start full training")
    root = run_dir(c)
    smoke_bundle = root / "smoke-bundle"
    cfg = trinity_config(c, smoke=True)
    job = Path(cfg["checkpoint_root_dir"]) / cfg["project"] / cfg["group"] / cfg["name"]
    pointer = job / "latest_checkpointed_iteration.txt"
    if (not pointer.is_file() or pointer.read_text().strip() != "2"
            or not (job / "global_step_2/.full_checkpoint").is_file()):
        raise ValueError("Full training requires a completed two-iteration smoke run; run smoke first")
    smoke = json.loads((smoke_bundle / "manifest.json").read_text())
    train = json.loads((train_bundle / "manifest.json").read_text())
    for key in ("config_sha256", "input_sha256", "runtime_contract"):
        if smoke.get(key) != train.get(key):
            raise ValueError(f"Smoke/training {key} differ; run a smoke test for these exact inputs")
    def code_hashes(manifest):
        return {k: v for k, v in manifest["source_sha256"].items()
                if k.startswith(("src/", "Trinity-RFT/", "verl_fork/", "plugins/", "plugin_entry/"))
                or k.startswith("kernel_deps/") or k.endswith(".lock.json")}
    if code_hashes(smoke) != code_hashes(train):
        raise ValueError("Training source differs from the completed smoke; use a new output_root and smoke again")
    if c["domain"] == "optmath":
        for rank in range(c["runtime"]["trainer_gpus"]):
            report = json.loads((root / "allocation-validation" / f"stress-rank{rank}.json").read_text())
            if (report.get("status") != "passed" or report.get("config_sha256") != fingerprint(c)
                    or report.get("train_bundle_manifest_sha256") != sha256(Path(train_bundle) / "manifest.json")
                    or len(report.get("cases", [])) != 2):
                raise ValueError("OptMATH full-length checkpoint-reload check missing or stale")


def reconcile(c, trinity):
    """Rewind to the last full checkpoint; retain displaced state for inspection."""
    job = Path(trinity["checkpoint_root_dir"]) / trinity["project"] / trinity["group"] / trinity["name"]
    queue = Path(trinity["buffer"]["trainer_input"]["experience_buffer"]["path"].removeprefix("sqlite:///"))
    pointer = job / "latest_checkpointed_iteration.txt"
    stamp = f"pre-resume-{time.time_ns()}"
    def preserve(file):
        if file.exists():
            file.rename(file.with_name(file.name + "." + stamp))
    if not pointer.exists():
        preserve(job)
        step = 0
    else:
        step = int(pointer.read_text().strip())
        if not 0 <= step <= trinity["trainer"]["total_steps"] or not (job / f"global_step_{step}/.full_checkpoint").exists():
            raise ValueError("Invalid completed-checkpoint pointer; inspect before resuming")
        if step == trinity["trainer"]["total_steps"]:
            return {"status": "complete", "iteration": step}
        for filename in ("trainer_meta.json", "explorer_meta.json", "latest_state_dict_iteration.txt"):
            preserve(job / filename)
        for file in job.glob("global_step_*"):
            suffix = file.name.removeprefix("global_step_")
            if suffix.isdigit() and int(suffix) > step:
                preserve(file)
        replay_hash = None
        if c["method"] in REPLAY_METHODS:
            from .replay import Replay
            kind = {"EliteReplay": "qd", "IID_Replay": "iid", "dph_js": "dph_js", "dph_f": "dph_f"}[c["method"]]
            source = trinity["algorithm"]["sample_strategy_args"]["replay_path"]
            rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
            replay_hash = Replay(rows, method=kind, seed=c["seed"]).fingerprint
        (job / "trainer_meta.json").write_text(json.dumps({"latest_iteration": step,
            "sample_strategy_state": {"reader": {"current_index": 0}, "replay_fingerprint": replay_hash}}))
        (job / "explorer_meta.json").write_text(json.dumps({"latest_iteration": step,
            "taskset_states": [{"current_index": step * trinity["buffer"]["batch_size"]}]}))
        (job / "latest_state_dict_iteration.txt").write_text(str(step) + "\n")
    for suffix in ("", "-wal", "-shm"):
        preserve(Path(str(queue) + suffix))
    return {"status": "resume" if step else "restart_from_initialization", "iteration": step}


def launch(c, directory, *, offline=False, seeds_only=False, stress=False):
    """Called only by explicit execution commands; never by plan/check."""
    env = os.environ.copy()
    values = {"PYTHONPATH": ":".join(map(str, [directory / "kernel_deps/site-packages",
        directory / "verl_fork", directory / "Trinity-RFT",
        ROOT / "external/trinity-runtime", ROOT / "external/python-deps", directory / "src"])),
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(ROOT / "external/hf-cache"), "WANDB_MODE": c["runtime"]["wandb_mode"],
        "TRINITY_PLUGIN_DIRS": str(directory / "plugin_entry"),
        "ELITEREPLAY_TOKENIZER_PATH": c["model"]["path"],
        **({"BIRD_DATABASE_ROOT": c["verifier"]["bird_database_root"],
            "BIRD_EVALUATOR_ROOT": c["verifier"]["bird_evaluator_root"],
            "BIRD_SQL_TIMEOUT_SECONDS": str(c["verifier"]["bird_timeout_seconds"])}
           if c["domain"] != "optmath" else {}),
        "NCCL_P2P_DISABLE": "0" if c["runtime"].get("nccl_p2p", True) else "1",
        "NCCL_CUMEM_ENABLE": "0", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "RAY_USAGE_STATS_ENABLED": "0",
        # Ray 2.53 initializes OpenTelemetry asynchronously, racing worker
        # setenv with native getenv (observed QueueStorage SIGSEGV). Disable
        # that exporter before Ray starts; trainer/TensorBoard metrics remain.
        "RAY_enable_open_telemetry": "0"}
    env.update(values)
    command = ["python", "-m", "evorl.baselines.execute", str(directory),
               "prepare-seeds" if seeds_only else "prepare" if offline else "train"]
    if stress:
        command = ["torchrun", "--standalone", "--nnodes=1", f"--nproc-per-node={c['runtime']['trainer_gpus']}",
                   "-m", "evorl.baselines.optmath_stress" if c["domain"] == "optmath" else "evorl.baselines.recovery_stress", str(directory)]
    if c["runtime"]["launcher"] == "apptainer":
        for key, value in values.items():
            env["APPTAINERENV_" + key] = value
        for key in ("CUDA_VISIBLE_DEVICES", "AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION",
                    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "WANDB_API_KEY",
                    "LITELLM_API_KEY", "LITELLM_BASE_URL", "API_ENDPOINT"):
            if key in env:
                env["APPTAINERENV_" + key] = env[key]
        env["APPTAINERENV_SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
        binds = ["--bind", f"{ROOT}:{ROOT}:rw"]
        extra_paths = [(ROOT / "external").resolve(), Path(c["output_root"]), path(c["model"]["path"]), path(c["data"]["root"]),
                       path(c["method_options"]["embedding_model"])]
        extra_paths += [path(c["verifier"][k]) for k in
                        ("bird_database_root", "bird_evaluator_root", "taco_image", "apptainer_image", "gurobipy_path")
                        if k in c["verifier"]]
        if c["domain"] == "optmath":
            extra_paths.append(path(c["verifier"]["highspy_path"]))
        # Offline descriptors may live beside the campaign, outside both the
        # source snapshot and the checkpoint output root (e.g. scratch recovery).
        for key in ("iid_taxonomy_path", "source_dir"):
            if c["offline"].get(key):
                extra_paths.append(path(c["offline"][key]))
        for source in extra_paths:
            if source.exists() and not source.is_relative_to(ROOT):
                binds += ["--bind", f"{source}:{source}:rw"]
        if c["domain"] in ("taco", "optmath"):
            pairs = [("/usr/bin/apptainer", "/usr/bin/apptainer"),
                ("/usr/libexec/apptainer", "/usr/libexec/apptainer"), ("/etc/apptainer", "/etc/apptainer"),
                ("/var/apptainer", "/var/apptainer"), ("/usr/bin/fuse-overlayfs", "/usr/bin/fuse-overlayfs"),
                ("/usr/bin/squashfuse", "/usr/bin/squashfuse"), ("/usr/bin/squashfuse_ll", "/usr/bin/squashfuse_ll")]
            pairs += [(f"/lib64/{name}", f"/lib/x86_64-linux-gnu/{name}") for name in
                      ("libsquashfuse.so.0", "libfuse.so.2", "libfuse3.so.3")]
            for source, destination in pairs:
                if not Path(source).exists():
                    raise FileNotFoundError(f"Required nested TACO sandbox component: {source}")
                binds += ["--bind", f"{source}:{destination}"]
        command = ["apptainer", "exec", "--nv", "--cleanenv", *binds, c["runtime"]["image"], *command]
    subprocess.run(command, env=env, cwd=ROOT, check=True)
    if not offline and not stress:
        trinity = yaml.safe_load((directory / "trinity.yaml").read_text())
        job = Path(trinity["checkpoint_root_dir"]) / trinity["project"] / trinity["group"] / trinity["name"]
        expected = trinity["trainer"]["total_steps"]
        pointer = job / "latest_checkpointed_iteration.txt"
        if not pointer.is_file() or int(pointer.read_text().strip()) != expected or not (job / f"global_step_{expected}/.full_checkpoint").is_file():
            raise RuntimeError("Training process exited without its expected final complete checkpoint")
