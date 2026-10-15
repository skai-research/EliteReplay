"""File-only configuration for the BIRD/TACO baseline suite.

No hyperparameters are accepted on the command line. Every run saves the
input YAML and the fully resolved Trinity YAML. Paths are relative to the
repository root, so invoking a runner from another directory is safe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("grpo", "maxrl", "setpo", "dph_f", "dph_js", "IID_Replay", "EliteReplay")
DPH_METHODS = ("dph_f", "dph_js")
REPLAY_METHODS = ("IID_Replay", "EliteReplay", *DPH_METHODS)


def path(value, root=ROOT):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(filename):
    value = yaml.safe_load(Path(filename).read_text())
    validate(value)
    return value


def validate(c):
    allowed = {"schema_version", "method", "domain", "seed", "output_root", "model", "data",
               "training", "method_options", "offline", "runtime", "verifier"}
    if not isinstance(c, dict) or set(c) != allowed or c["schema_version"] != 1:
        raise ValueError(f"Expected schema_version=1 and exactly these top-level keys: {sorted(allowed)}")
    if c["method"] not in METHODS or c["domain"] != "bird":
        raise ValueError("This public training release supports BIRD and the listed baseline methods")
    fields = {
        "model": {"path", "revision", "max_prompt_tokens", "max_response_tokens", "enable_thinking"},
        "data": {"root", "train_questions", "archive_questions"},
        "training": {"rollout_iterations", "prompts_per_iteration", "rollouts_per_prompt",
                     "prompts_per_optimizer_update", "replay_per_iteration", "learning_rate",
                     "warmup_iterations", "weight_decay", "betas", "epsilon", "grad_clip",
                     "clip_low", "clip_high", "temperature", "top_p", "top_k", "min_p",
                     "presence_penalty", "repetition_penalty", "save_every"},
        "runtime": {"launcher", "image", "trainer_gpus", "rollout_gpus", "max_tokens_per_gpu",
                    "rollout_memory_fraction", "max_num_seqs", "rollout_workers", "cpus",
                    "sync_timeout_seconds", "monitor", "wandb_mode"},
        "method_options": {"sft_weight", "js_weight", "setpo_weight", "setpo_tau", "embedding_model",
                           "embedding_device", "embedding_batch_size", "embedding_max_tokens",
                           "presample_rollouts", "preservation_threshold", "max_resample_groups"},
        "offline": {"candidates_per_question", "seed_samples", "rounds", "children_per_round",
                    "export_target", "minimum_examples", "per_question_cap", "generation_batch_size",
                    "taxonomy_size", "taxonomy_sample_cap", "feasible_labels", "label_confidence",
                    "labeler_model", "label_workers", "verifier_workers"},
        "verifier": {"bird_database_root", "bird_evaluator_root", "bird_timeout_seconds",
                     "taco_image", "wall_timeout_seconds", "per_test_timeout_seconds",
                     "cpu_seconds", "memory_bytes", "max_response_bytes"},
    }
    if c["domain"] == "optmath":
        fields["verifier"] = {"apptainer_image", "gurobipy_path", "highspy_path",
                              "wall_timeout_seconds", "max_response_bytes"}
    if "compact_zero_loss" in c["method_options"]:
        fields["method_options"].add("compact_zero_loss")
        if type(c["method_options"]["compact_zero_loss"]) is not bool:
            raise ValueError("compact_zero_loss must be boolean")
    if c["method"] == "dph_f":
        fields["method_options"].add("forward_weight")
    if "dynamic_sampling" in c["method_options"]:
        fields["method_options"].add("dynamic_sampling")
        if c["method"] not in DPH_METHODS or type(c["method_options"]["dynamic_sampling"]) is not bool:
            raise ValueError("dynamic_sampling requires a DPH method and a boolean")
    for name in ("labeler_region", "labeler_provider", "labeler_api_base"):
        if name in c["offline"]:
            fields["offline"].add(name)
    provider = c["offline"].get("labeler_provider", "bedrock")
    if provider not in ("litellm", "bedrock"):
        raise ValueError("Unknown offline.labeler_provider")
    if provider == "litellm":
        from evorl.litellm import base_url
        if not c["offline"].get("labeler_api_base"):
            raise ValueError("Pin offline.labeler_api_base for reproducible LiteLLM runs")
        base_url(c["offline"]["labeler_api_base"])
    for name, choices in {"sft_reduction": ("token_mean", "response_mean"),
                          "replay_sampling": ("question_random", "mode_cycles")}.items():
        if name in c["method_options"]:
            fields["method_options"].add(name)
            if c["method"] not in ("IID_Replay", "EliteReplay") or c["method_options"][name] not in choices:
                raise ValueError(f"Invalid CHORD {name}")
    for name in ("source_dir", "cell_descriptor", "attempt_aware"):
        if name in c["offline"]:
            fields["offline"].add(name)
            if c["method"] not in ("IID_Replay", "EliteReplay"):
                raise ValueError(f"offline.{name} is a CHORD archive option")
    if "source_dir" in c["offline"] and (not isinstance(c["offline"]["source_dir"], str) or not c["offline"]["source_dir"]):
        raise ValueError("offline.source_dir must be a nonempty path")
    if "iid_taxonomy_path" in c["offline"]:
        fields["offline"].add("iid_taxonomy_path")
        if c["method"] != "IID_Replay" or not isinstance(c["offline"]["iid_taxonomy_path"], str) or not c["offline"]["iid_taxonomy_path"]:
            raise ValueError("iid_taxonomy_path requires an IID control and a taxonomy file")
    if "seed_source_config" in c["offline"]:
        fields["offline"].add("seed_source_config")
        if c["method"] != "EliteReplay" or not isinstance(c["offline"]["seed_source_config"], str) or not c["offline"]["seed_source_config"]:
            raise ValueError("seed_source_config must identify a frozen QD seed configuration")
    if c["offline"].get("cell_descriptor", "primary") not in ("primary", "primary_coarse_v1"):
        raise ValueError("Unknown cell_descriptor")
    if type(c["offline"].get("attempt_aware", False)) is not bool:
        raise ValueError("attempt_aware must be boolean")
    for name in ("embedding_cpu_dtype", "embedding_cpu_threads"):
        if name in c["method_options"]:
            fields["method_options"].add(name)
    if c["method_options"].get("embedding_cpu_dtype", "float32") not in ("float32", "bfloat16"):
        raise ValueError("embedding_cpu_dtype must be float32 or bfloat16")
    threads = c["method_options"].get("embedding_cpu_threads", 2)
    if type(threads) is not int or not 0 < threads <= c["runtime"]["cpus"]:
        raise ValueError("embedding_cpu_threads must fit the CPU allocation")
    if "max_checkpoints_to_keep" in c["runtime"]:
        fields["runtime"].add("max_checkpoints_to_keep")
    if "kernel_directory" in c["runtime"]:
        fields["runtime"].add("kernel_directory")
        if not isinstance(c["runtime"]["kernel_directory"], str) or not c["runtime"]["kernel_directory"]:
            raise ValueError("runtime.kernel_directory must be a nonempty path")
    if "nccl_p2p" in c["runtime"]:
        fields["runtime"].add("nccl_p2p")
        if type(c["runtime"]["nccl_p2p"]) is not bool:
            raise ValueError("runtime.nccl_p2p must be a boolean")
    if "batch_timeout_seconds" in c["runtime"]:
        fields["runtime"].add("batch_timeout_seconds")
        value = c["runtime"]["batch_timeout_seconds"]
        if type(value) is not int or value < c["runtime"]["sync_timeout_seconds"]:
            raise ValueError("runtime.batch_timeout_seconds must be an integer at least sync_timeout_seconds")
    if "validation_only" in c["runtime"]:
        fields["runtime"].add("validation_only")
        if type(c["runtime"]["validation_only"]) is not bool:
            raise ValueError("runtime.validation_only must be a boolean")
    if "micro_batch_size_per_gpu" in c["runtime"]:
        fields["runtime"].add("micro_batch_size_per_gpu")
        size = c["runtime"]["micro_batch_size_per_gpu"]
        if type(size) is not int or size <= 0:
            raise ValueError("runtime.micro_batch_size_per_gpu must be a positive integer")
    for key, expected in fields.items():
        if not isinstance(c[key], dict):
            raise ValueError(f"{key} must be a mapping")
        if set(c[key]) != expected:
            raise ValueError(f"Invalid {key} keys; missing={expected-set(c[key])}, unknown={set(c[key])-expected}")
    t, r, o, m = c["training"], c["runtime"], c["offline"], c["method_options"]
    if type(r.get("max_checkpoints_to_keep", 0)) is not int or r.get("max_checkpoints_to_keep", 0) < 0:
        raise ValueError("max_checkpoints_to_keep must be a nonnegative integer")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not 0 < c["data"]["archive_questions"] <= c["data"]["train_questions"]:
        raise ValueError("Archive question count must fit within the training pool")
    for name in ("rollout_iterations", "prompts_per_iteration", "rollouts_per_prompt",
                 "prompts_per_optimizer_update", "save_every"):
        if type(t[name]) is not int or t[name] <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("trainer_gpus", "rollout_gpus", "max_tokens_per_gpu", "cpus"):
        if type(r[name]) is not int or r[name] <= 0:
            raise ValueError(f"runtime.{name} must be a positive integer")
    updates = t["prompts_per_iteration"] // t["prompts_per_optimizer_update"]
    if not updates or t["prompts_per_iteration"] % t["prompts_per_optimizer_update"]:
        raise ValueError("prompts_per_iteration must divide into complete optimizer batches")
    if t["prompts_per_optimizer_update"] % r["trainer_gpus"]:
        raise ValueError("prompts_per_optimizer_update must be divisible by trainer_gpus")
    if t["replay_per_iteration"] < 0 or t["replay_per_iteration"] % (updates * r["trainer_gpus"]):
        raise ValueError("Replay count must divide evenly across updates and trainer ranks")
    if (t["replay_per_iteration"] > 0) != (c["method"] in REPLAY_METHODS):
        raise ValueError("Only CHORD and DPH use replay, and they require a positive replay count")
    if t["rollouts_per_prompt"] < (3 if c["method"] == "setpo" else 2):
        raise ValueError("Insufficient rollouts for this estimator")
    if c["model"]["enable_thinking"] is not False:
        raise ValueError("This suite uses direct SQL/Python generation: enable_thinking must be false")
    if t["min_p"] != 0 or t["presence_penalty"] != 0 or t["repetition_penalty"] != 1:
        raise ValueError("This training backend supports neutral generation penalties only")
    if not 0 < t["temperature"] or not 0 < t["top_p"] <= 1 or t["learning_rate"] <= 0:
        raise ValueError("Invalid temperature, top_p or learning_rate")
    if not 0 < t["clip_low"] < 1 or t["clip_high"] <= 0 or t["epsilon"] != 1e-8:
        raise ValueError("Invalid PPO clipping or unsupported optimizer epsilon (must be 1e-8)")
    if r["launcher"] not in ("native", "apptainer") or r["monitor"] not in ("tensorboard", "wandb"):
        raise ValueError("Invalid runtime launcher/monitor")
    if r["wandb_mode"] not in ("offline", "online", "disabled"):
        raise ValueError("Invalid wandb_mode")
    if not 0 <= t["warmup_iterations"] <= t["rollout_iterations"] or t["weight_decay"] < 0 or t["grad_clip"] <= 0:
        raise ValueError("Invalid warmup, weight decay or gradient clipping")
    if len(t["betas"]) != 2 or any(not 0 <= b < 1 for b in t["betas"]):
        raise ValueError("AdamW betas must contain two values in [0, 1)")
    if type(t["top_k"]) is not int or t["top_k"] == 0 or t["top_k"] < -1:
        raise ValueError("top_k must be -1 (disabled) or a positive integer")
    if not 0 < r["rollout_memory_fraction"] < 1:
        raise ValueError("rollout_memory_fraction must lie between 0 and 1")
    for key in ("presample_rollouts", "preservation_threshold", "max_resample_groups", "embedding_batch_size", "embedding_max_tokens"):
        if type(m[key]) is not int or m[key] <= 0:
            raise ValueError(f"method_options.{key} must be a positive integer")
    if m["presample_rollouts"] < 2:
        raise ValueError("Presampling needs at least two responses per question")
    if not 0 <= m["sft_weight"] <= 1 or m["js_weight"] < 0 or m["setpo_weight"] < 0 or m["setpo_tau"] <= 0:
        raise ValueError("Invalid method coefficient")
    if c["method"] == "dph_f" and m["forward_weight"] < 0:
        raise ValueError("Invalid forward KL coefficient")
    if m["preservation_threshold"] != m["presample_rollouts"]:
        raise ValueError("DPH uses the all-correct variant; threshold must equal presample_rollouts")
    if o["candidates_per_question"] != o["seed_samples"] + o["rounds"] * o["children_per_round"]:
        raise ValueError("QD and IID candidate budgets must agree")
    if not 0 < o["minimum_examples"] <= o["export_target"] or o["per_question_cap"] < 1:
        raise ValueError("Invalid archive size/cap")
    for key in ("max_prompt_tokens", "max_response_tokens"):
        if type(c["model"][key]) is not int or c["model"][key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    if r["max_tokens_per_gpu"] < c["model"]["max_prompt_tokens"] + c["model"]["max_response_tokens"]:
        raise ValueError("max_tokens_per_gpu must fit a complete prompt+response")


def run_dir(c):
    return path(c["output_root"]) / c["domain"] / c["method"] / f"s{c['seed']}"


def offline_dir(c):
    if "source_dir" in c["offline"]:
        return path(c["offline"]["source_dir"])
    return run_dir(c) / "offline"


def offline_fingerprint(c):
    """Shared archives depend on generation inputs, never on the SFT coefficient.

    Keep the historical full-config contract for nonshared artifacts. Version 2
    makes source reuse explicit and pins all generation/verifier choices.
    """
    if "source_dir" not in c["offline"]:
        # This optional training wait budget cannot change offline generations.
        # Removing only the new field preserves existing legacy artifact hashes.
        legacy = {**c, "runtime": {k: v for k, v in c["runtime"].items()
                                  if k != "batch_timeout_seconds"}}
        return fingerprint(legacy)
    return fingerprint({"contract_version": 2, "method": c["method"], "domain": c["domain"],
        "seed": c["seed"], "model": c["model"], "data": c["data"], "verifier": c["verifier"],
        "offline": {k: v for k, v in c["offline"].items() if k not in ("source_dir", "seed_source_config")},
        "generation": {k: c["training"][k] for k in ("temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "repetition_penalty")}, "rollout_gpus": c["runtime"]["rollout_gpus"]})


def adapter_args(c):
    v = c["verifier"]
    args = {"max_response_bytes": v["max_response_bytes"]}
    if c["domain"] == "optmath":
        args["sandbox_config"] = {
            **{k: str(path(v[k])) for k in ("apptainer_image", "gurobipy_path", "highspy_path")},
            "workspace_root": str(run_dir(c) / "sandbox"),
            "artifact_root": str(run_dir(c) / "verifier_artifacts"),
            "wall_timeout_seconds": v["wall_timeout_seconds"],
        }
    if c["domain"] == "taco":
        args["sandbox_config"] = {
            "apptainer_image": str(path(v["taco_image"])),
            "workspace_root": str(run_dir(c) / "sandbox"),
            **{k: v[k] for k in ("wall_timeout_seconds", "per_test_timeout_seconds", "cpu_seconds", "memory_bytes")},
            "max_tests": None,
        }
    return args


def plan(c):
    t = c["training"]
    updates = t["prompts_per_iteration"] // t["prompts_per_optimizer_update"]
    return {"method": c["method"], "domain": c["domain"], "seed": c["seed"],
            "run_dir": str(run_dir(c)), "rollout_iterations": t["rollout_iterations"],
            "optimizer_updates": updates * t["rollout_iterations"],
            "online_responses_per_iteration": t["prompts_per_iteration"] * t["rollouts_per_prompt"],
            "total_accepted_online_responses": t["rollout_iterations"] * t["prompts_per_iteration"] * t["rollouts_per_prompt"],
            "total_replay_presentations": t["replay_per_iteration"] * t["rollout_iterations"],
            "dph_extra_rejection_samples": c["method"] in DPH_METHODS and c["method_options"].get("dynamic_sampling", True),
            "config_sha256": fingerprint(c)}


def trinity_config(c, *, smoke=False):
    """Compile explicit user settings; historical recipes are never inherited."""
    t, r, m, model = c["training"], c["runtime"], c["method_options"], c["model"]
    data = path(c["data"]["root"])
    directory = run_dir(c) / ("smoke" if smoke else "train")
    prompts = r["trainer_gpus"] * 2 if smoke else t["prompts_per_iteration"]
    update_prompts = r["trainer_gpus"] if smoke else t["prompts_per_optimizer_update"]
    replay = (prompts if t["replay_per_iteration"] else 0) if smoke else t["replay_per_iteration"]
    updates = prompts // update_prompts
    iterations = 2 if smoke else t["rollout_iterations"]
    # Qwen3.5's large vocabulary makes dense logits expensive. Accumulate one
    # complete sequence per GPU; the optimizer batch and loss normalization do
    # not change. Token-budget batching exceeded 80 GB in the second GPU step.
    micro_batch = r.get("micro_batch_size_per_gpu", 1)
    online = prompts * t["rollouts_per_prompt"]
    # Queue reads cover the whole batch, including queued waves of workers.
    # Keep each workflow bounded separately; DPH rejection sampling can make
    # full-batch assembly substantially slower than an individual workflow.
    batch_timeout = r.get("batch_timeout_seconds", r["sync_timeout_seconds"])
    aux_path = offline_dir(c) / ("references.jsonl" if c["method"] in DPH_METHODS else "archive.jsonl")
    train_path = offline_dir(c) / "exploration.parquet" if c["method"] in DPH_METHODS else data / "train.parquet"
    loss_args = {"method": c["method"], "dp_size": r["trainer_gpus"],
                 "optimizer_batch_size": (online + replay) // updates,
                 "sft_weight": m["sft_weight"], "js_weight": m["js_weight"],
                 "forward_weight": m.get("forward_weight", 0.01),
                 "clip_low": t["clip_low"], "clip_high": t["clip_high"]}
    if "sft_reduction" in m:
        loss_args["sft_reduction"] = m["sft_reduction"]
    rollout_args = {"temperature": t["temperature"], "top_p": t["top_p"],
                    "top_k": t["top_k"], "logprobs": 0, "max_tokens": model["max_response_tokens"]}
    return {
        "mode": "both", "project": "evorl-baselines", "group": c["domain"],
        "name": f"{c['method']}-s{c['seed']}", "checkpoint_root_dir": str(directory / "checkpoints"),
        "continue_from_checkpoint": True,
        "algorithm": {"algorithm_type": "grpo", "repeat_times": t["rollouts_per_prompt"],
            "optimizer": {"lr": t["learning_rate"], "lr_scheduler_type": "constant",
                          "lr_warmup_steps": t["warmup_iterations"], "weight_decay": t["weight_decay"],
                          "betas": t["betas"], "clip_grad": t["grad_clip"]},
            "kl_penalty_fn": "none", "kl_loss_fn": "none", "entropy_loss_fn": "none",
            "bypass_old_logprobs": True, "sample_strategy": "evorl_baselines",
            "sample_strategy_args": {"method": c["method"], "seed": c["seed"],
                "repeat_times": t["rollouts_per_prompt"], "online_count": online,
                "replay_count": replay, "replay_path": str(aux_path),
                "prompts_per_update": update_prompts, "dp_size": r["trainer_gpus"],
                "policy_model_path": str(path(model["path"])), **m,
                "embedding_model": str(path(m["embedding_model"]))},
            "policy_loss_fn": "evorl_baselines", "policy_loss_fn_args": loss_args},
        "model": {"model_path": str(path(model["path"])), "trust_remote_code": True,
            # veRL also uses this temperature when scoring training tokens.
            # Raw rollout/reference logprobs require unscaled trainer logits.
            "temperature": 1.0, "top_p": t["top_p"], "top_k": t["top_k"], "logprobs": 0,
            "max_prompt_tokens": model["max_prompt_tokens"], "max_response_tokens": model["max_response_tokens"],
            "max_model_len": model["max_prompt_tokens"] + model["max_response_tokens"],
            "enable_prompt_truncation": False, "enable_thinking": False},
        "cluster": {"node_num": 1, "gpu_per_node": r["trainer_gpus"] + r["rollout_gpus"]},
        "buffer": {"total_steps": iterations, "batch_size": prompts, "train_batch_size": online + replay,
            "explorer_input": {"taskset": {"name": "training", "storage_type": "EliteReplay_file",
                "path": str(train_path), "split": "train", "rollout_args": rollout_args,
                "workflow_args": {"domain": c["domain"], "adapter_args": adapter_args(c),
                    "dynamic_sampling": c["method"] in DPH_METHODS and m.get("dynamic_sampling", True), "exploration_path": str(train_path),
                    "seed": c["seed"], "max_resample_groups": m["max_resample_groups"]}},
                "eval_tasksets": [], "default_workflow_type": "evorl_binary_workflow"},
            "trainer_input": {"experience_buffer": {"name": "rollouts", "storage_type": "queue",
                "path": "sqlite:///" + str(directory / "buffer.db"), "max_read_timeout": batch_timeout}}},
        "explorer": {"eval_on_startup": False, "eval_interval": 25,
            "runner_per_model": min(r["rollout_workers"], prompts), "concurrent_mode": "sequential",
            "max_timeout": r["sync_timeout_seconds"],
            "rollout_model": {"engine_num": 1, "tensor_parallel_size": r["rollout_gpus"],
                "temperature": t["temperature"],
                "enable_prefix_caching": False, "enable_chunked_prefill": True, "enforce_eager": True,
                "gpu_memory_utilization": r["rollout_memory_fraction"], "dtype": "bfloat16", "seed": c["seed"],
                "extra_engine_args": {"max_num_seqs": r["max_num_seqs"], "logprobs_mode": "raw_logprobs",
                                      "limit_mm_per_prompt": {"image": 0, "video": 0}}}},
        "synchronizer": {"sync_method": "nccl", "sync_style": "fixed", "sync_interval": 1,
                         "sync_timeout": batch_timeout},
        "trainer": {"trainer_strategy": "fsdp2", "total_steps": iterations,
            "save_interval": 1 if smoke else t["save_every"], "save_hf_checkpoint": "last",
            "max_checkpoints_to_keep": r.get("max_checkpoints_to_keep", 0), "grad_clip": t["grad_clip"], "use_dynamic_bsz": micro_batch is None,
            "use_remove_padding": False, "balance_batch": False,
            "max_token_len_per_gpu": r["max_tokens_per_gpu"], "ulysses_sequence_parallel_size": 1,
            "use_torch_compile": False,
            # The pinned image has PyTorch SDPA, but no standalone flash_attn.
            # Dense, masked batches (use_remove_padding=False) support SDPA.
            "trainer_config": {"actor_rollout_ref": {
                "model": {"override_config": {"attn_implementation": "sdpa"}},
                # This must be explicit: Trinity's legacy defaults otherwise
                # overwrite the actor builder with a null optimizer override.
                "actor": {**({"ppo_micro_batch_size_per_gpu": micro_batch} if micro_batch else {}),
                    "optim": {"override_optimizer_config": {"foreach": False, "eps": t["epsilon"]}}},
                **({"ref": {"log_prob_micro_batch_size_per_gpu": micro_batch},
                    "rollout": {"log_prob_micro_batch_size_per_gpu": micro_batch}} if micro_batch else {})}}},
        "monitor": {"monitor_type": r["monitor"], "detailed_stats": True},
    }
