"""File-only benchmark protocol and model selection. No training imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import yaml

from elitereplay.project import ROOT
KS = (1, 2, 4, 8, 16, 32, 64, 128)


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(filename, value):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp-{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def load(filename):
    c = yaml.safe_load(Path(filename).read_text())
    expected = {"schema_version", "domain", "output_root", "data", "models", "sampling", "runtime", "verifier"}
    if not isinstance(c, dict) or set(c) != expected or c["schema_version"] != 1:
        raise ValueError("Invalid evaluation config schema")
    if c["domain"] not in ("bird", "spider"):
        raise ValueError("This release supports BIRD and Spider evaluation")
    c["runtime"].setdefault("overlap_verification", False)
    fields = {
        "data": {"split", "release", "source_file", "source_revision", "prepared_dir", "expected_questions", "training_data_root"},
        "sampling": {"samples_per_question", "k_values", "seed", "temperature", "top_p", "top_k", "max_prompt_tokens", "max_response_tokens", "enable_thinking", "questions_per_batch", "samples_per_block"},
        "runtime": {"launcher", "tensor_parallel_size", "gpu_memory_fraction", "max_num_seqs", "verifier_workers",
                    "enforce_eager", "enable_prefix_caching", "overlap_verification"},
        "verifier": {"bird_database_root", "bird_evaluator_root", "bird_timeout_seconds", "max_response_bytes"},
    }
    if c["runtime"].get("launcher") == "apptainer" or "image" in c["runtime"]:
        fields["runtime"].add("image")
    if c["domain"] == "spider":
        fields["verifier"] |= {"spider_database_root", "spider_timeout_seconds"}
        if c["verifier"].get("spider_timeout_seconds", 0) <= 0:
            raise ValueError("Positive Spider timeout required")
    if "smoke_problem_ids" in c["runtime"]:
        fields["runtime"].add("smoke_problem_ids")
        ids = c["runtime"]["smoke_problem_ids"]
        if not isinstance(ids, list) or len(ids) != 2 or any(not isinstance(x, str) for x in ids) or len(set(ids)) != 2:
            raise ValueError("runtime.smoke_problem_ids must contain two different official problem IDs")
    for key, keys in fields.items():
        if not isinstance(c[key], dict) or set(c[key]) != keys:
            raise ValueError(f"Invalid {key} keys: expected {sorted(keys)}")
    s, d, r = c["sampling"], c["data"], c["runtime"]
    if d["split"] not in ({"official_dev", "official_test"} if c["domain"] == "bird" else {"official_dev"} if c["domain"] == "spider" else {"official_test"}):
        raise ValueError("Use an explicit official split; train/validation panels are not main evaluation")
    if not isinstance(d["release"], str) or not d["release"].strip():
        raise ValueError("Record the dataset release")
    if (type(s["samples_per_question"]) is not int or s["samples_per_question"] not in (64, 128)
            or s["k_values"] != [k for k in KS if k <= s["samples_per_question"]]
            or s["enable_thinking"] is not False):
        raise ValueError("Main evaluation uses 64 or 128 direct samples and all power-of-two k values through that budget")
    for obj, keys in ((s, ("samples_per_block", "questions_per_batch", "max_prompt_tokens", "max_response_tokens")),
                      (r, ("tensor_parallel_size", "max_num_seqs", "verifier_workers")), (d, ("expected_questions",))):
        if any(type(obj[k]) is not int or obj[k] <= 0 for k in keys):
            raise ValueError("Counts and token limits must be positive integers")
    if s["samples_per_question"] % s["samples_per_block"] or not 0 < s["temperature"] or not 0 < s["top_p"] <= 1:
        raise ValueError("Invalid sampling settings")
    if type(s["seed"]) is not int or s["seed"] < 0 or type(s["top_k"]) is not int or s["top_k"] == 0 or s["top_k"] < -1:
        raise ValueError("Invalid seed/top_k")
    if r["launcher"] not in ("native", "apptainer") or not 0 < r["gpu_memory_fraction"] < 1:
        raise ValueError("Invalid runtime settings")
    if any(type(r[k]) is not bool for k in ("enforce_eager", "enable_prefix_caching", "overlap_verification")):
        raise ValueError("Inference optimization flags must be booleans")
    if not d["training_data_root"] or not d["prepared_dir"]:
        raise ValueError("Prepared evaluation and training provenance paths are required")
    if c["domain"] in ("bird", "spider") and not d["source_file"]:
        raise ValueError("BIRD needs the original official JSON path")
    v = c["verifier"]
    for key in ("bird_timeout_seconds", "max_response_bytes"):
        if not isinstance(v[key], (int, float)) or v[key] <= 0:
            raise ValueError(f"verifier.{key} must be positive")
    if not isinstance(c["models"], dict) or not c["models"]:
        raise ValueError("At least one named model is required")
    for name, m in c["models"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Model labels must be safe directory names")
        if set(m) != {"path", "checkpoint_marker"} or not m["path"]:
            raise ValueError("Models require path and optional checkpoint_marker")
    return c


def selected(c, name):
    if name not in c["models"]:
        raise ValueError(f"Unknown model {name!r}; choose one of {list(c['models'])}")
    # Other models' paths do not invalidate this model's resumable run.
    c = json.loads(json.dumps(c))
    c["models"] = {name: c["models"][name]}
    for section, keys in {"data": ["source_file", "prepared_dir", "training_data_root"],
                          "runtime": ["image"], "verifier": ["bird_database_root", "bird_evaluator_root"]}.items():
        for key in keys:
            if c[section].get(key) is not None:
                c[section][key] = str(path(c[section][key]))
    if c["domain"] == "spider":
        c["verifier"]["spider_database_root"] = str(path(c["verifier"]["spider_database_root"]))
    c["output_root"] = str(path(c["output_root"]))
    for key in ("path", "checkpoint_marker"):
        if c["models"][name][key] is not None:
            c["models"][name][key] = str(path(c["models"][name][key]))
    return c


def run_dir(c, name, smoke=False, *, smoke_samples=2):
    phase = (f"smoke{smoke_samples}" if smoke_samples != 2 else "smoke") if smoke else "main"
    return path(c["output_root"]) / c["domain"] / c["data"]["split"] / name / phase


def plan(c, name):
    return {"domain": c["domain"], "split": c["data"]["split"], "release": c["data"]["release"],
            "questions": c["data"]["expected_questions"], "model": name, "model_path": str(path(c["models"][name]["path"])),
            "samples_per_question": c["sampling"]["samples_per_question"],
            "total_samples": c["sampling"]["samples_per_question"] * c["data"]["expected_questions"],
            "k_values": c["sampling"]["k_values"], "output": str(run_dir(c, name)), "sampling": c["sampling"]}
